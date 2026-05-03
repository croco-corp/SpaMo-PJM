"""
MediaPipe keypoint extraction for PHOENIX-2014T.

Adapted from scripts/mediapipe_extract.py (PJM):
- Input: per-video PNG sequences {phoenix_root}/features/fullFrame-210x260px/{mode}/{key}/images*.png
- No LMDB crop_params (Phoenix is already signer-centered)
- Output: features/mediapipe_feat_p14t.h5  (same 258-dim schema as PJM mediapipe_feat_pjm.h5)

Usage:
    uv run python scripts/mediapipe_extract_p14t.py \\
        --phoenix-root /home/croco/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \\
        --output features/mediapipe_feat_p14t.h5 \\
        --num-workers 16
"""

import argparse
import logging
import os
import time
from multiprocessing import Process, Value
from pathlib import Path

import mediapipe as mp
import h5py
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    PoseLandmarker, PoseLandmarkerOptions,
    RunningMode,
)
from PIL import Image
import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POSE_DIM = 33 * 4
HAND_DIM = 21 * 3
FEAT_DIM = POSE_DIM + HAND_DIM * 2  # 258


def extract_landmarks(pose_result, hand_result) -> np.ndarray:
    vec = np.zeros(FEAT_DIM, dtype=np.float32)
    if pose_result.pose_landmarks:
        for i, lm in enumerate(pose_result.pose_landmarks[0]):
            vec[i * 4:i * 4 + 4] = [lm.x, lm.y, lm.z, lm.visibility]
    if hand_result.hand_landmarks:
        for landmarks, handedness_list in zip(hand_result.hand_landmarks, hand_result.handedness):
            label = handedness_list[0].category_name
            offset = POSE_DIM if label == "Left" else POSE_DIM + HAND_DIM
            for i, lm in enumerate(landmarks):
                vec[offset + i * 3:offset + i * 3 + 3] = [lm.x, lm.y, lm.z]
    return vec


def create_detectors(pose_model_path: str, hand_model_path: str):
    pose_det = PoseLandmarker.create_from_options(PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=pose_model_path),
        running_mode=RunningMode.IMAGE,
    ))
    hand_det = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=hand_model_path),
        num_hands=2,
        running_mode=RunningMode.IMAGE,
    ))
    return pose_det, hand_det


def list_p14t_videos(phoenix_root: str, modes=("train", "dev", "test")):
    base = Path(phoenix_root) / "features" / "fullFrame-210x260px"
    out = []
    for mode in modes:
        mdir = base / mode
        if not mdir.exists():
            continue
        for d in sorted(mdir.iterdir()):
            if d.is_dir():
                out.append((mode, d.name, d))
    return out


def run_worker(worker_id: int, video_chunk: list, part_path: str,
               already_done: set, flush_every: int, counter: Value,
               pose_model_path: str, hand_model_path: str):
    log = logging.getLogger(f"worker-{worker_id}")

    done = set(already_done)
    if os.path.exists(part_path):
        with h5py.File(part_path, "r") as f:
            done |= set(f.keys())

    pose_det, hand_det = create_detectors(pose_model_path, hand_model_path)

    newly = errors = 0
    start = time.time()

    with h5py.File(part_path, "a") as hf:
        for mode, key, frame_dir in video_chunk:
            if key in done:
                continue
            try:
                png_paths = sorted(frame_dir.glob("images*.png"))
                if not png_paths:
                    raise ValueError(f"No PNG frames in {frame_dir}")

                feats = []
                for p in png_paths:
                    rgb = np.array(Image.open(p).convert("RGB"))
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    pose_result = pose_det.detect(mp_img)
                    hand_result = hand_det.detect(mp_img)
                    feats.append(extract_landmarks(pose_result, hand_result))

                arr = np.stack(feats, axis=0)
                hf.create_dataset(key, data=arr, dtype="float32")
                newly += 1
                with counter.get_lock():
                    counter.value += 1

                if newly % flush_every == 0:
                    hf.flush()
                    elapsed = time.time() - start
                    log.info(f"{newly} done | {elapsed/60:.1f}min")

            except Exception as e:
                errors += 1
                log.warning(f"Error {key}: {e}")

        hf.flush()

    pose_det.close()
    hand_det.close()
    elapsed = time.time() - start
    log.info(f"Finished: newly={newly:,} errors={errors} time={elapsed/60:.1f}min")


def merge_parts(output_path: str, part_paths: list):
    logger.info("Merging parts...")
    with h5py.File(output_path, "a", locking=False) as hf_out:
        hf_out.attrs.setdefault("feat_dim", FEAT_DIM)
        hf_out.attrs.setdefault("description",
                                "PHOENIX-2014T: pose(33x4) + left_hand(21x3) + right_hand(21x3)")
        for part in part_paths:
            if not os.path.exists(part):
                continue
            with h5py.File(part, "r", locking=False) as hf_in:
                for key in hf_in:
                    if key not in hf_out:
                        hf_in.copy(key, hf_out)
            os.remove(part)
    logger.info("Merge done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phoenix-root",
                        default="/home/croco/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T")
    parser.add_argument("--output", default="features/mediapipe_feat_p14t.h5")
    parser.add_argument("--flush-every", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    parser.add_argument("--pose-model", default="models/pose_landmarker_full.task")
    parser.add_argument("--hand-model", default="models/hand_landmarker.task")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    videos = list_p14t_videos(args.phoenix_root)
    logger.info(f"Total Phoenix videos: {len(videos):,}")

    processed = set()
    if os.path.exists(args.output):
        with h5py.File(args.output, "r") as f:
            processed = set(f.keys())
        logger.info(f"Resuming — {len(processed):,} already done")

    remaining = [v for v in videos if v[1] not in processed]
    logger.info(f"To process: {len(remaining):,} | Workers: {args.num_workers}")

    if args.dry_run:
        logger.info("Dry run — exiting.")
        return

    chunks = [remaining[i::args.num_workers] for i in range(args.num_workers)]
    part_paths = [args.output + f".part{i}" for i in range(args.num_workers)]

    start = time.time()
    counter = Value("i", 0)
    procs = []
    for i, (chunk, part_path) in enumerate(zip(chunks, part_paths)):
        p = Process(
            target=run_worker,
            args=(i, chunk, part_path, processed, args.flush_every, counter,
                  args.pose_model, args.hand_model),
            daemon=True,
        )
        p.start()
        procs.append(p)

    with tqdm.tqdm(total=len(remaining), desc="mediapipe-p14t", mininterval=10) as pbar:
        last = 0
        while any(p.is_alive() for p in procs):
            current = counter.value
            pbar.update(current - last)
            last = current
            time.sleep(2)
        pbar.update(counter.value - last)

    for p in procs:
        p.join()

    merge_parts(args.output, part_paths)

    elapsed = time.time() - start
    logger.info(f"All done. Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
