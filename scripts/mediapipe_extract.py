"""
MediaPipe keypoint extraction for PJM corpus.

Extracts per-frame pose + both hands landmarks from cropped signer frames.
Output shape per video: [T, 258]  (33 pose × 4 + 21 left_hand × 3 + 21 right_hand × 3)
Missing hand detections are filled with zeros.

Usage:
    uv run python scripts/mediapipe_extract.py \
        --tars-dir /home/croco/CrocoSign/data/pjm_segments \
        --crop-params-path crop_params/crop_params.lmdb \
        --output features/mediapipe_feat_pjm.h5 \
        --num-workers 16
"""

import argparse
import io
import logging
import os
import tarfile
import time
from multiprocessing import Process, Value

import h5py
import lmdb
import mediapipe as mp
import msgpack
import numpy as np
from PIL import Image
from PIL.ImageOps import pad as resize_and_pad
from av import open as av_open
import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POSE_DIM = 33 * 4        # x, y, z, visibility
HAND_DIM = 21 * 3        # x, y, z
FEAT_DIM = POSE_DIM + HAND_DIM * 2   # 258


def get_video_frames(video_bytes: bytes) -> list:
    container = av_open(io.BytesIO(video_bytes))
    stream = container.streams.video[0]
    return [frame.to_image() for frame in container.decode(stream)]


def crop_frame(image: Image.Image, crop_params: dict, output_size=(224, 224)) -> Image.Image:
    cropped = image.crop((
        crop_params["x_start"], crop_params["y_start"],
        crop_params["x_end"],   crop_params["y_end"],
    ))
    return resize_and_pad(cropped, output_size)


def extract_landmarks(results) -> np.ndarray:
    """Pack MediaPipe Holistic results into a flat float32 vector of length 258."""
    vec = np.zeros(FEAT_DIM, dtype=np.float32)

    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            vec[i * 4:i * 4 + 4] = [lm.x, lm.y, lm.z, lm.visibility]

    offset = POSE_DIM
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            vec[offset + i * 3:offset + i * 3 + 3] = [lm.x, lm.y, lm.z]

    offset = POSE_DIM + HAND_DIM
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            vec[offset + i * 3:offset + i * 3 + 3] = [lm.x, lm.y, lm.z]

    return vec


def iter_tar_names(tars_dir: str, tar_names: list):
    """Yield (key, mp4_bytes) from a given list of tar filenames."""
    for tar_name in tar_names:
        tar_path = os.path.join(tars_dir, tar_name)
        with tarfile.open(tar_path) as tf:
            members = {m.name: m for m in tf.getmembers()}
            mp4_keys = [n[:-4] for n in members if n.endswith(".mp4")]
            for key in mp4_keys:
                f = tf.extractfile(members[key + ".mp4"])
                if f is not None:
                    yield key, f.read()


def run_worker(worker_id: int, tar_names: list, tars_dir: str,
               crop_params_path: str, part_path: str,
               already_done: set, flush_every: int, counter: Value):
    """Worker process: handles a subset of tar files, writes to part_path."""
    log = logging.getLogger(f"worker-{worker_id}")

    # Resume from existing part file
    done = set(already_done)
    if os.path.exists(part_path):
        with h5py.File(part_path, "r") as f:
            done |= set(f.keys())

    lmdb_env = lmdb.open(crop_params_path, readonly=True, lock=False)
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=True,
        model_complexity=1,
        enable_segmentation=False,
    )

    newly = errors = 0
    start = time.time()

    with h5py.File(part_path, "a") as hf:
        for key, mp4_bytes in iter_tar_names(tars_dir, tar_names):
            if key in done:
                continue
            try:
                frames = get_video_frames(mp4_bytes)
                with lmdb_env.begin(write=False) as txn:
                    raw = txn.get(key.encode())
                if raw is None:
                    raise ValueError(f"No crop params for {key}")
                crop_params = msgpack.unpackb(raw)

                feats = []
                for frame in frames:
                    cropped = crop_frame(frame, crop_params)
                    rgb = np.array(cropped.convert("RGB"))
                    result = holistic.process(rgb)
                    feats.append(extract_landmarks(result))

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

    lmdb_env.close()
    holistic.close()
    elapsed = time.time() - start
    log.info(f"Finished: newly={newly:,} errors={errors} time={elapsed/60:.1f}min")


def merge_parts(output_path: str, part_paths: list):
    """Merge all part H5 files into the final output, then delete parts."""
    logger.info("Merging parts...")
    with h5py.File(output_path, "a") as hf_out:
        hf_out.attrs.setdefault("feat_dim", FEAT_DIM)
        hf_out.attrs.setdefault("description", "pose(33x4) + left_hand(21x3) + right_hand(21x3)")
        for part in part_paths:
            if not os.path.exists(part):
                continue
            with h5py.File(part, "r") as hf_in:
                for key in hf_in:
                    if key not in hf_out:
                        hf_in.copy(key, hf_out)
            os.remove(part)
    logger.info("Merge done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tars-dir", default="/home/croco/CrocoSign/data/pjm_segments")
    parser.add_argument("--crop-params-path", default="crop_params/crop_params.lmdb")
    parser.add_argument("--output", default="features/mediapipe_feat_pjm.h5")
    parser.add_argument("--flush-every", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(),
                        help="Number of parallel processes (default: all CPUs)")
    parser.add_argument("--dry-run", action="store_true", help="Count videos and exit without processing")
    args = parser.parse_args()

    all_tars = sorted(f for f in os.listdir(args.tars_dir) if f.endswith(".tar"))

    total = sum(
        sum(1 for m in tarfile.open(os.path.join(args.tars_dir, f)).getmembers() if m.name.endswith(".mp4"))
        for f in all_tars
    )
    logger.info(f"Total videos: {total:,}")

    processed = set()
    if os.path.exists(args.output):
        with h5py.File(args.output, "r") as f:
            processed = set(f.keys())
        logger.info(f"Resuming — {len(processed):,} already done")

    remaining = total - len(processed)
    logger.info(f"To process: {remaining:,} | Workers: {args.num_workers}")

    if args.dry_run:
        logger.info("Dry run — exiting without processing.")
        return

    # Split tars across workers (round-robin to balance load)
    chunks = [all_tars[i::args.num_workers] for i in range(args.num_workers)]
    part_paths = [args.output + f".part{i}" for i in range(args.num_workers)]

    start = time.time()
    counter = Value("i", 0)
    procs = []
    for i, (chunk, part_path) in enumerate(zip(chunks, part_paths)):
        p = Process(
            target=run_worker,
            args=(i, chunk, args.tars_dir, args.crop_params_path,
                  part_path, processed, args.flush_every, counter),
            daemon=True,
        )
        p.start()
        procs.append(p)

    # Single aggregated progress bar in main process
    with tqdm.tqdm(total=remaining, desc="mediapipe", mininterval=10) as pbar:
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
