"""
Hand-crop ViT feature extraction for PHOENIX-2014T.

Adapted from scripts/hand_crop_vit_extract.py (PJM):
- Input: per-video PNG sequences {phoenix_root}/features/fullFrame-210x260px/{mode}/{key}/images*.png
- No LMDB crop_params (Phoenix is already signer-centered at 210x260)
- Output: features/hand_vit_feat_p14t.h5  (same schema as PJM hand_vit_feat_pjm.h5)

Usage:
    uv run python scripts/hand_crop_vit_extract_p14t.py \\
        --phoenix-root /home/croco/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T \\
        --output features/hand_vit_feat_p14t.h5 \\
        --num-workers 8
"""

import argparse
import logging
import os
import time
from multiprocessing import Process, Queue
from pathlib import Path

import h5py
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, CLIPVisionModel
import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

HAND_CROP_PAD = 0.25
FEAT_DIM = 2048
QUEUE_MAXSIZE = 128


def hand_bbox(landmarks, img_w: int, img_h: int, pad: float = HAND_CROP_PAD):
    xs = [lm.x * img_w for lm in landmarks]
    ys = [lm.y * img_h for lm in landmarks]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    pw = (x2 - x1) * pad
    ph = (y2 - y1) * pad
    return (int(max(0, x1 - pw)), int(max(0, y1 - ph)),
            int(min(img_w, x2 + pw)), int(min(img_h, y2 + ph)))


def list_p14t_videos(phoenix_root: str, modes=("train", "dev", "test")):
    """Return list of (mode, key, frame_dir) tuples for every Phoenix video."""
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


def cpu_worker(worker_id: int, video_chunk: list, out_queue: Queue,
               already_done: set, hand_model_path: str):
    log = logging.getLogger(f"cpu-{worker_id}")
    try:
        _cpu_worker_inner(worker_id, video_chunk, out_queue, already_done, log, hand_model_path)
    except Exception as e:
        log.error(f"Worker {worker_id} crashed: {e}")
    finally:
        out_queue.put(("done", worker_id))


def _cpu_worker_inner(worker_id, video_chunk, out_queue, already_done, log, hand_model_path):
    hands_detector = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=hand_model_path),
        num_hands=2,
        running_mode=RunningMode.IMAGE,
    ))

    for mode, key, frame_dir in video_chunk:
        if key in already_done:
            out_queue.put(("skip", key))
            continue
        try:
            png_paths = sorted(frame_dir.glob("images*.png"))
            if not png_paths:
                raise ValueError(f"No PNG frames in {frame_dir}")

            crops_list = []   # flat list of np.ndarray
            frame_sizes = []  # number of hand crops per frame
            for p in png_paths:
                rgb = np.array(Image.open(p).convert("RGB"))
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = hands_detector.detect(mp_img)
                h, w = rgb.shape[:2]
                n = 0
                if result.hand_landmarks:
                    for hand_lm in result.hand_landmarks:
                        x1, y1, x2, y2 = hand_bbox(hand_lm, w, h)
                        if x2 > x1 and y2 > y1:
                            crops_list.append(rgb[y1:y2, x1:x2].copy())
                            n += 1
                frame_sizes.append(n)

            out_queue.put(("ok", key, crops_list, frame_sizes))

        except Exception as e:
            out_queue.put(("error", key, str(e)))

    hands_detector.close()
    log.info("Worker finished.")


@torch.no_grad()
def encode_crops(crops: list, processor, model, device: str) -> np.ndarray:
    inputs = processor(images=crops, return_tensors="pt").to(device)
    hidden = model(**inputs, output_hidden_states=True).hidden_states[-1]
    cls = hidden[:, 0]
    patch_mean = hidden[:, 1:].mean(1)
    return torch.cat([cls, patch_mean], dim=-1).float().cpu().numpy()


def gpu_writer(out_queue: Queue, n_workers: int, args, processed: set):
    logger.info("Loading CLIP ViT-L/14...")
    processor = AutoImageProcessor.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    vit = CLIPVisionModel.from_pretrained(
        args.model_name, output_hidden_states=True, cache_dir=args.cache_dir,
    ).to(args.device).eval()
    logger.info("Model loaded. GPU writer ready.")

    zero_feat = np.zeros(FEAT_DIM, dtype=np.float32)
    done_workers = 0
    newly = errors = skipped = 0
    start = time.time()

    pbar = tqdm.tqdm(desc="hand-vit-p14t", mininterval=30)

    with h5py.File(args.output, "a") as hf:
        hf.attrs.setdefault("feat_dim", FEAT_DIM)
        hf.attrs.setdefault("description",
                            "CLIP ViT-L/14 on hand crops (PHOENIX-2014T), mean-pooled L+R, zero if no detection")

        while done_workers < n_workers:
            item = out_queue.get()
            tag = item[0]

            if tag == "done":
                done_workers += 1
                continue
            if tag == "skip":
                skipped += 1
                pbar.update(1)
                pbar.set_postfix(new=newly, skip=skipped, err=errors)
                continue
            if tag == "error":
                _, key, msg = item
                errors += 1
                logger.warning(f"Error {key}: {msg}")
                pbar.update(1)
                pbar.set_postfix(new=newly, skip=skipped, err=errors)
                continue

            _, key, crops_list, frame_sizes = item

            if crops_list:
                all_pil = [Image.fromarray(c) for c in crops_list]
                all_feats = []
                for i in range(0, len(all_pil), args.batch_size):
                    batch = all_pil[i:i + args.batch_size]
                    all_feats.append(encode_crops(batch, processor, vit, args.device))
                all_feats = np.concatenate(all_feats, axis=0)
            else:
                all_feats = np.empty((0, FEAT_DIM), dtype=np.float32)

            video_feats = []
            idx = 0
            for n in frame_sizes:
                if n > 0:
                    video_feats.append(all_feats[idx:idx + n].mean(axis=0))
                    idx += n
                else:
                    video_feats.append(zero_feat)

            arr = np.stack(video_feats, axis=0)
            hf.create_dataset(key, data=arr, dtype="float32")
            newly += 1
            pbar.update(1)
            pbar.set_postfix(new=newly, skip=skipped, err=errors)

            if newly % args.flush_every == 0:
                hf.flush()
                elapsed = time.time() - start
                logger.info(f"{newly + len(processed)} done | {elapsed/60:.1f}min")

        hf.flush()

    pbar.close()
    elapsed = time.time() - start
    logger.info(f"Done. Newly: {newly:,} | Errors: {errors} | Time: {elapsed/60:.1f} min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phoenix-root",
                        default="/home/croco/data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T")
    parser.add_argument("--output", default="features/hand_vit_feat_p14t.h5")
    parser.add_argument("--cache-dir", default="/home/croco/.cache/huggingface/hub")
    parser.add_argument("--model-name", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--flush-every", type=int, default=200)
    parser.add_argument("--num-workers", type=int, default=max(1, os.cpu_count() - 2))
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
    logger.info(f"To process: {len(remaining):,} | CPU workers: {args.num_workers}")

    if args.dry_run:
        logger.info("Dry run — exiting.")
        return

    chunks = [remaining[i::args.num_workers] for i in range(args.num_workers)]
    out_queue = Queue(maxsize=QUEUE_MAXSIZE)

    cpu_procs = []
    for i, chunk in enumerate(chunks):
        p = Process(
            target=cpu_worker,
            args=(i, chunk, out_queue, processed, args.hand_model),
            daemon=True,
        )
        p.start()
        cpu_procs.append(p)

    gpu_writer(out_queue, args.num_workers, args, processed)

    for p in cpu_procs:
        p.join()


if __name__ == "__main__":
    main()
