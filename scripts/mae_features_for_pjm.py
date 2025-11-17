import io
from typing import Callable
import numpy as np
import datasets

from scripts.mae_extract_feature import VideoMAEFeatureReader
from utils.helpers import sliding_window_for_list, frames_from_mpegts


def get_iterator_for_pjm(args, split: str) -> Callable:
    dataset = datasets.load_dataset(args.dataset_name, split=split, cache_dir=args.cache_dir)

    reader = VideoMAEFeatureReader(
        args.model_name,
        device=args.device,
        overlap_size=args.overlap_size,
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir,
    )

    dataset_iterator = iter(dataset)
    def iterate():
        REQUIRED_NUM_FRAMES_FOR_MAE = 16
        for row in dataset_iterator:
            video_bytes = row['mp4']
            video_id = row['__name__']

            video_buffer = io.BytesIO(video_bytes)
            frames = frames_from_mpegts(video_buffer, bytes_format="mpegts")

            if len(frames) < REQUIRED_NUM_FRAMES_FOR_MAE:
                frames.extend([frames[-1]] * (REQUIRED_NUM_FRAMES_FOR_MAE - len(frames)))

            frame_chunks = sliding_window_for_list(
                frames, window_size=REQUIRED_NUM_FRAMES_FOR_MAE, overlap_size=args.overlap_size
            )

            video_feats = []

            for j in range(0, len(frame_chunks), args.batch_size):
                video_batch = frame_chunks[j:min(j + args.batch_size, len(frame_chunks))]
                feats = reader.get_feats(video_batch).cpu().numpy()
                video_feats.append(feats)

            yield np.concatenate(video_feats, axis=0), video_id, None
            
    return iterate