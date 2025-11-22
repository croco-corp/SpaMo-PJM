import logging
from typing import Callable
import sys
import argparse
import os.path as osp

import h5py
import numpy as np
import torch
import tqdm
import lmdb
import datasets
from transformers import VideoMAEModel, VideoMAEImageProcessor

sys.path.append("./")

from utils.pjm.preprocessing import ImageConverter, get_video_frames
from utils.helpers import sliding_window_for_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True  # type: ignore


class VideoMAEFeatureReader(object):
    def __init__(
        self,
        model_name="MCG-NJU/videomae-large",
        cache_dir=None,
        device="cuda:0",
        overlap_size=0,
        nth_layer=-1,
    ):
        self.device = device
        self.overlap_size = overlap_size
        self.nth_layer = nth_layer

        self.image_processor = VideoMAEImageProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        self.model = VideoMAEModel.from_pretrained(model_name).to(self.device).eval()  # type: ignore

    @torch.no_grad()
    def get_feats(self, video):
        inputs = self.image_processor(images=video, return_tensors="pt").to(self.device)  # type: ignore

        outputs = self.model(**inputs, output_hidden_states=True).hidden_states

        outputs = outputs[self.nth_layer]
        outputs = outputs[:, 0]

        return outputs


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        help="path to huggingface dataset",
        type=str,
        default="croco-corp/pjm-segments",
    )
    parser.add_argument(
        "--crop-params-path",
        help="path to crop params lmdb file",
        type=str,
        default="crop_params/crop_params.lmdb",
    )
    parser.add_argument("--device", help="device to use", type=str, default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--overlap-size", type=int, default=8)
    parser.add_argument(
        "--nth-layer", help="which layer to extract features from", type=int, default=-1
    )
    parser.add_argument(
        "--cache-dir", help="cache directory for model", type=str, default=None
    )
    parser.add_argument(
        "--model-name",
        help="ViT model name",
        type=str,
        default="MCG-NJU/videomae-large",
    )
    parser.add_argument(
        "--save-dir", help="where to save the output", type=str, required=True
    )
    parser.add_argument(
        "--split",
        help="dataset split to process",
        type=str,
        default="train",
        choices=["train"], #["train", "val", "test"]
    )
    return parser


def get_iterator_for_pjm(args, split: str = "train") -> tuple[Callable, int]:
    batch_size = args.batch_size
    dataset = datasets.load_dataset(
        args.dataset_path, split=split, cache_dir=args.cache_dir
    )
    lmdb_env = lmdb.open(args.crop_params_path, readonly=True, lock=False)
    converter = ImageConverter(lmdb_env, (224, 224))

    reader = VideoMAEFeatureReader(
        args.model_name,
        device=args.device,
        overlap_size=args.overlap_size,
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir,
    )

    num = len(dataset)

    def iterate():
        try:
            REQUIRED_NUM_FRAMES_FOR_MAE = 16
            for rec in dataset:
                video_id = rec["__key__"]
                frames = get_video_frames(rec["mp4"])
                if not frames:
                    logger.warning(f"No frames for {video_id}, skipping")
                    continue

                processed_frames = converter.process_frames(frames, video_id)

                if len(processed_frames) < REQUIRED_NUM_FRAMES_FOR_MAE:
                    processed_frames.extend(
                        [processed_frames[-1]]
                        * (REQUIRED_NUM_FRAMES_FOR_MAE - len(processed_frames))
                    )

                frame_chunks = sliding_window_for_list(
                    processed_frames,
                    window_size=REQUIRED_NUM_FRAMES_FOR_MAE,
                    overlap_size=args.overlap_size,
                )

                video_feats = []
                for j in range(0, len(frame_chunks), batch_size):
                    video_batch = frame_chunks[j : j + batch_size]
                    feats = reader.get_feats(video_batch).cpu().numpy()
                    video_feats.append(feats)

                yield (
                    np.concatenate(video_feats, axis=0),
                    video_id,
                    None,
                )  # For now None, maybe in the future start time
        finally:
            lmdb_env.close()

    return iterate, num


def save_hdf5(save_in_every: int = 500):
    parser = get_parser()
    args = parser.parse_args()
    split = args.split 
    generator, num = get_iterator_for_pjm(args, split=split)
    iterator = generator()

    output_file = osp.join(args.save_dir, f"mae_feat_pjm_{split}.h5")
    if osp.exists(output_file):
        logger.error(f"Output file {output_file} already exists!")
        raise FileExistsError(f"Output file {output_file} already exists!")
    
    logger.info(f"Processing {num} videos for split '{split}'")

    with h5py.File(name=output_file, mode="w") as f:
        f.attrs["model"] = args.model_name
        f.attrs["overlap_size"] = args.overlap_size
        f.attrs["nth_layer"] = args.nth_layer
        f.attrs["dataset_name"] = "PJM"
        f.attrs["split"] = split
        f.attrs["num"] = num

        pbar = tqdm.tqdm(iterator, total=num, desc="Processing PJM")
        for i, mae_feat in enumerate(pbar):
            feats, video_id, _ = mae_feat

            ds = f.create_dataset(
                video_id,
                data=feats,
                dtype="float32",
            )
            assert len(feats.shape) == 2, f"Expected 2D features, got {feats.shape}"
            ds.attrs["num_frames"] = feats.shape[0]
            ds.attrs["features_dim"] = feats.shape[1]

            pbar.set_postfix({"video_id": video_id, "frames": feats.shape[0]})
            if (i + 1) % save_in_every == 0:
                f.flush()
                logger.info(f"Flushed at video {i + 1}/{num}")
        f.flush()


def main():
    save_hdf5(save_in_every=500)


if __name__ == "__main__":
    main()
