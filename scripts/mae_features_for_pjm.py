from typing import Union, Callable
import os
import sys
import argparse
import os.path as osp

import numpy as np
import torch
import tqdm
import lmdb
import datasets
from transformers import VideoMAEModel, VideoMAEImageProcessor

sys.path.append("./")

from utils.pjm.preprocessing import ImageConverter, get_video_frames
from utils.helpers import sliding_window_for_list

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
    parser.add_argument(
        "--scales",
        nargs="+",
        type=int,
        help="List of scales for S2-Wrapping",
        default=[],
    )
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
    return parser


def get_iterator_for_pjm(args, split: str = "train") -> Union[Callable, int]:
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
        REQUIRED_NUM_FRAMES_FOR_MAE = 16
        for rec in dataset:
            video_id = rec["__key__"]
            frames = get_video_frames(rec["mp4"])
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

            yield np.concatenate(video_feats, axis=0), video_id, None

    return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()
    generator, num = get_iterator_for_pjm(args)
    iterator = generator()
    split = "train"
    for mae_feat in tqdm.tqdm(iterator, total=num, desc=f"Processing PJM"):
        feats, video_id, st_ = mae_feat
        save_path = osp.join(args.save_dir, "mae_feat_pjm", f"{video_id}_overlap-{args.overlap_size}.npy")
        postfix = ""
        if args.overlap_size > 0:
            postfix = f"_overlap-{args.overlap_size}"
        if args.s2_mode != "":
            postfix = f"{postfix}_{args.s2_mode}"
        np.save(osp.join(save_path, f"{video_id}{postfix}.npy"), feats)


if __name__ == "__main__":
    main()
