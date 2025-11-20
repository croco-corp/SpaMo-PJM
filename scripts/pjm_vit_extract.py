import argparse
import os
import os.path as osp
import glob
import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, CLIPVisionModel
import datasets
import sys
import lmdb
sys.path.append('./')

from utils.s2wrapper import forward as multiscale_forward
from utils.helpers import read_video, get_img_list
from utils.pjm.preprocessing import get_video_frames, ImageConverter

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)



class ViTFeatureReader(object):
    def __init__(
        self, 
        model_name='openai/clip-vit-large-patch14', 
        cache_dir=None,
        device='cuda:0', 
        s2_mode='s2wrapping',
        scales=[1, 2],
        nth_layer=-1
    ):
        self.s2_mode = s2_mode
        self.device = device
        self.scales = scales
        self.nth_layer = nth_layer
        
        self.model = CLIPVisionModel.from_pretrained(
            model_name, output_hidden_states=True, cache_dir=cache_dir
        ).to(device).eval()
        
        self.image_processor = AutoImageProcessor.from_pretrained(model_name, do_center_crop=False, do_resize=False)
        print(self.image_processor.do_center_crop, self.image_processor.do_resize)

    @torch.no_grad()
    def forward_features(self, inputs):
        outputs = self.model(inputs).hidden_states
        outputs = outputs[self.nth_layer]
        return outputs

    @torch.no_grad()
    def get_feats(self, video):
        inputs = self.image_processor(list(video), return_tensors="pt").to(self.device).pixel_values
        if self.s2_mode == "s2wrapping":
            outputs = multiscale_forward(self.forward_features, inputs, scales=self.scales, num_prefix_token=1)
        else:
            outputs = self.forward_features(inputs)
        return outputs[:, 0]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', help='Path to huggingface dataset', type=str, default='croco-corp/pjm-segments')
    parser.add_argument('--crop-params-path', help='Path to crop params lmdb file', type=str, default='crop_params/crop_params.lmdb')
    parser.add_argument('--device', help='device to use', default='cuda:0')
    parser.add_argument('--s2_mode', default='')
    parser.add_argument('--scales', nargs='+', type=int, help='List of scales', default=[])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nth_layer', type=int, default=-1)
    parser.add_argument('--cache_dir', help='cache dir for model', default=None)
    
    parser.add_argument('--save_dir', help='where to save the output', required=True)
    parser.add_argument('--model_name', help='ViT model name', default='openai/clip-vit-large-patch14')

    return parser

def get_iterator(args):
    batch_size = args.batch_size
    dataset = datasets.load_dataset(args.dataset_path, split='train')
    lmdb_env = lmdb.open(args.crop_params_path)
    converter = ImageConverter(lmdb_env, (224,224))
    
    reader = ViTFeatureReader(
        args.model_name, 
        device=args.device, 
        s2_mode=args.s2_mode, 
        scales=args.scales,
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir
    )
    
    def iterate():
        for rec in dataset:
            frames = get_video_frames(rec['mp4'])
            processed = converter.process_frames(frames, rec['__key__'])
            frame_features = []
            for i in range(0, len(processed), batch_size):
                frame_batch = processed[i:i+batch_size]
                features = reader.get_feats(frame_batch).cpu().numpy()
                frame_features.append(features)
            
            yield np.concatenate(frame_features, axis=0), rec['__key__']
            
    return iterate


def main():
    parser = get_parser()
    args = parser.parse_args()

    generator = get_iterator(args)
    iterator = generator()

    for vit_feat in tqdm.tqdm(iterator):
        feats, id, st = vit_feat
        save_path = osp.join(args.save_dir, fname, m)
        
        postfix = ""
        if args.s2_mode != "":
            postfix = f"_{args.s2_mode}"
        if len(args.scales) == 3:
            postfix = f'{postfix}_large'
        if st is not None:
            postfix = f'_{st}{postfix}'
        
        np.save(osp.join(save_path, f'{id}{postfix}.npy'), feats)


if __name__ == "__main__":
    main()