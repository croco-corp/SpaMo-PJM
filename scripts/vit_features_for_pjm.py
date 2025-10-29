import argparse
import os
import os.path as osp
import glob
import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor 
import datasets
import io
import sys

def get_iterator_for_pjm(args, split: str):
    dataset = datasets.load_dataset(
        args.dataset_repo_id, 
        split=split
    )
    
    reader = ViTFeatureReader(
        args.model_name, 
        device=args.device, 
        s2_mode=args.s2_mode, 
        scales=args.scales,
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir
    )
    
    dataset_iterator = iter(dataset)
    def iterate():
        for row in dataset_iterator:
            video_bytes = row['mp4']
            video_id = row['__name__']
            
            video_buffer = io.BytesIO(video_bytes)
            frames = mpeg(video_buffer, bytes_format="mpegts")
        
            video_features = []
            for j in range(0, len(frames), args.batch_size):
                video_batch = frames[j:min(j + args.batch_size, len(frames))]
                feats = reader.get_feats(video_batch).cpu().numpy()
                video_features.append(feats)
            yield np.concatenate(video_features, axis=0), video_id
    
    return iterate
    