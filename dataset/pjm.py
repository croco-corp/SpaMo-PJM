import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader
import h5py

class PJMKorpusPreloaded(torch.utils.data.Dataset):
    def __init__(
        self,
        visual_features_path: str,
        spatiotemporal_features_path: str,
        texts_path: str,
    ):
        super().__init__()
        
        visual_features = h5py.File(visual_features_path, mode='r')
        spatiotemporal_features = h5py.File(spatiotemporal_features_path, mode='r')
        texts = h5py.File(texts_path, mode='r')
        
        self.idx_to_key = sorted(visual_features.keys())
        self.data = []
        for key in self.idx_to_key:
            vf = torch.tensor(visual_features[key][()])
            stf = torch.tensor(spatiotemporal_features[key][()])
            text = texts[key][()]
            self.data.append((vf, stf, text))
        
        visual_features.close()
        spatiotemporal_features.close()
        texts.close()
    
    def __getitem__(self, index: int) -> dict[str, any]:
       (item_visual_features, item_spatiotemporal_features, text) = self.data[index]
       key = self.idx_to_key[index]
       return {
            'pixel_value': item_visual_features,
            'glor_value': item_spatiotemporal_features,
            'text': text,
            'id': key,
            'num_frames': len(item_visual_features),
            'lang': 'Polish'
        }

    def __len__(self) -> int:
        return len(self.idx_to_key)

class PJMDatamodule(pl.LightningDataModule):
    def __init__(self, visual_features_path: str, spatiotemporal_features_path: str, texts_path: str, batch_size: int = 32, num_workers=4, val_split: float = 0.2):
        super().__init__()
        self.visual_features_path = visual_features_path
        self.spatiotemporal_features_path = spatiotemporal_features_path
        self.texts_path = texts_path
        self.batch_size = batch_size
        self.val_split = val_split
        self.dataset = None
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = PJMKorpusPreloaded(self.visual_features_path, self.spatiotemporal_features_path, self.texts_path)
        train_size = int((1 - self.val_split) * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)