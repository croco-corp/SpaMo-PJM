import torch
import numpy as np
import h5py

class PJMKorpus(torch.utils.data.Dataset):
    def __init__(
        self,
        visual_features_path: str,
        spatiotemporal_features_path: str,
        texts_path: str 
    ):
        super.__init__()
        self.visual_features = h5py.File(visual_features_path, mode='r')
        self.spatiotemporal_features = h5py.File(spatiotemporal_features_path, mode='r')
        self.texts = h5py.File(texts_path, mode='r')
        self.id_to_key = sorted(self.visual_features.keys())
    
    def _get_visual_features(self, key: str) -> torch.Tensor:
        vf = self.visual_features[key]
        vf = np.array(vf)
        
        return torch.tensor(vf)

    def _get_spatiotemporal_features(self, key: str) -> torch.Tensor:
        sf = self.spatiotemporal_features[key]
        sf = np.array(sf)

        return torch.tensor(sf)
    

    def __getitem__(self, index: int) -> dict[str, any]:
        key = self.id_to_key[index]
        item_visual_features = self._get_visual_features(key)
        item_spatiotemporal_features = self._get_spatiotemporal_features(key)
        text = self.texts[key]

        return {
            'pixel_value': item_visual_features,
            'glor_value': item_spatiotemporal_features,
            'text': text,
            'id': key,
            'num_frames': len(item_visual_features),
            'lang': 'Polish'
        }

    def __len__(self) -> int:
        return len(self.id_to_key)

