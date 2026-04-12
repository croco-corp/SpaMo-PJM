import pandas as pd
import torch
import h5py
import random
import pytorch_lightning as pl


class PJMKorpus(torch.utils.data.Dataset):
    def __init__(
        self,
        visual_features_path: str,
        motion_features_path: str,
        texts_path: str,
        device: str,
        max_frame_length: int = 512,
        keys: set | None = None,
        key_to_speaker: dict | None = None,
    ):
        super().__init__()

        visual_features = h5py.File(visual_features_path, mode='r')
        motion_features = h5py.File(motion_features_path, mode='r')
        texts = h5py.File(texts_path, mode='r')
        self.device = device
        self.data = []
        self.max_num_of_frames = max_frame_length
        speaker_map = key_to_speaker or {}

        all_keys = sorted(visual_features.keys())
        idx_to_key = [k for k in all_keys if k in keys] if keys else all_keys
        for key in idx_to_key:
            vf = visual_features[key][()]
            mf = motion_features[key][()]
            text = texts[key][()].decode()
            speaker_id = speaker_map.get(key, 'unknown')
            self.data.append((key, vf, mf, text, speaker_id))

        visual_features.close()
        motion_features.close()
        texts.close()

    def __getitem__(self, index: int) -> dict:
        key, item_visual_features, item_motion_features, text, speaker_id = self.data[index]
        item_visual_features = torch.from_numpy(item_visual_features)
        item_motion_features = torch.from_numpy(item_motion_features)
        num_of_frames = item_visual_features.size(dim=0)
        if num_of_frames > self.max_num_of_frames:
            start_index = random.randint(0, num_of_frames - self.max_num_of_frames)
            item_visual_features = item_visual_features[start_index:start_index + self.max_num_of_frames]
        return {
            'id': key,
            'pixel_value': item_visual_features,
            'num_frames': item_visual_features.size(dim=0),
            'glor_value': item_motion_features,
            'text': text,
            'lang': '',
            'speaker_id': speaker_id,
        }

    def __len__(self) -> int:
        return len(self.data)


def collate_data(batch):
    # Return the list of sample dicts as-is; get_inputs() handles batching internally
    return batch


class PJMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        visual_features_path: str,
        motion_features_path: str,
        texts_path: str,
        device: str,
        train_split_path: str,
        val_split_path: str,
        test_split_path: str | None = None,
        max_frame_length: int = 512,
        batch_size: int = 64,
        num_workers: int = 4,
        set_to_test: str = 'val',
    ):
        super().__init__()
        self.visual_features_path = visual_features_path
        self.motion_features_path = motion_features_path
        self.texts_path = texts_path
        self.device = device
        self.train_split_path = train_split_path
        self.val_split_path = val_split_path
        self.test_split_path = test_split_path
        self.max_frame_length = max_frame_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.set_to_test = set_to_test

    @staticmethod
    def _load_split(path: str):
        df = pd.read_csv(path)
        keys = set(df['key'].astype(str))
        key_to_speaker = dict(zip(df['key'].astype(str), df['speaker_id'].astype(str)))
        return keys, key_to_speaker

    def setup(self, stage=None):
        train_keys, train_k2s = self._load_split(self.train_split_path)
        val_keys,   val_k2s   = self._load_split(self.val_split_path)
        self.train_set = PJMKorpus(
            self.visual_features_path,
            self.motion_features_path,
            self.texts_path,
            device=self.device,
            max_frame_length=self.max_frame_length,
            keys=train_keys,
            key_to_speaker=train_k2s,
        )
        self.val_set = PJMKorpus(
            self.visual_features_path,
            self.motion_features_path,
            self.texts_path,
            device=self.device,
            max_frame_length=self.max_frame_length,
            keys=val_keys,
            key_to_speaker=val_k2s,
        )
        if self.test_split_path:
            test_keys, test_k2s = self._load_split(self.test_split_path)
            self.test_set = PJMKorpus(
                self.visual_features_path,
                self.motion_features_path,
                self.texts_path,
                device=self.device,
                max_frame_length=self.max_frame_length,
                keys=test_keys,
                key_to_speaker=test_k2s,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate_data,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_data,
        )

    def test_dataloader(self):
        if self.set_to_test == 'test' and hasattr(self, 'test_set'):
            dataset = self.test_set
        elif self.set_to_test == 'train':
            dataset = self.train_set
        else:
            dataset = self.val_set
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=collate_data,
        )


if __name__ == '__main__':
    dataset = PJMKorpus('features/vit_feat_pjm.h5', 'features/mae_feat_pjm.h5', 'features/texts_eng.h5', device='cpu')
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, collate_fn=collate_data)
    batch = next(iter(loader))
    print(f"Batch size: {len(batch)}, keys: {list(batch[0].keys())}")
