import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class EchoNetDataset(Dataset):
    def __init__(self, split_csv, max_frames=64, augment=False):
        self.df = pd.read_csv(split_csv)
        self.max_frames = max_frames
        self.augment = augment

    def __len__(self): 
        return len(self.df)

    def _pad_crop(self, v):
        T = v.shape[0]
        if T >= self.max_frames:
            s = np.random.randint(0, T - self.max_frames + 1)
            return v[s : s + self.max_frames]
        
        pad = np.zeros((self.max_frames - T, *v.shape[1:]), np.float32)
        return np.concatenate([v, pad])

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        v = self._pad_crop(np.load(full_path))
        
        if self.augment and np.random.rand() > 0.5:
            v = np.flip(v, axis=2).copy()
            
        vid = torch.FloatTensor(v).unsqueeze(1)    # (T, 1, H, W)
        ef = torch.FloatTensor([row.ef / 100.0])
        
        return vid, ef, row.patient_id