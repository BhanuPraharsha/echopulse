import numpy as np
import torch


class BlockMask:
    """Masks contiguous temporal blocks for better context learning."""

    def __init__(self, mask_ratio=0.50, block_size=4):
        self.mask_ratio = mask_ratio
        self.block_size = block_size

    def __call__(self, volume):  # volume: (T, W, H)
        T = volume.shape[0]  # number of time frames
        n_mask = int(T * self.mask_ratio)  # number of frames to block
        n_blocks = max(1, n_mask // self.block_size)  # number of masked blocks
        starts = np.random.choice(T - self.block_size, size=n_blocks, replace=False)
        mask = torch.zeros(T, dtype=torch.bool)
        for s in starts:
            mask[s : min(s + self.block_size, T)] = (
                True  # consecutive frames in each block to be dropped
            )
        masked = volume.clone()
        masked[mask] = 0.0  # pure black for the blocks
        return masked, mask  # returns the block-wise masked video and the mask used
