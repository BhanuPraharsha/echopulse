import torch , numpy as np

class RandomMask:
    '''Randomly masks "mask_ratio" fraction of the time frames.'''
    def __init__(self, mask_ratio=0.50):
        self.mask_ratio = mask_ratio


    def __call__(self, volume): #(T,H,W)
        T = volume.shape[0]
        n_mask = int(T*self.mask_ratio)
        indices = torch.randperm(T)[:n_mask]
        mask = torch.zeros(T, dtype=torch.bool) # blacks out mask_ratio fraction of the time frames
        masked = volume.clone()
        masked[mask] = 0.0
        return masked, mask # returns the randomly masked video and the mask used