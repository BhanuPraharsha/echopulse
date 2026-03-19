import torch, numpy as np
from typing import Optional 

class PhaseAwareMask:
    '''
    Phase-aligned cardiac masking strategy.
    Preferentially masks systolic frames (phase=1)(frames where the heart contracts).

    Args:
        mask_ratio: fraction of frames to mask(default 0.50)
        systole_frac: fraction of masked frames from the systole (default 0.70)
        phase_labels: per-frame labels: 0-diastole, 1=systole.
                      If None: middle third of the sequence treated as systole.
    '''

    def __init__(self, mask_ratio=0.50, systole_frac=0.70, phase_labels=None):
        self.mask_ratio = mask_ratio
        self.systole_frac = systole_frac # fraction of the masked frames that are systolic
        self.phase_labels = phase_labels # per-frame labels: 0-diastole, 1=systole(if known)

    def _estimate_phases(self, T):
        """Fallback: if phase_labels unknown
            first third--> diastole
            middle third--> systole
            last third--> diastole"""
        p = np.zeros(T, dtype=np.int64)
        p[T//3 : 2*T//3] = 1
        return p

    def __call__(self, volume, phase_labels=None):
        T = volume.shape[0]
        labels = np.asarray(
            phase_labels if phase_labels is not None else
            (self.phase_labels if self.phase_labels is not None
             else self._estimate_phases(T))
        )

        sys_idx = np.where(labels == 1)[0]
        dias_idx = np.where(labels == 0)[0]

        n_total = int(T * self.mask_ratio)
        n_sys = min(int(n_total * self.systole_frac), len(sys_idx))
        n_dias = min(n_total - n_sys, len(dias_idx))

        sel_sys = np.random.choice(sys_idx, n_sys, replace=False) if n_sys > 0 else np.array([])
        sel_dias = np.random.choice(dias_idx, n_dias, replace=False) if n_dias > 0 else np.array([])

        mask = torch.zeros(T, dtype=torch.bool)
        mask[np.concatenate([sel_sys, sel_dias]).astype(int)] = True

        masked = volume.clone()
        masked[mask] = 0.0
        return masked, mask