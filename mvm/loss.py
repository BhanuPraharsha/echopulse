from dataclasses import dataclass

import torch.nn as nn


@dataclass
class LossOut:
    total: float
    recon: float
    phase: float
    ef: float


class MVMLoss(nn.Module):
    """
    L_total = L_recon + alpha * L_phase + beta * L_EF
    alpha=0.30 (phase weight), beta=0.50 (EF weight)
    """

    def __init__(self, alpha=0.30, beta=0.50):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(
        self, recon_pred, recon_true, mask, phase_pred, phase_true, ef_pred, ef_true
    ):

        # recon: only on masked frames
        me = mask.unsqueeze(-1).unsqueeze(-1).expand_as(recon_pred)
        L_recon = self.mse(recon_pred[me], recon_true[me])

        # phase: cross-entropy all frames
        B, T, _ = phase_pred.shape
        L_phase = self.ce(phase_pred.reshape(B * T, -1), phase_true.reshape(B * T))

        # EF: mean squared error
        L_ef = self.mse(ef_pred.squeeze(-1), ef_true.squeeze(-1))

        L_total = L_recon + self.alpha * L_phase + self.beta * L_ef
        return (
            LossOut(L_total.item(), L_recon.item(), L_phase.item(), L_ef.item()),
            L_total,
        )
