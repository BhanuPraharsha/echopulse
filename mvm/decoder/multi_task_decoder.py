import torch.nn as nn
from einops import rearrange


class MultiTaskDecoder(nn.Module):
    """
    Three output heads:
    1. Reconstruction: predict pixel values for masked frames
    2. EF Regression:  predict ejection fraction (sigmoid output)
    3. Phase Classify: predict systole/diastole per frame
    """

    def __init__(self, embed_dim=768, n_patches=49, patch_pixels=256, n_phases=2):
        super().__init__()
        self.n_patches = n_patches

        self.recon_head = nn.Sequential(  # pixel reconstruction head
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, patch_pixels),
        )  # takes each patch token(768 dimensions) and predicts pixel values for it(256)

        self.ef_head = nn.Sequential(  # EF regression head
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # output in [0,1]
        )

        self.phase_head = nn.Sequential(  # phase classification head
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_phases),  # input: mean of all patch tokens for each frame
            # output: gives probability of each phase
        )

    def forward(self, encoder_out, T):
        # encoder_out: (B, T*P, E)
        B, TP, E = encoder_out.shape
        P = self.n_patches
        tokens = rearrange(encoder_out, "b (t p) e -> b t p e", t=T, p=P)

        # 1. Reconstruction per patch
        recon = self.recon_head(rearrange(tokens, "b t p e -> (b t) p e"))
        recon = rearrange(recon, "(b t) p px -> b t p px", b=B)

        # 2. EF via global mean pool
        ef = self.ef_head(tokens.mean(dim=[1, 2]))

        # 3. Phase per frame via patch mean pool
        phase = self.phase_head(tokens.mean(dim=2))

        return recon, ef, phase
