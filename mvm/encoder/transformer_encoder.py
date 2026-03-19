import torch
import torch.nn as nn
from einops import rearrange

"""
    B-> batch size(how many videos processed simultaneously)
    T-> time 
    C->colour channels(1 here (greyscale))
"""


class PatchEmbed(nn.Module):  # helper module
    def __init__(self, img_size=112, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, patch_size, patch_size
        )  # (B*T, 1, 112, 112)  →  (B*T, 768, 7, 7)

        self.n_patches = (img_size // patch_size) ** 2  # number of patches

    def forward(self, x):  # (B*T, C, H, W)
        return rearrange(
            self.proj(x), "bt e h w -> bt (h w) e"
        )  # runs the conv layer and rearranges the output


class CardiacTransformerEncoder(nn.Module):
    """
    Transformer encoder with HYBRID positional encoding:
      1. Absolute: learnable patch positional embedding
      2. Phase:    cardiac phase embedding (0=diastole / 1=systole)
      3. Temporal: learnable temporal positional embedding
    """

    def __init__(
        self,
        img_size=112,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        max_frames=64,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.n_patches = self.patch_embed.n_patches

        # Absolute positional embedding (per patch)
        self.abs_pos = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.abs_pos, std=0.02)

        # Cardiac phase embedding (broadcast over all patches of a frame)
        self.phase_embed = nn.Embedding(2, embed_dim)

        # Temporal positional embedding (per time step)
        self.temp_pos = nn.Parameter(torch.zeros(1, max_frames, embed_dim))
        nn.init.trunc_normal_(self.temp_pos, std=0.02)

        # Transformer backbone
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            enc_layer, depth, enable_nested_tensor=False
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, phase_labels):
        # x: (B, T, H, W)
        # phase_labels: (B, T) int64

        B, T, H, W = x.shape

        # Patch embed each frame
        x_flat = rearrange(x, "b t h w -> (b t) 1 h w")
        tokens = self.patch_embed(x_flat)  # (B*T, P, E)
        tokens = tokens + self.abs_pos  # add absolute pos

        # Add cardiac phase embedding (per frame, broadcast to patches)
        ph = self.phase_embed(phase_labels.reshape(-1))  # (B*T, E)
        tokens = tokens + ph.unsqueeze(1)  # (B*T, P, E)

        # Add temporal embedding
        tokens = rearrange(tokens, "(b t) p e -> b t p e", b=B)
        tokens = tokens + self.temp_pos[:, :T, :].unsqueeze(2)
        tokens = rearrange(tokens, "b t p e -> b (t p) e")

        return self.norm(self.transformer(tokens))
