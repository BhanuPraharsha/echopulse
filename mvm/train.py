import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from data.echonet_dataset import EchoNetDataset
from mvm.decoder.multi_task_decoder import MultiTaskDecoder
from mvm.encoder.transformer_encoder import CardiacTransformerEncoder
from mvm.loss import MVMLoss
from mvm.masking.phase_aware_mask import PhaseAwareMask


def build_targets(video, ps=16):
    B, T, C, H, W = video.shape
    p = video.reshape(B * T, C, H, W)
    p = p.unfold(2, ps, ps).unfold(3, ps, ps)
    nh, nw = H // ps, W // ps
    p = p.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B*T, nh, nw, C, ps, ps)
    return p.reshape(B, T, nh * nw, C * ps * ps)


def train_epoch(enc, dec, masker, loss_fn, loader, opt, dev, ep):
    enc.train()
    dec.train()
    total = 0
    scaler = torch.amp.GradScaler("cuda")  # speeds up training by using mixed precision
    for vid, ef, _ in tqdm(loader, desc=f"Epoch {ep}"):
        vid, ef = vid.to(dev), ef.to(dev)
        B, T, C, H, W = vid.shape
        v2d = vid.squeeze(2)  # (B, T, H, W)
        mvids, masks = [], []
        ph_labels = torch.zeros(B, T, dtype=torch.long).to(dev)
        for i in range(B):
            mv, m = masker(v2d[i])
            mvids.append(mv)
            masks.append(m)
            est = masker._estimate_phases(T)
            ph_labels[i] = torch.tensor(est, dtype=torch.long).to(dev)
        mvids = torch.stack(mvids).to(dev)
        masks = torch.stack(masks).to(dev)
        with torch.amp.autocast("cuda"):  # runs in float16
            enc_out = enc(mvids, ph_labels)  # encoder output
            recon, ef_p, ph_p = dec(enc_out, T)  # decoder output
            tgt = build_targets(vid)  # target video
            _, L = loss_fn(recon, tgt, masks, ph_p, ph_labels, ef_p, ef)  # loss
        opt.zero_grad()
        scaler.scale(L).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(dec.parameters()), 1.0
        )

        scaler.step(opt)  # updates weights
        scaler.update()
        total += L.item()  # total loss
    return total / len(loader)  # average loss


def validate_epoch(
    enc, dec, masker, loss_fn, loader, dev, ep
):  # same as train_epoch but no updating of weights
    enc.eval()
    dec.eval()
    total = 0

    with torch.no_grad():  # Freezes weights so it doesn't learn!
        for vid, ef, _ in tqdm(loader, desc=f"Val Epoch {ep}"):
            vid, ef = vid.to(dev), ef.to(dev)
            B, T, C, H, W = vid.shape
            v2d = vid.squeeze(2)  # (B, T, H, W)

            mvids, masks = [], []
            ph_labels = torch.zeros(B, T, dtype=torch.long).to(dev)
            for i in range(B):
                mv, m = masker(v2d[i])
                mvids.append(mv)
                masks.append(m)
                est = masker._estimate_phases(T)
                ph_labels[i] = torch.tensor(est, dtype=torch.long).to(dev)

            mvids = torch.stack(mvids).to(dev)
            masks = torch.stack(masks).to(dev)

            with torch.amp.autocast("cuda"):
                enc_out = enc(mvids, ph_labels)
                recon, ef_p, ph_p = dec(enc_out, T)
                tgt = build_targets(vid)
                _, L = loss_fn(recon, tgt, masks, ph_p, ph_labels, ef_p, ef)

            total += L.item()

    return total / len(loader)


def main(cfg):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project="echopulse", config=cfg, name=cfg.get("run_name", "run"))

    max_frames = cfg["encoder"]["max_frames"]

    train_ds = EchoNetDataset(
        "data/echonet/splits/train.csv", max_frames=max_frames, augment=True
    )
    val_ds = EchoNetDataset("data/echonet/splits/val.csv", max_frames=max_frames)

    tl = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
    )
    vl = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    enc = CardiacTransformerEncoder(**cfg["encoder"]).to(dev)
    dec = MultiTaskDecoder(**cfg["decoder"]).to(dev)
    masker = PhaseAwareMask(mask_ratio=cfg["mask_ratio"])
    loss_fn = MVMLoss(alpha=cfg["alpha"], beta=cfg["beta"])
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    ckpt = Path(cfg["output_dir"]) / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    start_ep = 1
    resume_path = ckpt / "best_model.pth"
    if resume_path.exists():
        ck = torch.load(resume_path, map_location=dev)
        enc.load_state_dict(ck["enc"])
        dec.load_state_dict(ck["dec"])
        start_ep = ck["ep"] + 1
        print(f"Resumed from epoch {start_ep-1}")
    for ep in range(start_ep, cfg["epochs"] + 1):
        train_loss = train_epoch(enc, dec, masker, loss_fn, tl, opt, dev, ep)

        # 2. Validate
        val_loss = validate_epoch(enc, dec, masker, loss_fn, vl, dev, ep)
        sched.step()
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": ep})
        if val_loss < best:
            best = val_loss
            torch.save(
                {"enc": enc.state_dict(), "dec": dec.state_dict(), "ep": ep},
                ckpt / "best_model.pth",
            )
            print(f"  Saved  (val_loss={best:.4f})")

    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="experiments/configs/baseline.yaml")
    a = p.parse_args()
    import yaml

    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    main(cfg)
