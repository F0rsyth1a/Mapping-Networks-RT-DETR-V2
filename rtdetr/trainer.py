import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional

from rtdetr.mapping_backbone import MappingBackbone
from rtdetr.spec import get_stage_groups


def _backbone_stability_loss(mapping: MappingBackbone, x: torch.Tensor, sigma: float = 0.01):
    """Stability: compare feature maps with z + noise."""
    original_state = {}
    for attr in ["latent", "latents"]:
        if hasattr(mapping, attr):
            v = getattr(mapping, attr)
            original_state[attr] = v
            break

    if mapping.layerwise:
        noisy_latents = nn.ParameterList(
            [nn.Parameter(l.data + torch.randn_like(l.data) * sigma) for l in mapping.latents]
        )
        mapping.latents = noisy_latents
    else:
        noisy_latent = nn.Parameter(mapping.latent.data + torch.randn_like(mapping.latent.data) * sigma)
        mapping.latent = noisy_latent

    with torch.no_grad():
        feats_noisy = mapping(x, return_smoothness=False)

    # restore
    for attr, v in original_state.items():
        setattr(mapping, attr, v)

    feats_clean = mapping(x, return_smoothness=False)

    loss = 0.0
    for fc, fn in zip(feats_clean, feats_noisy):
        loss += torch.nn.functional.mse_loss(fc, fn)
    return loss / len(feats_clean)


def _alignment_loss_det(mapping: MappingBackbone):
    if mapping.layerwise:
        return sum(z.pow(2).mean() * 0.0001 for z in mapping.latents)
    return mapping.latent.pow(2).mean() * 0.0001


def run_backbone_diagnostic(mapping: MappingBackbone, device: torch.device):
    mapping.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 640, 640, device=device)
        feats = mapping(dummy, return_smoothness=False)
        if isinstance(feats, tuple):
            feats = feats[0]
        print("  Backbone feature shapes:")
        for i, f in enumerate(feats):
            print(f"    feat[{i}]: {tuple(f.shape)}  mean={f.mean().item():.4f}  std={f.std().item():.4f}")
        n_train = mapping.count_trainable_params()
        n_target = mapping.count_target_params()
        print(f"  Trainable params: {n_train:,}")
        print(f"  Target conv params: {n_target:,}")
        if n_train > 0:
            print(f"  Compression ratio: {n_target / n_train:.1f}x")


def train_rtdetr_backbone(
    mapping: MappingBackbone,
    train_loader,
    cfg,
    device: torch.device,
):
    optimizer = optim.AdamW(
        mapping.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    if train_loader is not None:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr,
            steps_per_epoch=len(train_loader),
            epochs=cfg.epochs,
            pct_start=0.05,
        )
    else:
        scheduler = None

    run_backbone_diagnostic(mapping, device)

    if train_loader is None:
        print("No train_loader provided. Ending training.")
        return mapping

    best_loss = float("inf")
    os.makedirs(cfg.save_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        mapping.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)

            optimizer.zero_grad()

            feats, smooth_val = mapping(images, return_smoothness=True)

            task_loss = sum(f.abs().mean() for f in feats) * 0.0001

            stab_loss = _backbone_stability_loss(mapping, images, cfg.stability_sigma)
            align_loss = _alignment_loss_det(mapping)

            total = (
                task_loss
                + cfg.lambda_stability * stab_loss
                + cfg.lambda_smoothness * smooth_val
                + 0.001 * align_loss
            )

            total.backward()
            torch.nn.utils.clip_grad_norm_(mapping.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += total.item()
            n_batches += 1

            if batch_idx % cfg.log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{total.item():.4f}",
                    "task": f"{task_loss.item():.6f}",
                    "stab": f"{stab_loss.item():.6f}",
                    "smooth": f"{smooth_val.item():.4f}",
                })

        avg_loss = total_loss / max(1, n_batches)
        lr_val = scheduler.get_last_lr()[0] if scheduler else cfg.lr
        print(f"  Epoch {epoch}: loss={avg_loss:.4f}  lr={lr_val:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model": mapping.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(cfg.save_dir, f"{cfg.exp_name}_best.pt"))

    return mapping
