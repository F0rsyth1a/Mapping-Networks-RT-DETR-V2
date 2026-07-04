import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Optional

from rtdetr.mapping_backbone import MappingBackbone, _sanitize
from rtdetr.spec import get_stage_groups


def _delta_magnitude_per_stage(mapping: MappingBackbone) -> dict:
    mags = {}
    stages = get_stage_groups(mapping.depth)
    for stage in stages:
        total_mag = 0.0
        count = 0
        for name in mapping._name_list:
            if mapping.conv_stage.get(name) == stage:
                z = mapping._get_latent(stage)
                proj = mapping.projections[_sanitize(name)]
                with torch.no_grad():
                    raw = proj(z, mapping.alpha, return_smoothness=False)
                total_mag += raw.abs().mean().item()
                count += 1
        mags[stage] = total_mag / max(count, 1)
    return mags


def run_backbone_diagnostic(mapping: MappingBackbone, device: torch.device):
    mapping.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, 640, 640, device=device)
        feats = mapping(dummy)
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
    encoder: Optional[nn.Module],
    decoder: Optional[nn.Module],
    criterion: Optional[nn.Module],
    train_loader,
    cfg,
    device: torch.device,
    eval_fn=None,
    eval_every: int = 0,
    is_baseline: bool = False,
):
    if is_baseline:
        print("\n  *** BASELINE MODE: all params frozen, loss=baseline ***")
        mapping.eval()
        total_base = 0.0
        n = 0
        with torch.no_grad():
            for images, targets in tqdm(train_loader, desc="Baseline eval"):
                images = images.to(device)
                td = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                feats = mapping(images)
                encoded = encoder(feats)
                out = decoder(encoded, td)
                loss_dict = criterion(out, td)
                total_base += sum(v.item() for v in loss_dict.values() if isinstance(v, torch.Tensor))
                n += 1
        avg_base = total_base / max(1, n)
        print(f"  Baseline avg task_loss: {avg_base:.4f}")
        # also print ΔW mag
        mags = _delta_magnitude_per_stage(mapping)
        print(f"  ΔW magnitude per stage: {mags}")
        return mapping

    optimizer = optim.AdamW(mapping.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr,
        steps_per_epoch=len(train_loader),
        epochs=cfg.epochs, pct_start=0.05,
    )

    run_backbone_diagnostic(mapping, device)
    mags0 = _delta_magnitude_per_stage(mapping)
    print(f"  Initial ΔW magnitude per stage: {mags0}")

    best_loss = float("inf")
    os.makedirs(cfg.save_dir, exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        mapping.train()
        total_loss = 0.0
        total_task = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for images, targets in pbar:
            images = images.to(device)
            optimizer.zero_grad()

            feats_mapping, smooth_val = mapping(images, return_smoothness=True)

            if encoder is not None and decoder is not None and criterion is not None:
                td = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
                encoded = encoder(feats_mapping)
                decoder_out = decoder(encoded, td)
                loss_dict = criterion(decoder_out, td)
                task_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
            else:
                task_loss = sum(f.abs().mean() for f in feats_mapping) * 0.0001

            total = task_loss + cfg.lambda_smoothness * smooth_val

            total.backward()
            torch.nn.utils.clip_grad_norm_(mapping.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += total.item()
            if isinstance(task_loss, torch.Tensor):
                total_task += task_loss.item()
            n_batches += 1

            if n_batches % cfg.log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{total.item():.4f}",
                    "task": f"{task_loss.item():.4f}" if isinstance(task_loss, torch.Tensor) else f"{task_loss:.4f}",
                    "smooth": f"{smooth_val.item():.4f}",
                })

        avg_loss = total_loss / n_batches
        avg_task = total_task / n_batches
        lr_val = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch}: loss={avg_loss:.4f}  task={avg_task:.4f}  lr={lr_val:.2e}")

        if epoch % max(1, eval_every) == 0 or epoch == cfg.epochs:
            mags = _delta_magnitude_per_stage(mapping)
            mag_str = "  ".join(f"{k}={v:.4f}" for k, v in mags.items())
            print(f"  ΔW mag: {mag_str}")
            if eval_fn:
                metrics = eval_fn(mapping, encoder, decoder, train_loader, device)
                print(f"  mAP: {metrics}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch, "model": mapping.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, os.path.join(cfg.save_dir, f"{cfg.exp_name}_best.pt"))

    return mapping
