import argparse
import os
import torch
from config_rtdetr import RTDETRConfig
from rtdetr.spec import get_total_target_params
from rtdetr.mapping_backbone import MappingBackbone
from rtdetr.trainer import train_rtdetr_backbone


def main():
    parser = argparse.ArgumentParser(description="RT-DETR Backbone Mapping Training")
    parser.add_argument("--pretrained_ckpt", type=str, default="")
    parser.add_argument("--depth", type=int, default=18, choices=[18, 34, 50, 101])
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--layerwise", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--data_dir", type=str, default="./data/coco128")
    parser.add_argument("--output_gain", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp_name", type=str, default="mapping_rtdetr")
    args = parser.parse_args()

    cfg = RTDETRConfig(
        backbone_depth=args.depth,
        latent_dim=args.latent_dim,
        layerwise=args.layerwise,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        img_size=args.img_size,
        data_dir=args.data_dir,
        output_gain=args.output_gain,
        device=args.device,
        exp_name=args.exp_name,
        pretrained_ckpt=args.pretrained_ckpt,
    )

    print("=" * 60)
    print("RT-DETR Backbone Mapping Training")
    print("=" * 60)
    print(f"  Backbone: PResNet-{cfg.backbone_depth}-vd")
    print(f"  Latent dim: {cfg.latent_dim}")
    print(f"  Layerwise: {cfg.layerwise}")
    print(f"  Output gain: {cfg.output_gain}")
    print(f"  Target conv params: {get_total_target_params(cfg.backbone_depth):,}")
    print("=" * 60)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    pretrained_state = {}
    if cfg.pretrained_ckpt and os.path.exists(cfg.pretrained_ckpt):
        print(f"\nLoading pretrained checkpoint: {cfg.pretrained_ckpt}")
        ckpt = torch.load(cfg.pretrained_ckpt, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            pretrained_state = ckpt.get("model", ckpt.get("ema", ckpt))

    mapping = MappingBackbone(
        pretrained_state=pretrained_state,
        depth=cfg.backbone_depth,
        latent_dim=cfg.latent_dim,
        alpha=cfg.alpha,
        layerwise=cfg.layerwise,
        block_size=cfg.block_size,
        cache_projections=cfg.cache_projections,
        use_tanh=cfg.use_tanh,
        output_gain=cfg.output_gain,
    ).to(device)

    n_trainable = mapping.count_trainable_params()
    n_target = mapping.count_target_params()
    n_pretrained = sum(v.numel() for v in mapping.pretrained_conv.values())
    print(f"\n  Pretrained conv params loaded: {n_pretrained:,}")
    print(f"  Trainable params: {n_trainable:,}")
    print(f"  Target conv params: {n_target:,}")
    if n_trainable > 0:
        print(f"  Compression ratio: {n_target / n_trainable:.1f}x")

    train_loader = None
    train_rtdetr_backbone(mapping, train_loader, cfg, device)


if __name__ == "__main__":
    main()
