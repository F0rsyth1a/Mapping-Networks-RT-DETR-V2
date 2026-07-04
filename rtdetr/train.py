import argparse
import os
import sys
# Allow running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from config_rtdetr import RTDETRConfig
from rtdetr.spec import get_total_target_params
from rtdetr.mapping_backbone import MappingBackbone
from rtdetr.trainer import train_rtdetr_backbone


def _download_coco128(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)
    images_dir = os.path.join(data_dir, "images", "train2017")
    if os.path.exists(images_dir) and len(os.listdir(images_dir)) > 0:
        print(f"COCO128 already present in {data_dir}")
        return

    print(f"Downloading COCO128 to {data_dir}...")
    import zipfile, requests, io
    url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
    r = requests.get(url, timeout=120)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(data_dir)
    print("COCO128 download complete.")


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=100, img_size=640):
        self.num_samples = num_samples
        self.img_size = img_size
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        img = torch.randn(3, self.img_size, self.img_size)
        return img, {}


class ReferenceBackbone(torch.nn.Module):
    """Pretrained backbone forward without mapping modifications."""
    def __init__(self, pretrained_conv, bn_buffers, depth):
        super().__init__()
        self.pretrained_conv = pretrained_conv
        self.bn_buffers = bn_buffers
        self.depth = depth

    def forward(self, x):
        from rtdetr.functional_presnet import presnet_forward
        conv_w = {k: v.to(x.device) for k, v in self.pretrained_conv.items()}
        bn = {}
        for k, v in self.bn_buffers.items():
            bn[k] = {kk: vv if isinstance(vv, float) else vv.to(x.device) for kk, vv in v.items()}
        return presnet_forward(x, conv_w, bn, self.depth)


def main():
    parser = argparse.ArgumentParser(description="RT-DETR Backbone Mapping Training")
    parser.add_argument("--pretrained_ckpt", type=str, default="")
    parser.add_argument("--det_ckpt", type=str, default="", help="Full RT-DETR detection checkpoint (encoder+decoder)")
    parser.add_argument("--depth", type=int, default=18, choices=[18, 34, 50, 101])
    parser.add_argument("--latent_dim", type=int, default=1024)
    parser.add_argument("--layerwise", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--data_dir", type=str, default="./data/coco128")
    parser.add_argument("--output_gain", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--exp_name", type=str, default="mapping_rtdetr")
    parser.add_argument("--dummy", action="store_true", help="Use random dummy data for quick test")
    parser.add_argument("--dummy_samples", type=int, default=64)
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
    print(f"  Target conv params: {get_total_target_params(cfg.backbone_depth):,}")
    print(f"  Dummy data: {args.dummy}")
    print(f"  Detection ckpt: {args.det_ckpt or 'none'}")
    print("=" * 60)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)

    pretrained_state = {}
    if cfg.pretrained_ckpt and os.path.exists(cfg.pretrained_ckpt):
        print(f"\nLoading pretrained checkpoint: {cfg.pretrained_ckpt}")
        ckpt = torch.load(cfg.pretrained_ckpt, map_location="cpu", weights_only=False)
        pretrained_state = ckpt if isinstance(ckpt, dict) else {}

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

    encoder, decoder, criterion = None, None, None
    if args.det_ckpt and os.path.exists(args.det_ckpt) and not args.dummy:
        from rtdetr.encoder_decoder import load_frozen_ed
        encoder, decoder, criterion = load_frozen_ed(args.det_ckpt, device)

    if args.dummy or not os.path.exists(cfg.data_dir):
        print(f"\nUsing dummy dataset ({args.dummy_samples} random images)")
        encoder = decoder = criterion = None  # no detection modules in dummy mode
        train_ds = DummyDataset(num_samples=args.dummy_samples, img_size=cfg.img_size)
        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True
        )
    else:
        from rtdetr.coco_loader import build_coco_loader
        img_dir = os.path.join(cfg.data_dir, "images", "train2017")
        ann_file = os.path.join(cfg.data_dir, "annotations", "instances_train2017.json")
        if os.path.exists(img_dir):
            train_loader = build_coco_loader(
                img_dir, ann_file, cfg.batch_size, cfg.img_size, cfg.num_workers
            )
        else:
            print("Image directory not found, falling back to dummy data")
            train_ds = DummyDataset(num_samples=128, img_size=cfg.img_size)
            train_loader = torch.utils.data.DataLoader(
                train_ds, batch_size=cfg.batch_size, shuffle=True
            )

    train_rtdetr_backbone(
        mapping, encoder, decoder, criterion,
        train_loader, cfg, device,
    )


if __name__ == "__main__":
    main()
