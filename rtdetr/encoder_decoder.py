"""Load frozen RT-DETR encoder, decoder, and criterion from a full checkpoint."""
import os
import sys
import torchvision
import torchvision.transforms.v2 as T_v2
if not hasattr(torchvision, 'datapoints'):
    from torchvision import tv_tensors
    torchvision.datapoints = tv_tensors
if not hasattr(T_v2, 'ToImageTensor'):
    T_v2.ToImageTensor = type('ToImageTensor', (), {})
import torch
import torch.nn as nn
from typing import Dict, List, Tuple


def _setup_rtdetr_path():
    rtdetr_src = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "RT-DETR-main", "RT-DETR-main", "rtdetr_pytorch",
    )
    if rtdetr_src not in sys.path:
        sys.path.insert(0, rtdetr_src)


# Mock src.data before _setup_rtdetr_path, so it doesn't trigger the import chain
_dummy = type(sys)("mock")
sys.modules["src.data"] = _dummy
sys.modules["src.data.transforms"] = _dummy
sys.modules["src.data.coco"] = _dummy
sys.modules["src.data.coco.coco_dataset"] = _dummy
sys.modules["src.nn.backbone.regnet"] = _dummy
sys.modules["src.nn.backbone.dla"] = _dummy
sys.modules["src.nn.backbone.test_resnet"] = _dummy
sys.modules["src.nn.criterion"] = _dummy
sys.modules["src.nn.arch"] = _dummy
sys.modules["src.misc.visualizer"] = _dummy

_setup_rtdetr_path()

from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
from src.zoo.rtdetr.rtdetr_criterion import SetCriterion
from src.zoo.rtdetr.matcher import HungarianMatcher


R18VD_ENCODER_CFG = dict(
    in_channels=[128, 256, 512],
    feat_strides=[8, 16, 32],
    hidden_dim=256,
    use_encoder_idx=[2],
    num_encoder_layers=1,
    nhead=8,
    dim_feedforward=1024,
    dropout=0.0,
    enc_act='gelu',
    pe_temperature=10000,
    expansion=0.5,
    depth_mult=1,
    act='silu',
)

R18VD_DECODER_CFG = dict(
    num_classes=80,
    hidden_dim=256,
    num_queries=300,
    feat_channels=[256, 256, 256],
    feat_strides=[8, 16, 32],
    num_levels=3,
    num_decoder_layers=3,
    num_denoising=100,
    eval_idx=-1,
)

CRITERION_CFG = dict(
    weight_dict={'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2},
    losses=['vfl', 'boxes'],
    alpha=0.75,
    gamma=2.0,
    num_classes=80,
)

MATCHER_CFG = dict(
    weight_dict={'cost_class': 2, 'cost_bbox': 5, 'cost_giou': 2},
    alpha=0.25,
    gamma=2.0,
)


def create_encoder_decoder(device: torch.device) -> Tuple[nn.Module, nn.Module]:
    encoder = HybridEncoder(**R18VD_ENCODER_CFG)
    decoder = RTDETRTransformer(**R18VD_DECODER_CFG)
    return encoder.to(device), decoder.to(device)


def create_criterion() -> nn.Module:
    matcher = HungarianMatcher(
        weight_dict=MATCHER_CFG['weight_dict'],
        alpha=MATCHER_CFG['alpha'],
        gamma=MATCHER_CFG['gamma'],
    )
    return SetCriterion(
        matcher=matcher,
        weight_dict=CRITERION_CFG['weight_dict'],
        losses=CRITERION_CFG['losses'],
        alpha=CRITERION_CFG['alpha'],
        gamma=CRITERION_CFG['gamma'],
        num_classes=CRITERION_CFG['num_classes'],
    )


def load_frozen_ed(ckpt_path: str, device: torch.device) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Load encoder, decoder, criterion from checkpoint and freeze them."""
    print(f"Loading full RT-DETR checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    state = ckpt.get('model', ckpt.get('ema', {}).get('module', ckpt))
    if not state:
        raise KeyError("Checkpoint does not contain 'model' or 'ema' keys")

    encoder, decoder = create_encoder_decoder(device)

    encoder_state = {}
    decoder_state = {}

    for k, v in state.items():
        if k.startswith('encoder.'):
            encoder_state[k[len('encoder.'):]] = v
        elif k.startswith('decoder.'):
            decoder_state[k[len('decoder.'):]] = v

    encoder.load_state_dict(encoder_state, strict=False)
    decoder.load_state_dict(decoder_state, strict=False)

    for p in encoder.parameters():
        p.requires_grad = False
    for p in decoder.parameters():
        p.requires_grad = False

    encoder.eval()
    decoder.eval()

    criterion = create_criterion().to(device)
    for p in criterion.parameters():
        p.requires_grad = False

    print(f"  Encoder params: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")
    loaded_enc = sum(1 for _ in encoder_state)
    loaded_dec = sum(1 for _ in decoder_state)
    print(f"  Loaded encoder keys: {loaded_enc}, decoder keys: {loaded_dec}")

    return encoder, decoder, criterion


def run_ed_diagnostic(encoder, decoder, dummy_feats: List[torch.Tensor], targets=None):
    """Quick diagnostic to verify encoder/decoder produce valid outputs."""
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        encoded = encoder(dummy_feats)
        for i, f in enumerate(encoded):
            print(f"    encoded[{i}]: {tuple(f.shape)} mean={f.mean().item():.4f}")

        if targets is None:
            targets = [{
                'labels': torch.tensor([1], device=dummy_feats[0].device),
                'boxes': torch.tensor([[0.3, 0.3, 0.1, 0.1]], device=dummy_feats[0].device),
                'orig_size': torch.tensor([640, 640], device=dummy_feats[0].device),
                'size': torch.tensor([640, 640], device=dummy_feats[0].device),
            } for _ in range(dummy_feats[0].shape[0])]

        decoder_out = decoder(encoded, targets)
        if isinstance(decoder_out, dict):
            for k, v in decoder_out.items():
                if isinstance(v, torch.Tensor):
                    print(f"    decoder[{k}]: {tuple(v.shape)}")
