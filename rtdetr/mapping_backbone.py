import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from mapping_network import OnTheFlyProjection
from rtdetr.spec import build_backbone_specs, get_stage_groups
from rtdetr.functional_presnet import presnet_forward


def _sanitize(name: str) -> str:
    return name.replace(".", "_")


class MappingBackbone(nn.Module):
    def __init__(
        self,
        pretrained_state: Dict[str, torch.Tensor],
        depth: int = 18,
        latent_dim: int = 1024,
        alpha: float = 0.001,
        layerwise: bool = True,
        block_size: int = 8192,
        cache_projections: bool = False,
        use_tanh: bool = True,
        output_gain: float = 1.0,
    ):
        super().__init__()
        self.depth = depth
        self.alpha = alpha
        self.output_gain = output_gain
        self.layerwise = layerwise
        self.use_tanh = use_tanh

        self.conv_specs = build_backbone_specs(depth)
        stage_groups = get_stage_groups(depth)

        self.pretrained_conv: Dict[str, torch.Tensor] = {}
        self.bn_buffers: Dict[str, Dict[str, torch.Tensor]] = {}
        self.conv_shapes: Dict[str, Tuple[int, ...]] = {}
        self.conv_stage: Dict[str, str] = {}
        self.conv_fan_in: Dict[str, int] = {}
        self._name_list: List[str] = []

        for name, (w_shape, _) in self.conv_specs.items():
            self.conv_shapes[name] = w_shape
            fan_in = int(np.prod(w_shape[1:])) if len(w_shape) > 1 else w_shape[1]
            self.conv_fan_in[name] = fan_in
            self._name_list.append(name)

            for group in stage_groups:
                if name.startswith(group) or (group == "conv1" and name.startswith("conv1")):
                    self.conv_stage[name] = group
                    break

        self._extract_pretrained(pretrained_state, depth)

        self.projections = nn.ModuleDict()
        seed_counter = 0

        if layerwise:
            self.latents = nn.ParameterList(
                [nn.Parameter(torch.randn(1, latent_dim) * 1.0) for _ in stage_groups]
            )
            for group_idx, group in enumerate(stage_groups):
                group_layers = [n for n in self.conv_specs if self.conv_stage.get(n) == group]
                for name in group_layers:
                    self.projections[_sanitize(name)] = OnTheFlyProjection(
                        in_dim=latent_dim,
                        out_dim=int(np.prod(self.conv_shapes[name])),
                        seed=seed_counter,
                        block_size=block_size,
                        use_tanh=use_tanh,
                    )
                    seed_counter += 1
        else:
            self.latent = nn.Parameter(torch.randn(1, latent_dim) * 1.0)
            for name in self._name_list:
                self.projections[_sanitize(name)] = OnTheFlyProjection(
                    in_dim=latent_dim,
                    out_dim=int(np.prod(self.conv_shapes[name])),
                    seed=seed_counter,
                    block_size=block_size,
                    use_tanh=use_tanh,
                )
                seed_counter += 1

    def _extract_pretrained(self, state: Dict[str, torch.Tensor], depth: int):
        for name, (w_shape, _) in self.conv_specs.items():
            out_ch = w_shape[0]
            # Default BN buffers
            self.bn_buffers[name] = {
                "bn_w": torch.ones(out_ch),
                "bn_b": torch.zeros(out_ch),
                "bn_mean": torch.zeros(out_ch),
                "bn_var": torch.ones(out_ch),
                "bn_eps": torch.tensor(1e-5),
            }

        if not state:
            return

        for name in self.conv_specs:
            conv_key, bn_prefix = _find_ckpt_keys(name, depth, state)
            if conv_key and conv_key in state:
                self.pretrained_conv[name] = state[conv_key].detach().clone().float()
            if bn_prefix:
                w_key = f"{bn_prefix}.weight"
                b_key = f"{bn_prefix}.bias"
                m_key = f"{bn_prefix}.running_mean"
                v_key = f"{bn_prefix}.running_var"
                if w_key in state:
                    self.bn_buffers[name] = {
                        "bn_w": state[w_key].detach().clone().float(),
                        "bn_b": state[b_key].detach().clone().float(),
                        "bn_mean": state[m_key].detach().clone().float(),
                        "bn_var": state[v_key].detach().clone().float(),
                        "bn_eps": torch.tensor(1e-5),
                    }

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_target_params(self) -> int:
        return sum(int(np.prod(w)) for w, _ in self.conv_specs.values())

    def _get_latent(self, stage: str) -> torch.Tensor:
        if not self.layerwise:
            return self.latent
        stage_groups = get_stage_groups(self.depth)
        idx = stage_groups.index(stage)
        return self.latents[idx]

    def _generate_conv_weights(
        self, return_smoothness: bool = False
    ) -> Union[Dict[str, torch.Tensor], Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        adapted = {}
        smooth_vals = []

        for name in self._name_list:
            proj = self.projections[_sanitize(name)]
            stage = self.conv_stage[name]
            z = self._get_latent(stage)
            out = proj(z, self.alpha, return_smoothness)

            if return_smoothness:
                theta, s = out
                smooth_vals.append(s)
            else:
                theta = out

            w_shape = self.conv_shapes[name]
            fan_in = self.conv_fan_in[name]
            delta = theta.view(w_shape) * self.output_gain / (fan_in ** 0.5)

            if name in self.pretrained_conv:
                adapted[name] = self.pretrained_conv[name].to(delta.device) + delta
            else:
                adapted[name] = delta

        if return_smoothness and smooth_vals:
            return adapted, torch.stack(smooth_vals).mean()
        return adapted

    def forward(
        self, x: torch.Tensor, return_smoothness: bool = False
    ) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], torch.Tensor]]:
        if return_smoothness:
            adapted_weights, smooth_val = self._generate_conv_weights(return_smoothness=True)
        else:
            adapted_weights = self._generate_conv_weights(return_smoothness=False)

        device = x.device
        conv_w = {k: v.to(device) for k, v in adapted_weights.items()}
        bn = {}
        for k, v in self.bn_buffers.items():
            bn[k] = {kk: vv if isinstance(vv, float) else vv.to(device) for kk, vv in v.items()}

        feats = presnet_forward(x, conv_w, bn, self.depth)
        if return_smoothness:
            return feats, smooth_val
        return feats


def _find_ckpt_keys(
    name: str, depth: int, state: Dict[str, torch.Tensor]
) -> Tuple[Optional[str], Optional[str]]:
    """Find the checkpoint key for a conv layer, trying both with and without 'backbone.' prefix."""
    parts = name.split(".")
    block_nums = [2, 2, 2, 2] if depth == 18 else [3, 4, 6, 3]
    stage_names = ["res2", "res3", "res4", "res5"]
    sub = None

    if parts[0] == "conv1":
        # conv1.conv1_1 → conv1.conv1_1.conv
        tail = f"conv1.{parts[1]}.conv"
    else:
        for si, stage_name in enumerate(stage_names):
            if parts[0] == stage_name:
                block_idx = int(parts[1].replace("block", ""))
                sub = parts[2]
                if sub == "shortcut":
                    tail = f"res_layers.{si}.blocks.{block_idx}.short.conv"
                    # variant='d' with stride=2 wraps shortcut in Sequential:
                    # conv → pool+conv, so ckpt key has extra ".conv"
                    tail_alt = f"res_layers.{si}.blocks.{block_idx}.short.conv.conv"
                else:
                    tail = f"res_layers.{si}.blocks.{block_idx}.{sub}.conv"
                break
        else:
            return None, None

    candidates_conv = [tail, f"backbone.{tail}"]
    if sub == "shortcut":
        candidates_conv += [tail_alt, f"backbone.{tail_alt}"]
    conv_key = None
    for c in candidates_conv:
        if f"{c}.weight" in state:
            conv_key = f"{c}.weight"
            break
    if conv_key is None:
        for c in candidates_conv:
            if c in state:
                conv_key = c
                break

    bn_prefix = tail.replace(".conv", ".norm")
    bn_candidates = [bn_prefix, f"backbone.{bn_prefix}"]
    if sub == "shortcut":
        bn_alt = tail_alt.replace(".conv", ".norm")
        bn_candidates += [bn_alt, f"backbone.{bn_alt}"]
    bn_key = None
    for c in bn_candidates:
        if f"{c}.weight" in state:
            bn_key = c
            break

    return conv_key, bn_key
