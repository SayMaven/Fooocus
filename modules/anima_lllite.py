"""ControlNet-LLLite implementation for Anima (DiT) in Fooocus.

Adapted from kohya-ss/Anima-LLLite and ComfyUI-Anima-LLLite.
Provides lightweight low-rank spatial conditioning injected into DiT attention/MLP layers.
"""
from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Target classes within Anima DiT
TARGET_ATTENTION_CLASS = "Attention"
TARGET_MLP_CLASS = "GPT2FeedForward"
LLM_ADAPTER_NAME = "llm_adapter"
LLLITE_ARCH_VERSION = "2"

ATOMIC_SPECIFIERS: Tuple[str, ...] = (
    "self_attn_q_pre",
    "self_attn_kv_pre",
    "cross_attn_q_pre",
    "mlp_fc1_pre",
)

PRESETS: dict = {
    "self_attn_q": ("self_attn_q_pre",),
    "self_attn_qkv": ("self_attn_q_pre", "self_attn_kv_pre"),
    "self_attn_qkv_cross_q": ("self_attn_q_pre", "self_attn_kv_pre", "cross_attn_q_pre"),
}


def parse_target_layers(spec: str) -> Tuple[str, ...]:
    if not isinstance(spec, str):
        raise TypeError(f"target_layers must be str, got {type(spec).__name__}")
    spec = spec.strip()
    if not spec:
        raise ValueError("target_layers spec is empty")

    if spec in PRESETS:
        parts = list(PRESETS[spec])
    else:
        parts = [p.strip() for p in spec.split(",") if p.strip()]
        bad = [p for p in parts if p not in ATOMIC_SPECIFIERS]
        if bad:
            raise ValueError(
                f"unknown target_layers atomic specifier(s): {bad}. "
                f"valid atomic={list(ATOMIC_SPECIFIERS)}, presets={list(PRESETS)}"
            )

    return tuple(a for a in ATOMIC_SPECIFIERS if a in parts)


def _gn(channels: int) -> nn.GroupNorm:
    g = 8
    while g > 1 and channels % g != 0:
        g //= 2
    return nn.GroupNorm(g, channels)


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = _gn(ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)
        self.norm2 = _gn(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


ASPP_DEFAULT_DILATIONS: Tuple[int, ...] = (1, 2, 4, 8)


class _ASPP(nn.Module):
    def __init__(self, ch: int, dilations: Tuple[int, ...] = ASPP_DEFAULT_DILATIONS):
        super().__init__()
        assert len(dilations) >= 1, "ASPP needs at least one dilation"
        branches = []
        for d in dilations:
            if d == 1:
                conv = nn.Conv2d(ch, ch, kernel_size=1)
            else:
                conv = nn.Conv2d(ch, ch, kernel_size=3, padding=d, dilation=d)
            branches.append(nn.Sequential(conv, _gn(ch), nn.SiLU()))
        self.branches = nn.ModuleList(branches)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(nn.Conv2d(ch, ch, kernel_size=1), _gn(ch), nn.SiLU())

        n_branches = len(dilations) + 1
        self.proj = nn.Sequential(
            nn.Conv2d(ch * n_branches, ch, kernel_size=1), _gn(ch), nn.SiLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        outs = [b(x) for b in self.branches]
        g = self.global_conv(self.global_pool(x))
        g = F.interpolate(g, size=(h, w), mode="bilinear", align_corners=False)
        outs.append(g)
        return self.proj(torch.cat(outs, dim=1))


class _Conditioning1(nn.Module):
    def __init__(
        self,
        cond_dim: int,
        cond_emb_dim: int,
        n_resblocks: int,
        use_aspp: bool = False,
        aspp_dilations: Tuple[int, ...] = ASPP_DEFAULT_DILATIONS,
        cond_in_channels: int = 3,
    ):
        super().__init__()
        assert cond_dim % 2 == 0, f"cond_dim must be even, got {cond_dim}"
        assert cond_in_channels >= 1, f"cond_in_channels must be >= 1, got {cond_in_channels}"
        ch_half = cond_dim // 2

        self.cond_in_channels = cond_in_channels
        self.conv1 = nn.Conv2d(cond_in_channels, ch_half, kernel_size=4, stride=4, padding=0)
        self.norm1 = _gn(ch_half)
        self.conv2 = nn.Conv2d(ch_half, ch_half, kernel_size=3, stride=1, padding=1)
        self.norm2 = _gn(ch_half)
        self.conv3 = nn.Conv2d(ch_half, cond_dim, kernel_size=4, stride=4, padding=0)
        self.norm3 = _gn(cond_dim)

        self.resblocks = nn.ModuleList([_ResBlock(cond_dim) for _ in range(n_resblocks)])
        self.aspp = _ASPP(cond_dim, aspp_dilations) if use_aspp else None

        self.proj = nn.Conv2d(cond_dim, cond_emb_dim, kernel_size=1)
        self.out_norm = nn.LayerNorm(cond_emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = F.silu(self.norm2(self.conv2(h)))
        h = F.silu(self.norm3(self.conv3(h)))
        for rb in self.resblocks:
            h = rb(h)
        if self.aspp is not None:
            h = self.aspp(h)
        h = self.proj(h)
        b, c, hh, ww = h.shape
        h = h.view(b, c, hh * ww).permute(0, 2, 1).contiguous()
        h = self.out_norm(h)
        return h


class LLLiteModuleDiT(nn.Module):
    def __init__(
        self,
        name: str,
        org_module: nn.Linear,
        cond_emb_dim: int,
        mlp_dim: int,
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
    ):
        super().__init__()
        self.lllite_name = name
        self.org_module = [org_module]
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.multiplier = multiplier

        in_dim = org_module.in_features
        self.down = nn.Linear(in_dim, mlp_dim)
        self.mid = nn.Linear(mlp_dim + cond_emb_dim, mlp_dim)

        self.cond_to_film = nn.Linear(cond_emb_dim, 2 * mlp_dim)
        nn.init.zeros_(self.cond_to_film.weight)
        nn.init.zeros_(self.cond_to_film.bias)

        self.up = nn.Linear(mlp_dim, in_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        self.cond_emb: Optional[torch.Tensor] = None
        self.org_forward = None
        self.layer_idx: int = -1
        self._depth_embeds_ref: List[nn.Parameter] = []

    def apply_to(self):
        if self.org_forward is None:
            self.org_forward = self.org_module[0].forward
            self.org_module[0].forward = self.forward

    def restore(self):
        if self.org_forward is not None:
            self.org_module[0].forward = self.org_forward
            self.org_forward = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multiplier == 0.0 or self.cond_emb is None:
            return self.org_forward(x)

        orig_shape = x.shape
        is_5d = x.dim() == 5
        if is_5d:
            B, T, H, W, D = orig_shape
            x = x.reshape(B, T * H * W, D)

        cx = self.cond_emb

        if x.shape[0] != cx.shape[0]:
            if x.shape[0] % cx.shape[0] != 0:
                return self.org_forward(x.reshape(orig_shape) if is_5d else x)
            cx = cx.repeat(x.shape[0] // cx.shape[0], 1, 1)

        if x.shape[1] != cx.shape[1]:
            return self.org_forward(x.reshape(orig_shape) if is_5d else x)

        param_dtype = self.down.weight.dtype
        x_proc = x if x.dtype == param_dtype else x.to(param_dtype)
        if cx.dtype != param_dtype or cx.device != x.device:
            cx = cx.to(device=x.device, dtype=param_dtype)

        if self._depth_embeds_ref:
            depth_e = self._depth_embeds_ref[0][self.layer_idx]
            if depth_e.dtype != param_dtype or depth_e.device != x.device:
                depth_e = depth_e.to(device=x.device, dtype=param_dtype)
            cond_local = cx + depth_e
        else:
            cond_local = cx

        h = F.silu(self.down(x_proc))
        gb = self.cond_to_film(cond_local)
        gamma, beta = gb.chunk(2, dim=-1)

        m = self.mid(torch.cat([cond_local, h], dim=-1))
        m = m * (1 + gamma) + beta
        m = F.silu(m)

        if self.dropout is not None and self.training:
            m = F.dropout(m, p=self.dropout)

        out = self.up(m) * self.multiplier
        if out.dtype != x.dtype:
            out = out.to(x.dtype)

        y = self.org_forward(x + out)

        if is_5d:
            y = y.reshape(orig_shape[0], orig_shape[1], orig_shape[2], orig_shape[3], -1)
        return y


class ControlNetLLLiteDiT(nn.Module):
    def __init__(
        self,
        dit: nn.Module,
        cond_emb_dim: int = 32,
        mlp_dim: int = 64,
        target_layers: str = "self_attn_q",
        dropout: Optional[float] = None,
        multiplier: float = 1.0,
        cond_dim: int = 64,
        cond_resblocks: int = 1,
        use_aspp: bool = False,
        aspp_dilations: Tuple[int, ...] = ASPP_DEFAULT_DILATIONS,
        cond_in_channels: int = 3,
        inpaint_masked_input: bool = False,
    ):
        super().__init__()

        atomics = parse_target_layers(target_layers)
        self.cond_emb_dim = cond_emb_dim
        self.mlp_dim = mlp_dim
        self.target_layers = target_layers
        self.target_atomics = atomics
        self.dropout = dropout
        self.multiplier = multiplier
        self.cond_dim = cond_dim
        self.cond_resblocks = cond_resblocks
        self.use_aspp = use_aspp
        self.aspp_dilations = tuple(aspp_dilations) if use_aspp else ()
        self.cond_in_channels = cond_in_channels
        self.inpaint_masked_input = inpaint_masked_input

        self.conditioning1 = _Conditioning1(
            cond_dim, cond_emb_dim, cond_resblocks,
            use_aspp=use_aspp, aspp_dilations=aspp_dilations,
            cond_in_channels=cond_in_channels,
        )

        modules = self._create_modules(dit, cond_emb_dim, mlp_dim, atomics, dropout, multiplier)
        self.lllite_modules = nn.ModuleList(modules)

        n = len(self.lllite_modules)
        self.depth_embeds = nn.Parameter(torch.zeros(n, cond_emb_dim))
        for i, m in enumerate(self.lllite_modules):
            m.layer_idx = i
            m._depth_embeds_ref = [self.depth_embeds]

    @staticmethod
    def _attn_atomic_match(is_self_attn: bool, child_name: str, atomics: Tuple[str, ...]) -> bool:
        if "output_proj" in child_name:
            return False
        if is_self_attn:
            if child_name == "q_proj":
                return "self_attn_q_pre" in atomics
            if child_name in ("k_proj", "v_proj"):
                return "self_attn_kv_pre" in atomics
            return False
        else:
            if child_name == "q_proj":
                return "cross_attn_q_pre" in atomics
            return False

    def _create_modules(
        self,
        dit: nn.Module,
        cond_emb_dim: int,
        mlp_dim: int,
        atomics: Tuple[str, ...],
        dropout: Optional[float],
        multiplier: float,
    ) -> List[LLLiteModuleDiT]:
        modules: List[LLLiteModuleDiT] = []
        want_mlp_fc1 = "mlp_fc1_pre" in atomics
        any_attn = any(a in atomics for a in ("self_attn_q_pre", "self_attn_kv_pre", "cross_attn_q_pre"))

        for name, module in dit.named_modules():
            if LLM_ADAPTER_NAME in name:
                continue
            cls = module.__class__.__name__

            def _is_linear_like(mod):
                return (
                    hasattr(mod, "in_features")
                    and hasattr(mod, "out_features")
                    and callable(getattr(mod, "forward", None))
                )

            if any_attn and cls == TARGET_ATTENTION_CLASS:
                if not hasattr(module, "is_selfattn"):
                    continue
                is_self_attn = bool(module.is_selfattn)
                for child_name, child in module.named_children():
                    if not _is_linear_like(child):
                        continue
                    if not self._attn_atomic_match(is_self_attn, child_name, atomics):
                        continue
                    full_name = f"lllite_dit.{name}.{child_name}".replace(".", "_")
                    modules.append(
                        LLLiteModuleDiT(full_name, child, cond_emb_dim, mlp_dim, dropout, multiplier)
                    )

            elif want_mlp_fc1 and cls == TARGET_MLP_CLASS:
                child = getattr(module, "layer1", None)
                if not _is_linear_like(child):
                    continue
                full_name = f"lllite_dit.{name}.layer1".replace(".", "_")
                modules.append(
                    LLLiteModuleDiT(full_name, child, cond_emb_dim, mlp_dim, dropout, multiplier)
                )

        return modules

    def set_cond_image(self, cond_image: Optional[torch.Tensor]):
        if cond_image is None:
            for m in self.lllite_modules:
                m.cond_emb = None
            return
        cx = self.conditioning1(cond_image)
        for m in self.lllite_modules:
            m.cond_emb = cx

    def clear_cond_image(self):
        self.set_cond_image(None)

    def set_multiplier(self, multiplier: float):
        self.multiplier = multiplier
        for m in self.lllite_modules:
            m.multiplier = multiplier

    def apply_to(self):
        for m in self.lllite_modules:
            m.apply_to()

    def restore(self):
        for m in self.lllite_modules:
            m.restore()


_INTERNAL_MODULES_PREFIX = "lllite_modules."
_INTERNAL_COND_PREFIX = "conditioning1."
_INTERNAL_DEPTH_KEY = "depth_embeds"
_SAVED_COND_PREFIX = "lllite_conditioning1."
_SAVED_DEPTH_SUFFIX = ".depth_embed"


def _from_saved_state_dict(lllite: ControlNetLLLiteDiT, weights_sd: dict) -> dict:
    name_to_idx = {m.lllite_name: i for i, m in enumerate(lllite.lllite_modules)}
    n_modules = len(name_to_idx)
    out: dict = {}
    depth_slices: dict = {}

    for k, v in weights_sd.items():
        if k.startswith(_SAVED_COND_PREFIX):
            out[_INTERNAL_COND_PREFIX + k[len(_SAVED_COND_PREFIX):]] = v
            continue
        if k.endswith(_SAVED_DEPTH_SUFFIX):
            name = k[: -len(_SAVED_DEPTH_SUFFIX)]
            if name in name_to_idx:
                depth_slices[name_to_idx[name]] = v
                continue
        head, dot, tail = k.partition(".")
        if dot and head in name_to_idx:
            out[f"{_INTERNAL_MODULES_PREFIX}{name_to_idx[head]}.{tail}"] = v
            continue
        out[k] = v

    if depth_slices:
        missing = [i for i in range(n_modules) if i not in depth_slices]
        if missing:
            raise RuntimeError(f"depth_embed slices missing for module idx(es) {missing}")
        out[_INTERNAL_DEPTH_KEY] = torch.stack(
            [depth_slices[i] for i in range(n_modules)], dim=0
        )

    return out


def load_lllite_weights(lllite: ControlNetLLLiteDiT, file: str, strict: bool = False):
    if os.path.splitext(file)[1] == ".safetensors":
        from safetensors.torch import load_file
        weights_sd = load_file(file)
    else:
        weights_sd = torch.load(file, map_location="cpu")

    converted = _from_saved_state_dict(lllite, weights_sd)
    info = lllite.load_state_dict(converted, strict=strict)
    return info


def read_lllite_metadata(file: str) -> dict:
    if os.path.splitext(file)[1] != ".safetensors":
        return {}
    try:
        from safetensors import safe_open
        with safe_open(file, framework="pt") as f:
            meta = f.metadata()
        return meta or {}
    except Exception:
        return {}


def _target_cond_hw(latent_h: int, latent_w: int, patch_spatial: int = 2) -> tuple[int, int]:
    padded_h = ((latent_h + patch_spatial - 1) // patch_spatial) * patch_spatial
    padded_w = ((latent_w + patch_spatial - 1) // patch_spatial) * patch_spatial
    return padded_h * 8, padded_w * 8


def _prepare_cond_image(image: torch.Tensor, latent_h: int, latent_w: int,
                        device: torch.device, dtype: torch.dtype,
                        patch_spatial: int = 2) -> torch.Tensor:
    if image.ndim == 4 and image.shape[-1] == 3:
        img = image.permute(0, 3, 1, 2).contiguous()
    elif image.ndim == 3 and image.shape[-1] == 3:
        img = image.unsqueeze(0).permute(0, 3, 1, 2).contiguous()
    elif image.ndim == 4 and image.shape[1] == 3:
        img = image.contiguous()
    else:
        raise ValueError(f"Unexpected cond image shape: {tuple(image.shape)}")

    img = img[:1]
    target_h, target_w = _target_cond_hw(latent_h, latent_w, patch_spatial)
    if img.shape[-2] != target_h or img.shape[-1] != target_w:
        img = F.interpolate(img.float(), size=(target_h, target_w), mode="bicubic", align_corners=False)
        img = img.clamp(0.0, 1.0)
    img = img * 2.0 - 1.0
    return img.to(device=device, dtype=dtype)


def apply_lllite_to_model(
    model,
    weights_path: str,
    image: torch.Tensor,
    strength: float = 1.0,
    start_percent: float = 0.0,
    end_percent: float = 1.0,
    preserve_wrapper: bool = True
):
    """Patches a ComfyUI ModelPatcher instance with Anima-LLLite weights.
    
    Args:
        model: ComfyUI ModelPatcher wrapping Anima DiT.
        weights_path: Path to LLLite safetensors file.
        image: PyTorch tensor (B, H, W, 3) with range [0.0, 1.0].
        strength: Multiplier float.
        start_percent: Float 0.0 to 1.0.
        end_percent: Float 0.0 to 1.0.
        preserve_wrapper: Whether to delegate to existing unet wrapper.
    Returns:
        Patched model clone.
    """
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"LLLite weights not found: {weights_path}")

    meta = read_lllite_metadata(weights_path)
    ce_dim = int(meta.get("lllite.cond_emb_dim", 32))
    m_dim = int(meta.get("lllite.mlp_dim", 64))
    tl = meta.get("lllite.target_atomics", meta.get("lllite.target_layers", "self_attn_q"))
    cond_dim = int(meta.get("lllite.cond_dim", 64))
    cond_resblocks = int(meta.get("lllite.cond_resblocks", 1))
    use_aspp = str(meta.get("lllite.use_aspp", "false")).lower() == "true"
    aspp_dilations_meta = meta.get("lllite.aspp_dilations")
    if use_aspp and aspp_dilations_meta:
        aspp_dilations = tuple(int(d) for d in aspp_dilations_meta.split(",") if d.strip())
    else:
        aspp_dilations = ASPP_DEFAULT_DILATIONS
    cond_in_channels = int(meta.get("lllite.cond_in_channels", 3))

    inner = getattr(model, "model", None)
    if inner is None:
        raise RuntimeError("Input MODEL has no .model attribute (not a ModelPatcher?)")
    dit = getattr(inner, "diffusion_model", None)
    if dit is None:
        raise RuntimeError("MODEL.model has no .diffusion_model")

    patch_spatial = int(getattr(dit, "patch_spatial", 2))
    lllite = ControlNetLLLiteDiT(
        dit,
        cond_emb_dim=ce_dim,
        mlp_dim=m_dim,
        target_layers=tl,
        multiplier=strength,
        cond_dim=cond_dim,
        cond_resblocks=cond_resblocks,
        use_aspp=use_aspp,
        aspp_dilations=aspp_dilations,
        cond_in_channels=cond_in_channels,
        inpaint_masked_input=False,
    )
    load_lllite_weights(lllite, weights_path, strict=False)
    lllite.eval().requires_grad_(False)

    model_sampling = model.get_model_object("model_sampling")
    if model_sampling is not None and hasattr(model_sampling, "percent_to_sigma"):
        sigma_start = float(model_sampling.percent_to_sigma(start_percent))
        sigma_end = float(model_sampling.percent_to_sigma(end_percent))
    else:
        sigma_start = float("inf")
        sigma_end = 0.0

    src_image = image.detach().clone()
    cache = {"cond_image_pp": None, "key": None, "lllite_loaded_to": None}
    old_wrapper = model.model_options.get("model_function_wrapper")

    def _call_next(apply_model, input_x, timestep, c):
        if preserve_wrapper and old_wrapper is not None:
            return old_wrapper(apply_model, {"input": input_x, "timestep": timestep, "c": c})
        return apply_model(input_x, timestep, **c)

    def wrapper(apply_model, args):
        input_x = args["input"]
        timestep = args["timestep"]
        c = args["c"]

        sigma = float(timestep.max().item())
        if not (sigma_end <= sigma <= sigma_start):
            return _call_next(apply_model, input_x, timestep, c)

        latent_h, latent_w = int(input_x.shape[-2]), int(input_x.shape[-1])
        device = input_x.device
        dtype = input_x.dtype

        tag = (device, dtype)
        if cache["lllite_loaded_to"] != tag:
            lllite.to(device=device, dtype=dtype)
            cache["lllite_loaded_to"] = tag
            cache["cond_image_pp"] = None

        key = (latent_h, latent_w, device, dtype)
        if cache["key"] != key or cache["cond_image_pp"] is None:
            cache["cond_image_pp"] = _prepare_cond_image(
                src_image, latent_h, latent_w, device, dtype, patch_spatial
            )
            cache["key"] = key

        lllite.set_multiplier(strength)
        lllite.set_cond_image(cache["cond_image_pp"])
        lllite.apply_to()
        try:
            return _call_next(apply_model, input_x, timestep, c)
        finally:
            lllite.restore()
            lllite.clear_cond_image()

    m = model.clone()
    m.set_model_unet_function_wrapper(wrapper)
    return m
