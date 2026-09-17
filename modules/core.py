import os
import sys
import einops
import torch
import numpy as np

import ldm_patched.modules.model_management
import ldm_patched.modules.model_detection
import ldm_patched.modules.model_patcher
import ldm_patched.modules.utils
import ldm_patched.modules.controlnet
import modules.sample_hijack
import ldm_patched.modules.samplers
import ldm_patched.modules.latent_formats

from ldm_patched.modules.sd import load_checkpoint_guess_config
from ldm_patched.contrib.external import VAEDecode, EmptyLatentImage, VAEEncode, VAEEncodeTiled, VAEDecodeTiled, \
    ControlNetApplyAdvanced
from ldm_patched.contrib.external_freelunch import FreeU_V2
from ldm_patched.modules.sample import prepare_mask
from modules.lora import match_lora
from modules.util import get_file_from_folder_list
from ldm_patched.modules.lora import model_lora_keys_unet, model_lora_keys_clip
from modules.config import path_embeddings
from ldm_patched.contrib.external_model_advanced import ModelSamplingDiscrete, ModelSamplingContinuousEDM

opEmptyLatentImage = EmptyLatentImage()
opVAEDecode = VAEDecode()
opVAEEncode = VAEEncode()
opVAEDecodeTiled = VAEDecodeTiled()
opVAEEncodeTiled = VAEEncodeTiled()
opControlNetApplyAdvanced = ControlNetApplyAdvanced()
opFreeU = FreeU_V2()
opModelSamplingDiscrete = ModelSamplingDiscrete()
opModelSamplingContinuousEDM = ModelSamplingContinuousEDM()

_anima_reference_sampler_cache = {}
_anima_reference_sampler_announced = set()


class StableDiffusionModel:
    def __init__(self, unet=None, vae=None, clip=None, clip_vision=None, filename=None, vae_filename=None):
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision
        self.filename = filename
        self.vae_filename = vae_filename
        self.unet_with_lora = unet
        self.clip_with_lora = clip
        self.visited_loras = ''
        self.loras_to_load = []
        if self.unet_with_lora is not None:
            self.unet_with_lora.loras_to_load = []

        self.lora_key_map_unet = {}
        self.lora_key_map_clip = {}

        if self.unet is not None:
            self.lora_key_map_unet = model_lora_keys_unet(self.unet.model, self.lora_key_map_unet)
            self.lora_key_map_unet.update({x: x for x in self.unet.model.state_dict().keys()})

        if self.clip is not None:
            self.lora_key_map_clip = model_lora_keys_clip(self.clip.cond_stage_model, self.lora_key_map_clip)
            self.lora_key_map_clip.update({x: x for x in self.clip.cond_stage_model.state_dict().keys()})

    @torch.no_grad()
    @torch.inference_mode()
    def refresh_loras(self, loras):
        assert isinstance(loras, list)

        if self.visited_loras == str(loras):
            return

        self.visited_loras = str(loras)

        if self.unet is None:
            return

        print(f'Request to load LoRAs {str(loras)} for model [{self.filename}].')

        loras_to_load = []

        for filename, weight in loras:
            if filename == 'None':
                continue

            if os.path.exists(filename):
                lora_filename = filename
            else:
                lora_filename = get_file_from_folder_list(filename, modules.config.paths_loras)

            if not os.path.exists(lora_filename):
                print(f'Lora file not found: {lora_filename}')
                continue

            loras_to_load.append((lora_filename, weight))

        self.unet_with_lora = self.unet.clone() if self.unet is not None else None
        if self.unet_with_lora is not None:
            self.unet_with_lora.model_file = self.filename
            self.unet_with_lora.loras_to_load = list(loras_to_load)
        self.loras_to_load = list(loras_to_load)
        self.clip_with_lora = self.clip.clone() if self.clip is not None else None

        for lora_filename, weight in loras_to_load:
            lora_unmatch = ldm_patched.modules.utils.load_torch_file(lora_filename, safe_load=False)
            lora_unet, lora_unmatch = match_lora(lora_unmatch, self.lora_key_map_unet)
            lora_clip, lora_unmatch = match_lora(lora_unmatch, self.lora_key_map_clip)

            if len(lora_unmatch) > 12:
                # model mismatch
                continue

            if len(lora_unmatch) > 0:
                print(f'Loaded LoRA [{lora_filename}] for model [{self.filename}] '
                      f'with unmatched keys {list(lora_unmatch.keys())}')

            if self.unet_with_lora is not None and len(lora_unet) > 0:
                loaded_keys = self.unet_with_lora.add_patches(lora_unet, weight)
                print(f'Loaded LoRA [{lora_filename}] for UNet [{self.filename}] '
                      f'with {len(loaded_keys)} keys at weight {weight}.')
                for item in lora_unet:
                    if item not in loaded_keys:
                        print("UNet LoRA key skipped: ", item)

            if self.clip_with_lora is not None and len(lora_clip) > 0:
                loaded_keys = self.clip_with_lora.add_patches(lora_clip, weight)
                print(f'Loaded LoRA [{lora_filename}] for CLIP [{self.filename}] '
                      f'with {len(loaded_keys)} keys at weight {weight}.')
                for item in lora_clip:
                    if item not in loaded_keys:
                        print("CLIP LoRA key skipped: ", item)


@torch.no_grad()
@torch.inference_mode()
def apply_freeu(model, b1, b2, s1, s2):
    return opFreeU.patch(model=model, b1=b1, b2=b2, s1=s1, s2=s2)[0]


@torch.no_grad()
@torch.inference_mode()
def load_controlnet(ckpt_filename):
    return ldm_patched.modules.controlnet.load_controlnet(ckpt_filename)


@torch.no_grad()
@torch.inference_mode()
def apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent):
    return opControlNetApplyAdvanced.apply_controlnet(positive=positive, negative=negative, control_net=control_net,
        image=image, strength=strength, start_percent=start_percent, end_percent=end_percent)


@torch.no_grad()
@torch.inference_mode()
def load_model(ckpt_filename, vae_filename=None):
    unet, clip, vae, vae_filename, clip_vision = load_checkpoint_guess_config(ckpt_filename, embedding_directory=path_embeddings,
                                                                vae_filename_param=vae_filename)
    if unet is not None:
        unet.model_file = ckpt_filename
    return StableDiffusionModel(unet=unet, clip=clip, vae=vae, clip_vision=clip_vision, filename=ckpt_filename, vae_filename=vae_filename)


def _is_anima_model_patcher(model):
    return hasattr(model, "model") and model.model.__class__.__name__ == "Anima"


_is_anima_checkpoint_cache = {}


def is_anima_checkpoint_file(filename_or_path):
    if not filename_or_path or not isinstance(filename_or_path, str):
        return False
    fn_lower = filename_or_path.lower()
    if "anima" in fn_lower:
        return True

    if filename_or_path in _is_anima_checkpoint_cache:
        return _is_anima_checkpoint_cache[filename_or_path]

    full_path = filename_or_path
    if not os.path.isfile(full_path):
        try:
            import modules.default_pipeline as _dp
            import modules.config as _cfg
            full_path = _dp.get_file_from_folder_list(filename_or_path, _cfg.paths_checkpoints)
        except Exception:
            pass

    if full_path and os.path.isfile(full_path) and full_path.endswith('.safetensors'):
        try:
            from safetensors import safe_open
            with safe_open(full_path, framework="pt", device="cpu") as f:
                keys = f.keys()
                is_anima = any("llm_adapter" in k or "x_embedder" in k for k in keys)
                _is_anima_checkpoint_cache[filename_or_path] = is_anima
                _is_anima_checkpoint_cache[full_path] = is_anima
                return is_anima
        except Exception:
            pass

    _is_anima_checkpoint_cache[filename_or_path] = False
    return False


def is_anima_model(model):
    if model is None:
        return False
    if isinstance(model, str):
        return is_anima_checkpoint_file(model)
    if _is_anima_model_patcher(model):
        return True
    inner = getattr(model, "model", None)
    if inner is not None:
        if inner.__class__.__name__ == "Anima":
            return True
        diff = getattr(inner, "diffusion_model", None)
        if diff is not None and diff.__class__.__name__ in ("MiniTrainDIT", "CosmosTransformer", "Anima"):
            return True
    filename = getattr(model, "model_file", None) or getattr(model, "filename", None)
    if filename:
        return is_anima_checkpoint_file(filename)
    return False


_COMFY_AIMDO_STUBS = {
    "__init__.py": (
        '"""Lightweight stubs for optional ComfyUI AIMDO integrations.\n\n'
        'These placeholders are enough for the Anima sampler reference path,\n'
        'which only needs the Python imports to succeed.\n'
        '"""\n'
    ),
    "host_buffer.py": (
        '"""Host buffer stub used when AIMDO is unavailable."""\n\n\n'
        'class HostBuffer:\n'
        '    def __init__(self, size):\n'
        '        self.size = int(size)\n'
    ),
    "model_vbar.py": (
        '"""No-op fallback for the optional AIMDO virtual BAR helpers."""\n\n\n'
        'class ModelVBAR:\n'
        '    def __init__(self, size, device_index=None):\n'
        '        self.size = int(size)\n'
        '        self.device_index = device_index\n\n'
        '    def loaded_size(self):\n'
        '        return 0\n\n'
        '    def prioritize(self):\n'
        '        return None\n\n\n'
        'def vbar_fault(_vbar):\n'
        '    return None\n\n\n'
        'def vbar_signature_compare(_signature, _other_signature):\n'
        '    return True\n\n\n'
        'def vbar_unpin(_vbar):\n'
        '    return None\n\n\n'
        'def vbars_analyze():\n'
        '    return 0\n\n\n'
        'def vbars_reset_watermark_limits():\n'
        '    return None\n'
    ),
    "malloc_graph.py": (
        '"""Malloc graph stub used when AIMDO is unavailable."""\n\n\n'
        'class _MallocGraph:\n'
        '    def __init__(self):\n'
        '        self.rogue_count = 0\n'
        '        self._comfy_active = False\n'
        '        self._comfy_cuda_graph_modules = set()\n'
        '    def push(self):\n'
        '        pass\n'
        '    def pop(self):\n'
        '        return False\n'
        '    def pause(self, sync=False):\n'
        '        pass\n'
        '    def resume(self, sync=False):\n'
        '        pass\n'
        '    def abort(self):\n'
        '        pass\n\n\n'
        'def record(stream=None, assert_graph_breaks=False):\n'
        '    return _MallocGraph()\n'
    ),
    "control.py": (
        '"""Control stubs for optional AIMDO integrations."""\n\n\n'
        'def init(*args, **kwargs):\n'
        '    return None\n'
    ),
    "model_mmap.py": (
        '"""Model mmap stub used when AIMDO is unavailable."""\n\n\n'
        'class ModelMmap:\n'
        '    def __init__(self, *args, **kwargs):\n'
        '        pass\n'
    ),
    "storage.py": (
        '"""Storage stubs for optional AIMDO integrations."""\n\n\n'
        'def fast_disk(_path):\n'
        '    return None\n'
    ),
    "torch.py": (
        '"""Torch bridge stubs for optional AIMDO integrations."""\n\n'
        'import torch\n\n\n'
        'def aimdo_to_tensor(_vbar, device):\n'
        '    return torch.empty(0, device=device)\n\n\n'
        'def hostbuf_to_tensor(hostbuf):\n'
        '    return torch.empty(hostbuf.size, dtype=torch.uint8)\n'
    ),
    "vram_buffer.py": (
        '"""VRAM buffer stub used when AIMDO is unavailable."""\n\n\n'
        'class VRAMBuffer:\n'
        '    def __init__(self, size, device_index=None):\n'
        '        self.size = int(size)\n'
        '        self.device_index = device_index\n'
    ),
}


class _AimdoLoader:
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        fullname = module.__name__
        if fullname == "comfy_aimdo":
            module.__path__ = []
            module.__package__ = "comfy_aimdo"
            return

        sub = fullname.split(".", 1)[1] if "." in fullname else ""
        if sub == "malloc_graph":
            class _MallocGraph:
                def __init__(self):
                    self.rogue_count = 0
                    self._comfy_active = False
                    self._comfy_cuda_graph_modules = set()
                def push(self): pass
                def pop(self): return False
                def pause(self, sync=False): pass
                def resume(self, sync=False): pass
                def abort(self): pass

            module.record = lambda stream=None, assert_graph_breaks=False: _MallocGraph()
        elif sub == "host_buffer":
            class HostBuffer:
                def __init__(self, size):
                    self.size = int(size)
            module.HostBuffer = HostBuffer
        elif sub == "vram_buffer":
            class VRAMBuffer:
                def __init__(self, size, device_index=None):
                    self.size = int(size)
                    self.device_index = device_index
            module.VRAMBuffer = VRAMBuffer
        elif sub == "model_vbar":
            class ModelVBAR:
                def __init__(self, size, device_index=None):
                    self.size = int(size)
                    self.device_index = device_index
                def loaded_size(self): return 0
                def prioritize(self): return None
            module.ModelVBAR = ModelVBAR
            module.vbar_fault = lambda _v: None
            module.vbar_signature_compare = lambda _s, _o: True
            module.vbar_unpin = lambda _v: None
            module.vbars_analyze = lambda: 0
            module.vbars_reset_watermark_limits = lambda: None
        elif sub == "storage":
            module.fast_disk = lambda _p: None
        elif sub == "control":
            module.init = lambda *a, **k: None
        elif sub == "model_mmap":
            class ModelMmap:
                def __init__(self, *a, **k): pass
            module.ModelMmap = ModelMmap
        elif sub == "torch":
            module.aimdo_to_tensor = lambda _v, device: torch.empty(0, device=device)
            module.hostbuf_to_tensor = lambda h: torch.empty(h.size, dtype=torch.uint8)


class _AimdoFinder:
    def find_spec(self, fullname, path, target=None):
        if fullname == "comfy_aimdo" or fullname.startswith("comfy_aimdo."):
            from importlib.machinery import ModuleSpec
            return ModuleSpec(fullname, _AimdoLoader(), is_package=(fullname == "comfy_aimdo"))
        return None


def _register_aimdo_in_memory():
    import types
    if not any(isinstance(f, _AimdoFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AimdoFinder())

    if "comfy_aimdo" not in sys.modules:
        m = types.ModuleType("comfy_aimdo")
        m.__path__ = []
        m.__package__ = "comfy_aimdo"
        sys.modules["comfy_aimdo"] = m
    else:
        m = sys.modules["comfy_aimdo"]
        if not hasattr(m, "__path__") or not isinstance(m.__path__, list):
            m.__path__ = []
        if not hasattr(m, "__package__"):
            m.__package__ = "comfy_aimdo"

    if "comfy_aimdo.malloc_graph" not in sys.modules:
        mg = types.ModuleType("comfy_aimdo.malloc_graph")
        class _MallocGraph:
            def __init__(self):
                self.rogue_count = 0
                self._comfy_active = False
                self._comfy_cuda_graph_modules = set()
            def push(self): pass
            def pop(self): return False
            def pause(self, sync=False): pass
            def resume(self, sync=False): pass
            def abort(self): pass

        mg.record = lambda stream=None, assert_graph_breaks=False: _MallocGraph()
        m.malloc_graph = mg
        sys.modules["comfy_aimdo.malloc_graph"] = mg

    if "comfy_aimdo.storage" not in sys.modules:
        s = types.ModuleType("comfy_aimdo.storage")
        s.fast_disk = lambda _path: None
        m.storage = s
        sys.modules["comfy_aimdo.storage"] = s

    if "comfy_aimdo.host_buffer" not in sys.modules:
        hb = types.ModuleType("comfy_aimdo.host_buffer")
        class HostBuffer:
            def __init__(self, size):
                self.size = int(size)
        hb.HostBuffer = HostBuffer
        m.host_buffer = hb
        sys.modules["comfy_aimdo.host_buffer"] = hb

    if "comfy_aimdo.vram_buffer" not in sys.modules:
        vb = types.ModuleType("comfy_aimdo.vram_buffer")
        class VRAMBuffer:
            def __init__(self, size, device_index=None):
                self.size = int(size)
                self.device_index = device_index
        vb.VRAMBuffer = VRAMBuffer
        m.vram_buffer = vb
        sys.modules["comfy_aimdo.vram_buffer"] = vb

    if "comfy_aimdo.model_vbar" not in sys.modules:
        mv = types.ModuleType("comfy_aimdo.model_vbar")
        class ModelVBAR:
            def __init__(self, size, device_index=None):
                self.size = int(size)
                self.device_index = device_index
            def loaded_size(self): return 0
            def prioritize(self): return None
        mv.ModelVBAR = ModelVBAR
        mv.vbar_fault = lambda _v: None
        mv.vbar_signature_compare = lambda _s, _o: True
        mv.vbar_unpin = lambda _v: None
        mv.vbars_analyze = lambda: 0
        mv.vbars_reset_watermark_limits = lambda: None
        m.model_vbar = mv
        sys.modules["comfy_aimdo.model_vbar"] = mv

    if "comfy_aimdo.torch" not in sys.modules:
        ct = types.ModuleType("comfy_aimdo.torch")
        ct.aimdo_to_tensor = lambda _v, device: torch.empty(0, device=device)
        ct.hostbuf_to_tensor = lambda h: torch.empty(h.size, dtype=torch.uint8)
        m.torch = ct
        sys.modules["comfy_aimdo.torch"] = ct

    if "comfy_aimdo.control" not in sys.modules:
        ctrl = types.ModuleType("comfy_aimdo.control")
        ctrl.init = lambda *a, **k: None
        m.control = ctrl
        sys.modules["comfy_aimdo.control"] = ctrl

    if "comfy_aimdo.model_mmap" not in sys.modules:
        mm = types.ModuleType("comfy_aimdo.model_mmap")
        class ModelMmap:
            def __init__(self, *a, **k): pass
        mm.ModelMmap = ModelMmap
        m.model_mmap = mm
        sys.modules["comfy_aimdo.model_mmap"] = mm

    # Ensure broken comfy_kitchen in-memory stub is not present so ComfyUI uses its native safe fallback via ImportError
    if "comfy_kitchen" in sys.modules:
        ck = sys.modules.get("comfy_kitchen")
        if not isinstance(getattr(ck, "__file__", None), str) or not isinstance(getattr(ck, "__path__", None), (list, tuple)):
            sys.modules.pop("comfy_kitchen", None)
    if "comfy_kitchen.tensor" in sys.modules:
        ckt = sys.modules.get("comfy_kitchen.tensor")
        if not isinstance(getattr(ckt, "__file__", None), str):
            sys.modules.pop("comfy_kitchen.tensor", None)


_register_aimdo_in_memory()


def _default_anima_comfy_root():
    if os.path.isdir("/content"):
        return "/content/ComfyUI"
    fooocus_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(fooocus_root, "comfyui_tmp")


def _ensure_aimdo_stubs_on_disk(comfy_root):
    stub_dir = os.path.join(comfy_root, "comfy_aimdo")
    os.makedirs(stub_dir, exist_ok=True)
    for name, body in _COMFY_AIMDO_STUBS.items():
        target = os.path.join(stub_dir, name)
        if not os.path.exists(target) or os.path.getsize(target) == 0:
            with open(target, "w", encoding="utf-8") as f:
                f.write(body)
    m = sys.modules.get("comfy_aimdo")
    if m is not None and hasattr(m, "__path__") and isinstance(m.__path__, list):
        if stub_dir not in m.__path__:
            m.__path__.append(stub_dir)


def _bootstrap_anima_comfy_reference(comfy_root):
    import subprocess as _sp
    comfy_sd = os.path.join(comfy_root, "comfy", "sd.py")
    if not os.path.exists(comfy_sd):
        parent = os.path.dirname(comfy_root) or "."
        os.makedirs(parent, exist_ok=True)
        print(f"[Anima] Cloning ComfyUI reference into {comfy_root} (shallow clone)...")
        _sp.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/comfyanonymous/ComfyUI.git", comfy_root],
            check=True,
        )
    _ensure_aimdo_stubs_on_disk(comfy_root)
    os.environ["FOOOCUS_ANIMA_COMFY_ROOT"] = comfy_root
    print(f"[Anima] FOOOCUS_ANIMA_COMFY_ROOT={comfy_root}")
    return comfy_root


def _get_anima_reference_comfy_root(auto_bootstrap=False):
    fooocus_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.environ.get("FOOOCUS_ANIMA_COMFY_ROOT"),
        "/content/ComfyUI",
        os.path.join(fooocus_root, "comfyui_tmp"),
    ]
    for root in candidates:
        if not root:
            continue
        if not os.path.exists(os.path.join(root, "comfy", "sd.py")):
            continue
        _ensure_aimdo_stubs_on_disk(root)
        return root
    if auto_bootstrap:
        try:
            return _bootstrap_anima_comfy_reference(_default_anima_comfy_root())
        except Exception as e:
            print(f"[Anima] Failed to auto-bootstrap ComfyUI reference: {e}")
    return None


def _load_anima_reference_modules():
    _register_aimdo_in_memory()
    comfy_root = _get_anima_reference_comfy_root()
    if comfy_root is None:
        return None, None
    if comfy_root not in sys.path:
        sys.path.insert(0, comfy_root)
    import comfy.sample
    import comfy.sd

    return comfy.sample, comfy.sd


_anima_lora_cache = {}


def _get_anima_lora_dict(lora_path, comfy_utils):
    if not os.path.exists(lora_path):
        return None
    try:
        mtime = os.path.getmtime(lora_path)
    except Exception:
        mtime = 0
    cached = _anima_lora_cache.get(lora_path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    lora_dict = comfy_utils.load_torch_file(lora_path, safe_load=True)
    _anima_lora_cache[lora_path] = (mtime, lora_dict)
    return lora_dict


def _get_anima_reference_model(model):
    ckpt_filename = getattr(model, "model_file", None)
    if not ckpt_filename and hasattr(model, "filename"):
        ckpt_filename = model.filename
    if not ckpt_filename:
        try:
            import modules.default_pipeline as _dp
            ckpt_filename = getattr(_dp.model_base, "filename", None)
        except Exception:
            pass
    if not ckpt_filename:
        return None

    comfy_sample, comfy_sd = _load_anima_reference_modules()
    if comfy_sample is None or comfy_sd is None:
        return None

    cached = _anima_reference_sampler_cache.get(ckpt_filename)
    if cached is None:
        cached, _clip, _vae, _clipvision = comfy_sd.load_checkpoint_guess_config(
            ckpt_filename,
            output_vae=False,
            output_clip=False,
            output_clipvision=False,
            embedding_directory=path_embeddings,
        )
        _anima_reference_sampler_cache[ckpt_filename] = cached

    # Free duplicate DiT weights from Fooocus so only ONE copy of the 4.5 GB model exists in VRAM
    try:
        import modules.default_pipeline as _dp
        if hasattr(_dp, "model_base") and getattr(_dp.model_base, "unet", None) is not None:
            base_unet = _dp.model_base.unet
            if hasattr(base_unet, "model") and hasattr(base_unet.model, "diffusion_model"):
                if base_unet.model.diffusion_model is not cached.model.diffusion_model:
                    old_diff = base_unet.model.diffusion_model
                    base_unet.model.diffusion_model = cached.model.diffusion_model
                    del old_diff
        if hasattr(model, "model") and hasattr(model.model, "diffusion_model"):
            if model.model.diffusion_model is not cached.model.diffusion_model:
                old_diff = model.model.diffusion_model
                model.model.diffusion_model = cached.model.diffusion_model
                del old_diff
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as e:
        pass

    out_model = cached.clone()

    loras_to_load = getattr(model, "loras_to_load", None)
    if loras_to_load is None:
        try:
            import modules.default_pipeline as _dp
            loras_to_load = getattr(getattr(_dp.model_base, "unet_with_lora", None), "loras_to_load", None)
            if loras_to_load is None:
                loras_to_load = getattr(_dp.model_base, "loras_to_load", None)
        except Exception:
            pass

    if loras_to_load:
        import comfy.utils
        for lora_path, weight in loras_to_load:
            if not weight:
                continue
            try:
                lora_dict = _get_anima_lora_dict(lora_path, comfy.utils)
                if lora_dict is not None:
                    patched_model, _ = comfy_sd.load_lora_for_models(out_model, None, lora_dict, weight, 0)
                    if patched_model is not None:
                        out_model = patched_model
                        print(f"[AnimaSampler] Applied LoRA to Comfy reference: {os.path.basename(lora_path)} (weight={weight})")
            except Exception as e:
                print(f"[AnimaSampler] Failed to apply LoRA {lora_path}: {e}")

    # Apply ControlNet-LLLite tasks if present
    lllite_tasks = getattr(model, "lllite_tasks", None)
    if lllite_tasks is None:
        try:
            import modules.default_pipeline as _dp
            lllite_tasks = getattr(getattr(_dp.model_base, "unet_with_lora", None), "lllite_tasks", None)
            if lllite_tasks is None:
                lllite_tasks = getattr(_dp.model_base, "lllite_tasks", None)
            if lllite_tasks is None:
                lllite_tasks = getattr(_dp, "anima_lllite_tasks", None)
        except Exception:
            pass

    if lllite_tasks:
        import modules.anima_lllite as anima_lllite
        for task in lllite_tasks:
            try:
                weights_path, img_tensor, weight, stop_at = task
                if weights_path and os.path.isfile(weights_path) and img_tensor is not None:
                    out_model = anima_lllite.apply_lllite_to_model(
                        out_model,
                        weights_path=weights_path,
                        image=img_tensor,
                        strength=float(weight),
                        start_percent=0.0,
                        end_percent=float(stop_at)
                    )
                    print(f"[AnimaSampler] Applied ControlNet-LLLite: {os.path.basename(weights_path)} (strength={weight}, stop_at={stop_at})")
            except Exception as e:
                print(f"[AnimaSampler] Failed to apply ControlNet-LLLite: {e}")

    # Force ComfyUI to evaluate CFG (positive & negative) sequentially with batch_size=1
    # on VRAM-constrained GPUs (<= 18 GB like Tesla T4), cutting peak DiT self-attention memory in half
    # and preventing CUDA OOM / allocator freeze during Upscale 1.5x / 2x.
    try:
        base_model_obj = out_model.model
        orig_mem_req = getattr(base_model_obj, "_orig_memory_required", None)
        if orig_mem_req is None:
            orig_mem_req = base_model_obj.memory_required
            base_model_obj._orig_memory_required = orig_mem_req

        def _anima_memory_required(input_shape, cond_shapes=None):
            if cond_shapes is None:
                cond_shapes = {}
            batch = input_shape[0] if len(input_shape) > 0 else 1
            if batch > 1:
                return 1e12
            try:
                return orig_mem_req(input_shape, cond_shapes=cond_shapes)
            except Exception:
                return 0

        base_model_obj.memory_required = _anima_memory_required
        print("[AnimaSampler] Sequential CFG batching enabled (batch=1 mode to protect VRAM)")
    except Exception as e:
        pass

    # Ensure ComfyUI's Wan21 latent_format process_in / process_out always receives 5D latents (B, C, T, H, W)
    # so PyTorch broadcasting never matches batch (dim 0) with channels (dim 1), which would turn 1 frame into 16 frames
    try:
        lf = getattr(out_model.model, "latent_format", None)
        if lf is not None and hasattr(lf, "process_in"):
            orig_process_in = getattr(lf, "_orig_process_in", None)
            if orig_process_in is None:
                orig_process_in = lf.process_in
                lf._orig_process_in = orig_process_in
            def _safe_process_in(latent):
                if hasattr(latent, "ndim") and latent.ndim == 4:
                    latent = latent.unsqueeze(2)
                return orig_process_in(latent)
            lf.process_in = _safe_process_in

        if lf is not None and hasattr(lf, "process_out"):
            orig_process_out = getattr(lf, "_orig_process_out", None)
            if orig_process_out is None:
                orig_process_out = lf.process_out
                lf._orig_process_out = orig_process_out
            def _safe_process_out(latent):
                if hasattr(latent, "ndim") and latent.ndim == 4:
                    latent = latent.unsqueeze(2)
                return orig_process_out(latent)
            lf.process_out = _safe_process_out
    except Exception as e:
        pass

    return out_model


def _can_use_anima_reference_sampler(model, refiner):
    if refiner is not None:
        return False
    if not _is_anima_model_patcher(model):
        return False
    # Auto-bootstrap the ComfyUI reference checkout the first time we hit this for an Anima model.
    # Fooocus' standard sampler does not support Anima's 5D (B,C,T,H,W) latents, so without
    # this the run crashes in anisotropic.adaptive_anisotropic_filter.
    if _get_anima_reference_comfy_root(auto_bootstrap=True) is None:
        return False
    return True


@torch.no_grad()
@torch.inference_mode()
def generate_empty_latent(width=1024, height=1024, batch_size=1):
    return opEmptyLatentImage.generate(width=width, height=height, batch_size=batch_size)[0]


@torch.no_grad()
@torch.inference_mode()
def decode_vae(vae, latent_image, tiled=False):
    if hasattr(vae, "first_stage_model") and hasattr(vae.first_stage_model, "to") and hasattr(vae, "device"):
        vae.first_stage_model.to(vae.device)
    samples = latent_image.get("samples") if isinstance(latent_image, dict) else latent_image
    is_5d = samples is not None and getattr(samples, "ndim", 4) == 5
    if tiled and not is_5d:
        return opVAEDecodeTiled.decode(samples=latent_image, vae=vae, tile_size=512)[0]
    else:
        return opVAEDecode.decode(samples=latent_image, vae=vae)[0]


@torch.no_grad()
@torch.inference_mode()
def encode_vae(vae, pixels, tiled=False):
    if hasattr(vae, "first_stage_model") and hasattr(vae.first_stage_model, "to") and hasattr(vae, "device"):
        vae.first_stage_model.to(vae.device)
    is_3d_vae = getattr(vae, "latent_dim", 2) == 3 or (
        hasattr(vae, "first_stage_model") and getattr(vae.first_stage_model, "latent_dim", 2) == 3
    )
    if tiled and not is_3d_vae:
        return opVAEEncodeTiled.encode(pixels=pixels, vae=vae, tile_size=512)[0]
    else:
        return opVAEEncode.encode(pixels=pixels, vae=vae)[0]


@torch.no_grad()
@torch.inference_mode()
def encode_vae_inpaint(vae, pixels, mask):
    assert mask.ndim == 3 and pixels.ndim == 4
    assert mask.shape[-1] == pixels.shape[-2]
    assert mask.shape[-2] == pixels.shape[-3]

    w = mask.round()[..., None]
    pixels = pixels * (1 - w) + 0.5 * w

    latent = vae.encode(pixels)
    B, C, H, W = latent.shape

    latent_mask = mask[:, None, :, :]
    latent_mask = torch.nn.functional.interpolate(latent_mask, size=(H * 8, W * 8), mode="bilinear").round()
    latent_mask = torch.nn.functional.max_pool2d(latent_mask, (8, 8)).round().to(latent)

    return latent, latent_mask


class VAEApprox(torch.nn.Module):
    def __init__(self):
        super(VAEApprox, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, (7, 7))
        self.conv2 = torch.nn.Conv2d(8, 16, (5, 5))
        self.conv3 = torch.nn.Conv2d(16, 32, (3, 3))
        self.conv4 = torch.nn.Conv2d(32, 64, (3, 3))
        self.conv5 = torch.nn.Conv2d(64, 32, (3, 3))
        self.conv6 = torch.nn.Conv2d(32, 16, (3, 3))
        self.conv7 = torch.nn.Conv2d(16, 8, (3, 3))
        self.conv8 = torch.nn.Conv2d(8, 3, (3, 3))
        self.current_type = None

    def forward(self, x):
        extra = 11
        x = torch.nn.functional.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
        x = torch.nn.functional.pad(x, (extra, extra, extra, extra))
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(x, 0.1)
        return x


VAE_approx_models = {}


@torch.no_grad()
@torch.inference_mode()
def get_previewer(model):
    global VAE_approx_models

    from modules.config import path_vae_approx

    latent_format = getattr(getattr(model, 'model', None), 'latent_format', None)
    if latent_format is None:
        try:
            import modules.default_pipeline as _dp
            latent_format = getattr(getattr(_dp.model_base, 'unet', None), 'latent_format', None)
        except Exception:
            pass

    latent_channels = getattr(latent_format, 'latent_channels', 4)

    # For models with non-4-channel latents (e.g., Anima with 16ch Wan21 latents),
    # use direct linear RGB projection via latent_rgb_factors for zero-VRAM live preview.
    if latent_channels != 4:
        rgb_factors = getattr(latent_format, 'latent_rgb_factors', None)
        rgb_bias = getattr(latent_format, 'latent_rgb_factors_bias', None)
        if rgb_factors is None:
            try:
                from ldm_patched.modules.latent_formats import Wan21
                w = Wan21()
                rgb_factors = w.latent_rgb_factors
                rgb_bias = w.latent_rgb_factors_bias
            except Exception:
                pass

        if rgb_factors is not None:
            factors_np = np.array(rgb_factors, dtype=np.float32)
            bias_np = np.array(rgb_bias, dtype=np.float32) if rgb_bias is not None else None

            @torch.no_grad()
            @torch.inference_mode()
            def anima_preview_function(x0, step, total_steps):
                with torch.no_grad():
                    if x0 is None:
                        return None
                    if hasattr(x0, "ndim") and x0.ndim == 5:
                        x0 = x0[:, :, 0, :, :]
                    latent = x0[0].detach().float().cpu().numpy()
                    latent_hwc = np.transpose(latent, (1, 2, 0))
                    rgb = np.matmul(latent_hwc, factors_np)
                    if bias_np is not None:
                        rgb = rgb + bias_np
                    rgb_scaled = np.clip(((rgb + 1.0) / 2.0) * 255.0, 0, 255).astype(np.uint8)
                    return np.repeat(np.repeat(rgb_scaled, 2, axis=0), 2, axis=1)

            return anima_preview_function
        return None

    is_sdxl = isinstance(getattr(model, 'model', None) and model.model.latent_format, ldm_patched.modules.latent_formats.SDXL)
    vae_approx_filename = os.path.join(path_vae_approx, 'xlvaeapp.pth' if is_sdxl else 'vaeapp_sd15.pth')

    if vae_approx_filename in VAE_approx_models:
        VAE_approx_model = VAE_approx_models[vae_approx_filename]
    else:
        sd = torch.load(vae_approx_filename, map_location='cpu', weights_only=True)
        VAE_approx_model = VAEApprox()
        VAE_approx_model.load_state_dict(sd)
        del sd
        VAE_approx_model.eval()

        if ldm_patched.modules.model_management.should_use_fp16():
            VAE_approx_model.half()
            VAE_approx_model.current_type = torch.float16
        else:
            VAE_approx_model.float()
            VAE_approx_model.current_type = torch.float32

        VAE_approx_model.to(ldm_patched.modules.model_management.get_torch_device())
        VAE_approx_models[vae_approx_filename] = VAE_approx_model

    @torch.no_grad()
    @torch.inference_mode()
    def preview_function(x0, step, total_steps):
        with torch.no_grad():
            x_sample = x0.to(VAE_approx_model.current_type)
            x_sample = VAE_approx_model(x_sample) * 127.5 + 127.5
            x_sample = einops.rearrange(x_sample, 'b c h w -> b h w c')[0]
            x_sample = x_sample.cpu().numpy().clip(0, 255).astype(np.uint8)
            return x_sample

    return preview_function


@torch.no_grad()
@torch.inference_mode()
def ksampler(model, positive, negative, latent, seed=None, steps=30, cfg=7.0, sampler_name='dpmpp_2m_sde_gpu',
             scheduler='karras', denoise=1.0, disable_noise=False, start_step=None, last_step=None,
             force_full_denoise=False, callback_function=None, refiner=None, refiner_switch=-1,
             previewer_start=None, previewer_end=None, sigmas=None, noise_mean=None, disable_preview=False):

    if sigmas is not None:
        sigmas = sigmas.clone().to(ldm_patched.modules.model_management.get_torch_device())

    latent_image = latent["samples"]
    is_anima_sampler = _can_use_anima_reference_sampler(model, refiner)
    orig_latent_ndim = getattr(latent_image, "ndim", 4)
    if is_anima_sampler and orig_latent_ndim == 4:
        latent_image = latent_image.unsqueeze(2)

    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
    else:
        batch_inds = latent["batch_index"] if "batch_index" in latent else None
        noise = ldm_patched.modules.sample.prepare_noise(latent_image, seed, batch_inds)

    if isinstance(noise_mean, torch.Tensor):
        noise = noise + noise_mean - torch.mean(noise, dim=1, keepdim=True)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]
        if is_anima_sampler and noise_mask is not None and getattr(noise_mask, "ndim", 4) == 4:
            noise_mask = noise_mask.unsqueeze(2)

    previewer = get_previewer(model)

    if previewer_start is None:
        previewer_start = 0

    if previewer_end is None:
        previewer_end = steps

    def callback(step, x0, x, total_steps):
        ldm_patched.modules.model_management.throw_exception_if_processing_interrupted()
        y = None
        if previewer is not None and not disable_preview:
            y = previewer(x0, previewer_start + step, previewer_end)
        if callback_function is not None:
            callback_function(previewer_start + step, x0, x, previewer_end, y)

    disable_pbar = False

    if _can_use_anima_reference_sampler(model, refiner):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            try:
                free_mem, total_mem = torch.cuda.mem_get_info()
                print(f"[AnimaSampler] GPU VRAM before sampling: {free_mem / (1024**2):.1f} MB free / {total_mem / (1024**2):.1f} MB total")
            except Exception:
                pass

        comfy_sample, _comfy_sd = _load_anima_reference_modules()
        reference_model = _get_anima_reference_model(model)
        if reference_model is not None and comfy_sample is not None:
            model_file = getattr(model, "model_file", "<unknown>")
            if model_file not in _anima_reference_sampler_announced:
                comfy_root = _get_anima_reference_comfy_root()
                print(f"[AnimaSampler] Using Comfy reference sampler from {comfy_root}")
                _anima_reference_sampler_announced.add(model_file)
            try:
                samples = comfy_sample.sample(
                    model=reference_model,
                    noise=noise,
                    steps=steps,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    scheduler=scheduler,
                    positive=positive,
                    negative=negative,
                    latent_image=latent_image,
                    denoise=denoise,
                    disable_noise=disable_noise,
                    start_step=start_step,
                    last_step=last_step,
                    force_full_denoise=force_full_denoise,
                    noise_mask=noise_mask,
                    sigmas=sigmas,
                    callback=callback,
                    disable_pbar=disable_pbar,
                    seed=seed,
                )
            finally:
                try:
                    reference_model.unpatch_model()
                except Exception:
                    pass
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            out = latent.copy()
            if is_anima_sampler and orig_latent_ndim == 4 and getattr(samples, "ndim", 4) == 5:
                samples = samples.squeeze(2)
            out["samples"] = samples
            return out

    modules.sample_hijack.current_refiner = refiner
    modules.sample_hijack.refiner_switch_step = refiner_switch
    ldm_patched.modules.samplers.sample = modules.sample_hijack.sample_hacked

    try:
        samples = ldm_patched.modules.sample.sample(model,
                                                    noise, steps, cfg, sampler_name, scheduler,
                                                    positive, negative, latent_image,
                                                    denoise=denoise, disable_noise=disable_noise,
                                                    start_step=start_step,
                                                    last_step=last_step,
                                                    force_full_denoise=force_full_denoise, noise_mask=noise_mask,
                                                    callback=callback,
                                                    disable_pbar=disable_pbar, seed=seed, sigmas=sigmas)

        out = latent.copy()
        out["samples"] = samples
    finally:
        modules.sample_hijack.current_refiner = None

    return out


@torch.no_grad()
@torch.inference_mode()
def pytorch_to_numpy(x):
    return [np.clip(255. * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]


@torch.no_grad()
@torch.inference_mode()
def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y
