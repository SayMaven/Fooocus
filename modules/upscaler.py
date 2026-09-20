import os
from collections import OrderedDict

import modules.core as core
import torch
import ldm_patched.modules.utils
from ldm_patched.contrib.external_upscale_model import ImageUpscaleWithModel
from ldm_patched.pfn import model_loading
from ldm_patched.pfn.architecture.RRDB import RRDBNet as ESRGAN
from modules.config import path_upscale_models, downloading_upscale_model

opImageUpscaleWithModel = ImageUpscaleWithModel()
model = None
loaded_model_filename = None


def _resolve_upscale_path(model_str):
    if not model_str or model_str in ['Default (Fooocus)', 'Default', 'default', 'fooocus_upscaler', 'None', '']:
        return None
    if os.path.exists(model_str):
        return model_str
    custom_path = os.path.join(path_upscale_models, model_str)
    if os.path.exists(custom_path):
        return custom_path
    for ext in ['', '.safetensors', '.pt', '.pth']:
        if os.path.exists(custom_path + ext):
            return custom_path + ext
    base_name = os.path.splitext(model_str)[0]
    if os.path.exists(path_upscale_models):
        for f in os.listdir(path_upscale_models):
            if (f.startswith(base_name) or base_name in f) and f.endswith(('.pth', '.safetensors', '.bin', '.pt')):
                return os.path.join(path_upscale_models, f)
    return None


def get_upscale_model_filename(model_name=None):
    if model_name:
        resolved = _resolve_upscale_path(str(model_name).strip())
        if resolved:
            return resolved

    import modules.config as config
    config_default = getattr(config, 'default_upscale_model', None)
    if config_default:
        resolved = _resolve_upscale_path(str(config_default).strip())
        if resolved:
            return resolved

    default_path = os.path.join(path_upscale_models, 'fooocus_upscaler_s409985e5.bin')
    if os.path.exists(default_path):
        return default_path

    return downloading_upscale_model()


def perform_upscale(img, upscale_model_path=None):
    global model, loaded_model_filename

    target_model_filename = get_upscale_model_filename(upscale_model_path)
    print(f'Upscaling image with shape {str(img.shape)} using {os.path.basename(target_model_filename)} ...')

    if model is None or loaded_model_filename != target_model_filename:
        print(f'[Upscaler] Loading upscale model: {os.path.basename(target_model_filename)}')
        sd = ldm_patched.modules.utils.load_torch_file(target_model_filename, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = ldm_patched.modules.utils.state_dict_prefix_replace(sd, {"module.": ""})

        sdo = OrderedDict()
        for k, v in sd.items():
            if isinstance(k, str) and 'residual_block_' in k:
                sdo[k.replace('residual_block_', 'RDB')] = v
            else:
                sdo[k] = v
        del sd

        try:
            model = model_loading.load_state_dict(sdo).eval()
        except Exception:
            model = ESRGAN(sdo).eval()

        del sdo
        model.cpu()
        loaded_model_filename = target_model_filename

    import gc
    gc.collect()

    img_tensor = core.numpy_to_pytorch(img)
    upscaled = opImageUpscaleWithModel.upscale(model, img_tensor)[0]
    del img_tensor
    img_out = core.pytorch_to_numpy(upscaled)[0]
    del upscaled

    if model is not None:
        try:
            model.cpu()
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return img_out


