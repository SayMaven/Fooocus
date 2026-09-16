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


def get_upscale_model_filename(model_name=None):
    if model_name:
        model_str = str(model_name).strip()
        if model_str not in ['Default (Fooocus)', 'Default', 'default', 'fooocus_upscaler', 'None', '']:
            if os.path.exists(model_str):
                return model_str
            custom_path = os.path.join(path_upscale_models, model_str)
            if os.path.exists(custom_path):
                return custom_path

    import modules.config as config
    config_default = getattr(config, 'default_upscale_model', None)
    if config_default:
        config_str = str(config_default).strip()
        if config_str not in ['Default (Fooocus)', 'Default', 'default', 'fooocus_upscaler', 'None', '']:
            if os.path.exists(config_str):
                return config_str
            custom_path = os.path.join(path_upscale_models, config_str)
            if os.path.exists(custom_path):
                return custom_path

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

        model.cpu()
        loaded_model_filename = target_model_filename

    img = core.numpy_to_pytorch(img)
    img = opImageUpscaleWithModel.upscale(model, img)[0]
    img = core.pytorch_to_numpy(img)[0]

    if model is not None:
        try:
            model.cpu()
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return img


