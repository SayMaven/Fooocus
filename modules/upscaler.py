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


def get_upscale_model_filename():
    import modules.config as config
    custom_model = getattr(config, 'default_upscale_model', None)
    if custom_model:
        custom_path = os.path.join(path_upscale_models, custom_model)
        if os.path.exists(custom_path):
            return custom_path

    if os.path.exists(path_upscale_models):
        for f in os.listdir(path_upscale_models):
            if f.endswith(('.pth', '.safetensors', '.bin', '.pt')) and f != 'fooocus_upscaler_s409985e5.bin':
                return os.path.join(path_upscale_models, f)

    return downloading_upscale_model()


def perform_upscale(img, upscale_model_path=None):
    global model, loaded_model_filename

    print(f'Upscaling image with shape {str(img.shape)} ...')

    target_model_filename = upscale_model_path or get_upscale_model_filename()

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


