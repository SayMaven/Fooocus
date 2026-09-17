import os
from urllib.parse import urlparse
from typing import Optional


def _get_civitai_token() -> Optional[str]:
    token = os.environ.get("CIVITAI_API_TOKEN") or os.environ.get("CIVITAI_TOKEN")
    if not token:
        env_paths = [
            os.path.join(os.getcwd(), ".env"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"),
        ]
        for env_path in env_paths:
            if os.path.isfile(env_path):
                try:
                    with open(env_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#") or "=" not in line:
                                continue
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k in ("CIVITAI_API_TOKEN", "CIVITAI_TOKEN") and v:
                                token = v
                                os.environ[k] = v
                                return token
                except Exception:
                    pass
    return token


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    domain = os.environ.get("HF_MIRROR", "https://huggingface.co").rstrip('/')
    url = str.replace(url, "https://huggingface.co", domain, 1)

    civitai_token = _get_civitai_token()
    if civitai_token and ("civitai.com" in url or "civitai.red" in url) and "token=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}token={civitai_token}"

    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        from urllib.error import HTTPError
        from torch.hub import download_url_to_file
        try:
            download_url_to_file(url, cached_file, progress=progress)
        except HTTPError as e:
            if e.code == 401:
                raise RuntimeError(
                    f"\n[Civitai Error 401: Unauthorized]\n"
                    f"Model download failed for {url}\n"
                    f"Civitai requires an API token to download this model.\n"
                    f"Please either:\n"
                    f"1. Add '&token=YOUR_CIVITAI_TOKEN' to the download URL in your preset.\n"
                    f"2. Set the environment variable CIVITAI_API_TOKEN in Colab:\n"
                    f"   %env CIVITAI_API_TOKEN=your_token_here\n"
                ) from e
            raise
    return cached_file


_is_anima_checkpoint_cache = {}


def is_anima_checkpoint_file(filename_or_path: Optional[str]) -> bool:
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


def is_anima_model(model) -> bool:
    if model is None:
        return False
    if isinstance(model, str):
        return is_anima_checkpoint_file(model)
    if hasattr(model, "model") and getattr(model.model, "__class__", None) and model.model.__class__.__name__ == "Anima":
        return True
    inner = getattr(model, "model", None)
    if inner is not None:
        if getattr(inner, "__class__", None) and inner.__class__.__name__ == "Anima":
            return True
        diff = getattr(inner, "diffusion_model", None)
        if diff is not None and getattr(diff, "__class__", None) and diff.__class__.__name__ in ("MiniTrainDIT", "CosmosTransformer", "Anima"):
            return True
    filename = getattr(model, "model_file", None) or getattr(model, "filename", None)
    if filename:
        return is_anima_checkpoint_file(filename)
    return False
