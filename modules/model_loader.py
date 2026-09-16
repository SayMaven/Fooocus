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
