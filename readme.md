<div align=center>
<img src="https://raw.githubusercontent.com/SayMaven/ColabFoocus/main/assets/arale.png">
</div>

# Fooocus (SayMaven Fork - SDXL & Anima DiT Dual-Architecture)

[>>> Click Here to Install Fooocus <<<](#download)

Fooocus is an image generating software (based on [Gradio](https://www.gradio.app/) <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a>).

Fooocus presents a rethinking of image generator designs. The software is offline, open source, and free, while at the same time, similar to many online image generators like Midjourney, manual tweaking is not needed, and users only need to focus on prompts and images. Fooocus has also simplified installation: between pressing "download" and generating the first image, the number of needed mouse clicks is strictly limited to less than 3. Minimal GPU memory requirement is 4GB (Nvidia).

## 🚀 SayMaven Fork Highlights: SDXL & Anima DiT Support

This repository is an active fork extending Fooocus with **Dual-Architecture Support** and full compatibility with **modern Google Colab runtimes** (Python 3.13, CUDA 12.8+, PyTorch 2.x, Transformers 5.x):

- **Dual-Architecture Backbone**:
  - **SDXL Pipeline (UNet)**: Preserves complete, flawless SDXL compatibility (CLIP-L + OpenCLIP BigG, 4-channel latent, standard SDXL VAE, refiners, and Fooocus V2 expansion).
  - **Anima Pipeline (DiT - Diffusion Transformer)**: Full integration of the 28-block DiT architecture (`MiniTrainDIT` / `CosmosTransformer`), Qwen3-0.6B text conditioning with Danbooru tag padding (512 token attention sink), and Wan21 16-channel 3D VAE (`WanVAE`).
- **ComfyUI Reference Sampler Bootstrap**:
  - Direct execution via ComfyUI reference sampler (`euler_ancestral` + `simple`, shift 3.0, multiplier 1.0) with custom LoRA support and in-memory caching (`_anima_lora_cache`).
- **Memory & VRAM Optimizations for Colab (Tesla T4 16GB)**:
  - **Weight Unification**: Merges identical DiT model weights on GPU, returning **~4.5 GB VRAM** instantly.
  - **Sequential CFG Batching (`batch=1`)**: Evaluates positive and negative conditionings sequentially, slashing RoPE activation peaks by **~50%** and enabling **High-Res Upscale (1.5x / 2.0x)** without memory freeze or CUDA OOM.
  - **Zero CPU Offloading**: Eliminates DiT RAM offloading, completely avoiding Linux OOM Killer terminations.
- **Zero-VRAM Live Step Preview for Anima**:
  - Instant live preview during DiT sampling steps using CPU-based linear Wan21 RGB latent projection.
- **Custom Upscaler Dropdown & Robust Model Loader**:
  - WebUI dropdown selector in the *Upscale or Variation* and *Enhance* tabs supporting any custom super-resolution model (`RealESRGAN_x4Plus_Anime_6B`, `4x-UltraSharp`, `Remacri`, etc.) placed in `models/upscale_models/`.
  - Smart download prevention: Eliminates redundant Hugging Face downloads when custom models are chosen.
  - Magic-byte verification in model loader to automatically detect PyTorch `.pt`/`.pth` archives even if mistakenly named `.safetensors`.
- **Anima ControlNet-LLLite Integration**:
  - Seamless structural, lineart, and pose conditioning for DiT models via **ControlNet-LLLite** (Low-Rank Light Adaptation ControlNet) directly in the *Image Prompt* tab.
  - Lightweight model footprint (~52 MB) with automatic lazy downloading (`any-test-like-v2.safetensors` from `kohyass/anima-pencil-xl-controlnet-lllite`).
  - Calibrated default parameters (`strength: 0.2`, `stop_at: 0.4`) tuned to preserve crisp outlines and structural control without over-saturation across both base diffusion and high-res upscale passes.
  - Dynamic WebUI switching: automatically restricts Image Prompt methods to `["Anima-LLLite"]` for Anima models while preserving the full suite (`ImagePrompt`, `FaceSwap`, `PyraCanny`, `CPDS`) for SDXL models.
- **YOLO Anime Detection & ADetailer-Style Multi-Tab Enhancement**:
  - **Native Anime Detection Models**: Integrated ONNX models specifically trained for anime/illustration characters (DeepGHS & Hysts):
    - `yolov8n-animeface.onnx` (Default mask generation model for both Inpaint & Enhance tabs)
    - `yolov8s-animeface.onnx`
    - `face_yolov8n.onnx`
    - `yolov8n-eyes.onnx` (Anime eye detection)
    - `hand_yolov8n.onnx` (Anime hand & finger detection)
    - `person_yolov8n.onnx` (Anime full-body character detection)
  - **Dynamic Custom Model Auto-Discovery**: Drop any custom YOLO `.onnx` or `.pt` model into `models/detection/` to automatically register it in the WebUI.
  - **Seam-Free Elliptical Feathering**: Replaced harsh rectangular bounding-box cuts with smooth elliptical mask contours (`cv2.ellipse`), Gaussian feathering (`k=31`), and a 2D boundary window taper inside `InpaintWorker`, completely eliminating hard boundary cut lines.
  - **Calibrated DiT Inpaint Defaults**:
    - Default Inpaint Method: **`Improve Detail (face, hand, eyes, etc.)`**
    - Default Denoise Strength: **`0.3`** (was 0.5; preserves 70% of character anatomy/pose while refining fine facial/hand micro-details)
    - Default Respective Field: **`0.0`** (tight crop for maximum localized detail resolution)
    - Inpaint Engine: **`None`** (native DiT crop-and-stitch bypasses incompatible SDXL inpaint heads)
  - **Sequential Multi-Tab Enhancement Pipeline**: Chain detection models across tabs (e.g. `#1 Anime Face` $\rightarrow$ `#2 Hands` $\rightarrow$ `#3 Eyes`) with zero-VRAM ONNX inference and smart auto-skip when target objects are not present.
  - **Persistent Prompt LoRA Retention**: LoRAs defined via prompt syntax (`<lora:name:weight>`) are preserved across all sequential enhancement passes, preventing character LoRAs from unloading when custom enhancement prompts (e.g. `beautiful eyes`) are used.
- **Enhanced 3-Stage Gallery Zoom Viewer**:
  - Intuitive 3-stage inspection flow: **Grid Thumbnails** $\rightarrow$ **Canvas Focused View** $\rightarrow$ **Fullscreen Lightbox Modal**.
  - Centered fullscreen lightbox with keyboard arrow navigation (`ArrowLeft` / `ArrowRight`) and quick escape (`Esc`), eliminating UI cut-offs when previewing high-resolution generations.
- **Fast Startup & Colab Auto-Bypass**:
  - Pre-built binary wheel integration for `numpy-1.26.4` on Python 3.13, skipping 7-minute Meson/Ninja compilations.
  - CLI flag `--skip-pip` (or `SKIP_PIP=1`) to skip package dependency loops on warm runtimes.
  - Automated Civitai token authentication (`CIVITAI_API_TOKEN` / `CIVITAI_TOKEN`).
  - Network dependency locking (`starlette==0.37.2`, `fastapi==0.112.2`) ensuring unbroken Gradio websocket connectivity.

# Features

Below is a quick list using Midjourney's examples:

| Midjourney | Fooocus |
| - | - |
| High-quality text-to-image without needing much prompt engineering or parameter tuning. <br> (Unknown method) | High-quality text-to-image without needing much prompt engineering or parameter tuning. <br> (Fooocus has an offline GPT-2 based prompt processing engine and lots of sampling improvements so that results are always beautiful, no matter if your prompt is as short as “house in garden” or as long as 1000 words) |
| V1 V2 V3 V4 | Input Image -> Upscale or Variation -> Vary (Subtle) / Vary (Strong)|
| U1 U2 U3 U4 | Input Image -> Upscale or Variation -> Upscale (1.5x) / Upscale (2x) |
| Inpaint / Up / Down / Left / Right (Pan) | Input Image -> Inpaint or Outpaint -> Inpaint / Up / Down / Left / Right <br> (Fooocus uses its own inpaint algorithm and inpaint models so that results are more satisfying than all other software that uses standard SDXL inpaint method/model) |
| Image Prompt | Input Image -> Image Prompt <br> (Fooocus uses its own image prompt algorithm so that result quality and prompt understanding are more satisfying than all other software that uses standard SDXL methods like standard IP-Adapters or Revisions) |
| --style | Advanced -> Style |
| --stylize | Advanced -> Advanced -> Guidance |
| --niji | [Multiple launchers: "run.bat", "run_anime.bat", and "run_realistic.bat".](https://github.com/lllyasviel/Fooocus/discussions/679) <br> Fooocus support SDXL models on Civitai <br> (You can google search “Civitai” if you do not know about it) |
| --quality | Advanced -> Quality |
| --repeat | Advanced -> Image Number |
| Multi Prompts (::) | Just use multiple lines of prompts |
| Prompt Weights | You can use " I am (happy:1.5)". <br> Fooocus uses A1111's reweighting algorithm so that results are better than ComfyUI if users directly copy prompts from Civitai. (Because if prompts are written in ComfyUI's reweighting, users are less likely to copy prompt texts as they prefer dragging files) <br> To use embedding, you can use "(embedding:file_name:1.1)" |
| --no | Advanced -> Negative Prompt |
| --ar | Advanced -> Aspect Ratios |
| InsightFace | Input Image -> Image Prompt -> Advanced -> FaceSwap |
| Describe | Input Image -> Describe |

Below is a quick list using LeonardoAI's examples:

| LeonardoAI | Fooocus |
| - | - |
| Prompt Magic | Advanced -> Style -> Fooocus V2 |
| Advanced Sampler Parameters (like Contrast/Sharpness/etc) | Advanced -> Advanced -> Sampling Sharpness / etc |
| User-friendly ControlNets | Input Image -> Image Prompt -> Advanced |

Also, [click here to browse the advanced features.](https://github.com/lllyasviel/Fooocus/discussions/117)

# Download

### Windows

You can directly download Fooocus with:

**[>>> Click here to download <<<](https://github.com/SayMaven/Fooocus/releases/download/v2.5.6-saymaven/Fooocus_win64_SayMaven_v2.5.6.7z)**

After you download the file, please uncompress it and then run the "run.bat".

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/c49269c4-c274-4893-b368-047c401cc58c)

The first time you launch the software, it will automatically download models:

1. It will download [default models](#models) to the folder "Fooocus\models\checkpoints" given different presets. You can download them in advance if you do not want automatic download.
2. Note that if you use inpaint, at the first time you inpaint an image, it will download [Fooocus's own inpaint control model from here](https://huggingface.co/lllyasviel/fooocus_inpaint/resolve/main/inpaint_v26.fooocus.patch) as the file "Fooocus\models\inpaint\inpaint_v26.fooocus.patch" (the size of this file is 1.28GB).

After Fooocus 2.1.60, you will also have `run_anime.bat` and `run_realistic.bat`. They are different model presets (and require different models, but they will be automatically downloaded). [Check here for more details](https://github.com/lllyasviel/Fooocus/discussions/679).

After Fooocus 2.3.0 you can also switch presets directly in the browser. Keep in mind to add these arguments if you want to change the default behavior:
* Use `--disable-preset-selection` to disable preset selection in the browser.
* Use `--always-download-new-model` to download missing models on preset switch. Default is fallback to `previous_default_models` defined in the corresponding preset, also see terminal output.

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/d386f817-4bd7-490c-ad89-c1e228c23447)

If you already have these files, you can copy them to the above locations to speed up installation.

Note that if you see **"MetadataIncompleteBuffer" or "PytorchStreamReader"**, then your model files are corrupted. Please download models again.

Below is a test on a relatively low-end laptop with **16GB System RAM** and **6GB VRAM** (Nvidia 3060 laptop). The speed on this machine is about 1.35 seconds per iteration. Pretty impressive – nowadays laptops with 3060 are usually at very acceptable price.

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/938737a5-b105-4f19-b051-81356cb7c495)

Note that the minimal requirement is **4GB Nvidia GPU memory (4GB VRAM)** and **8GB system memory (8GB RAM)**. This requires using Microsoft’s Virtual Swap technique, which is automatically enabled by your Windows installation in most cases, so you often do not need to do anything about it. However, if you are not sure, or if you manually turned it off (would anyone really do that?), or **if you see any "RuntimeError: CPUAllocator"**, you can enable it here:

<details>
<summary>Click here to see the image instructions. </summary>

![image](https://github.com/lllyasviel/Fooocus/assets/19834515/2a06b130-fe9b-4504-94f1-2763be4476e9)

**And make sure that you have at least 40GB free space on each drive if you still see "RuntimeError: CPUAllocator" !**

</details>

Please open an issue if you use similar devices but still cannot achieve acceptable performances.

Note that the [minimal requirement](#minimal-requirement) for different platforms is different.

See also the common problems and troubleshoots [here](troubleshoot.md).

### Colab (Modern Runtime Support - SayMaven Fork)

This fork is actively maintained on branch `anima-support` (Dual-Architecture: SDXL + Anima DiT) and `colab-support` (Stable SDXL) for Google Colab's modern runtime (Python 3.13+, CUDA 12.8+, Transformers 5.x, PyTorch 2.x).

**Quick Start in Google Colab (Anima DiT & SDXL Dual-Architecture):**
```bash
%cd /content
!if [ ! -d "Fooocus" ]; then git clone -b anima-support https://github.com/SayMaven/Fooocus.git; fi
%cd /content/Fooocus
!python entry_with_update.py --share --always-high-vram --preset anima --skip-pip --port 7866
```

**Quick Start for Standard SDXL:**
```bash
%cd /content
!if [ ! -d "Fooocus" ]; then git clone -b anima-support https://github.com/SayMaven/Fooocus.git; fi
%cd /content/Fooocus
!python entry_with_update.py --share --always-high-vram --skip-pip --port 7866
```

*Tips:*
- Using `--always-high-vram` shifts resource allocation from RAM to VRAM and achieves the overall best balance between performance, flexibility, and stability on the default Tesla T4 instance.
- Using `--skip-pip` bypasses package dependency re-checks during subsequent launches, cutting startup time down to seconds.
- You can switch presets anytime with `--preset anima`, `--preset anime`, `--preset realistic`, etc.

### Linux (Using Anaconda)

If you want to use Anaconda/Miniconda, you can:

    git clone -b anima-support https://github.com/SayMaven/Fooocus.git
    cd Fooocus
    conda env create -f environment.yaml
    conda activate fooocus
    pip install -r requirements_versions.txt

Then download the models into `Fooocus/models/checkpoints`, or **let Fooocus automatically download the models** using the launcher:

    conda activate fooocus
    python entry_with_update.py

Or, if you want to open a remote port, use:

    conda activate fooocus
    python entry_with_update.py --listen

Use `python entry_with_update.py --preset anima` for Fooocus Anima (DiT) Edition, or `--preset anime` / `--preset realistic`.

### Linux (Using Python Venv)

Using Python venv:

    git clone -b anima-support https://github.com/SayMaven/Fooocus.git
    cd Fooocus
    python3 -m venv fooocus_env
    source fooocus_env/bin/activate
    pip install -r requirements_versions.txt

Launch the software with:

    source fooocus_env/bin/activate
    python entry_with_update.py

Or, if you want to open a remote port, use:

    source fooocus_env/bin/activate
    python entry_with_update.py --listen

Use `python entry_with_update.py --preset anima` for Fooocus Anima (DiT) Edition, or `--preset anime` / `--preset realistic`.

### Linux (Using native system Python)

If you have Python (and Pip) installed on your system:

    git clone -b anima-support https://github.com/SayMaven/Fooocus.git
    cd Fooocus
    pip3 install -r requirements_versions.txt

Launch the software with:

    python3 entry_with_update.py

Or, if you want to open a remote port, use:

    python3 entry_with_update.py --listen

Use `python entry_with_update.py --preset anima` for Fooocus Anima (DiT) Edition, or `--preset anime` / `--preset realistic`.

### Linux (AMD GPUs)

Note that the [minimal requirement](#minimal-requirement) for different platforms is different.

Same with the above instructions. You need to change torch to the AMD version

    pip uninstall torch torchvision torchaudio torchtext functorch xformers 
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

AMD is not intensively tested, however. The AMD support is in beta.

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus Anime/Realistic Edition.

### Windows (AMD GPUs)

Note that the [minimal requirement](#minimal-requirement) for different platforms is different.

Same with Windows. Download the software and edit the content of `run.bat` as:

    .\python_embeded\python.exe -m pip uninstall torch torchvision torchaudio torchtext functorch xformers -y
    .\python_embeded\python.exe -m pip install torch-directml
    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --directml
    pause

Then run the `run.bat`.

AMD is not intensively tested, however. The AMD support is in beta.

For AMD, use `.\python_embeded\python.exe Fooocus\entry_with_update.py --directml --preset anime` or `.\python_embeded\python.exe Fooocus\entry_with_update.py --directml --preset realistic` for Fooocus Anime/Realistic Edition.

### Mac

Note that the [minimal requirement](#minimal-requirement) for different platforms is different.

Mac is not intensively tested. Below is an unofficial guideline for using Mac. You can discuss problems [here](https://github.com/lllyasviel/Fooocus/pull/129).

You can install Fooocus on Apple Mac silicon (M1 or M2) with macOS 'Catalina' or a newer version. Fooocus runs on Apple silicon computers via [PyTorch](https://pytorch.org/get-started/locally/) MPS device acceleration. Mac Silicon computers don't come with a dedicated graphics card, resulting in significantly longer image processing times compared to computers with dedicated graphics cards.

1. Install the conda package manager and pytorch nightly. Read the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) Apple Developer guide for instructions. Make sure pytorch recognizes your MPS device.
1. Open the macOS Terminal app and clone this repository with `git clone -b anima-support https://github.com/SayMaven/Fooocus.git`.
1. Change to the new Fooocus directory, `cd Fooocus`.
1. Create a new conda environment, `conda env create -f environment.yaml`.
1. Activate your new conda environment, `conda activate fooocus`.
1. Install the packages required by Fooocus, `pip install -r requirements_versions.txt`.
1. Launch Fooocus by running `python entry_with_update.py`. (Some Mac M2 users may need `python entry_with_update.py --disable-offload-from-vram` to speed up model loading/unloading.) The first time you run Fooocus, it will automatically download the Stable Diffusion SDXL models and will take a significant amount of time, depending on your internet connection.

Use `python entry_with_update.py --preset anime` or `python entry_with_update.py --preset realistic` for Fooocus Anime/Realistic Edition.

### Docker

See [docker.md](docker.md)

### Download Previous Version

See the guidelines [here](https://github.com/lllyasviel/Fooocus/discussions/1405).

## Minimal Requirement

Below is the minimal requirement for running Fooocus locally. If your device capability is lower than this spec, you may not be able to use Fooocus locally. (Please let us know, in any case, if your device capability is lower but Fooocus still works.)

| Operating System  | GPU                          | Minimal GPU Memory           | Minimal System Memory     | [System Swap](troubleshoot.md) | Note                                                                       |
|-------------------|------------------------------|------------------------------|---------------------------|--------------------------------|----------------------------------------------------------------------------|
| Windows/Linux     | Nvidia RTX 4XXX              | 4GB                          | 8GB                       | Required                       | fastest                                                                    |
| Windows/Linux     | Nvidia RTX 3XXX              | 4GB                          | 8GB                       | Required                       | usually faster than RTX 2XXX                                               |
| Windows/Linux     | Nvidia RTX 2XXX              | 4GB                          | 8GB                       | Required                       | usually faster than GTX 1XXX                                               |
| Windows/Linux     | Nvidia GTX 1XXX              | 8GB (&ast; 6GB uncertain)    | 8GB                       | Required                       | only marginally faster than CPU                                            |
| Windows/Linux     | Nvidia GTX 9XX               | 8GB                          | 8GB                       | Required                       | faster or slower than CPU                                                  |
| Windows/Linux     | Nvidia GTX < 9XX             | Not supported                | /                         | /                              | /                                                                          |
| Windows           | AMD GPU                      | 8GB    (updated 2023 Dec 30) | 8GB                       | Required                       | via DirectML (&ast; ROCm is on hold), about 3x slower than Nvidia RTX 3XXX |
| Linux             | AMD GPU                      | 8GB                          | 8GB                       | Required                       | via ROCm, about 1.5x slower than Nvidia RTX 3XXX                           |
| Mac               | M1/M2 MPS                    | Shared                       | Shared                    | Shared                         | about 9x slower than Nvidia RTX 3XXX                                       |
| Windows/Linux/Mac | only use CPU                 | 0GB                          | 32GB                      | Required                       | about 17x slower than Nvidia RTX 3XXX                                      |

&ast; AMD GPU ROCm (on hold): The AMD is still working on supporting ROCm on Windows.

&ast; Nvidia GTX 1XXX 6GB uncertain: Some people report 6GB success on GTX 10XX, but some other people report failure cases.

*Note that Fooocus is only for extremely high quality image generating. We will not support smaller models to reduce the requirement and sacrifice result quality.*

## Troubleshoot

See the common problems [here](troubleshoot.md).

## Default Models
<a name="models"></a>

Given different goals, the default models and configs of Fooocus are different:

| Task      | Windows | Linux args | Main Model                  | Refiner | Config                                                                         |
|-----------| --- | --- |-----------------------------| --- |--------------------------------------------------------------------------------|
| General (SDXL)   | run.bat |  | juggernautXL_v8Rundiffusion | not used | [here](presets/default.json)   |
| Realistic (SDXL) | run_realistic.bat | --preset realistic | realisticStockPhoto_v20     | not used | [here](presets/realistic.json) |
| Anime (SDXL)     | run_anime.bat | --preset anime | animaPencilXL_v500          | not used | [here](presets/anime.json)     |
| Anima (DiT)      | run_anima.bat | --preset anima | Ob_animaV4                  | not used | [here](presets/anima.json)     |
| Anima Base (DiT) | | --preset anima_base_v1 | anima-base-v1.0             | not used | [here](presets/anima_base_v1.json) |
| Hassaku Anima (DiT) | | --preset hassaku_anima_v01 | hassakuAnima_v01            | not used | [here](presets/hassaku_anima_v01.json) |
| Wai Anima (DiT)  | | --preset wai_anima         | waiANIMA_v10Base10          | not used | [here](presets/wai_anima.json) |

Note that the download is **automatic** - you do not need to do anything if the internet connection is okay. However, you can download them manually if you (or move them from somewhere else) have your own preparation.

## 🔑 Civitai API Token Authentication

Many checkpoints and LoRAs on Civitai (such as early-access, age-gated, or restricted models) require account authentication. Without a token, downloading them directly will trigger an `HTTP 401 Unauthorized` error.

Fooocus supports **automated Civitai token authentication** via the `CIVITAI_API_TOKEN` (or `CIVITAI_TOKEN`) environment variable or a local `.env` file. When configured, Fooocus automatically appends your token to any `civitai.com` or `civitai.red` download request.

### How to configure:

#### 1. Via `.env` File (Recommended for Local Windows / Linux)
Create a `.env` file in the Fooocus root directory (already included in `.gitignore` to protect your privacy):
```ini
CIVITAI_API_TOKEN=your_civitai_token_here
```

#### 2. Via Google Colab Notebook
Set the environment variable in a notebook cell before launching Fooocus:
```python
%env CIVITAI_API_TOKEN=your_civitai_token_here
```
*(Or use Colab's built-in **Secrets** 🔑 feature in the left sidebar).*

#### 3. Via Windows Batch Script (`run.bat`)
Add this line before the python launch command:
```cmd
set CIVITAI_API_TOKEN=your_civitai_token_here
```

#### 4. Via Terminal (PowerShell / Bash)
- **PowerShell (Windows):**
  ```powershell
  $env:CIVITAI_API_TOKEN = "your_civitai_token_here"
  ```
- **Bash (Linux/macOS):**
  ```bash
  export CIVITAI_API_TOKEN="your_civitai_token_here"
  ```

> **Where to get your Civitai API Token:**
> 1. Log in to [Civitai.com](https://civitai.com).
> 2. Go to **Account Settings** -> **API Keys**.
> 3. Click **Add API Key**, give it a name, and copy your token.

## 🔍 Custom Upscalers & Super-Resolution Models

Fooocus allows you to use custom super-resolution models (such as `RealESRGAN_x4Plus_Anime_6B`, `4x-UltraSharp`, `Remacri`, `DAT`, etc.) alongside the default Fooocus upscaler.

### How to Use:
1. Place your upscaler models into the `models/upscale_models/` directory (supports `.pth`, `.pt`, `.safetensors`, and `.bin`).
2. Open the WebUI and navigate to **Input Image** -> **Upscale or Variation** (or the **Enhance** tab).
3. Select your desired model from the **Upscaler** dropdown:
   - **`Default (Fooocus)`**: Uses Fooocus's built-in upscaler model (`fooocus_upscaler_s409985e5.bin`).
   - **Custom Models**: Any models detected in `models/upscale_models/` appear directly in the list (e.g. `RealESRGAN_x4Plus_Anime_6B.pt`).
4. If you add new models while Fooocus is running, simply click **`🔄 Refresh All Files`** in the Model tab to update the list without restarting.

### Key Advantages:
- **Smart Download Prevention**: When you select a custom upscaler, Fooocus will never download the default 32 MB model from Hugging Face. If `Default (Fooocus)` is chosen, it only downloads once if the file is missing locally.
- **Multi-Scale Compatibility**: Even if a custom model has a fixed scale (e.g., 4x), Fooocus automatically upscales and precisely resamples the image to your requested scaling target (`Upscale (Fast 2x)`, `Upscale (1.5x)`, or `Upscale (2x)`), followed by the high-resolution diffusion refinement pass.
- **Magic Header Detection**: If a PyTorch `.pt` or `.pth` model was saved with a `.safetensors` extension, Fooocus automatically inspects the first 8 magic bytes and safely routes it to `torch.load` to avoid `SafetensorError: header too large` crashes.

## 🎨 Anima ControlNet-LLLite (Structure & Pose Conditioning)

Unlike SDXL models that rely on heavy UNet ControlNets or IP-Adapters, the Anima DiT architecture uses **ControlNet-LLLite** (developed by Kohya-ss) for structural guidance:

### Key Highlights:
- **Ultra-Lightweight**: Models are only ~52 MB (compared to 1.4–2.5 GB for traditional ControlNets), loading in milliseconds with negligible VRAM overhead.
- **How to Use**:
  1. Check **Input Image** -> **Image Prompt**.
  2. Load your reference pose, lineart, or sketch image.
  3. Select **Anima-LLLite** (automatically selected when using Anima presets).
  4. Fooocus will automatically download the default `any-test-like-v2.safetensors` model on first use.
- **Optimized Defaults**: Default parameters are calibrated to **`Stop At: 0.4`** and **`Weight: 0.2`**. This prevents over-saturation or burned colors when combined with Upscale (1.5x / 2.0x), where LLLite operates on both generation and upscale refinement passes.
- **Custom LLLite Models**: You can drop additional Anima LLLite models into `models/controlnet/` or `models/prompt_expansion/controlnet/`.

## 🖼️ 3-Stage Gallery Zoom & Lightbox Viewer

Fooocus includes a fluid 3-stage image viewing workflow designed for evaluating generated outputs at any zoom level:

1. **Stage 1 — Grid Thumbnails**: View all generated batch outputs side-by-side in a responsive grid. Clicking any thumbnail seamlessly switches the gallery to that image in the canvas view.
2. **Stage 2 — Canvas Detailed View**: Inspect the image fitted comfortably within the WebUI workspace while retaining immediate access to generation settings, seeds, and metadata.
3. **Stage 3 — Fullscreen Lightbox Modal**: Click the canvas image to open the centered fullscreen lightbox overlay.
   - **Keyboard Navigation**: Press `ArrowLeft` / `ArrowRight` to cycle through your generation history without closing the viewer.
   - **Quick Close**: Press `Esc`, click the dark background, or click `(X)` to exit back to the canvas view.

## UI Access and Authentication
In addition to running on localhost, Fooocus can also expose its UI in two ways: 
* Local UI listener: use `--listen` (specify port e.g. with `--port 8888`). 
* API access: use `--share` (registers an endpoint at `.gradio.live`).

In both ways the access is unauthenticated by default. You can add basic authentication by creating a file called `auth.json` in the main directory, which contains a list of JSON objects with the keys `user` and `pass` (see example in [auth-example.json](./auth-example.json)).

## List of Architecture Optimizations
<a name="tech_list"></a>

<details>
<summary>Click to view built-in sampling and pipeline optimizations.</summary>

1. **Prompt Expansion**: Offline GPT-2 dynamic prompt enhancement engine ("Fooocus V2").
2. **Native Refiner Swap**: Seamless momentum-preserving model swap inside a single k-sampler for SDXL.
3. **Negative ADM Guidance**: Compensates for lack of cross-attention contrast in SDXL highest resolution level.
4. **Self-Attention Guidance (SAG)**: Anisotropic kernel variation of SAG for artifact prevention and structural preservation.
5. **Style Normalization**: Balanced multi-style blending and automatic A1111-compatible prompt emphasizing.
6. **Rectified Flow Matching**: Shifted Euler Ancestral scheduling tuned for DiT (Anima) models.
7. **Dual-Path VAE Decode**: Specialized handling for both 4-channel 2D KL-Autoencoder and 16-channel 3D Causal WanVAE.
8. **Sequential CFG Batching**: Sequential positive/unconditioned passes preventing high-resolution activation spikes.
9. **Zero-VRAM Step Preview**: Instant linear RGB latent projection preview for DiT latents.
10. **DiT ControlNet-LLLite Injection**: Low-rank linear adapters and ASPP conditioning extractor for DiT models.
11. **3-Stage Image Inspection**: Gradio gallery pass-through to canvas view with centered fullscreen lightbox modal.
</details>

## Customization

After the first time you run Fooocus, a config file will be generated at `Fooocus\config.txt`. This file can be edited to change the model path or default parameters.

For example, an edited `Fooocus\config.txt` (this file will be generated after the first launch) may look like this:

```json
{
    "path_checkpoints": "D:\\Fooocus\\models\\checkpoints",
    "path_loras": "D:\\Fooocus\\models\\loras",
    "path_embeddings": "D:\\Fooocus\\models\\embeddings",
    "path_vae_approx": "D:\\Fooocus\\models\\vae_approx",
    "path_upscale_models": "D:\\Fooocus\\models\\upscale_models",
    "path_inpaint": "D:\\Fooocus\\models\\inpaint",
    "path_controlnet": "D:\\Fooocus\\models\\controlnet",
    "path_clip_vision": "D:\\Fooocus\\models\\clip_vision",
    "path_fooocus_expansion": "D:\\Fooocus\\models\\prompt_expansion\\fooocus_expansion",
    "path_outputs": "D:\\Fooocus\\outputs",
    "default_model": "realisticStockPhoto_v10.safetensors",
    "default_refiner": "",
    "default_upscale_model": "Default (Fooocus)",
    "default_loras": [["lora_filename_1.safetensors", 0.5], ["lora_filename_2.safetensors", 0.5]],
    "default_cfg_scale": 3.0,
    "default_sampler": "dpmpp_2m",
    "default_scheduler": "karras",
    "default_negative_prompt": "low quality",
    "default_positive_prompt": "",
    "default_styles": [
        "Fooocus V2",
        "Fooocus Photograph",
        "Fooocus Negative"
    ]
}
```

Many other keys, formats, and examples are in `Fooocus\config_modification_tutorial.txt` (this file will be generated after the first launch).

Consider twice before you really change the config. If you find yourself breaking things, just delete `Fooocus\config.txt`. Fooocus will go back to default.

A safer way is just to try "run_anime.bat" or "run_realistic.bat" - they should already be good enough for different tasks.

### All CMD Flags

```
entry_with_update.py  [-h] [--listen [IP]] [--port PORT]
                      [--disable-header-check [ORIGIN]]
                      [--web-upload-size WEB_UPLOAD_SIZE]
                      [--hf-mirror HF_MIRROR]
                      [--external-working-path PATH [PATH ...]]
                      [--output-path OUTPUT_PATH]
                      [--temp-path TEMP_PATH] [--cache-path CACHE_PATH]
                      [--in-browser] [--disable-in-browser]
                      [--gpu-device-id DEVICE_ID]
                      [--async-cuda-allocation | --disable-async-cuda-allocation]
                      [--disable-attention-upcast]
                      [--all-in-fp32 | --all-in-fp16]
                      [--unet-in-bf16 | --unet-in-fp16 | --unet-in-fp8-e4m3fn | --unet-in-fp8-e5m2]
                      [--vae-in-fp16 | --vae-in-fp32 | --vae-in-bf16]
                      [--vae-in-cpu]
                      [--clip-in-fp8-e4m3fn | --clip-in-fp8-e5m2 | --clip-in-fp16 | --clip-in-fp32]
                      [--directml [DIRECTML_DEVICE]]
                      [--disable-ipex-hijack]
                      [--preview-option [none,auto,fast,taesd]]
                      [--attention-split | --attention-quad | --attention-pytorch]
                      [--disable-xformers]
                      [--always-gpu | --always-high-vram | --always-normal-vram | --always-low-vram | --always-no-vram | --always-cpu [CPU_NUM_THREADS]]
                      [--always-offload-from-vram]
                      [--pytorch-deterministic] [--disable-server-log]
                      [--debug-mode] [--is-windows-embedded-python]
                      [--disable-server-info] [--multi-user] [--share]
                      [--preset PRESET] [--disable-preset-selection]
                      [--language LANGUAGE]
                      [--disable-offload-from-vram] [--theme THEME]
                      [--disable-image-log] [--disable-analytics]
                      [--disable-metadata] [--disable-preset-download]
                      [--disable-enhance-output-sorting]
                      [--enable-auto-describe-image]
                      [--always-download-new-model]
                      [--skip-pip]
                      [--rebuild-hash-cache [CPU_NUM_THREADS]]
```

## Inline Prompt Features

### Wildcards

Example prompt: `__color__ flower`

Processed for positive and negative prompt.

Selects a random wildcard from a predefined list of options, in this case the `wildcards/color.txt` file. 
The wildcard will be replaced with a random color (randomness based on seed). 
You can also disable randomness and process a wildcard file from top to bottom by enabling the checkbox `Read wildcards in order` in Developer Debug Mode.

Wildcards can be nested and combined, and multiple wildcards can be used in the same prompt (example see `wildcards/color_flower.txt`).

### Array Processing

Example prompt: `[[red, green, blue]] flower`

Processed only for positive prompt.

Processes the array from left to right, generating a separate image for each element in the array. In this case 3 images would be generated, one for each color.
Increase the image number to 3 to generate all 3 variants.

Arrays can not be nested, but multiple arrays can be used in the same prompt.
Does support inline LoRAs as array elements!

### Inline LoRAs

Example prompt: `flower <lora:sunflowers:1.2>`

Processed only for positive prompt.

Applies a LoRA to the prompt. The LoRA file must be located in the `models/loras` directory.

## Advanced Features

[Click here to browse the advanced features.](https://github.com/lllyasviel/Fooocus/discussions/117)

## Forks

Below are some Forks to Fooocus:

| Fooocus' forks |
| - |
| [fenneishi/Fooocus-Control](https://github.com/fenneishi/Fooocus-Control) </br>[runew0lf/RuinedFooocus](https://github.com/runew0lf/RuinedFooocus) </br> [MoonRide303/Fooocus-MRE](https://github.com/MoonRide303/Fooocus-MRE) </br> [mashb1t/Fooocus](https://github.com/mashb1t/Fooocus) </br> and so on ... |

## Thanks

Many thanks to [twri](https://github.com/twri) and [3Diva](https://github.com/3Diva) and [Marc K3nt3L](https://github.com/K3nt3L) for creating additional SDXL styles available in Fooocus. 

The project starts from a mixture of [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and [ComfyUI](https://github.com/comfyanonymous/ComfyUI) codebases.

Also, thanks [daswer123](https://github.com/daswer123) for contributing the Canvas Zoom!

## Update Log

The log is [here](update_log.md).

## Localization/Translation/I18N

You can put json files in the `language` folder to translate the user interface.

For example, below is the content of `Fooocus/language/example.json`:

```json
{
  "Generate": "生成",
  "Input Image": "入力画像",
  "Advanced": "고급",
  "SAI 3D Model": "SAI 3D Modèle"
}
```

If you add `--language example` arg, Fooocus will read `Fooocus/language/example.json` to translate the UI.

For example, you can edit the ending line of Windows `run.bat` as

    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --language example

Or `run_anime.bat` as

    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --language example --preset anime

Or `run_realistic.bat` as

    .\python_embeded\python.exe -s Fooocus\entry_with_update.py --language example --preset realistic

For practical translation, you may create your own file like `Fooocus/language/jp.json` or `Fooocus/language/cn.json` and then use flag `--language jp` or `--language cn`. Apparently, these files do not exist now. **We need your help to create these files!**

Note that if no `--language` is given and at the same time `Fooocus/language/default.json` exists, Fooocus will always load `Fooocus/language/default.json` for translation. By default, the file `Fooocus/language/default.json` does not exist.
