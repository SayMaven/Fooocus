import os
import csv
import numpy as np

from PIL import Image
from modules.config import path_clip_vision
from modules import config
from modules.model_loader import load_file_from_url


TAGGER_MODELS = {
    "wd-eva02-large-tagger-v3": {
        "repo": "SmilingWolf/wd-eva02-large-tagger-v3",
        "model_file": "model.onnx",
        "csv_file": "selected_tags.csv",
    },
    "wd-v1-4-moat-tagger-v2": {
        "repo": "lllyasviel/misc",
        "model_file": "wd-v1-4-moat-tagger-v2.onnx",
        "csv_file": "wd-v1-4-moat-tagger-v2.csv",
    },
    "wd-vit-large-tagger-v3": {
        "repo": "SmilingWolf/wd-vit-large-tagger-v3",
        "model_file": "model.onnx",
        "csv_file": "selected_tags.csv",
    },
    "wd-swinv2-tagger-v3": {
        "repo": "SmilingWolf/wd-swinv2-tagger-v3",
        "model_file": "model.onnx",
        "csv_file": "selected_tags.csv",
    },
    "wd-convnext-tagger-v3": {
        "repo": "SmilingWolf/wd-convnext-tagger-v3",
        "model_file": "model.onnx",
        "csv_file": "selected_tags.csv",
    },
    "wd-vit-tagger-v3": {
        "repo": "SmilingWolf/wd-vit-tagger-v3",
        "model_file": "model.onnx",
        "csv_file": "selected_tags.csv",
    },
}

kaomojis = {
    "0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", "=_=",
    ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", "u_u", "x_x", "|_|", "||_||",
}

global_model = None
global_csv = None
current_model_name = None


def get_tagger_files(model_name="wd-eva02-large-tagger-v3"):
    if model_name not in TAGGER_MODELS:
        model_name = "wd-eva02-large-tagger-v3"

    info = TAGGER_MODELS[model_name]
    repo = info["repo"]
    remote_model = info["model_file"]
    remote_csv = info["csv_file"]

    local_model_name = f"{model_name}.onnx"
    local_csv_name = f"{model_name}.csv"

    local_model_path = os.path.join(path_clip_vision, local_model_name)
    local_csv_path = os.path.join(path_clip_vision, local_csv_name)

    if not (os.path.exists(local_model_path) and os.path.exists(local_csv_path)):
        try:
            model_onnx_filename = load_file_from_url(
                url=f'https://huggingface.co/{repo}/resolve/main/{remote_model}',
                model_dir=path_clip_vision,
                file_name=local_model_name,
            )
            model_csv_filename = load_file_from_url(
                url=f'https://huggingface.co/{repo}/resolve/main/{remote_csv}',
                model_dir=path_clip_vision,
                file_name=local_csv_name,
            )
        except Exception as e:
            # Fallback if offline and old moat tagger is present
            moat_model_path = os.path.join(path_clip_vision, "wd-v1-4-moat-tagger-v2.onnx")
            moat_csv_path = os.path.join(path_clip_vision, "wd-v1-4-moat-tagger-v2.csv")
            if os.path.exists(moat_model_path) and os.path.exists(moat_csv_path):
                print(f"[WD14Tagger] Download failed ({e}), falling back to local wd-v1-4-moat-tagger-v2")
                return "wd-v1-4-moat-tagger-v2", moat_model_path, moat_csv_path
            raise e
    else:
        model_onnx_filename = local_model_path
        model_csv_filename = local_csv_path

    return model_name, model_onnx_filename, model_csv_filename


def default_interrogator(image_rgb, threshold=0.35, character_threshold=0.85, exclude_tags=""):
    global global_model, global_csv, current_model_name

    configured_model = getattr(config, 'default_anime_tagger', 'wd-eva02-large-tagger-v3')
    model_name, model_onnx_filename, model_csv_filename = get_tagger_files(configured_model)

    import onnxruntime as ort
    from onnxruntime import InferenceSession

    if global_model is not None and current_model_name == model_name:
        model = global_model
    else:
        model = InferenceSession(model_onnx_filename, providers=ort.get_available_providers())
        global_model = model
        current_model_name = model_name
        global_csv = None

    input_meta = model.get_inputs()[0]
    try:
        height = int(input_meta.shape[1]) if input_meta.shape[1] is not None and int(input_meta.shape[1]) > 0 else 448
    except Exception:
        height = 448

    image = Image.fromarray(image_rgb)  # RGB
    ratio = float(height) / max(image.size)
    new_size = tuple([int(x * ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(image, ((height - new_size[0]) // 2, (height - new_size[1]) // 2))

    image = np.array(square).astype(np.float32)
    image = image[:, :, ::-1]  # RGB -> BGR
    image = np.expand_dims(image, 0)

    if global_csv is not None and current_model_name == model_name:
        csv_lines = global_csv
    else:
        csv_lines = []
        with open(model_csv_filename, encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                csv_lines.append(row)
        global_csv = csv_lines

    general_indices = []
    character_indices = []
    tags = []
    for line_num, row in enumerate(csv_lines):
        if len(row) < 3:
            continue
        cat = row[2].strip()
        if cat == "0":
            general_indices.append(line_num)
        elif cat == "4":
            character_indices.append(line_num)
        tags.append(row[1])

    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input_meta.name: image})[0][0]

    general = [(tags[i], probs[i]) for i in general_indices if probs[i] > threshold]
    character = [(tags[i], probs[i]) for i in character_indices if probs[i] > character_threshold]

    all_tags = character + general
    remove = [s.strip().lower() for s in exclude_tags.split(",") if s.strip()]
    filtered_tags = [tag for tag in all_tags if tag[0].lower() not in remove]

    def format_tag(name):
        if name not in kaomojis:
            name = name.replace('_', ' ')
        return name.replace("(", "\\(").replace(")", "\\)")

    res = ", ".join((format_tag(item[0]) for item in filtered_tags))
    return res

