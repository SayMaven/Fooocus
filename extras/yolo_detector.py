import os
try:
    import cv2
except ImportError:
    cv2 = None
import numpy as np
import modules.config

_sessions = {}


def _get_onnx_session(model_path: str):
    global _sessions
    if model_path in _sessions:
        return _sessions[model_path]

    import onnxruntime as ort

    available_providers = ort.get_available_providers()
    providers = []
    if 'CUDAExecutionProvider' in available_providers:
        providers.append('CUDAExecutionProvider')
    providers.append('CPUExecutionProvider')

    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        print(f"[YOLO] Failed to load session with providers {providers}, falling back to CPU: {e}")
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

    _sessions[model_path] = session
    return session


def letterbox(img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2.0, dh / 2.0

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def detect_yolo_boxes(image: np.ndarray, model_path: str, conf_threshold: float = 0.35, nms_threshold: float = 0.45):
    """
    Run YOLO ONNX detection on an image.
    Returns list of [x1, y1, x2, y2] bounding boxes in original image coordinates.
    """
    if not os.path.exists(model_path):
        print(f"[YOLO] Model file not found: {model_path}")
        return []

    session = _get_onnx_session(model_path)
    inputs = session.get_inputs()
    input_name = inputs[0].name
    input_shape = inputs[0].shape

    target_h = input_shape[2] if len(input_shape) >= 4 and isinstance(input_shape[2], int) else 640
    target_w = input_shape[3] if len(input_shape) >= 4 and isinstance(input_shape[3], int) else 640

    orig_h, orig_w = image.shape[:2]
    img_padded, ratio, (dw, dh) = letterbox(image, new_shape=(target_h, target_w))

    # Convert RGB to float32 NCHW
    blob = img_padded.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))
    blob = np.expand_dims(blob, axis=0)

    try:
        outputs = session.run(None, {input_name: blob})
    except Exception as e:
        print(f"[YOLO] Inference error: {e}")
        return []

    preds = outputs[0]  # Shape typically (1, 4+classes, 8400) or (1, 8400, 4+classes)
    if preds.ndim == 3:
        if preds.shape[1] < preds.shape[2]:
            preds = np.transpose(preds, (0, 2, 1))

    preds = preds[0]  # Shape (N, 4+classes)
    if preds.shape[1] < 5:
        print(f"[YOLO] Unexpected prediction dimension: {preds.shape}")
        return []

    boxes_cxcywh = preds[:, :4]
    class_scores = preds[:, 4:]

    if class_scores.shape[1] == 1:
        confidences = class_scores[:, 0]
    else:
        confidences = np.max(class_scores, axis=1)

    mask = confidences >= conf_threshold
    boxes_filtered = boxes_cxcywh[mask]
    confidences_filtered = confidences[mask]

    if len(boxes_filtered) == 0:
        return []

    # Convert cx, cy, w, h to x1, y1, w, h for cv2.dnn.NMSBoxes
    cx = boxes_filtered[:, 0]
    cy = boxes_filtered[:, 1]
    w = boxes_filtered[:, 2]
    h = boxes_filtered[:, 3]

    x1 = (cx - w / 2.0 - dw) / ratio
    y1 = (cy - h / 2.0 - dh) / ratio
    bw = w / ratio
    bh = h / ratio

    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    bw = np.clip(bw, 1, orig_w - x1)
    bh = np.clip(bh, 1, orig_h - y1)

    cv_boxes = [[int(x1[i]), int(y1[i]), int(bw[i]), int(bh[i])] for i in range(len(x1))]
    conf_list = [float(c) for c in confidences_filtered]

    indices = cv2.dnn.NMSBoxes(cv_boxes, conf_list, conf_threshold, nms_threshold)
    if len(indices) == 0:
        return []

    if isinstance(indices, np.ndarray):
        indices = indices.flatten()

    final_boxes = []
    for idx in indices:
        bx, by, bw_i, bh_i = cv_boxes[idx]
        final_boxes.append([bx, by, bx + bw_i, by + bh_i])

    return final_boxes


def generate_yolo_mask(image: np.ndarray, model_name: str, conf_threshold: float = 0.35,
                       padding_ratio: float = 0.1) -> tuple[np.ndarray, int]:
    """
    Generate a 3-channel uint8 mask (0 or 255) for all detected objects.
    Returns (mask_image, detection_count).
    """
    model_path = modules.config.downloading_yolo_detection_model(model_name)
    if not os.path.exists(model_path):
        print(f"[YOLO] Could not find or download model: {model_name}")
        return np.zeros_like(image), 0

    boxes = detect_yolo_boxes(image, model_path, conf_threshold=conf_threshold)
    H, W = image.shape[:2]
    mask_2d = np.zeros((H, W), dtype=np.uint8)

    for box in boxes:
        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1

        # Apply padding ratio
        pad_x = int(bw * padding_ratio)
        pad_y = int(bh * padding_ratio)

        px1 = max(0, x1 - pad_x)
        py1 = max(0, y1 - pad_y)
        px2 = min(W, x2 + pad_x)
        py2 = min(H, y2 + pad_y)

        cv2.rectangle(mask_2d, (px1, py1), (px2, py2), 255, -1)

    mask_3d = np.dstack([mask_2d, mask_2d, mask_2d])
    return mask_3d, len(boxes)
