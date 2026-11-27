"""PPE detector with two interchangeable backends.

* ``UltralyticsDetector`` - ``.pt`` weights through Ultralytics/PyTorch (training, evaluation).
* ``OnnxDetector``        - ``.onnx`` weights through ONNX Runtime with hand-written
  pre/post-processing (letterbox, box decoding, class-aware NMS). No PyTorch needed,
  which keeps the serving image small and start-up fast.

Use :func:`load_detector` to pick the backend from the file extension.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from siteguard.types import CLASS_NAMES, Detections


class Detector(Protocol):
    names: list[str]

    def predict(self, image_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.5) -> Detections: ...


# --------------------------------------------------------------------------- pre/post-processing
def letterbox(
    img: np.ndarray, size: int, color: int = 114, auto: bool = False, stride: int = 32
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize keeping aspect ratio and pad (centred), as Ultralytics does.

    ``auto=False`` pads to ``size`` x ``size`` (static-shape models). ``auto=True`` only pads
    up to a multiple of ``stride`` - a 3:2 photo becomes 960x640 instead of 960x960, so the
    network processes ~33% fewer pixels (needs a model exported with dynamic shapes).
    """
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = round(w * r), round(h * r)
    dw, dh = size - nw, size - nh
    if auto:
        dw, dh = dw % stride, dh % stride
    dw, dh = dw / 2, dh / 2
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color, color, color))
    return out, r, (left, top)


def decode_yolo(
    output: np.ndarray, conf: float, iou: float, max_det: int = 300
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a YOLOv8/11 head output of shape (1, 4 + nc, anchors) -> boxes (xyxy), scores, class ids."""
    pred = output[0].T  # (anchors, 4 + nc)
    scores_all = pred[:, 4:]
    cls = scores_all.argmax(axis=1)
    scores = scores_all[np.arange(len(pred)), cls]
    keep = scores >= conf
    if not keep.any():
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    pred, cls, scores = pred[keep], cls[keep], scores[keep]
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    # class-aware NMS: a helmet box must not suppress the overlapping Person box
    idx = cv2.dnn.NMSBoxesBatched(
        np.stack([xyxy[:, 0], xyxy[:, 1], w, h], axis=1).tolist(), scores.tolist(), cls.tolist(), conf, iou
    )
    idx = np.asarray(idx, dtype=int).reshape(-1)[:max_det]
    return xyxy[idx].astype(np.float32), scores[idx].astype(np.float32), cls[idx].astype(int)


# --------------------------------------------------------------------------- backends
class OnnxDetector:
    def __init__(
        self, path: str | Path, providers: list[str] | None = None, threads: int | None = None, imgsz: int | None = None
    ):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=providers or ["CPUExecutionProvider"])
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        meta = self.session.get_modelmeta().custom_metadata_map
        # dynamic-shape export -> rectangular letterbox; size from metadata unless overridden
        self.dynamic = not isinstance(inp.shape[2], int)
        if self.dynamic:
            meta_sz = ast.literal_eval(meta.get("imgsz", "[640, 640]"))
            self.imgsz = int(imgsz or max(meta_sz))
        else:
            self.imgsz = int(inp.shape[2])
        names = meta.get("names")
        if names:  # Ultralytics stores a Python dict literal: "{0: 'helmet', 1: 'gloves', ...}"
            parsed = ast.literal_eval(names)
            self.names = [parsed[i] for i in range(len(parsed))]
        else:
            self.names = list(CLASS_NAMES)

    def predict(self, image_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.5) -> Detections:
        img, r, (px, py) = letterbox(image_bgr, self.imgsz, auto=self.dynamic)
        blob = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)[None] / 255.0
        out = self.session.run(None, {self.input_name: blob})[0]
        xyxy, scores, cls = decode_yolo(out, conf, iou)
        if len(xyxy):
            xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - px) / r
            xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - py) / r
            h, w = image_bgr.shape[:2]
            xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
            xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
        return Detections(xyxy, scores, cls, self.names)


class UltralyticsDetector:
    def __init__(self, path: str | Path, imgsz: int = 640, device: str = "cpu"):
        from ultralytics import YOLO

        self.model = YOLO(str(path))
        self.names = [self.model.names[i] for i in range(len(self.model.names))]
        self.imgsz = imgsz
        self.device = device

    def predict(self, image_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.5) -> Detections:
        r = self.model.predict(image_bgr, conf=conf, iou=iou, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        b = r.boxes
        return Detections(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int), self.names)


def load_detector(path: str | Path, **kwargs) -> Detector:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    if path.suffix == ".onnx":
        return OnnxDetector(path, **kwargs)
    if path.suffix == ".pt":
        return UltralyticsDetector(path, **kwargs)
    raise ValueError(f"Unsupported model format: {path.suffix} (use .pt or .onnx)")
