"""Export the trained model to ONNX, check it matches PyTorch, and benchmark both.

    python scripts/export_onnx.py --model models/ppe_yolo11n_640.pt

Writes ``models/siteguard.onnx`` and adds an ``export`` section to ``models/metrics.json``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from siteguard.compliance import iou_matrix  # noqa: E402
from siteguard.detector import OnnxDetector, UltralyticsDetector  # noqa: E402


def agreement(a, b, iou_thr=0.5) -> float:
    """Share of boxes in ``a`` that have a same-class box in ``b`` with IoU >= iou_thr."""
    if len(a) == 0:
        return 1.0
    iou = iou_matrix(a.xyxy, b.xyxy)
    same = a.class_id[:, None] == b.class_id[None, :]
    return float(((iou >= iou_thr) & same).any(axis=1).mean())


def bench(det, images, runs=1) -> float:
    det.predict(images[0])  # warm-up
    times = []
    for img in images:
        for _ in range(runs):
            t0 = time.perf_counter()
            det.predict(img)
            times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "ppe_yolo11n_640.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n", type=int, default=40, help="test images used for parity + benchmark")
    args = ap.parse_args()

    from ultralytics import YOLO

    pt = Path(args.model)
    # dynamic H/W: lets the runtime use a rectangular letterbox (fewer pixels for non-square photos)
    exported = Path(YOLO(str(pt)).export(format="onnx", imgsz=args.imgsz, simplify=True, dynamic=True, device="cpu"))
    onnx_path = ROOT / "models" / "siteguard.onnx"
    shutil.move(exported, onnx_path)
    print(f"ONNX -> {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    paths = [s for s in (ROOT / "data" / "test.txt").read_text().split() if s][: args.n]
    images = [cv2.imread(p) for p in paths]
    torch_det = UltralyticsDetector(pt, imgsz=args.imgsz)
    onnx_det = OnnxDetector(onnx_path, imgsz=args.imgsz)  # our own pre/post-processing, no PyTorch

    found_fwd, found_bwd, max_px, max_conf = [], [], 0.0, 0.0
    n_t = n_o = 0
    for img in images:
        t, o = torch_det.predict(img), onnx_det.predict(img)
        n_t += len(t)
        n_o += len(o)
        found_fwd.append(agreement(t, o, iou_thr=0.5))
        found_bwd.append(agreement(o, t, iou_thr=0.5))
        if len(t) and len(o):  # worst coordinate / confidence deviation over matched boxes
            iou = iou_matrix(t.xyxy, o.xyxy)
            iou[t.class_id[:, None] != o.class_id[None, :]] = 0
            j = iou.argmax(axis=1)
            ok = iou.max(axis=1) >= 0.5
            if ok.any():
                max_px = max(max_px, float(np.abs(t.xyxy[ok] - o.xyxy[j[ok]]).max()))
                max_conf = max(max_conf, float(np.abs(t.confidence[ok] - o.confidence[j[ok]]).max()))
    parity = {
        "images": len(images),
        "boxes_pytorch": n_t,
        "boxes_onnx": n_o,
        "pytorch_boxes_found_by_onnx": round(float(np.mean(found_fwd)), 4),
        "onnx_boxes_found_by_pytorch": round(float(np.mean(found_bwd)), 4),
        "max_box_deviation_px": round(max_px, 2),
        "max_confidence_deviation": round(max_conf, 4),
    }
    print("parity:", parity)

    latency = {"pytorch_ms": round(bench(torch_det, images), 1), "onnx_ms": round(bench(onnx_det, images), 1)}
    print("median latency per image (incl. pre/post-processing):", latency)

    metrics_path = ROOT / "models" / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics["export"] = {
        "onnx_file": onnx_path.name,
        "onnx_mb": round(onnx_path.stat().st_size / 1e6, 2),
        "pt_mb": round(pt.stat().st_size / 1e6, 2),
        "imgsz": args.imgsz,
        "parity": parity,
        "latency_cpu": latency,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
