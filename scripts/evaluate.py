"""Evaluate a trained model on the held-out test split.

Two levels:

1. **Detection** - COCO-style mAP per class (Ultralytics ``val``), at the training
   resolution and at a higher inference resolution (small objects).
2. **Compliance** - what the end user cares about. For every *ground-truth* worker the
   helmet/vest status is derived from the labels, then compared with what the full
   system (detector + rules) says about the same worker: "of the workers with no
   helmet, how many did we flag?". Reported with the default thresholds and with the
   thresholds tuned on the validation split (``configs/thresholds.json``).

    python scripts/evaluate.py --model models/ppe_yolo11n_640.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from siteguard import ComplianceConfig, load_detector  # noqa: E402
from siteguard.labels import read_labels  # noqa: E402
from siteguard.metrics import compliance_confusion, summarize  # noqa: E402


def detection_metrics(model_path: Path, data: str, imgsz: int, name: str) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    r = model.val(
        data=data,
        split="test",
        imgsz=imgsz,
        batch=8,
        device="cpu",
        plots=True,
        project=str(ROOT / "runs" / "eval"),
        name=name,
        exist_ok=True,
        verbose=False,
    )
    names = [model.names[i] for i in range(len(model.names))]
    per_class = {
        names[int(c)]: {
            "precision": round(float(r.box.p[i]), 3),
            "recall": round(float(r.box.r[i]), 3),
            "mAP50": round(float(r.box.ap50[i]), 3),
            "mAP50-95": round(float(r.box.ap[i]), 3),
        }
        for i, c in enumerate(r.box.ap_class_index)
    }
    return {
        "imgsz": imgsz,
        "mAP50": round(float(r.box.map50), 4),
        "mAP50-95": round(float(r.box.map), 4),
        "precision": round(float(r.box.mp), 4),
        "recall": round(float(r.box.mr), 4),
        "per_class": per_class,
        "save_dir": str(r.save_dir),
    }


def predict_split(model_path: Path, split: str, limit: int | None = None, conf: float = 0.05, imgsz: int | None = None):
    """Run the detector once at a low threshold; compliance thresholds are applied afterwards."""
    detector = load_detector(model_path, **({"imgsz": imgsz} if imgsz and Path(model_path).suffix == ".pt" else {}))
    images = [s for s in (ROOT / "data" / f"{split}.txt").read_text().split() if s][:limit]
    samples = []
    for path in images:
        img = cv2.imread(path)
        h, w = img.shape[:2]
        samples.append((read_labels(path, w, h), detector.predict(img, conf=conf)))
    return samples


def compliance_report(samples, cfg: ComplianceConfig) -> dict:
    confusion, total, found = compliance_confusion(samples, cfg)
    return {
        "min_confidence": cfg.min_confidence,
        "class_confidence": cfg.class_confidence,
        "workers": total,
        "worker_detection_recall": round(found / max(1, total), 3),
        "items": {it: summarize(c) for it, c in confusion.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "ppe_yolo11n_640.pt"))
    ap.add_argument("--data", default=str(ROOT / "configs" / "ppe.yaml"))
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 960])
    ap.add_argument(
        "--compliance-sizes",
        type=int,
        nargs="+",
        default=[640, 960],
        help="inference sizes for the compliance evaluation (thresholds from configs/thresholds_<size>.json)",
    )
    ap.add_argument("--skip-detection", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="only the first N test images (quick check)")
    ap.add_argument("--out", default=str(ROOT / "models" / "metrics.json"))
    args = ap.parse_args()

    model = Path(args.model)
    out_path = Path(args.out)
    result = json.loads(out_path.read_text()) if out_path.exists() else {}
    result["model"] = model.name

    if not args.skip_detection:
        result["detection"] = {}
        for sz in args.imgsz:
            print(f"== detection metrics @ {sz}px")
            m = detection_metrics(model, args.data, sz, f"{model.stem}_test_{sz}")
            result["detection"][str(sz)] = m
            print(f"   mAP50={m['mAP50']}  mAP50-95={m['mAP50-95']}")
        base = result["detection"][str(args.imgsz[0])]
        result["test"] = {k: base[k] for k in ("mAP50", "mAP50-95", "precision", "recall")}
        for fname in (
            "confusion_matrix_normalized.png",
            "BoxPR_curve.png",
            "val_batch0_pred.jpg",
            "val_batch0_labels.jpg",
        ):
            src = Path(base["save_dir"]) / fname
            if src.exists():
                shutil.copy2(src, ROOT / "docs" / fname)

    result["compliance"] = {}
    for sz in args.compliance_sizes:
        print(f"== compliance metrics (test split) @ {sz}px")
        samples = predict_split(model, "test", args.limit, imgsz=sz)
        res = {"default": compliance_report(samples, ComplianceConfig())}
        th_path = ROOT / "configs" / f"thresholds_{sz}.json"
        if th_path.exists():
            res["tuned"] = compliance_report(samples, ComplianceConfig.from_file(th_path))
        result["compliance"][str(sz)] = res
        for mode, rep in res.items():
            for it, m in rep["items"].items():
                print(
                    f"   [{sz} {mode:7s}] {it:7s} recall={m['violation_recall']:.3f} "
                    f"precision={m['violation_precision']:.3f} F2={m['violation_f2']:.3f} "
                    f"false-alarms={m['false_alarm_rate']:.3f} silent-miss={m['silent_miss_rate']:.3f}"
                )
            print(f"   [{sz} {mode:7s}] workers detected: {rep['worker_detection_recall']}")

    out_path.write_text(json.dumps(result, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
