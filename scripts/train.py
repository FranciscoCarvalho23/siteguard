"""Fine-tune a COCO-pretrained YOLO11 model on the PPE dataset.

    python scripts/train.py                              # defaults used for the shipped model
    python scripts/train.py --model yolo11s.pt --epochs 100 --imgsz 960 --device 0   # GPU

The best checkpoint (by val mAP50-95) is copied to ``models/ppe_<model>_<imgsz>.pt``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt", help="pretrained weights to start from")
    ap.add_argument("--data", default=str(ROOT / "configs" / "ppe.yaml"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of train set (quick experiments)")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = args.model
    local = ROOT / "models" / "pretrained" / Path(weights).name
    if local.exists():
        weights = str(local)

    name = args.name or f"{Path(args.model).stem}_{args.imgsz}"
    t0 = time.time()
    model = YOLO(weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        fraction=args.fraction,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        cache="ram",  # decode the 2048px JPEGs once instead of every epoch
        cos_lr=True,
        close_mosaic=5,  # last epochs without mosaic: closer to real images
        project=str(ROOT / "runs" / "train"),
        name=name,
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    elapsed = time.time() - t0

    run_dir = ROOT / "runs" / "train" / name
    best = run_dir / "weights" / "best.pt"
    out = ROOT / "models" / f"ppe_{name}.pt"
    out.parent.mkdir(exist_ok=True)
    shutil.copy2(best, out)
    meta = {
        "source_weights": args.model,
        "train_args": vars(args),
        "train_time_s": round(elapsed, 1),
        "run_dir": str(run_dir),
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"\nBest weights -> {out}  ({elapsed / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
