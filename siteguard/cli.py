"""Audit a folder of site photos from the command line.

python -m siteguard.cli photos/ --out audit/                 # annotated images + report.csv + report.json
python -m siteguard.cli photo.jpg --required helmet,vest,boots --strict
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

from siteguard.annotate import draw
from siteguard.compliance import ComplianceConfig, evaluate
from siteguard.detector import load_detector

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def collect(sources: list[str]) -> list[Path]:
    files: list[Path] = []
    for s in sources:
        p = Path(s)
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXT)
        elif p.suffix.lower() in IMAGE_EXT:
            files.append(p)
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="siteguard", description="PPE compliance audit for site photos")
    ap.add_argument("sources", nargs="+", help="images and/or folders")
    ap.add_argument("--model", default=str(ROOT / "models" / "siteguard.onnx"))
    ap.add_argument("--out", default="audit")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument(
        "--thresholds",
        default=None,
        help="per-class thresholds (default: configs/thresholds_<model input size>.json; 'none' to disable)",
    )
    ap.add_argument("--required", default="helmet,vest")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--no-images", action="store_true", help="only write the reports")
    args = ap.parse_args(argv)

    files = collect(args.sources)
    if not files:
        print("No images found.", file=sys.stderr)
        return 1
    detector = load_detector(args.model)
    th = args.thresholds or ROOT / "configs" / f"thresholds_{getattr(detector, 'imgsz', 640)}.json"
    cfg = ComplianceConfig.from_file(
        None if args.thresholds == "none" else th,
        required=tuple(args.required.split(",")),
        strict=args.strict,
        min_confidence=args.conf,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows, details = [], []
    t0 = time.time()
    for i, f in enumerate(files, 1):
        img = cv2.imread(str(f))
        if img is None:
            print(f"skip (unreadable): {f}", file=sys.stderr)
            continue
        report = evaluate(detector.predict(img, conf=cfg.detector_confidence), cfg)
        c = report.counts
        rows.append(
            {
                "image": str(f),
                "workers": len(report.people),
                **c,
                "unassigned_violations": len(report.unassigned_violations),
                "has_violation": report.has_violation,
            }
        )
        details.append({"image": str(f), **report.to_dict()})
        if not args.no_images:
            cv2.imwrite(str(out / f"{f.stem}_checked.jpg"), draw(img, report))
        flag = "VIOLATION" if report.has_violation else "ok"
        print(f"[{i}/{len(files)}] {f.name}: {len(report.people)} workers, {c['violation']} violation(s) -> {flag}")

    with open(out / "report.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "report.json").write_text(json.dumps(details, indent=2))
    flagged = sum(r["has_violation"] for r in rows)
    print(f"\n{len(rows)} images in {time.time() - t0:.1f}s - {flagged} with violations. Reports in {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
