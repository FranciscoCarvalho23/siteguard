"""Tune per-class confidence thresholds for the violation classes on the VALIDATION split.

A missed "no helmet" is worse than a false alarm, so thresholds for ``no_helmet`` /
``no_vest`` (and the matching positive classes) are chosen to maximise the
recall-weighted F2 score of violation detection. The test split is never touched
here - it is only used afterwards by ``evaluate.py`` to report the result.

    python scripts/tune_thresholds.py --model models/ppe_yolo11n_640.pt --imgsz 960
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import predict_split  # noqa: E402
from siteguard import ComplianceConfig  # noqa: E402
from siteguard.compliance import NEGATIVE  # noqa: E402
from siteguard.metrics import compliance_confusion, summarize  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "ppe_yolo11n_640.pt"))
    ap.add_argument("--items", default="helmet,vest")
    ap.add_argument("--base", type=float, default=0.25, help="threshold for every other class")
    ap.add_argument("--imgsz", type=int, default=960, help="inference size the thresholds are tuned for")
    ap.add_argument("--out", default=None, help="default: configs/thresholds_<imgsz>.json")
    args = ap.parse_args()

    items = args.items.split(",")
    print("Running detector on the validation split ...")
    samples = predict_split(Path(args.model), "val", imgsz=args.imgsz)
    grid = np.round(np.arange(0.05, 0.55, 0.05), 2)

    person_grid = [0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    def tune_item(it: str, t_person: float):
        best, rows = None, []
        for t_neg in grid:
            for t_pos in grid:
                cc = {NEGATIVE[it]: float(t_neg), it: float(t_pos), "Person": t_person}
                cfg = ComplianceConfig(required=(it,), min_confidence=args.base, class_confidence=cc)
                s = summarize(compliance_confusion(samples, cfg)[0][it])
                rows.append((float(t_neg), float(t_pos), s))
                # maximise F2; tie-break on precision so we don't pick needlessly low thresholds
                key = (s["violation_f2"], s["violation_precision"])
                if best is None or key > best[0]:
                    best = (key, float(t_neg), float(t_pos), s)
        return best, rows

    # The Person threshold is shared by all items: pick the one with the best mean F2.
    results = {}
    for t_person in person_grid:
        results[t_person] = {it: tune_item(it, t_person) for it in items}
        mean_f2 = np.mean([results[t_person][it][0][3]["violation_f2"] for it in items])
        print(f"Person>={t_person:.2f}: mean violation F2 = {mean_f2:.3f}")
    t_person = max(person_grid, key=lambda t: np.mean([results[t][it][0][3]["violation_f2"] for it in items]))

    chosen: dict[str, float] = {"Person": t_person}
    table = {}
    for it in items:
        (_, t_neg, t_pos, s), _ = results[t_person][it]
        _, default_rows = results[person_grid[0]][it]
        default = next(r[2] for r in default_rows if r[0] == args.base and r[1] == args.base)
        chosen[NEGATIVE[it]] = t_neg
        chosen[it] = t_pos
        table[it] = {"default": default, "tuned": s, "thresholds": {NEGATIVE[it]: t_neg, it: t_pos, "Person": t_person}}
        print(
            f"{it:7s} default  F2={default['violation_f2']:.3f} (R={default['violation_recall']:.3f} "
            f"P={default['violation_precision']:.3f})"
        )
        print(
            f"{it:7s} tuned    F2={s['violation_f2']:.3f} (R={s['violation_recall']:.3f} "
            f"P={s['violation_precision']:.3f})  {NEGATIVE[it]}>={t_neg}  {it}>={t_pos}  Person>={t_person}"
        )

    out = {
        "min_confidence": args.base,
        "class_confidence": chosen,
        "tuned_on": "val",
        "objective": "violation F2",
        "validation": table,
        "imgsz": args.imgsz,
    }
    out_path = Path(args.out or ROOT / "configs" / f"thresholds_{args.imgsz}.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
