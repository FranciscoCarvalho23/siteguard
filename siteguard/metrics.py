"""Worker-level compliance metrics (shared by evaluation and threshold tuning)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

import numpy as np

from siteguard.compliance import ComplianceConfig, evaluate, iou_matrix
from siteguard.types import Detections


def match_people(gt: np.ndarray, pred: np.ndarray, thr: float = 0.5) -> dict[int, int]:
    """Greedy IoU matching of ground-truth to predicted person boxes -> {gt_idx: pred_idx}."""
    iou = iou_matrix(gt.reshape(-1, 4), pred.reshape(-1, 4))
    pairs: dict[int, int] = {}
    while iou.size and iou.max() >= thr:
        g, p = np.unravel_index(iou.argmax(), iou.shape)
        pairs[int(g)] = int(p)
        iou[g, :] = -1
        iou[:, p] = -1
    return pairs


def compliance_confusion(samples: Iterable[tuple[Detections, Detections]], cfg: ComplianceConfig) -> tuple[dict, int, int]:
    """samples: (ground_truth, prediction) per image.

    Returns per-item Counter of (gt_state, predicted_state), number of GT workers and
    how many of them were matched to a predicted worker. Ground-truth states use the
    labels as perfect detections with the same compliance rules.
    """
    gt_cfg = ComplianceConfig(required=cfg.required, min_confidence=0.0, min_containment=cfg.min_containment)
    confusion = {it: Counter() for it in cfg.required}
    total = found = 0
    for gt_det, pred_det in samples:
        gt, pred = evaluate(gt_det, gt_cfg), evaluate(pred_det, cfg)
        pairs = match_people(np.array([p.box for p in gt.people]), np.array([p.box for p in pred.people]))
        total += len(gt.people)
        found += len(pairs)
        for gi, gp in enumerate(gt.people):
            for it in cfg.required:
                p_state = pred.people[pairs[gi]].items[it].status if gi in pairs else "no-person"
                confusion[it][(gp.items[it].status, p_state)] += 1
        # predicted "workers" with no real worker behind them (e.g. an empty forklift seat):
        # if they get flagged, that is a false alarm and must count against precision
        matched = set(pairs.values())
        for pi, pp in enumerate(pred.people):
            if pi not in matched:
                for it in cfg.required:
                    confusion[it][("no-worker", pp.items[it].status)] += 1
    return confusion, total, found


def summarize(c: Counter) -> dict:
    gt_missing = sum(v for (g, _), v in c.items() if g == "missing")
    pred_missing = sum(v for (_, p), v in c.items() if p == "missing")
    gt_ok = sum(v for (g, _), v in c.items() if g == "ok")
    tp = c[("missing", "missing")]
    recall = tp / max(1, gt_missing)
    precision = tp / max(1, pred_missing)
    f2 = 5 * precision * recall / max(1e-9, 4 * precision + recall)
    return {
        "violation_recall": round(recall, 3),  # of workers really without the item, share flagged
        "violation_precision": round(precision, 3),  # of the flags raised, share that were real
        "violation_f2": round(f2, 3),  # recall-weighted: a missed violation costs more
        "false_alarm_rate": round(c[("ok", "missing")] / max(1, gt_ok), 3),
        "spurious_worker_flags": c[("no-worker", "missing")],
        # the dangerous error: worker without the item reported as wearing it
        "silent_miss_rate": round(c[("missing", "ok")] / max(1, gt_missing), 3),
        # recall if "unverified" workers are also escalated (strict mode)
        "strict_recall": round((tp + c[("missing", "unknown")]) / max(1, gt_missing), 3),
        "ok_recall": round(c[("ok", "ok")] / max(1, gt_ok), 3),
        "gt_missing": gt_missing,
        "gt_ok": gt_ok,
        "confusion": {f"{g}->{p}": v for (g, p), v in sorted(c.items())},
    }
