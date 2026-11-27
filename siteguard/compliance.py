"""Turn raw PPE detections into per-worker compliance decisions.

The detector outputs independent boxes (``Person``, ``helmet``, ``no_helmet``,
``vest``, ...). A safety system needs answers per *worker*: "is this person
wearing a helmet?". This module:

1. assigns each PPE box to the person it belongs to, using *containment*
   (share of the item's area inside the person box) and the *body region*
   where the item is expected (helmet near the top, vest on the torso);
2. resolves conflicts when an item overlaps several people (closest match wins,
   each item belongs to at most one person);
3. decides a status per required item: ``ok`` / ``missing`` (explicit
   ``no_*`` detection) / ``unknown`` (nothing visible - occluded, cropped, too small);
4. aggregates to a person status: ``compliant`` / ``violation`` / ``unverified``.

``unknown`` is deliberately separate from ``missing``: flagging every
partially-visible worker as a violation produces alert fatigue, while silently
treating them as compliant hides risk. ``strict=True`` upgrades unknown to violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from siteguard.types import Detections

# Expected vertical position of each item, as a fraction of the person box height
# (0 = top of the box, 1 = bottom). The item's centre must fall in this band.
BODY_REGIONS: dict[str, tuple[float, float]] = {
    "helmet": (-0.15, 0.30),  # may stick out slightly above the person box
    "goggles": (-0.05, 0.30),
    "vest": (0.10, 0.80),
    "gloves": (0.20, 0.95),
    "boots": (0.65, 1.10),
}
NEGATIVE = {
    "helmet": "no_helmet",
    "vest": "no_vest",
    "goggles": "no_goggle",
    "gloves": "no_gloves",
    "boots": "no_boots",
}


@dataclass
class ComplianceConfig:
    required: tuple[str, ...] = ("helmet", "vest")
    min_confidence: float = 0.25
    # per-class overrides, e.g. {"no_helmet": 0.15}: a missed violation costs more than a false alarm
    class_confidence: dict[str, float] = field(default_factory=dict)
    min_containment: float = 0.5  # share of the item box inside the person box
    strict: bool = False  # treat "unknown" as a violation

    @property
    def detector_confidence(self) -> float:
        """Lowest threshold in use - the detector must not drop boxes the rules still need."""
        return min([self.min_confidence, *self.class_confidence.values()])

    @classmethod
    def from_file(cls, path, **overrides) -> ComplianceConfig:
        """Load tuned thresholds (``configs/thresholds.json``); missing file -> defaults."""
        import json
        from pathlib import Path

        data = {}
        if path and Path(path).exists():
            raw = json.loads(Path(path).read_text())
            data = {
                "min_confidence": raw.get("min_confidence", 0.25),
                "class_confidence": raw.get("class_confidence", {}),
            }
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    def __post_init__(self):
        unknown = set(self.required) - set(NEGATIVE)
        if unknown:
            raise ValueError(f"Unsupported PPE item(s): {sorted(unknown)}; choose from {sorted(NEGATIVE)}")
        self.required = tuple(self.required)


@dataclass
class ItemStatus:
    status: str  # ok | missing | unknown
    confidence: float | None = None
    box: list[float] | None = None


@dataclass
class PersonResult:
    box: list[float]
    confidence: float
    status: str  # compliant | violation | unverified
    items: dict[str, ItemStatus] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "box": self.box,
            "confidence": self.confidence,
            "status": self.status,
            "items": {k: v.__dict__ for k, v in self.items.items()},
        }


@dataclass
class ComplianceReport:
    people: list[PersonResult]
    unassigned_violations: list[dict]  # e.g. "no_helmet" on a person the detector missed

    @property
    def counts(self) -> dict[str, int]:
        c = {"compliant": 0, "violation": 0, "unverified": 0}
        for p in self.people:
            c[p.status] += 1
        return c

    @property
    def has_violation(self) -> bool:
        return self.counts["violation"] > 0 or bool(self.unassigned_violations)

    def to_dict(self) -> dict:
        return {
            "summary": {
                **self.counts,
                "people": len(self.people),
                "unassigned_violations": len(self.unassigned_violations),
                "has_violation": self.has_violation,
            },
            "people": [p.to_dict() for p in self.people],
            "unassigned_violations": self.unassigned_violations,
        }


# --------------------------------------------------------------------------- geometry
def containment(items: np.ndarray, people: np.ndarray) -> np.ndarray:
    """(I, P) matrix: fraction of each item box's area that lies inside each person box."""
    if len(items) == 0 or len(people) == 0:
        return np.zeros((len(items), len(people)), dtype=np.float32)
    ix1 = np.maximum(items[:, None, 0], people[None, :, 0])
    iy1 = np.maximum(items[:, None, 1], people[None, :, 1])
    ix2 = np.minimum(items[:, None, 2], people[None, :, 2])
    iy2 = np.minimum(items[:, None, 3], people[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area = np.clip((items[:, 2] - items[:, 0]) * (items[:, 3] - items[:, 1]), 1e-6, None)
    return inter / area[:, None]


def relative_height(items: np.ndarray, people: np.ndarray) -> np.ndarray:
    """(I, P) matrix: vertical position of each item centre inside each person box (0 top, 1 bottom)."""
    if len(items) == 0 or len(people) == 0:
        return np.zeros((len(items), len(people)), dtype=np.float32)
    cy = (items[:, 1] + items[:, 3]) / 2
    h = np.clip(people[:, 3] - people[:, 1], 1e-6, None)
    return (cy[:, None] - people[None, :, 1]) / h[None, :]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-6, None)


# --------------------------------------------------------------------------- assignment
def _assign(items: np.ndarray, people: np.ndarray, region: tuple[float, float], min_containment: float) -> np.ndarray:
    """Return, for each item, the index of the person it belongs to (-1 = none).

    Score = containment; candidates outside the expected body region are rejected.
    Greedy one-to-one isn't needed (a person can have two gloves), but each item
    goes to its single best person - this avoids one helmet "covering" two
    overlapping workers.
    """
    cont = containment(items, people)
    rel = relative_height(items, people)
    valid = (cont >= min_containment) & (rel >= region[0]) & (rel <= region[1])
    score = np.where(valid, cont, -1.0)
    if score.size == 0:
        return np.full(len(items), -1, dtype=int)
    # tie-break towards the smaller (closer, more specific) person box
    areas = (people[:, 2] - people[:, 0]) * (people[:, 3] - people[:, 1])
    score = score - 1e-9 * areas[None, :]
    best = score.argmax(axis=1)
    best[score.max(axis=1) < 0] = -1
    return best


def _threshold_mask(det: Detections, cfg: ComplianceConfig) -> np.ndarray:
    thr = np.full(len(det), cfg.min_confidence, dtype=np.float32)
    for name, value in cfg.class_confidence.items():
        if name in det.names:
            thr[det.class_id == det.names.index(name)] = value
    return det.confidence >= thr


def evaluate(det: Detections, cfg: ComplianceConfig | None = None) -> ComplianceReport:
    cfg = cfg or ComplianceConfig()
    det = det.filter(_threshold_mask(det, cfg))
    p_idx = det.where("Person")
    people = det.xyxy[p_idx]
    results = [
        PersonResult([round(float(v), 1) for v in det.xyxy[i]], round(float(det.confidence[i]), 3), "unverified")
        for i in p_idx
    ]
    unassigned: list[dict] = []

    for item in cfg.required:
        region = BODY_REGIONS[item]
        # best positive / negative evidence per person
        evidence: dict[int, dict[str, tuple[float, list[float]]]] = {}
        for label in (item, NEGATIVE[item]):
            idx = det.where(label)
            owner = _assign(det.xyxy[idx], people, region, cfg.min_containment)
            for k, o in zip(idx, owner, strict=True):
                conf = float(det.confidence[k])
                box = [round(float(v), 1) for v in det.xyxy[k]]
                if o < 0:
                    if label == NEGATIVE[item]:
                        unassigned.append({"class": label, "confidence": round(conf, 3), "box": box})
                    continue
                prev = evidence.setdefault(int(o), {}).get(label)
                if prev is None or conf > prev[0]:
                    evidence[int(o)][label] = (conf, box)

        for pi, res in enumerate(results):
            ev = evidence.get(pi, {})
            pos, neg = ev.get(item), ev.get(NEGATIVE[item])
            if pos and (not neg or pos[0] >= neg[0]):
                res.items[item] = ItemStatus("ok", round(pos[0], 3), pos[1])
            elif neg:
                res.items[item] = ItemStatus("missing", round(neg[0], 3), neg[1])
            else:
                res.items[item] = ItemStatus("unknown")

    for res in results:
        states = {s.status for s in res.items.values()}
        if "missing" in states or (cfg.strict and "unknown" in states):
            res.status = "violation"
        elif states <= {"ok"}:
            res.status = "compliant"
        else:
            res.status = "unverified"
    return ComplianceReport(results, unassigned)
