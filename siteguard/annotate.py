"""Draw compliance results on an image (BGR)."""

from __future__ import annotations

import cv2
import numpy as np

from siteguard.compliance import ComplianceReport
from siteguard.types import Detections

STATUS_COLOR = {"compliant": (60, 180, 75), "violation": (40, 40, 230), "unverified": (0, 170, 255)}
ITEM_COLOR = {"ok": (60, 180, 75), "missing": (40, 40, 230)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _label(img, text, x, y, color, scale):
    th = max(1, int(round(scale * 2)))
    (tw, tht), base = cv2.getTextSize(text, FONT, scale, th)
    y = max(int(y), tht + 4)
    cv2.rectangle(img, (int(x), y - tht - 6), (int(x) + tw + 6, y + base - 2), color, -1)
    cv2.putText(img, text, (int(x) + 3, y - 3), FONT, scale, (255, 255, 255), th, cv2.LINE_AA)


def draw(
    image: np.ndarray, report: ComplianceReport, detections: Detections | None = None, show_all: bool = False
) -> np.ndarray:
    """Person boxes coloured by status; their helmet/vest evidence drawn inside."""
    img = image.copy()
    h, w = img.shape[:2]
    scale = max(0.45, min(1.6, max(h, w) / 1400))
    thick = max(2, int(round(scale * 3)))

    if show_all and detections is not None:  # raw detections, thin grey
        for i in range(len(detections)):
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            cv2.rectangle(img, (x1, y1), (x2, y2), (180, 180, 180), 1)

    for p in report.people:
        color = STATUS_COLOR[p.status]
        x1, y1, x2, y2 = map(int, p.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        tags = []
        for item, st in p.items.items():
            mark = {"ok": "+", "missing": "NO", "unknown": "?"}[st.status]
            tags.append(f"{mark}{item}" if st.status != "missing" else f"NO {item}")
            if st.box is not None:
                bx1, by1, bx2, by2 = map(int, st.box)
                cv2.rectangle(img, (bx1, by1), (bx2, by2), ITEM_COLOR[st.status], max(1, thick - 1))
        _label(img, f"{p.status.upper()}  {' '.join(tags)}", x1, y1, color, scale * 0.6)

    for v in report.unassigned_violations:
        x1, y1, x2, y2 = map(int, v["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), STATUS_COLOR["violation"], thick)
        _label(img, v["class"], x1, y1, STATUS_COLOR["violation"], scale * 0.55)

    c = report.counts
    banner = (
        f"workers: {len(report.people)}   compliant: {c['compliant']}   "
        f"violation: {c['violation']}   unverified: {c['unverified']}"
    )
    _label(img, banner, 8, int(30 * scale) + 10, (30, 30, 30), scale * 0.7)
    return img
