"""SiteGuard - PPE compliance detection on construction sites."""

from siteguard.compliance import ComplianceConfig, ComplianceReport, evaluate
from siteguard.detector import OnnxDetector, UltralyticsDetector, load_detector
from siteguard.types import CLASS_NAMES, Detections

__version__ = "1.0.0"

__all__ = [
    "CLASS_NAMES",
    "ComplianceConfig",
    "ComplianceReport",
    "Detections",
    "OnnxDetector",
    "UltralyticsDetector",
    "evaluate",
    "load_detector",
]
