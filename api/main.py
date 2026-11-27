"""SiteGuard REST API.

    uvicorn api.main:app --port 8000          # then open http://localhost:8000

Environment:
    SITEGUARD_MODEL    path to .onnx (default) or .pt weights
    SITEGUARD_THREADS  ONNX Runtime intra-op threads (default: all cores)
    SITEGUARD_THRESHOLDS  per-class thresholds (default: configs/thresholds_<input size>.json)
"""

from __future__ import annotations

import base64
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from siteguard import ComplianceConfig, evaluate, load_detector
from siteguard.annotate import draw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models" / "siteguard.onnx"
CONFIGS = ROOT / "configs"
MAX_BYTES = 15 * 1024 * 1024
STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.environ.get("SITEGUARD_MODEL", DEFAULT_MODEL))
    kwargs = {}
    if model_path.suffix == ".onnx" and os.environ.get("SITEGUARD_THREADS"):
        kwargs["threads"] = int(os.environ["SITEGUARD_THREADS"])
    STATE["detector"] = load_detector(model_path, **kwargs)
    STATE["model_path"] = model_path
    # thresholds were tuned per inference size -> pick the file matching the model input
    imgsz = getattr(STATE["detector"], "imgsz", 640)
    th_path = Path(os.environ.get("SITEGUARD_THRESHOLDS", CONFIGS / f"thresholds_{imgsz}.json"))
    STATE["class_confidence"] = ComplianceConfig.from_file(th_path).class_confidence
    metrics = model_path.with_name("metrics.json")
    STATE["metrics"] = json.loads(metrics.read_text()) if metrics.exists() else None
    # warm-up so the first real request is not slow
    STATE["detector"].predict(np.zeros((480, 640, 3), np.uint8))
    yield
    STATE.clear()


app = FastAPI(
    title="SiteGuard",
    description="PPE compliance detection: who on site is missing a helmet or vest?",
    version="1.0.0",
    lifespan=lifespan,
)


async def _read_image(file: UploadFile) -> np.ndarray:
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"Image larger than {MAX_BYTES // (1024 * 1024)} MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) if data else None
    if img is None:
        raise HTTPException(400, "Could not decode image (send a JPEG/PNG/WebP file)")
    return img


def _config(required: str, strict: bool, conf: float) -> ComplianceConfig:
    items = tuple(s.strip() for s in required.split(",") if s.strip())
    try:
        # tuned per-class thresholds (from the validation split) apply on top of `conf`
        return ComplianceConfig(
            required=items, strict=strict, min_confidence=conf, class_confidence=dict(STATE.get("class_confidence", {}))
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


def _run(img: np.ndarray, conf: float, iou: float, cfg: ComplianceConfig):
    t0 = time.perf_counter()
    det = STATE["detector"].predict(img, conf=cfg.detector_confidence, iou=iou)
    t1 = time.perf_counter()
    report = evaluate(det, cfg)
    t2 = time.perf_counter()
    return det, report, {"inference": round((t1 - t0) * 1000, 1), "compliance": round((t2 - t1) * 1000, 2)}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "api" / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "detector" in STATE}


@app.get("/v1/model")
def model_info():
    det = STATE["detector"]
    return {
        "file": STATE["model_path"].name,
        "backend": type(det).__name__,
        "input_size": getattr(det, "imgsz", None),
        "class_confidence": STATE.get("class_confidence", {}),
        "classes": det.names,
        "metrics": STATE["metrics"],
    }


@app.post("/v1/detect")
async def detect(
    file: UploadFile = File(..., description="Image (JPEG/PNG/WebP)"),
    conf: float = Query(0.25, ge=0.01, le=0.99, description="Detection confidence threshold"),
    iou: float = Query(0.5, ge=0.1, le=0.95, description="NMS IoU threshold"),
    required: str = Query("helmet,vest", description="Comma-separated PPE: helmet, vest, goggles, gloves, boots"),
    strict: bool = Query(False, description="Count 'unknown' (not visible) PPE as a violation"),
    include_detections: bool = Query(False, description="Also return every raw detection"),
    return_image: bool = Query(False, description="Return the annotated image as base64 JPEG"),
):
    img = await _read_image(file)
    cfg = _config(required, strict, conf)
    det, report, latency = _run(img, conf, iou, cfg)
    body = {
        "image": {"width": img.shape[1], "height": img.shape[0]},
        "latency_ms": latency,
        "compliance": report.to_dict(),
    }
    if include_detections:
        body["detections"] = det.filter(det.confidence >= conf).to_list()
    if return_image:
        ok, buf = cv2.imencode(".jpg", _fit(draw(img, report)), [cv2.IMWRITE_JPEG_QUALITY, 85])
        body["image_base64"] = base64.b64encode(buf.tobytes()).decode()
    return body


@app.post("/v1/detect/annotated", responses={200: {"content": {"image/jpeg": {}}}}, response_class=Response)
async def detect_annotated(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=0.99),
    iou: float = Query(0.5, ge=0.1, le=0.95),
    required: str = Query("helmet,vest"),
    strict: bool = Query(False),
):
    img = await _read_image(file)
    _, report, latency = _run(img, conf, iou, _config(required, strict, conf))
    ok, buf = cv2.imencode(".jpg", _fit(draw(img, report)), [cv2.IMWRITE_JPEG_QUALITY, 88])
    c = report.counts
    headers = {
        "X-Workers": str(len(report.people)),
        "X-Violations": str(c["violation"]),
        "X-Inference-Ms": str(latency["inference"]),
    }
    return Response(buf.tobytes(), media_type="image/jpeg", headers=headers)


def _fit(img: np.ndarray, max_side: int = 1600) -> np.ndarray:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
