# Model card — SiteGuard PPE detector

## Model

| | |
|---|---|
| Architecture | YOLO11n (2.6 M parameters, 6.5 GFLOPs at 640 px), COCO-pretrained, fine-tuned |
| Classes | helmet, gloves, vest, boots, goggles, Person, no_helmet, no_goggle, no_gloves, no_boots, forklift, no_vest |
| Training | 20 epochs, 640 px, batch 8, AdamW (lr 6.25e-4, picked by Ultralytics `optimizer=auto`), cosine LR, mosaic off for the last 5 epochs, seed 0, CPU (110 min) |
| Best epoch | 15 (val mAP50-95 0.542) |
| Artefacts | `ppe_yolo11n_640.pt` (5.5 MB, PyTorch), `siteguard.onnx` (11 MB, FP32, dynamic H/W, used at 960 px) |

## Data

Ultralytics Construction-PPE, CC BY 4.0: 1,000 images of construction sites and warehouses, ~15 boxes per image.

| Split | Images | Source | Used for |
|---|---|---|---|
| train | 700 | original train | weights |
| val | 100 | original train (seed 42) | best epoch, threshold tuning |
| test | 200 | original val | final numbers only |

Objects are small: the median `no_helmet` box covers 0.06% of the image.

## Intended use

Decision support for site safety staff reviewing photos or camera snapshots: highlight workers who may be missing a helmet or high-visibility vest so a person can check. **Not** for automated disciplinary action, identification of individuals, or as the sole safety control.

## Performance (test split)

Detection: mAP50 0.739 @640 px, 0.766 @960 px.

Shipped configuration (960 px, thresholds from `configs/thresholds_960.json`):

| | Helmet | Vest |
|---|---|---|
| Violations caught (recall) | 71.8% | 70.9% |
| Flags that are real (precision) | 86.8% | 84.1% |
| Missing item reported as OK | 4.0% | 3.9% |
| Compliant workers wrongly flagged | 3.9% | 2.7% |
| Recall in strict mode (unverified escalated) | 89.1% | 86.6% |

Workers detected: 93.8%.

## Limitations and risks

- Small, homogeneous dataset; performance on real CCTV (low resolution, night, rain, unusual angles, other PPE colours) is unknown and likely lower.
- Missed workers (6%) are never assessed. Distant or occluded workers are often *unverified*.
- False "persons" (empty vehicle seats) can produce false alarms.
- The dataset was not audited for demographic balance; performance across skin tones, body types or headwear (e.g. turbans, hijabs) was not measured and should be before any deployment.
- Privacy: images of workers are personal data — process on-premises where possible, keep no images by default, and inform workers.
