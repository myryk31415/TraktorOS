# TraktorOS — Real-Time Safety System for Autonomous Agricultural Machinery

## By Team undark

---

**Code**: [https://github.com/myryk31415/TraktorOS](https://github.com/myryk31415/TraktorOS) **Live demo**: [http://34.210.69.60](http://34.210.69.60)

---

## 1. Overview

#FIXME
TraktorOS is a multi-layered computer vision pipeline combining on-device object detection, monocular depth estimation, and cloud-based scene analysis to prevent hazardous situations in autonomous farming. Rather than building a single detection model, we process the image in multiple ways to enable selfdriving vehicles to make safe navigation decisions.

Our system combines three analysis layers:

1. **Image Quality Assessment** — BRISQUE and classical CV metrics to determine if sensor input is reliable enough for safe operation
2. **On-Device Analysis** — Faster R-CNN / YOLO11x for object detection + MiDaS for depth estimation, producing instant action recommendations (stop, honk, steer, continue)
3. **Thorough Analysis** — Amazon Bedrock (Nova Pro) for scene-level understanding: ground conditions, path analysis, vegetation/maintenance detection

The entire system is deployed on AWS EC2 with a web-based dashboard, CI/CD via GitHub Actions, and a live demo accessible at [http://34.210.69.60](http://34.210.69.60).

---

## 2. Technical Approach

### 2.1 Detection Pipeline

We use a modular design which enables the models to be switched easily, on the website we provide two example models:

- **Faster R-CNN (ResNet-50 FPN)** pretrained on COCO, filtered to agriculture-relevant classes: persons, vehicles, animals and field obstacles.
- **YOLO11x** for higher-throughput detection

Both run locally on the machine with no external API calls, ensuring data stays within the vehicle as required.

### 2.2 Depth Estimation

**MiDaS** provides monocular depth estimation for every frame. For each detected object, we sample the depth in the lower half of the bounding box (feet/base area) and classify proximity

### 2.3 Tractor Path Modeling

The UI provides a configurable perspective trapezoid representing the tractor's forward path, defined by `Tractor width` and `Horizon line`. The calculated path is used to determine the relevance of detected objects for the navigation.

### 2.4 Action Decision Tree

The core innovation is a deterministic decision tree that combines detection class, depth, and path position into concrete tractor commands:

**Pre-checks:**
- Image quality insufficient (blurry, poor BRISQUE) → **STOP**
- No objects detected → **CONTINUE**

**Moving objects** (person, dog, horse, sheep, cow, bird):

| Proximity | In Path | Action |
|-----------|---------|--------|
| Very close | Yes | STOP |
| Very close | No | HONK |
| Nearby | Yes | STOP |
| Nearby | No | HONK |
| Far | Yes | HONK |
| Far | No | — |

**Stationary objects** (car, truck, bicycle, motorcycle, etc.):

| Proximity | In Path | Action |
|-----------|---------|--------|
| Very close | Yes | STOP |
| Nearby | Yes, left side | CORRECT RIGHT |
| Nearby | Yes, right side | CORRECT LEFT |
| Far | Yes | CONTINUE |
| Any | No | CONTINUE |

**Multi-object rules:**
- STOP overrides all other actions
- HONK can combine with steering corrections
- Obstacles on both sides → STOP (cannot dodge)
- CONTINUE only if no other action applies

### 2.5 Thorough Analysis (Amazon Bedrock)

For deeper scene understanding, we use Amazon Bedrock's Nova Pro model to analyze:

- **Ground assessment**: Surface type, traversability (safe/caution/unsafe), hazards
- **Path prediction**: Detect if on a trail/road, upcoming turns and their direction/distance
- **Maintenance detection**: Overhanging branches, damaged fences, blocked drainage, erosion — issues requiring farmer attention

These results feed back into the action recommendations: unsafe ground triggers STOP, detected turns generate TURN actions, and maintenance items are flagged.

### 2.6 Image Quality Gate

Before trusting any detection, we assess image quality using:
- **Laplacian variance** for blur/sharpness detection
- **BRISQUE** (Blind/Referenceless Image Spatial Quality Evaluator) for perceptual quality
- **NIMA** (Neural Image Assessment) for aesthetic/technical quality

If quality is insufficient, the system recommends STOP — because unreliable sensor data means unreliable decisions.

---

## 3. Architecture

```
User → nginx (:80) → Static Frontend (HTML/CSS/JS)
                   → Flask API (:5000) → Faster R-CNN / YOLO11x (object detection)
                                       → MiDaS (depth estimation)
                                       → BRISQUE + NIMA (quality assessment)
                                       → Amazon Bedrock Nova Pro (scene analysis)
```

- **Frontend**: Bootstrap 5 dashboard with real-time canvas rendering of detections, depth maps, and action cards
- **Backend**: Flask service running PyTorch models locally on the onboard machine
- **Infrastructure**: EC2 t3.xlarge (4 vCPU, 16GB RAM), nginx reverse proxy, GitHub Actions CI/**CD**
- **Auto-deploy**: Every push to `main` triggers deployment to EC2 via SSH

---

## 4. Evaluation

### Detection Performance
We use pretrained models (Faster R-CNN on COCO, YOLO11x) which provide strong baseline performance on person detection. The COCO-pretrained Faster R-CNN achieves ~37.9 mAP on the COCO validation set, with person detection being one of its strongest categories.

### System Latency (Timing Breakdown)
The `/detect` endpoint returns per-request timing:
- **Image decode**: ~30-50ms
- **Faster R-CNN inference**: ~800-1500ms (CPU)
- **MiDaS depth estimation**: ~400-800ms (CPU)
- **Post-processing**: ~50-100ms
- **Total**: ~1.5-2.5s per frame on t3.xlarge (CPU-only)

With GPU acceleration (e.g., ml.g4dn instance), inference would drop to ~100-200ms total.

### Quality Assessment
BRISQUE scores correlate with detection reliability — images scoring >60 (poor) show measurably lower detection confidence, validating our quality gate approach.

### Action Decision Accuracy
The decision tree is deterministic and auditable. Every action can be traced back to specific detections, their depth estimates, and path positions. This transparency is critical for safety certification in agricultural autonomy.

---

## 5. Innovation & Future Vision

### Edge Deployment
The decision tree is pure JavaScript running client-side — it requires zero additional resources. On a real harvester, the detection models run on an onboard GPU (e.g., NVIDIA Jetson), while the decision tree runs on the vehicle's main controller. The Bedrock analysis would only be used when connectivity is available, as a secondary validation layer.

### Handling False Positives
Our multi-layered approach inherently reduces false positive impact:
- The **depth estimation** layer filters out detections that are far away and not in the path
- The **path corridor** ignores objects outside the tractor's trajectory
- The **decision tree** differentiates between moving and stationary objects — a scarecrow (stationary, not moving) gets a steering correction rather than an emergency stop
- The **quality gate** prevents decisions based on degraded sensor input

### Beyond Human Detection
The system already detects vehicles, animals, and obstacles. The Bedrock analysis adds ground assessment and maintenance detection. Future extensions:
- **Crop health monitoring** using the same camera feed
- **Fence line detection** for autonomous boundary following
- **Weather condition assessment** for operational decisions
- **Fleet coordination** — sharing detected hazards across multiple machines
