# Traktoros — Hybrid Autonomous Tractor Steering Agent Using Visual Language Action Model
## By Team undark

---

**Code**: [https://github.com/myryk31415/TraktorOS](https://github.com/myryk31415/TraktorOS) **Live demo**: [http://34.210.69.60](http://34.210.69.60)

---

## 1. Overview

TraktorOS is a safety system for autonomous agricultural machinery that goes beyond detection — it decides what the tractor should do. We combine multiple computer vision techniques into a pipeline that takes a camera image and outputs concrete actions: stop, honk, steer, or continue.

The system processes each image through three layers:

1. **Image Quality Gate** — BRISQUE, NIMA, and classical CV metrics verify the sensor input is reliable. If not, the tractor stops.
2. **On-Device Analysis** — Object detection (Faster R-CNN / YOLO11x) and monocular depth estimation (MiDaS) identify obstacles, estimate their distance, and determine if they are in the tractor's path. A decision tree translates this into real-time actions.

Both of these run entirely on the tractor's onboard hardware — no internet connection required, enabling split-second decisions in the field.

3. **Thorough Analysis** — When connectivity is available, Amazon Bedrock (Nova Pro) provides deeper scene understanding: ground conditions, upcoming turns, and maintenance issues like overhanging branches that the farmer should address.

The entire system runs on AWS EC2 with automated deployment via GitHub Actions. A live demo is available at [http://34.210.69.60](http://34.210.69.60). The EC2 instance serves as a stand-in for the onboard computer — in production, the same code would run directly on the tractor's hardware.

---

## 2. Technical Approach

The end goal of our pipeline is to produce concrete actions for the tractor: **STOP** (emergency halt), **HONK** (audible warning), **CORRECT LEFT/RIGHT** (steer to avoid an obstacle), **CONTINUE** (path is clear), or **MAINTENANCE** (flag an issue for the farmer to address later, e.g. overhanging branches, damaged fences). Every component described below feeds into this decision.

### 2.1 Image Quality Gate

Before trusting any detection, we assess image quality using:
- **Laplacian variance** for blur/sharpness detection
- **BRISQUE** (Blind/Referenceless Image Spatial Quality Evaluator) for perceptual quality
- **NIMA** (Neural Image Assessment) for aesthetic/technical quality

If quality is insufficient, the system recommends STOP — because unreliable sensor data means unreliable decisions.

### 2.2 Detection Pipeline

Once the image passes the quality gate, we run object detection. In machine learning, this problem is generally approached through two distinct architectures: one-step detectors (mapping raw pixels directly to labeled bounding boxes) and two-step detectors (proposing regions of interest before classifying them).

Initially, we considered YOLO (You Only Look Once), a prominent state-of-the-art one-step solution. For a rigorous performance comparison, we also evaluated Faster R-CNN with a ResNet-50 backbone, a classic two-step architecture. It is important to note that for this evaluation, we utilized pre-trained models rather than training from scratch, allowing us to leverage features learned from large-scale datasets like COCO.

While one-step models are traditionally prized for their speed, our trials yielded surprising results:

As for accuracy, ResNet-50 consistently outperformed various YOLO configurations (including the 'S' and 'XL' variants) in detection precision.
Furthermore, contrary to the general expectation that two-step architectures are slower, the pre-trained ResNet-50 implementation demonstrated faster inference times in our specific testing environment.

Based on this superior balance of accuracy and speed, we have selected Faster R-CNN ResNet-50 as our primary model.
Both run locally on the machine with no external API calls, ensuring data stays within the vehicle as required.

### 2.3 Depth Estimation

Knowing *what* is in the frame is not enough — we also need to know *how far away* it is. A person 50 meters ahead requires a different response than one 2 meters in front of the tractor. Since we work with a single camera, we use **MiDaS** for monocular depth estimation. For each detected object, we sample the depth in the lower half of the bounding box (feet/base area) and classify proximity into three levels. This depth information feeds directly into the action decision tree.

### 2.4 Tractor Path Modeling

The UI provides a configurable perspective trapezoid representing the tractor's forward path, defined by `Tractor width` and `Horizon line`. The calculated path is used to determine the relevance of detected objects for the navigation.

### 2.5 Action Decision Tree

The core innovation is a deterministic decision tree that translates detections into concrete tractor commands. The logic is built around two key insights about object behavior:

**Stationary objects** (vehicles, fences, rocks, etc.) are predictable — they won't move into our path. We detect whether they are in our corridor. If an obstacle is in the path but still reasonably far away, we attempt to steer around it. If it is directly in front of us with no room to maneuver, we stop. Objects outside the path are ignored.

**Moving objects** (people, animals) are unpredictable — even if they are currently beside the tractor, they might step into its path at any moment. That's why we honk whenever we are approaching or passing by a person or animal, regardless of whether they are in the corridor. If a moving object is in our path and close enough that a collision is imminent, we stop immediately.

Additional rules ensure safe behavior in edge cases: if the image quality is too poor to trust detections, we stop. If obstacles block both sides so we cannot dodge, we stop.

### 2.6 Thorough Analysis (Amazon Bedrock)

For deeper scene understanding, we use Amazon Bedrock's Nova Pro model to analyze:

- **Ground assessment**: Surface type, traversability (safe/caution/unsafe), hazards
- **Path prediction**: Detect if on a trail/road, upcoming turns and their direction/distance
- **Maintenance detection**: Overhanging branches, damaged fences, blocked drainage, erosion — issues requiring farmer attention

These results feed back into the action recommendations: unsafe ground triggers STOP, detected turns generate TURN actions, and maintenance items are flagged.

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

## 5. Future Vision

### Time Series Data
Currently each frame is analyzed independently. By incorporating temporal context — tracking objects across frames — we can predict trajectories, distinguish a person walking toward the path from one walking away, and reduce false positives from momentary misdetections. Time series data also enables speed estimation, which directly improves the accuracy of our action decisions.

### Fine-Tuned Models
Our current pipeline uses pretrained models (COCO weights). Fine-tuning on agriculture-specific datasets like the one provided would dramatically improve detection in the conditions that matter most: people partially hidden by crops, dust-obscured animals, and farm equipment in unusual positions. The modular architecture makes swapping models straightforward — the decision tree and UI remain unchanged.

### Real-Time Streaming
The current system processes individual uploaded images. The natural next step is real-time video streaming from the tractor's camera, with continuous action recommendations updating live. This would turn TraktorOS from a demo into an operational safety system — a constant feed of detections, depth estimates, and actions rendered on a dashboard mounted in the cabin or fed directly into the vehicle's control system.
