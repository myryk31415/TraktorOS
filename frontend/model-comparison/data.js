window.MODEL_COMPARISON_DATA = {
  metadata: {
    description: "Comparison of selected pretrained object detection models with bounding box prediction capability",
    created: "2026-04-11",
    benchmark_dataset: "MS COCO (Common Objects in Context)",
    notes: [
      "All AP/mAP values follow COCO evaluation protocol unless noted otherwise.",
      "AP (or mAP) without qualifier = AP@[IoU=0.50:0.95], the primary COCO metric.",
      "AP50 = AP at IoU threshold 0.50. AP75 = AP at IoU threshold 0.75.",
      "Latency measured on NVIDIA T4 GPU with TensorRT FP16 unless noted.",
      "F1 is not a standard COCO benchmark metric; mAP integrates over the full precision-recall curve across IoU thresholds."
    ]
  },
  models: [
    {
      model_name: "RF-DETR Large",
      family: "RF-DETR",
      architecture_type: "transformer",
      detection_type: "end-to-end (NMS-free)",
      backbone: "DINOv2 ViT",
      paper_title: "RF-DETR: Neural Architecture Search for Real-Time Detection Transformers",
      paper_url: "https://arxiv.org/abs/2511.09554",
      year_released: 2025,
      license: "Apache 2.0",
      code_url: "https://github.com/roboflow/rf-detr",
      params_millions: 128,
      input_resolution: 728,
      gflops: null,
      latency_ms: 40.0,
      fps: 25,
      coco: { split: "val2017 / test-dev", ap: 60.5, ap50: null, ap75: null },
      source_notes: [
        "128M parameters",
        "728 input resolution",
        "60.5 AP on COCO"
      ]
    },
    {
      model_name: "RF-DETR Small",
      family: "RF-DETR",
      architecture_type: "transformer",
      detection_type: "end-to-end (NMS-free)",
      backbone: "DINOv2 ViT",
      paper_title: "RF-DETR: Neural Architecture Search for Real-Time Detection Transformers",
      paper_url: "https://arxiv.org/abs/2511.09554",
      year_released: 2025,
      license: "Apache 2.0",
      code_url: "https://github.com/roboflow/rf-detr",
      params_millions: null,
      input_resolution: null,
      gflops: null,
      latency_ms: 3.52,
      fps: 284,
      coco: { split: "val2017", ap: 53.0, ap50: 72.1, ap75: null },
      rf100vl: { ap: 59.6, ap50: 85.9 },
      source_notes: [
        "53.0 AP on COCO",
        "72.1 AP50 on COCO",
        "3.52 ms latency on T4"
      ]
    },
    {
      model_name: "YOLOv12",
      family: "YOLO",
      architecture_type: "attention-centric CNN",
      detection_type: "one-stage (anchor-based, uses NMS)",
      backbone: "R-ELAN with Area Attention + FlashAttention",
      paper_title: "YOLOv12: Attention-Centric Real-Time Object Detectors",
      paper_url: "https://arxiv.org/abs/2502.12524",
      year_released: 2025,
      license: "AGPL-3.0",
      code_url: "https://github.com/sunsmarterjie/yolov12",
      variants: [
        { model_name: "YOLOv12-N", params_millions: 2.6, gflops: 6.5, ap: 40.6, latency_ms: 1.64, fps: 610 },
        { model_name: "YOLOv12-S", params_millions: 9.3, gflops: 21.4, ap: 48.0, latency_ms: 2.61, fps: 383 },
        { model_name: "YOLOv12-M", params_millions: 20.2, gflops: 67.5, ap: 52.5, latency_ms: 4.86, fps: 206 },
        { model_name: "YOLOv12-L", params_millions: 26.4, gflops: 88.9, ap: 53.7, latency_ms: 6.77, fps: 148 },
        { model_name: "YOLOv12-X", params_millions: 59.1, gflops: 199.0, ap: 55.2, latency_ms: 11.79, fps: 85 }
      ],
      coco_split: "val2017 @ 640×640",
      performance_source: "https://arxiv.org/html/2502.12524v1"
    },
    {
      model_name: "Faster R-CNN ResNet50 FPN v1",
      family: "Faster R-CNN",
      architecture_type: "two-stage CNN",
      detection_type: "two-stage (anchor-based, uses NMS)",
      backbone: "ResNet-50 + FPN",
      paper_title: "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks",
      paper_url: "https://arxiv.org/abs/1506.01497",
      year_released: 2015,
      license: "BSD-3-Clause (PyTorch / TorchVision)",
      code_url: "https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py",
      torchvision_model_name: "fasterrcnn_resnet50_fpn",
      torchvision_weights: "FasterRCNN_ResNet50_FPN_Weights.COCO_V1",
      params_millions: 41.76,
      gflops: 134.38,
      latency_ms: null,
      fps: null,
      coco: { split: "val2017", ap: 37.0, ap50: null, ap75: null },
      source_notes: ["TorchVision pretrained baseline", "AP@[.50:.95] = 37.0"]
    },
    {
      model_name: "Faster R-CNN ResNet50 FPN v2",
      family: "Faster R-CNN",
      architecture_type: "two-stage CNN",
      detection_type: "two-stage (anchor-based, uses NMS)",
      backbone: "ResNet-50 + FPN",
      paper_title: "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (v2 training recipe)",
      paper_url: "https://arxiv.org/abs/1912.02424",
      year_released: 2022,
      license: "BSD-3-Clause (PyTorch / TorchVision)",
      code_url: "https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py",
      torchvision_model_name: "fasterrcnn_resnet50_fpn_v2",
      torchvision_weights: "FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1",
      params_millions: 43.71,
      gflops: 280.371,
      latency_ms: null,
      fps: null,
      coco: { split: "val2017", ap: 46.7, ap50: null, ap75: null },
      source_notes: ["TorchVision v2 baseline", "AP@[.50:.95] = 46.7"]
    }
  ]
};
