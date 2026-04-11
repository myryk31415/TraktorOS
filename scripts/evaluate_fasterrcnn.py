#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
)
from rfdetr import RFDETRLarge, RFDETRSmall
from ultralytics import YOLO

from coco_dataset import CocoDetectionDataset

import boto3

BUCKET_NAME = "Traktoros-training-data"

MODEL_TYPES = [
    # "fasterrcnn",
    "fasterrcnn_v2",
    # "rfdetr-large",
    # "rfdetr-small",
    # "yolo11-x",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained Faster R-CNN (COCO) on COCO-style annotations."
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Specify, if local dataset should be used.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/HackHPI2026_release"),
        help="Dataset root that contains 'annotation/' and 'data/' subdirectories.",
    )
    parser.add_argument(
        "--annotation-root",
        type=Path,
        default=None,
        help="Override annotation directory. Defaults to <dataset-root>/annotation.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Override image directory. Defaults to <dataset-root>/data.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
        help="Minimum prediction score to keep a detection.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for a true positive match.",
    )
    parser.add_argument(
        "--person-category-id",
        type=int,
        default=1,
        help=(
            "Ground-truth category id that represents 'person'. "
            "Set to -1 to evaluate all ground-truth boxes regardless of label."
        ),
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=200,
        help="Optional cap for quick evaluation runs.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Show images with gt and predicted boxes whenever false positives occur.",
    )
    parser.add_argument(
        "--fail-dir",
        type=str,
        default="artifacts/failure",
        help=(
            "Directory to save images with false positives/negatives when --verbose is set. "
            "Images will be annotated with green (GT) and red (predicted) boxes."
        ),
    )
    return parser.parse_args()


def validate_inputs(annotation_root: Path, image_root: Path) -> list[Path]:
    if not annotation_root.exists():
        raise FileNotFoundError(
            f"Annotation directory not found: {annotation_root}. "
            "Pass --annotation-root or --dataset-root pointing to your HackHPI data."
        )

    if not image_root.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_root}. "
            "Pass --image-root or --dataset-root pointing to your HackHPI data."
        )

    annotation_files = sorted(annotation_root.rglob("*.json"))
    if not annotation_files:
        raise FileNotFoundError(
            f"No JSON annotation files found under: {annotation_root}"
        )

    return annotation_files


def box_iou_one_to_many(one_box: torch.Tensor, many_boxes: torch.Tensor) -> torch.Tensor:
    """Compute IoU between one box [4] and many boxes [N, 4]."""
    x1 = torch.maximum(one_box[0], many_boxes[:, 0])
    y1 = torch.maximum(one_box[1], many_boxes[:, 1])
    x2 = torch.minimum(one_box[2], many_boxes[:, 2])
    y2 = torch.minimum(one_box[3], many_boxes[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter_area = inter_w * inter_h

    one_area = (one_box[2] - one_box[0]).clamp(min=0) * (one_box[3] - one_box[1]).clamp(min=0)
    many_area = (many_boxes[:, 2] - many_boxes[:, 0]).clamp(min=0) * (
        many_boxes[:, 3] - many_boxes[:, 1]
    ).clamp(min=0)

    union = one_area + many_area - inter_area
    return inter_area / union.clamp(min=1e-6)


def image_tensor_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    return image_tensor.detach().cpu().clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()


def init_model(model_name: str, device: torch.device) -> Any:
    if model_name == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        model.eval()
        model.to(device)
        return model

    if model_name == "fasterrcnn_v2":
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        model.eval()
        model.to(device)
        return model

    if model_name == "rfdetr-large":
        return RFDETRLarge()

    if model_name == "rfdetr-small":
        return RFDETRSmall()

    if model_name == "yolo11-x":
        model = YOLO("yolo11x.pt")
        try:
            model.to(str(device))
        except Exception:
            pass
        return model

    raise ValueError(f"Unsupported model type: {model_name}")


def prediction_to_tensors(
    model: Any,
    model_name: str,
    image_tensor: torch.Tensor,
    device: torch.device,
    confidence_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if model_name in {"fasterrcnn", "fasterrcnn_v2"}:
        inputs = [image_tensor.to(device)]
        prediction = model(inputs)[0]
        return (
            prediction["boxes"].detach().cpu(),
            prediction["labels"].detach().cpu(),
            prediction["scores"].detach().cpu(),
        )

    image_np = image_tensor_to_numpy(image_tensor)

    if model_name.startswith("rfdetr"):
        detections = model.predict(image_np, threshold=confidence_threshold)

        boxes_np = getattr(detections, "xyxy", np.zeros((0, 4), dtype=np.float32))
        labels_np = getattr(detections, "class_id", np.zeros((0,), dtype=np.int64))
        scores_np = getattr(detections, "confidence", np.zeros((0,), dtype=np.float32))

        boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
        labels = torch.as_tensor(labels_np, dtype=torch.int64)
        scores = torch.as_tensor(scores_np, dtype=torch.float32)
        return boxes, labels, scores

    if model_name == "yolo11-x":
        result = model.predict(source=image_np, conf=confidence_threshold, verbose=False)[0]
        boxes = result.boxes.xyxy.detach().cpu().to(dtype=torch.float32)
        labels = result.boxes.cls.detach().cpu().to(dtype=torch.int64)
        scores = result.boxes.conf.detach().cpu().to(dtype=torch.float32)
        return boxes, labels, scores

    raise ValueError(f"Unsupported model type: {model_name}")


def predicted_person_class_id(model_name: str) -> int:
    if model_name in {"fasterrcnn", "fasterrcnn_v2"}:
        return 1
    return 0


def _box_to_tuple(box: torch.Tensor) -> tuple[float, float, float, float]:
    return tuple(float(value) for value in box.tolist())


def visualize_detections(
    image_tensor: torch.Tensor,
    gt_boxes: torch.Tensor,
    pred_boxes: torch.Tensor,
    image_title: str,
    output_path: Path,
) -> None:
    image = Image.fromarray(
        (image_tensor.detach().cpu().clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy())
    )
    draw = ImageDraw.Draw(image)

    for gt_box in gt_boxes:
        draw.rectangle(_box_to_tuple(gt_box), outline="lime", width=3)

    for pred_box in pred_boxes:
        draw.rectangle(_box_to_tuple(pred_box), outline="red", width=3)

    draw.text((8, 8), image_title, fill="yellow")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def greedy_match(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float,
) -> tuple[int, int, int, list[float]]:
    """Greedy one-to-one matching using descending IoU for each prediction."""
    if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
        return 0, 0, 0, []
    if pred_boxes.numel() == 0:
        return 0, 0, int(gt_boxes.shape[0]), []
    if gt_boxes.numel() == 0:
        return 0, int(pred_boxes.shape[0]), 0, []

    gt_used = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    tp = 0
    matched_ious: list[float] = []

    for pred_box in pred_boxes:
        ious = box_iou_one_to_many(pred_box, gt_boxes)
        ious[gt_used] = -1.0
        best_iou, best_idx = torch.max(ious, dim=0)
        if best_iou.item() >= iou_threshold and not gt_used[best_idx]:
            gt_used[best_idx] = True
            tp += 1
            matched_ious.append(float(best_iou.item()))

    fp = int(pred_boxes.shape[0] - tp)
    fn = int(gt_boxes.shape[0] - tp)
    return tp, fp, fn, matched_ious


def evaluate(
    model: Any,
    model_name: str,
    annotation_files: list[Path],
    image_root: Path,
    device: torch.device,
    confidence_threshold: float,
    iou_threshold: float,
    person_category_id: int,
    max_images: int | None,
    verbose: bool,
    fail_dir: str,
) -> dict[str, float | int]:
    fail_dir = Path(fail_dir)
    if verbose:
        fail_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_matched_ious: list[float] = []

    with torch.no_grad():
        for annotation_file in annotation_files:
            dataset = CocoDetectionDataset(annotation_file=annotation_file, image_root=image_root, is_local=False)

            for image_index, (image_tensor, target) in enumerate(dataset):
                if max_images is not None and total_images >= max_images:
                    break

                pred_boxes, pred_labels, pred_scores = prediction_to_tensors(
                    model=model,
                    model_name=model_name,
                    image_tensor=image_tensor,
                    device=device,
                    confidence_threshold=confidence_threshold,
                )

                pred_keep = (
                    (pred_labels == predicted_person_class_id(model_name))
                    & (pred_scores >= confidence_threshold)
                )
                filtered_pred_boxes = pred_boxes[pred_keep]

                gt_boxes = target["boxes"].detach().cpu()

                tp, fp, fn, matched_ious = greedy_match(
                    pred_boxes=filtered_pred_boxes,
                    gt_boxes=gt_boxes,
                    iou_threshold=iou_threshold,
                )

                total_tp += tp
                total_fp += fp
                total_fn += fn
                all_matched_ious.extend(matched_ious)

                if verbose and (fp > 0 or fn > 0):
                    image_info = dataset._images[image_index]
                    image_title = f"image_id={int(image_info['id'])} fp={fp} tp={tp} fn={fn}"
                    output_path = fail_dir / (
                        f"{Path(annotation_file).stem}_img{int(image_info['id'])}_"
                        f"idx{total_images}_fp{fp}_fn{fn}.png"
                    )
                    visualize_detections(
                        image_tensor=image_tensor,
                        gt_boxes=gt_boxes,
                        pred_boxes=filtered_pred_boxes,
                        image_title=image_title,
                        output_path=output_path,
                    )

                total_images += 1

            if max_images is not None and total_images >= max_images:
                break

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mean_iou = sum(all_matched_ious) / len(all_matched_ious) if all_matched_ious else 0.0

    return {
        "images_evaluated": total_images,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
    }

def get_all_s3_keys(prefix):
    s3 = boto3.client('s3')
    keys = []
    
    # Paginator erstellen, um das 1000-Limit zu umgehen
    paginator = s3.get_paginator('list_objects_v2')
    
    # Rekursives Durchlaufen durch Angabe des Prefix
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                keys.append(obj['Key'])
                
    return keys

def main() -> None:
    args = parse_args()

    if args.local:
        dataset_root = args.dataset_root
        annotation_root = args.annotation_root or (dataset_root / "annotation")
        image_root = args.image_root or (dataset_root / "data")

        annotation_files = validate_inputs(
            annotation_root=annotation_root,
            image_root=image_root,
        )
    else:
        annotation_root = "data/HackHPI2026_release/annotation/"
        image_root = "data/HackHPI2026_release/data/"

        annotation_files = get_all_s3_keys(annotation_root)

    device = torch.device(
        "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Annotations found: {len(annotation_files)}")
    print(f"Image root: {image_root}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"IoU threshold: {args.iou_threshold}")
    if args.person_category_id < 0:
        print("Ground-truth filtering: all categories")
    else:
        print(f"Ground-truth filtering: category_id == {args.person_category_id}")

    overall_start = time.perf_counter()

    for model_name in MODEL_TYPES:
        print(f"\nEvaluating model: {model_name}")
        model = init_model(model_name=model_name, device=device)

        model_start = time.perf_counter()

        results = evaluate(
            model=model,
            model_name=model_name,
            annotation_files=annotation_files,
            image_root=image_root,
            device=device,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
            person_category_id=args.person_category_id,
            max_images=args.max_images,
            verbose=args.verbose,
            fail_dir=args.fail_dir,
        )

        elapsed_seconds = time.perf_counter() - model_start
        images_evaluated = int(results["images_evaluated"])
        images_per_second = (
            images_evaluated / elapsed_seconds if elapsed_seconds > 0 else 0.0
        )

        print("Evaluation results")
        print(f"Images evaluated: {results['images_evaluated']}")
        print(f"TP: {results['tp']}")
        print(f"FP: {results['fp']}")
        print(f"FN: {results['fn']}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 score: {results['f1']:.4f}")
        print(f"Mean IoU (matched TPs): {results['mean_iou']:.4f}")
        print(f"Elapsed time (s): {elapsed_seconds:.2f}")
        print(f"Throughput (img/s): {images_per_second:.2f}")

    overall_elapsed_seconds = time.perf_counter() - overall_start
    print(f"\nTotal elapsed time (s): {overall_elapsed_seconds:.2f}")


if __name__ == "__main__":
    main()
