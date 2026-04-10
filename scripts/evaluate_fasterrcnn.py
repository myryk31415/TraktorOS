#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)

from coco_dataset import CocoDetectionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pretrained Faster R-CNN (COCO) on COCO-style annotations."
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
        default=20,
        help="Optional cap for quick evaluation runs.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
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


def filter_ground_truth(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    person_category_id: int,
) -> torch.Tensor:
    if person_category_id < 0:
        return gt_boxes
    keep_mask = gt_labels == person_category_id
    return gt_boxes[keep_mask]


def evaluate(
    model: torch.nn.Module,
    annotation_files: list[Path],
    image_root: Path,
    device: torch.device,
    confidence_threshold: float,
    iou_threshold: float,
    person_category_id: int,
    max_images: int | None,
) -> dict[str, float | int]:
    total_images = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_matched_ious: list[float] = []

    with torch.no_grad():
        for annotation_file in annotation_files:
            dataset = CocoDetectionDataset(annotation_file=annotation_file, image_root=image_root)

            for image_tensor, target in dataset:
                if max_images is not None and total_images >= max_images:
                    break

                inputs = [image_tensor.to(device)]
                prediction = model(inputs)[0]

                pred_labels = prediction["labels"].detach().cpu()
                pred_scores = prediction["scores"].detach().cpu()
                pred_boxes = prediction["boxes"].detach().cpu()

                pred_keep = (pred_labels == 1) & (pred_scores >= confidence_threshold)
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


def main() -> None:
    args = parse_args()

    dataset_root = args.dataset_root
    annotation_root = args.annotation_root or (dataset_root / "annotation")
    image_root = args.image_root or (dataset_root / "data")

    annotation_files = validate_inputs(
        annotation_root=annotation_root,
        image_root=image_root,
    )

    device = torch.device(
        "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.eval()
    model.to(device)

    print("Evaluating pretrained Faster R-CNN")
    print(f"Device: {device}")
    print(f"Annotations found: {len(annotation_files)}")
    print(f"Image root: {image_root}")
    print(f"Confidence threshold: {args.confidence_threshold}")
    print(f"IoU threshold: {args.iou_threshold}")
    if args.person_category_id < 0:
        print("Ground-truth filtering: all categories")
    else:
        print(f"Ground-truth filtering: category_id == {args.person_category_id}")

    results = evaluate(
        model=model,
        annotation_files=annotation_files,
        image_root=image_root,
        device=device,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        person_category_id=args.person_category_id,
        max_images=args.max_images,
    )

    print("\nEvaluation results")
    print(f"Images evaluated: {results['images_evaluated']}")
    print(f"TP: {results['tp']}")
    print(f"FP: {results['fp']}")
    print(f"FN: {results['fn']}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 score: {results['f1']:.4f}")
    print(f"Mean IoU (matched TPs): {results['mean_iou']:.4f}")


if __name__ == "__main__":
    main()
