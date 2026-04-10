from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.tv_tensors import BoundingBoxes


class CocoDetectionDataset(torch.utils.data.Dataset):
    """Minimal COCO-style detection dataset.

    This loader keeps preprocessing intentionally small: it reads RGB images,
    converts them to float tensors in the range [0, 1], and returns COCO boxes
    in [x1, y1, x2, y2] format.
    """

    def __init__(self, annotation_file: str | Path, image_root: str | Path):
        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root)

        with self.annotation_file.open("r", encoding="utf-8") as file_handle:
            data = json.load(file_handle)

        self._images = sorted(data.get("images", []), key=lambda item: item["id"])
        self._annotations_by_image_id: dict[int, list[dict[str, Any]]] = {}

        for annotation in data.get("annotations", []):
            bbox = annotation.get("bbox", [])
            if len(bbox) != 4:
                continue

            x, y, width, height = bbox
            if width <= 0 or height <= 0:
                continue

            image_id = int(annotation["image_id"])
            self._annotations_by_image_id.setdefault(image_id, []).append(annotation)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        image_info = self._images[index]
        image_path = self._resolve_image_path(image_info)

        image = Image.open(image_path).convert("RGB")
        image_tensor = self._to_tensor(image)

        annotations = self._annotations_by_image_id.get(int(image_info["id"]), [])
        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        iscrowd_values: list[int] = []

        for annotation in annotations:
            x, y, width, height = annotation["bbox"]
            x1 = float(x)
            y1 = float(y)
            x2 = float(x + width)
            y2 = float(y + height)
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(int(annotation.get("category_id", 0)))
            areas.append(float(annotation.get("area", width * height)))
            iscrowd_values.append(int(annotation.get("iscrowd", 0)))

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            area_tensor = torch.tensor(areas, dtype=torch.float32)
            iscrowd_tensor = torch.tensor(iscrowd_values, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": BoundingBoxes(
                boxes_tensor,
                format="XYXY",
                canvas_size=(image.height, image.width),
            ),
            "labels": labels_tensor,
            "image_id": torch.tensor(int(image_info["id"]), dtype=torch.int64),
            "area": area_tensor,
            "iscrowd": iscrowd_tensor,
        }

        return image_tensor, target

    def _resolve_image_path(self, image_info: dict[str, Any]) -> Path:
        file_name = Path(str(image_info["file_name"]))

        if file_name.is_absolute() and file_name.exists():
            return file_name

        direct_candidate = self.image_root / file_name
        if direct_candidate.exists():
            return direct_candidate

        basename = file_name.name
        for candidate in self.image_root.rglob(basename):
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(f"Could not find image for COCO file_name: {file_name}")

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        buffer = torch.ByteTensor(torch.ByteStorage.from_buffer(image.tobytes()))
        tensor = buffer.view(image.height, image.width, 3).permute(2, 0, 1).contiguous()
        return tensor.float().div_(255.0)