from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ImageFile

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from imgutils.detect import detect_faces

from app.config import FACE_DIR

MAX_SIZE = 2048

LEFT_EXPAND_RATIO = 0.2
RIGHT_EXPAND_RATIO = 0.2
TOP_EXPAND_RATIO = 0.6
BOTTOM_EXPAND_RATIO = 0.1


@dataclass
class FaceCropResult:
    crop_path: str
    raw_bbox: Tuple[int, int, int, int]          # 原图坐标
    expanded_bbox: Tuple[int, int, int, int]     # 原图坐标
    score: float
    width: int
    height: int


def resize_if_needed(img: Image.Image, max_size: int = MAX_SIZE) -> tuple[Image.Image, float]:
    w, h = img.size
    if max(w, h) <= max_size:
        return img, 1.0

    ratio = max_size / max(w, h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return resized, ratio


def expand_face_box(
    box: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    box_w = x1 - x0
    box_h = y1 - y0

    left_expand = box_w * LEFT_EXPAND_RATIO
    right_expand = box_w * RIGHT_EXPAND_RATIO
    top_expand = box_h * TOP_EXPAND_RATIO
    bottom_expand = box_h * BOTTOM_EXPAND_RATIO

    ex0 = max(0, int(x0 - left_expand))
    ey0 = max(0, int(y0 - top_expand))
    ex1 = min(image_width, int(x1 + right_expand))
    ey1 = min(image_height, int(y1 + bottom_expand))

    return ex0, ey0, ex1, ey1


def scale_box_back(
    box: Tuple[int, int, int, int],
    ratio: float,
    orig_w: int,
    orig_h: int,
) -> Tuple[int, int, int, int]:
    if ratio == 1.0:
        return box

    x0, y0, x1, y1 = box
    ox0 = max(0, min(orig_w, int(round(x0 / ratio))))
    oy0 = max(0, min(orig_h, int(round(y0 / ratio))))
    ox1 = max(0, min(orig_w, int(round(x1 / ratio))))
    oy1 = max(0, min(orig_h, int(round(y1 / ratio))))
    return ox0, oy0, ox1, oy1


def detect_and_crop_faces(image_path: str, image_id: int) -> List[FaceCropResult]:
    os.makedirs(FACE_DIR, exist_ok=True)

    origin_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = origin_img.size

    resized_img, ratio = resize_if_needed(origin_img, MAX_SIZE)

    detections = detect_faces(resized_img)
    print(f"[DEBUG] face detections for image {image_id}: {detections}, resize_ratio={ratio}")

    results: List[FaceCropResult] = []

    for idx, (box, label, score) in enumerate(detections):
        x0, y0, x1, y1 = [int(v) for v in box]

        # 先在 resized 图上算黄框
        expanded_box_resized = expand_face_box(
            (x0, y0, x1, y1),
            resized_img.width,
            resized_img.height,
        )

        # 再映射回原图坐标
        raw_box_original = scale_box_back((x0, y0, x1, y1), ratio, orig_w, orig_h)
        expanded_box_original = scale_box_back(expanded_box_resized, ratio, orig_w, orig_h)

        # crop 一律从原图裁，保证 embedding 也是原图质量
        crop = origin_img.crop(expanded_box_original)

        crop_filename = f"image_{image_id}_face_{idx}.png"
        crop_path = str(Path(FACE_DIR) / crop_filename)
        crop.save(crop_path)

        results.append(
            FaceCropResult(
                crop_path=crop_path,
                raw_bbox=raw_box_original,
                expanded_bbox=expanded_box_original,
                score=float(score),
                width=crop.width,
                height=crop.height,
            )
        )

    return results