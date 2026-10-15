"""Candidate baseline and optional trained YOLO adapter."""

import csv
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["porosity", "crack", "inclusion"]
COLORS = {"porosity": "#20d6ad", "crack": "#ffb454", "inclusion": "#93b7ff"}


@dataclass(frozen=True)
class Detection:
    label: str
    xyxy: tuple[int, int, int, int]
    confidence: float | None = None


def candidates(image: Image.Image, sensitivity: int = 22, min_area: int = 18) -> list[Detection]:
    """Locate dark local anomalies. Labels are shape heuristics, NOT learned classes.

    No pseudo confidence is emitted. Dark-on-bright is an explicit assumption;
    reverse-polarity images require a separate preprocessing/validation study.
    """
    gray = np.asarray(image.convert("L"))
    background = cv2.GaussianBlur(gray, (0, 0), 11)
    residual = background.astype(np.int16) - gray.astype(np.int16)
    mask = (residual > sensitivity).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > gray.size * 0.12:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
        aspect = max(rw, rh) / max(min(rw, rh), 1)
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / max(perimeter**2, 1)
        label = "crack" if aspect > 3.5 else ("porosity" if circularity > 0.65 else "inclusion")
        found.append(Detection(label, (x, y, x + w, y + h)))
    return sorted(found, key=lambda d: (d.xyxy[1], d.xyxy[0]))


def load_model(weights: str | Path):
    path = Path(weights)
    if not path.is_file() or path.suffix.lower() != ".pt":
        raise ValueError("请指定已训练的本地 .pt 权重文件。")
    from ultralytics import YOLO

    model = YOLO(str(path))
    if model.task != "detect" or list(model.names.values()) != CLASSES:
        raise ValueError(f"模型必须是检测模型，且类别顺序为 {CLASSES}；不能使用 COCO 通用权重。")
    return model


def predict(model, image: Image.Image, confidence: float = 0.25, imgsz: int = 640) -> list[Detection]:
    if not 0 < confidence <= 1:
        raise ValueError("confidence 必须在 (0, 1] 范围内。")
    result = model.predict(image, conf=confidence, imgsz=imgsz, verbose=False)[0]
    width, height = image.size
    found = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        bounds = (max(0, int(x1)), max(0, int(y1)), min(width, int(x2)), min(height, int(y2)))
        found.append(Detection(model.names[int(box.cls.item())], bounds, float(box.conf.item())))
    return found


def annotate(image: Image.Image, detections: list[Detection], *, demo: bool = False) -> Image.Image:
    output = image.convert("RGB").copy()
    draw = ImageDraw.Draw(output)
    for item in detections:
        color = COLORS.get(item.label, "#ffffff")
        draw.rectangle(item.xyxy, outline=color, width=2)
        label = item.label if item.confidence is None else f"{item.label} {item.confidence:.2f}"
        x, y = item.xyxy[:2]
        draw.text((x + 2, max(18, y - 14)), label, fill=color, stroke_width=1, stroke_fill="#111827")
    if demo:
        draw.rectangle((0, 0, output.width, 18), fill="#332716")
        draw.text((8, 2), "DEMO | synthetic / heuristic candidates | NOT a trained model", fill="#ffcc80")
    return output


def report(detections: list[Detection], image: Image.Image, *, engine: str, source: str) -> dict:
    return {
        "schema_version": 1,
        "source": source,
        "image_size": list(image.size),
        "engine": engine,
        "coordinate_system": "xyxy pixels, origin top-left; x2/y2 exclusive",
        "status": "candidates_found" if detections else "no_candidates_found",
        "note": "No candidates does not certify absence of defects. Heuristic labels require review.",
        "detections": [asdict(item) for item in detections],
    }


def as_json(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)


def as_csv(detections: list[Detection]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(["label", "confidence", "x1", "y1", "x2", "y2"])
    for item in detections:
        writer.writerow([item.label, "" if item.confidence is None else item.confidence, *item.xyxy])
    return output.getvalue()
