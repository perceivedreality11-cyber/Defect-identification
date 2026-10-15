"""Strict checks before spending time on model training."""

import hashlib
from pathlib import Path

import yaml
from PIL import Image

from ndt.detection import CLASSES

SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def resolved_config(config_path: Path) -> dict:
    """Avoid framework-dependent handling of relative dataset roots."""
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("dataset.yaml 必须为映射。")
    root = Path(config.get("path", "."))
    config["path"] = str((root if root.is_absolute() else config_path.parent / root).resolve())
    return config


def validate_dataset(config_path: Path) -> dict:
    config_path = config_path.resolve()
    config = resolved_config(config_path)
    names = config.get("names")
    if isinstance(names, dict):
        names = [names.get(i) for i in range(len(names))]
    if names != CLASSES:
        raise ValueError(f"类别及顺序必须为 {CLASSES}。")
    root = Path(config["path"])
    seen: dict[str, str] = {}
    counts = {}
    for split in ("train", "val", "test"):
        if split == "test" and split not in config:
            continue
        folder = config.get(split)
        if not isinstance(folder, str):
            raise ValueError(f"{split} 必须是 images/{split} 目录路径。")
        image_dir = (root / folder).resolve()
        if image_dir.name != split or image_dir.parent.name != "images":
            raise ValueError(f"请使用 images/{split} 与 labels/{split} 的标准目录结构。")
        images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in SUFFIXES)
        if not images:
            raise ValueError(f"{split} 没有图像：{image_dir}")
        objects, negatives = 0, 0
        label_paths = set()
        for path in images:
            with Image.open(path) as image:
                image.load()
                if image.mode in ("I", "F") or image.mode.startswith("I;16"):
                    raise ValueError(f"{path}: 请先使用 ndt prepare 转为统一的 8 位训练图像。")
                pixels = image.convert("RGB")
                digest = hashlib.sha256(str(pixels.size).encode() + pixels.tobytes()).hexdigest()
            if digest in seen:
                raise ValueError(f"重复图像或数据泄漏：{path} 与 {seen[digest]}")
            seen[digest] = str(path)
            label = (
                image_dir.parent.parent / "labels" / split / path.relative_to(image_dir).with_suffix(".txt")
            )
            if label in label_paths:
                raise ValueError(f"多个图像共用同一标注：{label}")
            label_paths.add(label)
            if not label.is_file():
                raise ValueError(f"缺少标注：{label}；正常样本也需要空 txt 文件。")
            rows = label.read_text(encoding="utf-8").strip().splitlines()
            negatives += not rows
            for line_number, row in enumerate(rows, 1):
                try:
                    fields = row.split()
                    if len(fields) != 5:
                        raise ValueError("需要五列")
                    class_id = int(fields[0])
                    cx, cy, w, h = map(float, fields[1:])
                    if class_id not in range(len(CLASSES)) or not (0 < w <= 1 and 0 < h <= 1):
                        raise ValueError("类别或框尺寸无效")
                    if not (
                        w / 2 - 1e-6 <= cx <= 1 - w / 2 + 1e-6 and h / 2 - 1e-6 <= cy <= 1 - h / 2 + 1e-6
                    ):
                        raise ValueError("标注框越界或包含非有限数值")
                except ValueError as exc:
                    raise ValueError(f"{label}:{line_number}: {exc}") from exc
                objects += 1
        counts[split] = {"images": len(images), "objects": objects, "negative_images": negatives}
        if split == "train" and objects == 0:
            raise ValueError("训练集没有任何缺陷标注。")
    return counts
