"""Procedural grayscale phantoms, strictly for exercising software pipelines."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ndt.detection import CLASSES, Detection


def sample(seed: int = 42, size: int = 512, clean: bool = False) -> tuple[Image.Image, list[Detection]]:
    if size < 128:
        raise ValueError("size 至少为 128。")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    center = size * rng.uniform(0.43, 0.57)
    field = 110 + 75 * np.exp(-(((xx - center) / (size * 0.23)) ** 2))
    field += 8 * np.sin(yy / size * 3) + rng.normal(0, 2, (size, size))
    image = Image.fromarray(np.clip(field, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    found = []
    if not clean:
        # Separated vertical regions prevent accidental overlap of synthetic labels.
        for index, label in enumerate(CLASSES):
            x = int(size * rng.uniform(0.36, 0.58))
            y = int(size * (0.17 + index * 0.27) + rng.uniform(-0.03, 0.03) * size)
            dark = int(rng.integers(40, 75))
            if label == "porosity":
                r = max(4, int(size * rng.uniform(0.015, 0.027)))
                bounds = (x - r, y - r, x + r + 1, y + r + 1)
                draw.ellipse((x - r, y - r, x + r, y + r), fill=dark)
            elif label == "crack":
                length = int(size * rng.uniform(0.09, 0.16))
                thickness = max(2, size // 180)
                points = [(x, y), (x + length // 3, y + 4), (x + length * 2 // 3, y - 3), (x + length, y)]
                draw.line(points, fill=dark, width=thickness)
                bounds = (x - thickness, y - 3 - thickness, x + length + thickness + 1, y + 4 + thickness + 1)
            else:
                w, h = int(size * 0.055), int(size * 0.035)
                draw.polygon(
                    [(x, y), (x + w, y + h // 3), (x + w * 2 // 3, y + h), (x - 3, y + h * 2 // 3)], fill=dark
                )
                bounds = (x - 3, y, x + w + 1, y + h + 1)
            found.append(Detection(label, bounds))
    return image.filter(ImageFilter.GaussianBlur(0.5)).convert("RGB"), found


def make_dataset(output: Path, count: int = 60, seed: int = 42, size: int = 512) -> Path:
    import yaml

    if count < 9:
        raise ValueError("至少生成 9 张图像，保证三个划分都非空。")
    if output.exists() and any(output.iterdir()):
        raise ValueError("目标目录非空，请选择新目录，避免混合或覆盖数据。")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    train_end, val_end = int(count * 0.7), int(count * 0.85)
    manifest = {"synthetic": True, "seed": seed, "size": size, "samples": []}
    for rank, index in enumerate(order):
        split = "train" if rank < train_end else ("val" if rank < val_end else "test")
        image_dir, label_dir = output / "images" / split, output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        image, boxes = sample(seed + int(index), size, clean=int(index) % 5 == 0)
        name = f"synthetic_{int(index):05d}"
        image.save(image_dir / f"{name}.png")
        lines = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy
            lines.append(
                f"{CLASSES.index(box.label)} {(x1 + x2) / 2 / size:.7f} {(y1 + y2) / 2 / size:.7f} {(x2 - x1) / size:.7f} {(y2 - y1) / size:.7f}"
            )
        (label_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        manifest["samples"].append(
            {"id": name, "split": split, "seed": seed + int(index), "negative": not boxes}
        )
    config = {
        "path": str(output.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": CLASSES,
    }
    destination = output / "dataset.yaml"
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return destination
