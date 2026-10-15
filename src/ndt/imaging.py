"""Shared display transform, also used when preparing training images."""

from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps


def load_image(source: str | Path | BinaryIO) -> Image.Image:
    """Read PNG/JPEG/TIFF; percentile-window high-bit-depth grayscale for display.

    This intentionally discards absolute detector intensity. Keep raw source files
    for scientific processing and apply the same transform to training and inference.
    """
    with Image.open(source) as original:
        original = ImageOps.exif_transpose(original)
        if original.width * original.height > 25_000_000:
            raise ValueError("图像超过 2500 万像素，请先裁剪为检测区域。")
        if original.mode in ("I", "F") or original.mode.startswith("I;16"):
            a = np.asarray(original, dtype=np.float32)
            if not np.isfinite(a).all():
                raise ValueError("图像包含非有限灰度值。")
            lo, hi = np.percentile(a, (1, 99))
            a = np.zeros_like(a) if hi <= lo else np.clip((a - lo) / (hi - lo), 0, 1) * 255
            return Image.fromarray(a.astype(np.uint8)).convert("RGB")
        return original.convert("RGB")
