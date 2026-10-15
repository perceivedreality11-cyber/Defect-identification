"""Create the README figure from actual baseline outputs, without a trained model."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ndt.detection import annotate, candidates
from ndt.synthetic import sample


def main():
    image, _ = sample()
    detected = candidates(image)
    overlay = annotate(image, detected, demo=True)
    canvas = Image.new("RGB", (1104, 670), "#0c1220")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=23)
    small = ImageFont.load_default(size=15)
    draw.text((28, 22), "NDT LENS / INDUSTRIAL X-RAY", fill="#20d6ad", font=font)
    draw.text(
        (28, 62),
        "Procedural phantom  |  Heuristic candidates  |  No trained model",
        fill="#9aaecb",
        font=small,
    )
    draw.text((28, 108), "01 / INPUT", fill="#e8eef8", font=small)
    draw.text((564, 108), "02 / CANDIDATE REGIONS", fill="#e8eef8", font=small)
    canvas.paste(image, (28, 138))
    canvas.paste(overlay, (564, 138))
    destination = Path(__file__).resolve().parents[1] / "docs/demo.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)
    print(destination)


if __name__ == "__main__":
    main()
