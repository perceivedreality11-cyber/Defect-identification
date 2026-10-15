import io
import json
import shutil

import numpy as np
import pytest
from PIL import Image

from ndt.dataset import resolved_config, validate_dataset
from ndt.detection import Detection, annotate, as_csv, as_json, candidates, load_model, report
from ndt.imaging import load_image
from ndt.synthetic import make_dataset, sample


def test_phantom_reproducible_and_baseline_localizes():
    image, expected = sample()
    assert image.tobytes() == sample()[0].tobytes()
    detected = candidates(image)
    assert len(expected) == 3
    for box in expected:
        x1, y1, x2, y2 = box.xyxy
        assert any(
            d.xyxy[0] <= (x1 + x2) / 2 <= d.xyxy[2] and d.xyxy[1] <= (y1 + y2) / 2 <= d.xyxy[3]
            for d in detected
        )
    assert all(d.confidence is None for d in detected)


def test_flat_image_has_no_candidates():
    assert candidates(Image.new("RGB", (256, 256), "gray")) == []


def test_16_bit_window_and_constant_image():
    for array in [
        np.arange(65536, dtype=np.uint16).reshape(256, 256),
        np.ones((32, 32), dtype=np.uint16) * 1000,
    ]:
        buffer = io.BytesIO()
        Image.fromarray(array).save(buffer, format="TIFF")
        buffer.seek(0)
        image = load_image(buffer)
        assert image.mode == "RGB"
        assert image.size == (array.shape[1], array.shape[0])
        assert np.asarray(image).min() == 0


def test_dataset_valid_and_negatives(tmp_path):
    config = make_dataset(tmp_path / "data", count=30)
    stats = validate_dataset(config)
    assert sum(s["images"] for s in stats.values()) == 30
    assert sum(s["negative_images"] for s in stats.values()) == 6
    assert set(stats) == {"train", "val", "test"}
    with pytest.raises(ValueError, match="非空"):
        make_dataset(config.parent)


def test_relative_dataset_root_uses_yaml_location(tmp_path):
    import yaml

    original = make_dataset(tmp_path / "data", count=12)
    config = yaml.safe_load(original.read_text())
    config["path"] = "data"
    relocated = tmp_path / "relative.yaml"
    relocated.write_text(yaml.safe_dump(config))
    assert resolved_config(relocated)["path"] == str((tmp_path / "data").resolve())
    assert validate_dataset(relocated)["train"]["images"] == 8


def test_duplicate_pixels_across_splits_are_rejected(tmp_path):
    config = make_dataset(tmp_path / "data", count=12)
    source = next((config.parent / "images/train").glob("*.png"))
    destination = config.parent / "images/test/leaked.png"
    shutil.copyfile(source, destination)
    (config.parent / "labels/test/leaked.txt").write_text("")
    with pytest.raises(ValueError, match="数据泄漏"):
        validate_dataset(config)


@pytest.mark.parametrize(
    "row", ["0 0.99 0.5 0.2 0.1", "3 0.5 0.5 0.1 0.1", "0 nan 0.5 0.1 0.1", "0 0.5 0.5 -0.1 0.1", "0 0.5"]
)
def test_bad_labels_rejected(tmp_path, row):
    config = make_dataset(tmp_path / "data", count=12)
    label = next((config.parent / "labels/train").glob("*.txt"))
    label.write_text(row)
    with pytest.raises(ValueError):
        validate_dataset(config)


def test_missing_labels_rejected(tmp_path):
    config = make_dataset(tmp_path / "data", count=12)
    next((config.parent / "labels/train").glob("*.txt")).unlink()
    with pytest.raises(ValueError, match="缺少标注"):
        validate_dataset(config)


def test_report_roundtrip_and_overlay_does_not_mutate_input():
    image, detections = sample()
    original = image.tobytes()
    annotated = annotate(image, detections, demo=True)
    assert image.tobytes() == original
    assert annotated.tobytes() != original
    payload = json.loads(as_json(report(detections, image, engine="heuristic_demo", source="test.png")))
    assert payload["detections"][0]["confidence"] is None
    assert payload["image_size"] == [512, 512]
    assert len(as_csv(detections).splitlines()) == 4


def test_no_findings_is_not_a_pass():
    payload = report([], Image.new("RGB", (10, 10)), engine="heuristic_demo", source="flat.png")
    assert payload["status"] == "no_candidates_found"
    assert "certify" in payload["note"]


def test_no_silent_model_download(tmp_path):
    with pytest.raises(ValueError, match="本地"):
        load_model(tmp_path / "does-not-exist.pt")


def test_yolo_adapter_boxes_and_confidence():
    # Contract test using tensor-like objects; real model integration is a separate smoke run.
    from ndt.detection import predict

    class Tensor:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self

        def tolist(self):
            return self.value

        def item(self):
            return self.value

    class Box:
        xyxy = [Tensor([-1, 2, 200, 90])]
        cls = Tensor(0)
        conf = Tensor(0.8)

    class Result:
        boxes = [Box()]

    class Model:
        names = {0: "porosity"}

        def predict(self, image, **kwargs):
            return [Result()]

    assert predict(Model(), Image.new("RGB", (100, 100))) == [Detection("porosity", (0, 2, 100, 90), 0.8)]
