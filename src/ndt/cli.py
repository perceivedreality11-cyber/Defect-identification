"""Command-line entry points. Run `ndt --help`."""

import argparse
from pathlib import Path

import yaml

from ndt.dataset import SUFFIXES, resolved_config, validate_dataset
from ndt.detection import annotate, as_csv, as_json, candidates, load_model, predict, report
from ndt.imaging import load_image
from ndt.synthetic import make_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="工业 X 射线缺陷识别研究工具")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo-data", help="生成合成 YOLO 数据，仅用于流程验证")
    demo.add_argument("--output", type=Path, default=Path("data/synthetic"))
    demo.add_argument("--count", type=int, default=60)
    demo.add_argument("--seed", type=int, default=42)
    check = commands.add_parser("check-data", help="检查标注和重复图像")
    check.add_argument("--data", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="将单个图像目录统一转换为 8 位 PNG（不修改原图）")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    infer = commands.add_parser("predict", help="检测单张图像，保存 PNG/JSON/CSV")
    infer.add_argument("--source", type=Path, required=True)
    infer.add_argument("--weights", type=Path)
    infer.add_argument("--demo", action="store_true", help="启用形态学候选区域演示")
    infer.add_argument("--conf", type=float, default=0.25)
    infer.add_argument("--output", type=Path, default=Path("outputs/prediction"))
    for mode in ("train", "evaluate"):
        command = commands.add_parser(mode)
        command.add_argument("--data", type=Path, required=True)
        command.add_argument("--device", default="cpu")
        command.add_argument("--imgsz", type=int, default=640)
        command.add_argument("--batch", type=int, default=8)
        if mode == "train":
            command.add_argument("--model", default="yolo11n.pt")
            command.add_argument("--epochs", type=int, default=100)
            command.add_argument("--seed", type=int, default=42)
        else:
            command.add_argument("--weights", type=Path, required=True)
            command.add_argument("--split", choices=["val", "test"], default="test")
    args = parser.parse_args()
    try:
        if args.command == "demo-data":
            print(make_dataset(args.output, args.count, args.seed))
        elif args.command == "check-data":
            print(as_json(validate_dataset(args.data)))
        elif args.command == "prepare":
            images = sorted(p for p in args.source.iterdir() if p.suffix.lower() in SUFFIXES)
            if not images:
                raise ValueError("源目录没有可转换图像。")
            if args.output.exists() and any(args.output.iterdir()):
                raise ValueError("输出目录必须为空。")
            if len({p.stem for p in images}) != len(images):
                raise ValueError("文件名主干重复，转换将冲突；请先重命名。")
            args.output.mkdir(parents=True, exist_ok=True)
            for path in images:
                load_image(path).save(args.output / f"{path.stem}.png")
            print(f"已转换 {len(images)} 张图像。请在转换后的图像上标注。")
        elif args.command == "predict":
            if bool(args.weights) == bool(args.demo):
                raise ValueError("必须且只能选择 --weights 或 --demo。")
            image = load_image(args.source)
            detections = (
                candidates(image) if args.demo else predict(load_model(args.weights), image, args.conf)
            )
            args.output.mkdir(parents=True, exist_ok=True)
            annotate(image, detections, demo=args.demo).save(args.output / "annotated.png")
            payload = report(
                detections, image, engine="heuristic_demo" if args.demo else "yolo", source=args.source.name
            )
            (args.output / "detections.json").write_text(as_json(payload), encoding="utf-8")
            (args.output / "detections.csv").write_text(as_csv(detections), encoding="utf-8-sig")
            print(as_json(payload))
        else:
            counts = validate_dataset(args.data)
            from ultralytics import YOLO

            resolved = Path("outputs/configs/dataset.resolved.yaml")
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(yaml.safe_dump(resolved_config(args.data)), encoding="utf-8")
            common = dict(
                data=str(resolved.resolve()),
                device=args.device,
                imgsz=args.imgsz,
                batch=args.batch,
                workers=0,
            )
            if args.command == "train":
                model = YOLO(args.model)
                model.train(
                    **common,
                    epochs=args.epochs,
                    seed=args.seed,
                    deterministic=True,
                    project=str(Path("runs/train").resolve()),
                    name="ndt",
                    hsv_h=0,
                    hsv_s=0,
                    hsv_v=0.1,
                    fliplr=0.5,
                    flipud=0,
                    degrees=3,
                    mosaic=0,
                    mixup=0,
                )
            else:
                if args.split not in counts:
                    raise ValueError(f"数据配置未提供 {args.split} 集。")
                model = load_model(args.weights)
                metrics = model.val(
                    **common, split=args.split, project=str(Path("runs/evaluate").resolve()), name="ndt"
                )
                destination = Path(metrics.save_dir) / "metrics.json"
                payload = {
                    "split": args.split,
                    "data": str(args.data.resolve()),
                    "weights": str(args.weights.resolve()),
                    "metrics": metrics.results_dict,
                }
                destination.write_text(as_json(payload), encoding="utf-8")
                print(destination)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
