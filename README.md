# NDT Lens · 工业 X 射线缺陷识别

面向工业射线图像的缺陷检测研究项目：**合成演示 → 数据检查 → YOLO 训练 → 独立评估 → 图像检测与结果导出**。

![合成射线图像的候选区域演示](docs/demo.png)

> 当前版本提供可运行的流程和形态学基线，**不包含经过真实工业数据训练的权重，也没有真实数据准确率结论**。图中的类别是形状启发式猜测；合成样本不能证明工业检测能力。

## 功能

- Streamlit 中文工作台：原图与检测框对照、区域表格、PNG / JSON / CSV 下载。
- 无需模型即可运行的候选区域演示；演示结果不生成虚构置信度。
- 按 `porosity` 气孔、`crack` 裂纹、`inclusion` 夹杂三个初始类别训练 YOLO11。
- 检查缺失标注、非法类别、框越界和重复像素图像，减少数据泄漏。
- 独立 test 集评估，保存 Precision、Recall、mAP 等模型评估结果。
- PNG / JPEG / TIFF 输入；16 位灰度图使用固定的 1–99% 百分位窗转换为 8 位。

## 快速开始

建议 Python 3.10–3.12。首次安装需要访问 PyPI；演示运行不需要联网或 GPU。

```powershell
git clone https://github.com/perceivedreality11-cyber/Defect-identification.git
cd Defect-identification
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[ui,dev]"
streamlit run app.py
```

macOS / Linux 使用 `source .venv/bin/activate` 激活环境。Windows 若脚本执行受限，可直接运行 `.\.venv\Scripts\python.exe -m streamlit run app.py`，无需修改系统策略。

打开终端显示的本地地址，默认展示合成射线图像。演示模式按局部灰度差定位暗异常，按圆度和长宽比给出形态猜测。阈值适用于此演示，实际图像需单独标定；亮缺陷、散射、焊缝边缘与复杂纹理可能造成漏检或误报。

## 命令行演示

```powershell
ndt demo-data --count 60 --output data/synthetic
ndt check-data --data data/synthetic/dataset.yaml
ndt predict --demo --source data/synthetic/images/train/synthetic_00028.png --output outputs/demo
```

输出 `annotated.png`、`detections.json`、`detections.csv`。框坐标为原图像素的 `x1, y1, x2, y2`，左上角为原点，右下边界不包含在框内。无候选区域的状态为 `no_candidates_found`，不代表工件合格。

## 接入真实数据

自行取得具有使用授权的工业 X 射线数据，并由熟悉工艺的人员复核标注。项目不打包或自动下载第三方工业数据。

```text
data/xray/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

每张图对应同名 `.txt`，每行一个框：

```text
# class_id x_center y_center width height（实际 txt 中不要包含注释）
0 0.500000 0.400000 0.050000 0.060000
```

类别从 0 开始，坐标除以图像宽高归一化到 0–1。正常图像保留空标注文件。`configs/dataset.example.yaml` 已设置上述路径和类别。

**按工件、拍摄批次或原始底片分组划分 train / val / test，再切片或增强**，避免同工件的近似图跨集合。检查工具能发现相同像素，不能发现所有相似片段；分组需要保留采集元数据并人工核查。test 集只用于最终评估，不用于调阈值或选择权重。

对高位深输入，先统一转换，再在转换结果上标注：

```powershell
ndt prepare --source data/raw_tiff --output data/prepared
ndt check-data --data configs/dataset.example.yaml
```

`prepare` 转换一个平面目录，不递归，不拷贝标注，拒绝覆盖非空目录；输出仍保持原始宽高。推理使用同一灰度窗。原始图像应单独保留，百分位转换会丢失绝对灰度信息。当前不支持 DICOM 或多帧序列分析（TIFF 只读取首帧）。

## 训练和评估

```powershell
python -m pip install -e ".[train]"
# CPU 可运行；有可用 CUDA 环境时将 cpu 改为 0。
ndt train --data configs/dataset.example.yaml --epochs 100 --device cpu --batch 8
ndt evaluate --data configs/dataset.example.yaml --weights runs/train/ndt/weights/best.pt --split test
ndt predict --source path/to/xray.png --weights runs/train/ndt/weights/best.pt
```

首次训练默认下载通用 `yolo11n.pt` 初始化权重，需要联网。这些权重必须经过缺陷数据训练后才能用于本项目检测。推理入口检查本地文件、检测任务和类别顺序，拒绝直接使用 COCO 类别权重。仅加载可信来源的 `.pt` 文件。

训练结果位于 `runs/train/ndt*`，重复训练可能自动递增目录名，请以训练日志中的实际路径为准。评估的 `metrics.json`、混淆矩阵和曲线位于 `runs/evaluate/ndt*`。合成数据可用于短训练检查：

```powershell
ndt train --data data/synthetic/dataset.yaml --epochs 1 --imgsz 128 --batch 2 --device cpu --model yolo11n.yaml
```

这只验证训练代码，不产生可用于实际检测的模型。正式实验需记录数据版本、分组策略、随机种子、依赖版本、参数、每类 Precision / Recall、mAP50、mAP50–95，并在独立真实数据上复核漏检。当前不提供虚构的结果表。

## 开发与验证

```powershell
python -m pip install -e ".[ui,dev]"
ruff check .
pytest -q
```

测试覆盖合成图确定性、候选定位、正常图、16 位图像转换、标注与泄漏检查、导出、预测适配器以及界面错误状态。GitHub Actions 在 Python 3.10 / 3.12 上执行检查；不在 CI 中下载或训练模型。

```text
app.py                 中文检测工作台
src/ndt/               图像处理、候选检测、YOLO 适配、数据检查与命令行
configs/               真实数据配置模板
tests/                 单元测试与界面测试
docs/                  演示图与实验记录
```

数据、权重、运行结果、虚拟环境与密钥文件默认排除出 Git。分享模型或数据时单独确认其授权，并记录数据来源。

## 许可与参考

代码采用 **AGPL-3.0-only**；Ultralytics 代码和模型受其自身许可约束，详情见 [Ultralytics 许可说明](https://www.ultralytics.com/license)。

- [Ultralytics Python API](https://docs.ultralytics.com/usage/python/)
- [YOLO 检测标注格式](https://docs.ultralytics.com/datasets/detect/)
- [模型验证指标](https://docs.ultralytics.com/modes/val/)
- [Streamlit 文档](https://docs.streamlit.io/)

本项目用于研究和辅助分析。模型输出应由合格人员结合原始图像和工艺要求复核。
