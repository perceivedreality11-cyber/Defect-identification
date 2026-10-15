"""Run with: streamlit run app.py"""

import io
from pathlib import Path

import streamlit as st
from PIL import UnidentifiedImageError

from ndt.detection import annotate, as_csv, as_json, candidates, load_model, predict, report
from ndt.imaging import load_image
from ndt.synthetic import sample

st.set_page_config(page_title="NDT Lens · X 射线缺陷识别", page_icon="🔬", layout="wide")
st.markdown(
    """<style>
.stApp {background: #0c1220; color: #e8eef8;}
.block-container {padding-top: 2.5rem; max-width: 1450px;}
h1 {letter-spacing: -.04em;} [data-testid="stMetric"] {background: #151e30; border: 1px solid #27344e; padding: 18px; border-radius: 12px;}
</style>""",
    unsafe_allow_html=True,
)
st.caption("NDT LENS  /  INDUSTRIAL RADIOGRAPHY  /  v0.1")
st.title("让缺陷，有迹可循。")
st.write("工业 X 射线图像的候选区域定位与模型检测工作台")


@st.cache_resource
def cached_model(path: str, modified: int, size: int):
    return load_model(path)


with st.sidebar:
    st.subheader("检测设置")
    mode = st.radio("运行方式", ["演示 · 候选区域", "已训练模型 · YOLO"])
    demo = mode.startswith("演示")
    if demo:
        st.info("按局部灰度和形状寻找候选区域。类别为形态学猜测，不提供置信度。")
        sensitivity = st.slider("灰度差阈值", 5, 60, 22)
        min_area = st.slider("最小候选面积（像素）", 3, 200, 18)
        weights = ""
        confidence = 0.25
    else:
        weights = st.text_input("本地权重路径", "runs/train/ndt/weights/best.pt")
        confidence = st.slider("置信度阈值", 0.05, 0.95, 0.25)
        st.caption("仅加载可信来源、按本项目类别训练的权重。")
    st.divider()
    st.caption("初始类别：气孔 porosity · 裂纹 crack · 夹杂 inclusion")
    st.caption("输入：PNG / JPEG / TIFF。16 位 TIFF 使用 1–99% 灰度窗转换。")

uploaded = st.file_uploader("上传检测图像", type=["png", "jpg", "jpeg", "tif", "tiff"])
use_sample = st.checkbox("使用内置合成示例", value=uploaded is None, disabled=uploaded is not None)
if uploaded is None and not use_sample:
    st.info("上传一张图像，或打开内置合成示例。")
    st.stop()
try:
    image = load_image(uploaded) if uploaded else sample()[0]
except (ValueError, OSError, UnidentifiedImageError) as exc:
    st.error(f"无法读取图像：{exc}")
    st.stop()
source = uploaded.name if uploaded else "synthetic-demo.png"
if demo:
    st.warning("演示模式：当前结果仅为候选区域；合成图像不代表真实工业数据性能。")
elif not uploaded:
    st.info("当前输入为合成图像；模型在此输入的结果不能用于报告真实数据性能。")

try:
    with st.spinner("正在分析图像…"):
        if demo:
            detections = candidates(image, sensitivity, min_area)
        else:
            path = Path(weights)
            if not path.is_file():
                st.info("尚未找到训练权重。请按 README 训练模型，或切换到演示模式。")
                st.stop()
            info = path.stat()
            model = cached_model(str(path.resolve()), info.st_mtime_ns, info.st_size)
            detections = predict(model, image, confidence)
except (ValueError, OSError, ImportError, RuntimeError) as exc:
    st.error(f"检测未完成：{exc}")
    st.stop()

left, middle, right = st.columns(3)
left.metric("候选区域" if demo else "检测框", len(detections))
middle.metric("图像尺寸", f"{image.width} × {image.height}")
right.metric("检测引擎", "形态学演示" if demo else "YOLO")
before, after = st.columns(2)
with before:
    st.subheader("01 / 原始图像")
    st.image(image, width="stretch")
with after:
    st.subheader("02 / 定位结果")
    annotated = annotate(image, detections, demo=demo)
    st.image(annotated, width="stretch")
st.subheader("区域明细")
if detections:
    st.dataframe(
        [
            {
                "类别": d.label,
                "置信度": d.confidence,
                "x1": d.xyxy[0],
                "y1": d.xyxy[1],
                "x2": d.xyxy[2],
                "y2": d.xyxy[3],
            }
            for d in detections
        ],
        hide_index=True,
        width="stretch",
    )
else:
    st.info("当前参数下未发现候选区域；这不等于图像无缺陷。")
payload = report(detections, image, engine="heuristic_demo" if demo else "yolo", source=source)
png = io.BytesIO()
annotated.save(png, format="PNG")
a, b, c = st.columns(3)
a.download_button("下载标注图像", png.getvalue(), "annotated.png", "image/png")
b.download_button("下载 JSON 报告", as_json(payload), "detections.json", "application/json")
c.download_button("下载 CSV 明细", as_csv(detections).encode("utf-8-sig"), "detections.csv", "text/csv")
st.caption("研究与辅助分析工具。模型输出需要人工复核，不能作为工件合格判定的唯一依据。")
