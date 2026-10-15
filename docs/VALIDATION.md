# 本地验证记录

验证日期：2026-10-15。运行环境：Windows、Python 3.14.6、CPU。

| 检查 | 结果 |
|---|---|
| `pytest -q` | 17 项通过，包含 Streamlit AppTest |
| `ruff check .` | 通过 |
| 合成数据生成与检查 | 12 张，train / val / test 为 8 / 2 / 2 |
| CLI 候选检测 | PNG、JSON、CSV 成功生成 |
| CPU 训练 | YOLO11n 从配置随机初始化，1 epoch、imgsz=128、batch=2；成功保存权重 |
| 独立 test 集评估 | 成功生成 metrics.json 与评估图 |
| 已训练权重推理 | 成功加载本项目三类权重，输出报告 |
| 浏览器页面检查 | 默认演示、参数控件、图像和区域表格正常显示 |

短训练只验证代码路径，测试集 mAP 为 0，不能作为有效缺陷识别模型使用。生成的权重和合成数据不纳入 Git；真实工业数据实验尚未进行。浏览器文件选择上传的额外检查被自动权限审核超时中断，未声称该项检查完成。

实际使用的主要依赖版本：

```text
streamlit==1.66.0
ultralytics==8.4.177
torch==2.13.0+cpu
numpy==2.5.1
Pillow==12.3.0
opencv-python-headless==5.0.0.93
PyYAML==6.0.3
pytest==9.1.1
```

项目声明兼容 Python >=3.10；GitHub Actions 配置对 3.10 / 3.12 运行检查。本记录只证明上述本地环境的验证结果。
