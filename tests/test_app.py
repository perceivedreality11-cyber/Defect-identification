from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_demo_renders_and_missing_weights_is_actionable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value != "0"
    assert any("演示模式" in warning.value for warning in app.warning)
    app.sidebar.radio[0].set_value("已训练模型 · YOLO").run()
    assert not app.exception
    assert any("尚未找到训练权重" in info.value for info in app.info)
