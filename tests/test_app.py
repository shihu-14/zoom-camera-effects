from types import SimpleNamespace

import numpy as np
import pytest

from zoom_camera_effects import app
from zoom_camera_effects.effects import EffectConfig


@pytest.mark.parametrize("command", ["edge", "invalid", None])
def test_loop_updates_config_then_sends_frame_before_overlay(
    monkeypatch, tmp_path, capsys, command
):
    events = []
    ui_bases = []
    sent = []
    displayed = []
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    processor = SimpleNamespace(effect_config=EffectConfig())

    def set_config(config):
        processor.effect_config = config

    def process(image):
        events.append("process")
        assert processor.effect_config.mode == "thermal"
        return SimpleNamespace(frame_bgr=image)

    def read_config(base):
        events.append("read")
        if command == "invalid":
            raise ValueError("bad command")
        return EffectConfig(mode=command)

    def consume_config(base):
        events.append("consume")
        ui_bases.append(base.mode)
        return EffectConfig(mode="thermal")

    def send(image):
        events.append("send")
        sent.append(image.copy())

    def render(image, config):
        events.append("render")
        return np.full_like(image, 255)

    processor.set_effect_config = set_config
    processor.process = process
    reader = SimpleNamespace(
        reset_current=lambda: events.append("reset"),
        read_config=read_config,
    )
    overlay = SimpleNamespace(
        handle_mouse=lambda *args: None,
        consume_pending_config=consume_config,
        render=render,
    )
    monkeypatch.setattr(app, "EffectControlReader", lambda path: reader)
    monkeypatch.setattr(app, "OverlayControlUI", lambda: overlay)
    for name in ("namedWindow", "moveWindow", "setMouseCallback"):
        monkeypatch.setattr(app.cv2, name, lambda *args: None)
    monkeypatch.setattr(
        app.cv2, "imshow", lambda name, image: displayed.append(image.copy())
    )
    monkeypatch.setattr(app.cv2, "waitKey", lambda delay: -1)
    monkeypatch.setattr(app.cv2, "getWindowProperty", lambda *args: 1)
    config = app.AppConfig(
        control_file=tmp_path / "control.txt" if command is not None else None,
        mirror=False,
        max_frames=1,
    )
    capture = SimpleNamespace(read=lambda: (True, frame.copy()))
    writer = SimpleNamespace(send_bgr=send)
    assert app._loop(capture, processor, writer, config) == 0
    assert ui_bases == ["edge" if command == "edge" else "blur"]
    prefix = ["reset", "read"] if command is not None else []
    assert events == prefix + ["consume", "process", "send", "render"]
    assert np.array_equal(sent[0], frame)
    assert np.all(displayed[0] == 255)
    assert ("bad command" in capsys.readouterr().out) == (command == "invalid")
