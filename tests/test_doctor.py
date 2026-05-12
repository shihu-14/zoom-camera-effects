from finger_quad_blur.doctor import CheckResult, run_doctor


def test_run_doctor_returns_failure_when_any_check_fails(monkeypatch):
    monkeypatch.setattr(
        "finger_quad_blur.doctor._check_imports",
        lambda: CheckResult("dependencies", True, "ready"),
    )
    monkeypatch.setattr(
        "finger_quad_blur.doctor._check_mediapipe_hands",
        lambda: CheckResult("hand tracker", True, "ready"),
    )
    monkeypatch.setattr(
        "finger_quad_blur.doctor._check_virtual_camera",
        lambda _width, _height, _fps: CheckResult("virtual camera", False, "missing"),
    )
    monkeypatch.setattr("finger_quad_blur.doctor.platform.system", lambda: "Linux")
    lines = []

    result = run_doctor(print_line=lines.append)

    assert result == 1
    assert lines[-1] == "fail: virtual camera: missing"


def test_run_doctor_returns_success_when_checks_pass(monkeypatch):
    monkeypatch.setattr(
        "finger_quad_blur.doctor._check_imports",
        lambda: CheckResult("dependencies", True, "ready"),
    )
    monkeypatch.setattr(
        "finger_quad_blur.doctor._check_mediapipe_hands",
        lambda: CheckResult("hand tracker", True, "ready"),
    )
    monkeypatch.setattr(
        "finger_quad_blur.doctor._check_virtual_camera",
        lambda _width, _height, _fps: CheckResult("virtual camera", True, "ready"),
    )
    monkeypatch.setattr("finger_quad_blur.doctor.platform.system", lambda: "Linux")

    assert run_doctor(print_line=lambda _line: None) == 0
