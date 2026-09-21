# Zoom Camera Effects Refactoring Plan

> **For agentic workers:** 実装を依頼された後に `superpowers:executing-plans` を使い、チェック項目ごとに進める。この文書の作成・コミット・push は実装開始の承認を意味しない。

**Goal:** カメラ映像・操作方法・外部インターフェースを維持しながら、CLI、UI、実行ループの責務を小さく分け、変更箇所と検証範囲を明確にする。

**Architecture:** 既存の `cli → app → processor → effects` の流れを維持する。CLI の設定生成と実行ループの設定更新は既存ファイル内の非公開関数へ抽出する。UI だけは、独立している描画補助と設定値操作を内部モジュールへ移す。

**Tech Stack:** Python 3.10+、MediaPipe、NumPy、OpenCV、pyvirtualcam、pytest。依存の追加・更新は行わない。

**Spec:** この文書の「依頼内容と設計条件」。今回は計画のみを作成する依頼のため、設計案と実施手順を一つの文書にまとめる。

**状態:** 提案。以下の実装タスクはすべて未着手。

## 依頼内容と設計条件

- 専用の通常 Git ブランチ `refactor/plan-camera-effects-cleanup` で作業する。worktree は作成しない。
- 今回の成果物はリファクタリング計画。製品コード、テスト、依存設定は変更しない。
- Codex の貢献をコミットの共同作者として記録する。
- 対象箇所の指定がないため、保守性の改善を目的とし、動作を変えない最小限の整理を提案する。
- 実装開始はユーザーからの次の依頼を待つ。計画中のコード例は実装案であり、適用済みコードではない。

## Global Constraints

- Python の対応下限 `>=3.10` と `pyproject.toml` の依存範囲を維持する。
- `zoom-camera-effects`、`python3 -m zoom_camera_effects`、macOS ランチャーの起動方法を維持する。
- `AppConfig`、`EffectConfig`、`OverlayControlUI`、`FrameProcessor` など既存の非公開でない名前・シグネチャ・import パスを変更しない。
- 13種類の effect、3種類の scope、既定値、CLI オプション、ヘルプ文、終了コードを維持する。
- `monochrome`、既存 CLI 別名、制御ファイルの別名を維持する。既存仕様を廃止する設計変更は今回の範囲に含めない。
- 制御ファイルの JSON／テキスト形式、未知キーの扱い、既存設定とのマージ、mtime 判定、起動時の reset、原子的な置換を維持する。
- `finger` の検出失敗時は次の処理フレームで効果を解除する。`full`／`partial` の効果は手の検出に依存させない。
- 手の平滑化、3D 点と平面計算、画素処理、アニメーション位相を変更しない。
- UI はローカル表示にだけ描画し、仮想カメラ出力に混入させない。ミラー処理とフレーム送信の順序を維持する。
- 公開 API の変更、依存追加、機能追加、処理の高速化、UI デザイン変更は別途相談が必要。

## 調査結果

調査基点: `0b8197187ef1ea42a0a1552cb8775467af533b94`（2026-09-21 に確認）。参照行番号はこのコミットのもの。

| 箇所 | 確認できた状態 | 今回の扱い |
| --- | --- | --- |
| `src/zoom_camera_effects/ui.py`（882行） | 描画、ヒット領域、マウス操作、頂点編集、数値・色の変換が同居 | 状態を持たない補助処理を2つの内部モジュールに抽出 |
| `src/zoom_camera_effects/cli.py`（482行） | `main()` がコマンド分岐、入力変換、設定生成、起動を担当 | 設定生成を同じファイルの関数へ抽出 |
| `src/zoom_camera_effects/app.py`（158行） | `_loop()` 内で制御ファイル・UI の設定更新とフレーム処理を実施 | 設定更新のまとまりを同じファイルの関数へ抽出 |
| `effects.py`（443行）、`control.py`（218行） | effect 定義・画素処理、設定の入力変換をそれぞれ担当 | 初回の実装対象から外す |
| `processor.py`、`detection.py`、`geometry.py` | フレーム処理・検出判定・幾何処理が既に分離 | 現状維持、既存テストで回帰を確認 |
| `tests/` | 9ファイル、112件のテストが成功。`test_app.py` は存在しない | 実行ループを変更する前に、カメラを使わない境界テストを追加 |
| `pyproject.toml` | pytest 設定あり。lint／型チェックの設定なし | 既存の検証方法を使用し、新しいツールを導入しない |

`ui.py` の行数だけを減らす大規模なクラス再設計は行わない。とくに頂点編集、タイマー、ヒット領域の優先順位は密接に結び付いているため、初回は `OverlayControlUI` 内に残す。

### 検討した選択肢

| 選択肢 | 利点 | 負担・リスク | 判断 |
| --- | --- | --- | --- |
| 内部関数の抽出と、UI 補助処理の機械的な移動 | 既存テストを使い、段階ごとに確認できる | UI 本体の状態管理は残る | 推奨 |
| effect 定義を共通レジストリに統合し、CLI／UI／制御入力を自動生成 | 設定の重複を減らせる | 入力経路ごとに異なる変換・制限・ヘルプ表現を誤って統一する危険がある | 初回は採用しない |
| UI を複数の controller／renderer に全面分割 | 責務を細かく分けられる | 状態同期とイベント順序の変更範囲が大きい | 必要性を再評価してから別計画にする |

## Review Focus

1. CLI の不正な色・領域入力がカメラ起動前に終了コード2になること。Task 1 のパラメータ化テストで確認する。
2. UI 描画が入力映像を書き換えないこと。Task 2 で全 effect／scope の組合せを確認する。
3. 数値スライダーの丸め、部分領域の頂点操作、3秒を超えたときの非表示が変わらないこと。Task 2 で既存の該当テストを維持する。
4. 同一フレームの設定更新が制御ファイル→UI の順で処理され、不正な制御入力でも UI の更新へ進むこと。Task 3 の偽 reader／UI によるテストで確認する。
5. 仮想カメラへの送信が UI 描画より先であり、UI を含まないこと。Task 3 のフレームテストで確認する。

## 変更予定ファイルと依存方向

| ファイル | 将来の変更 |
| --- | --- |
| `src/zoom_camera_effects/cli.py` | `_build_app_config(args: argparse.Namespace) -> AppConfig` を追加して `main()` から呼ぶ |
| `tests/test_cli.py` | 不正入力時に `run_app()` が呼ばれないことを追加検証 |
| `src/zoom_camera_effects/ui.py` | 既存の公開名・状態管理・クラス内描画メソッドを維持し、補助処理を内部モジュールから import |
| `src/zoom_camera_effects/_ui_drawing.py`（新規） | 描画補助・矩形計算のみ。UI 本体や設定型に依存しない |
| `src/zoom_camera_effects/_ui_options.py`（新規） | 数値・選択肢の定義と `EffectConfig` の値操作。UI 本体に依存しない |
| `tests/test_ui.py` | 入力映像不変のテストを追加。既存の UI 操作テストを維持 |
| `src/zoom_camera_effects/app.py` | `_update_effect_config(processor, control_reader, overlay_ui) -> None` を抽出 |
| `tests/test_app.py`（新規） | `_loop()` を偽カメラ・偽 writer・偽 UI で検証 |

依存方向は `ui → _ui_drawing`、`ui → _ui_options → effects` とする。抽出先から `ui` へ逆向きの import を作らない。`effects` から CLI／UI への依存も作らない。

## Task 1: CLI の設定生成を分離する

**対象:** `cli.py:27–90`、`tests/test_cli.py`。

**インターフェース:** `_build_parser().parse_args()` の `argparse.Namespace` を消費し、既存の `AppConfig` を返す非公開関数を追加する。`main() -> int` は維持する。

- [ ] 変更前に `python3 -m pytest tests/test_cli.py` を実行する。
- [ ] 次の入力境界テストを `tests/test_cli.py` に追加し、現在の実装でも成功することを確認する。これは既存動作を固定するテストであり、意図的に失敗させるための仕様変更はしない。

```python
@pytest.mark.parametrize("options", [
    ["--color", "invalid"],
    ["--fill-color", "invalid"],
    ["--scope", "partial", "--area", "0,0", "1,0"],
])
def test_invalid_effect_input_does_not_start_camera(monkeypatch, options):
    def unexpected_run_app(config):
        pytest.fail("camera startup must not run for invalid input")

    monkeypatch.setattr("zoom_camera_effects.cli.run_app", unexpected_run_app)
    monkeypatch.setattr("sys.argv", ["zoom_camera_effects", *options])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
```

- [ ] 現在の色・領域の変換と `AppConfig` 構築を、次の関数へそのまま移す。既定値を新たに定義しない。

```python
def _build_app_config(args: argparse.Namespace) -> AppConfig:
    outline_color_bgr = _parse_color_bgr(args.color)
    outline_fill_color_bgr = _parse_color_bgr(args.fill_color)
    area_points = _parse_area_points(args.area)
    return AppConfig(
        camera_index=args.camera_index,
        width=args.width,
        height=args.height,
        fps=args.fps,
        preview=args.preview,
        virtual_camera=not args.preview,
        mirror=args.mirror,
        max_frames=args.max_frames,
        smoothing_factor=args.smoothing_factor,
        detection=DetectionConfig(
            min_hand_score=args.min_hand_score,
            min_point_score=args.min_point_score,
            point_bounds_margin=args.point_bounds_margin,
            require_distinct_handedness=args.require_distinct_handedness,
        ),
        effect=EffectConfig(
            mode=_normalize_effect_mode(args.effect),
            scope=_normalize_effect_scope(args.scope),
            area_points=area_points,
            kernel_size=args.kernel,
            mosaic_block_size=args.block_size,
            edge_low_threshold=args.threshold_low,
            edge_high_threshold=args.threshold_high,
            thermal_colormap=args.colormap,
            noise_strength=args.strength,
            outline_thickness=args.thickness,
            outline_color_bgr=outline_color_bgr,
            outline_fill=args.fill,
            outline_fill_color_bgr=outline_fill_color_bgr,
        ),
        control_file=None if args.no_control else args.control_file,
        ui=args.ui,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
```

- [ ] `main()` の `--set-effect` と `--doctor` の早期 return はその位置に残す。変換・構築ブロックを次へ置き換え、続く `run_app()` と `RuntimeError` の処理を維持する。

```python
    try:
        config = _build_app_config(args)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
```

- [ ] `python3 -m pytest tests/test_cli.py`、`python3 -m pytest` を実行する。ヘルプは変更前後に `PYTHONPATH=src python3 -m zoom_camera_effects --help` を実行して一時ファイルへ保存し、`diff -u` で差分がないことを確認する。
- [ ] CLI とそのテストだけを stage し、`refactor: separate CLI configuration construction` でコミットする。

**完了条件:** 同じ CLI 入力が同じ設定・終了コード・ヘルプになる。設定生成がカメラやファイルへの書込みを伴わない。

## Task 2: UI の独立した補助処理を抽出する

**対象:** `ui.py:49–105,565–590,641–690,695–867`、新規の内部モジュール2ファイル、`tests/test_ui.py`。

**インターフェース:** `OverlayControlUI.render(frame_bgr, config)`、`handle_mouse(event, x, y, flags, param)`、`consume_pending_config(base_config)` を維持する。移動する補助関数の引数・戻り値・本体は変更しない。

- [ ] `tests/test_ui.py` に `pytest` と `EFFECT_MODES`／`EFFECT_SCOPES` の import を追加し、次のテストが移動前に成功することを確認する。

```python
@pytest.mark.parametrize("mode", EFFECT_MODES)
@pytest.mark.parametrize("scope", EFFECT_SCOPES)
def test_overlay_render_preserves_source_frame(mode, scope):
    frame = np.full((480, 640, 3), 64, dtype=np.uint8)
    original = frame.copy()
    ui = OverlayControlUI(now=lambda: 0.0)
    output = ui.render(frame, EffectConfig(mode=mode, scope=scope))
    assert np.array_equal(frame, original)
    assert output.shape == frame.shape
    assert output.dtype == frame.dtype
    assert not np.shares_memory(output, frame)
```

- [ ] `_ui_drawing.py` を作成し、次の関数を元の順序で移す。依存は `cv2` と `numpy` のみ。

```text
_draw_button, _draw_arrow_button, _draw_value_pill,
_draw_centered_text, _fit_text, _draw_translucent_rect,
_draw_gear_icon, _put_text, _rect_start, _rect_end, _clip_rect
```

- [ ] `ui.py` は次を import する。`_fit_text` は描画モジュール内だけで使うため、`ui.py` には不要な再公開用ラッパーを追加しない。

```python
from ._ui_drawing import (
    _clip_rect,
    _draw_arrow_button,
    _draw_button,
    _draw_centered_text,
    _draw_gear_icon,
    _draw_translucent_rect,
    _draw_value_pill,
    _put_text,
    _rect_end,
    _rect_start,
)
```

- [ ] `_ui_options.py` を作成し、以下の定義を移す。依存は `dataclasses` と既存の `effects` の型・定数・色変換。数値範囲・丸め・列挙順は編集しない。

```text
COLOR_CHOICES, COLORMAP_CHOICES, NumericOption, NUMERIC_OPTIONS,
EFFECT_OPTIONS, _set_numeric_from_slider, _cycle_option,
_cycle_option_label, _cycle_option_value, _cycle_value,
_format_numeric_value, _color_label
```

- [ ] `ui.py` は次を import する。既存の非公開でない型・定数もこの import で元の位置から利用できる状態を維持する。旧実装を複製したまま残さない。

```python
from ._ui_options import (
    COLOR_CHOICES,
    COLORMAP_CHOICES,
    EFFECT_OPTIONS,
    NUMERIC_OPTIONS,
    NumericOption,
    _cycle_option,
    _cycle_option_label,
    _cycle_option_value,
    _format_numeric_value,
    _set_numeric_from_slider,
)
```

- [ ] `HitRegion`、全 `OverlayControlUI` メソッド、`_pixel_to_normalized_point`、`_area_points_to_pixels`、`_line_hit_rect`、`_center_rect`、`_point_in_rect`、UI の `main()` は元のファイルに残す。状態同期や描画順を変更しない。
- [ ] `python3 -m pytest tests/test_ui.py tests/test_effects.py`、`python3 -m pytest` を実行する。既存テストの slider／drag／add-delete／timeout の期待値を変更しない。
- [ ] UI と内部モジュール2ファイル、そのテストだけを stage し、`refactor: extract stateless overlay UI helpers` でコミットする。

**完了条件:** UI 本体は操作状態とレイアウトを担当し、描画補助・設定値変換を独立して読める。入力フレーム、色、レイアウト、ヒット領域の優先順位が維持される。

## Task 3: 実行ループの設定更新を分離する

**対象:** `app.py:80–158`、新規 `tests/test_app.py`。

**インターフェース:** `FrameProcessor`、`EffectControlReader | None`、`OverlayControlUI | None` を消費し、processor の既存 setter で設定を更新する非公開関数を追加する。`run_app(config: AppConfig) -> int` は変更しない。

- [ ] 次のテストを `tests/test_app.py` に追加する。実機・MediaPipe・OBS を起動せず、現在の `_loop()` で全ケースが成功することを確認する。

```python
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
    monkeypatch.setattr(app.cv2, "imshow", lambda name, image: displayed.append(image.copy()))
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
```

- [ ] 次の関数を `app.py` に追加する。処理順序、比較条件、ログ文、捕捉する例外型を維持する。

```python
def _update_effect_config(
    processor: FrameProcessor,
    control_reader: EffectControlReader | None,
    overlay_ui: OverlayControlUI | None,
) -> None:
    if control_reader is not None:
        try:
            effect_config = control_reader.read_config(processor.effect_config)
        except ValueError as exc:
            print(f"warning: ignoring runtime control command: {exc}")
            effect_config = None
        if effect_config is not None and effect_config != processor.effect_config:
            processor.set_effect_config(effect_config)
            print(f"effect switched: {effect_config.mode}")

    if overlay_ui is not None:
        effect_config = overlay_ui.consume_pending_config(processor.effect_config)
        if effect_config is not None and effect_config != processor.effect_config:
            processor.set_effect_config(effect_config)
```

- [ ] `_loop()` の `while True:` 直下の設定更新2ブロックを、次の1行で置き換える。`capture.read()` 以降と初期化・終了処理には手を加えない。

```python
        _update_effect_config(processor, control_reader, overlay_ui)
```

- [ ] `python3 -m pytest tests/test_app.py tests/test_control.py tests/test_processor.py`、`python3 -m pytest` を実行する。
- [ ] app とそのテストだけを stage し、`refactor: isolate runtime effect configuration updates` でコミットする。

**完了条件:** 設定更新の順序と失敗時の継続動作をテストで固定し、フレームループから独立して読める。映像処理・送信・UI 表示・終了の順序は維持される。

## 実装時の最終確認

- [ ] `python3 -m pytest` が全件成功する。
- [ ] `git diff --check` が成功する。
- [ ] `PYTHONPATH=src python3 -m zoom_camera_effects --help` の変更前後に差分がない。
- [ ] `python3 -m pip wheel --no-deps --no-build-isolation . --wheel-dir /tmp/zoom-camera-effects-wheel-check` で既存のビルド設定を確認する。出力された wheel に新規内部モジュール2つが含まれることを確認する。
- [ ] 実装段階では `--preview --max-frames 30` と、仮想カメラへの `--max-frames 30` を試す。カメラ・OBS が利用できない場合は未検証と記録し、pytest の成功で実機確認済みとは扱わない。
- [ ] 実機で mirror、effect／scope の切替、slider、部分領域の追加・移動・削除、UI の開閉と3秒超の非表示を確認する。会議アプリ側へ UI が映らないことも確認する。
- [ ] 変更ファイルを限定してコミットし、同じ作業ブランチを push する。main へのマージはこの計画には含めない。

lint／型チェックは現時点でプロジェクトに設定がないため、導入を完了条件に追加しない。各タスクは独立したコミットとし、不具合があれば該当コミットを revert できる大きさに保つ。

## 初回の対象外

- effect レジストリの導入、全入力経路の統一、CLI ヘルプの自動生成。
- 画像処理アルゴリズム、フレームコピー削減、検出の間引き、スレッド化。
- MediaPipe API の移行、Python／依存ライブラリの更新。
- カメラ・writer のリソース管理方式の変更、macOS ランチャーの改修。
- 3D 点／平面推定や既存 CLI 別名の削除。
- CI、formatter、linter、型検査ツールの新規導入。

これらは性能測定、対応環境、利用状況、仕様変更の判断が必要になるため、今回の構造整理と混ぜない。

## 計画作成時の検証記録

- `python3 --version`: Python 3.11.8。
- `python3 -m pytest`: **112 passed in 12.92s**。
- `PYTHONPATH=src python3 -m zoom_camera_effects --help`: 終了コード0、140行のヘルプを出力。
- テスト環境の主要パッケージ: pytest 9.0.3、NumPy 1.26.4、opencv-contrib-python 4.11.0.86、MediaPipe 0.10.21、pyvirtualcam 0.15.0。
- この計画作成ではカメラ／仮想カメラの起動、アプリのビルド、実装タスクの適用は行っていない。

未確定なのは実装を開始する時期と、この提案の採否。計画のみという依頼に従い、ここで実装へ進まない。
