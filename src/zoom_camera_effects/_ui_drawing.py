"""Stateless drawing and rectangle helpers for the overlay controls."""

from __future__ import annotations

import cv2
import numpy as np


def _draw_button(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    text: str,
    *,
    active: bool,
) -> None:
    fill = (50, 86, 118) if active else (36, 44, 52)
    border = (114, 176, 224) if active else (82, 96, 110)
    text_color = (248, 252, 255) if active else (214, 224, 232)
    cv2.rectangle(image, _rect_start(rect), _rect_end(rect), fill, -1)
    cv2.rectangle(image, _rect_start(rect), _rect_end(rect), border, 1)
    text = _fit_text(text, rect[2] - 12, 0.42)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    text_x = rect[0] + max(6, (rect[2] - text_size[0]) // 2)
    text_y = rect[1] + (rect[3] + text_size[1]) // 2
    _put_text(image, text, (text_x, text_y), 0.42, text_color, 1)


def _draw_arrow_button(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    direction: int,
) -> None:
    fill = (36, 44, 52)
    border = (82, 96, 110)
    icon = (225, 236, 246)
    cv2.rectangle(image, _rect_start(rect), _rect_end(rect), fill, -1)
    cv2.rectangle(image, _rect_start(rect), _rect_end(rect), border, 1)
    center_x = rect[0] + rect[2] // 2
    center_y = rect[1] + rect[3] // 2
    if direction < 0:
        points = np.array(
            [
                (center_x - 5, center_y),
                (center_x + 5, center_y - 8),
                (center_x + 5, center_y + 8),
            ],
            dtype=np.int32,
        )
    else:
        points = np.array(
            [
                (center_x + 5, center_y),
                (center_x - 5, center_y - 8),
                (center_x - 5, center_y + 8),
            ],
            dtype=np.int32,
        )
    cv2.fillConvexPoly(image, points, icon, cv2.LINE_AA)


def _draw_value_pill(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    value: str,
) -> None:
    fill = (42, 53, 64)
    border = (92, 111, 128)
    text_color = (236, 244, 250)
    cv2.rectangle(image, _rect_start(rect), _rect_end(rect), fill, -1)
    cv2.rectangle(image, _rect_start(rect), _rect_end(rect), border, 1)
    text = _fit_text(value, rect[2] - 12, 0.44)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    text_x = rect[0] + max(6, (rect[2] - text_size[0]) // 2)
    text_y = rect[1] + (rect[3] + text_size[1]) // 2
    _put_text(image, text, (text_x, text_y), 0.44, text_color, 1)


def _draw_centered_text(
    image: np.ndarray,
    text: str,
    rect: tuple[int, int, int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    text_x = rect[0] + max(1, (rect[2] - text_size[0]) // 2)
    text_y = rect[1] + (rect[3] + text_size[1]) // 2
    _put_text(image, text, (text_x, text_y), scale, color, thickness)


def _fit_text(text: str, max_width: int, scale: float) -> str:
    if cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] <= max_width:
        return text
    while len(text) > 3:
        candidate = text[:-1] + "."
        if (
            cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
            <= max_width
        ):
            return candidate
        text = text[:-1]
    return text


def _draw_translucent_rect(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    color: tuple[int, int, int],
    *,
    alpha: float,
) -> None:
    x, y, width, height = _clip_rect(rect, image.shape[1], image.shape[0])
    if width <= 0 or height <= 0:
        return
    roi = image[y : y + height, x : x + width]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def _draw_gear_icon(
    image: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
) -> None:
    cx, cy = center
    for angle in np.linspace(0, np.pi * 2, 8, endpoint=False):
        inner = (
            int(cx + np.cos(angle) * radius * 0.72),
            int(cy + np.sin(angle) * radius * 0.72),
        )
        outer = (
            int(cx + np.cos(angle) * radius * 1.12),
            int(cy + np.sin(angle) * radius * 1.12),
        )
        cv2.line(image, inner, outer, color, 2, cv2.LINE_AA)
    cv2.circle(image, center, radius, color, 2, cv2.LINE_AA)
    cv2.circle(image, center, max(3, radius // 3), color, 2, cv2.LINE_AA)


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _rect_start(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    return (rect[0], rect[1])


def _rect_end(rect: tuple[int, int, int, int]) -> tuple[int, int]:
    return (rect[0] + rect[2], rect[1] + rect[3])


def _clip_rect(
    rect: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = rect
    x = min(max(x, 0), image_width)
    y = min(max(y, 0), image_height)
    width = min(width, image_width - x)
    height = min(height, image_height - y)
    return x, y, width, height
