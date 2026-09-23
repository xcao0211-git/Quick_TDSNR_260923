"""与 UI 无关的 cursor 和 SNR 组合原语。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Iterable

import numpy as np

from .time_response import find_ui_center


MAIN_CURSOR_METHODS = frozenset({"peak", "half_height_center"})


@dataclass(frozen=True)
class WindowPeak:
    cursor_delta: int
    cursor_time_ps: float
    window_start_ps: float
    window_end_ps: float
    peak_time_ps: float
    signed_value: float
    abs_value: float


def find_main_cursor(y_data: np.ndarray, method: str) -> int:
    absolute = np.abs(np.asarray(y_data, dtype=float))
    if absolute.size == 0:
        raise ValueError("波形为空，无法定位main cursor")
    if method not in MAIN_CURSOR_METHODS:
        raise ValueError(f"未知main cursor方法：{method}")
    peak_idx = int(np.argmax(absolute))
    return peak_idx if method == "peak" else find_ui_center(y_data, peak_idx)


def make_cursor_times(main_time_ps: float, ui_ps: float, num_pre: int, num_post: int) -> dict[int, float]:
    if num_pre < 0 or num_post < 0:
        raise ValueError("pre/post cursor数量不能为负数")
    if ui_ps <= 0 or not math.isfinite(ui_ps):
        raise ValueError("ui必须为有限正数")
    return {delta: float(main_time_ps + delta * ui_ps) for delta in range(-int(num_pre), int(num_post) + 1)}


def find_window_peak(
    time_ps: np.ndarray,
    y_data: np.ndarray,
    cursor_delta: int,
    cursor_time_ps: float,
    half_width_ps: float,
) -> WindowPeak:
    time_ps = np.asarray(time_ps, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    if time_ps.ndim != 1 or y_data.ndim != 1 or time_ps.size != y_data.size or not time_ps.size:
        raise ValueError("time_ps 与 y_data 必须是等长非空一维数组")
    if np.any(np.diff(time_ps) <= 0) or half_width_ps < 0:
        raise ValueError("时间轴必须递增且窗口半宽不能为负数")
    start_ps = float(cursor_time_ps - half_width_ps)
    end_ps = float(cursor_time_ps + half_width_ps)
    if end_ps > float(time_ps[-1]):
        raise ValueError(
            f"cursor {cursor_delta:+d} 的窗口 [{start_ps:.3f}, {end_ps:.3f}] ps 超出波形范围 "
            f"[{time_ps[0]:.3f}, {time_ps[-1]:.3f}] ps"
        )
    if end_ps < float(time_ps[0]):
        return WindowPeak(int(cursor_delta), float(cursor_time_ps), start_ps, end_ps, float(cursor_time_ps), 0.0, 0.0)
    first = int(np.searchsorted(time_ps, max(start_ps, float(time_ps[0])), side="left"))
    last = int(np.searchsorted(time_ps, end_ps, side="right"))
    first = min(max(first, 0), len(time_ps) - 1)
    last = min(max(last, first + 1), len(time_ps))
    local = np.abs(y_data[first:last])
    idx = first + int(np.argmax(local))
    signed = float(y_data[idx])
    return WindowPeak(int(cursor_delta), float(cursor_time_ps), start_ps, end_ps, float(time_ps[idx]), signed, abs(signed))


def combine_noise(
    signal: float,
    direct_cursor_values: Iterable[float],
    aggressor_cursor_values: Iterable[Iterable[float]],
) -> tuple[float, float, float, float]:
    direct_noise = float(sum(abs(float(value)) for value in direct_cursor_values))
    aggressor_noises = [float(sum(abs(float(value)) for value in values)) for values in aggressor_cursor_values]
    xtalk_noise = float(math.sqrt(sum(value**2 for value in aggressor_noises)))
    total_noise = float(math.hypot(direct_noise, xtalk_noise))
    snr = float(abs(float(signal)) / total_noise) if total_noise > 0 else math.inf
    return direct_noise, xtalk_noise, total_noise, snr
