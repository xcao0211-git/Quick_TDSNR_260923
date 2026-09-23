"""与具体 UI 无关的传输函数时域数值原语。"""

from __future__ import annotations

import numpy as np


def next_power_of_two(value: int) -> int:
    """返回不小于 ``value`` 的 2 的幂，最小为 4。"""
    return 1 << max(2, int(value) - 1).bit_length()


def linear_interpolate_extrapolate(
    x_src: np.ndarray,
    y_src: np.ndarray,
    x_new: np.ndarray,
) -> np.ndarray:
    """一维线性插值，边界外沿首末两段斜率外推。"""
    x_src = np.asarray(x_src, dtype=float)
    y_src = np.asarray(y_src, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    if x_src.size != y_src.size:
        raise ValueError("x_src 与 y_src 长度必须一致")
    if x_src.size == 0:
        return np.zeros_like(x_new)
    if x_src.size == 1:
        return np.full_like(x_new, float(y_src[0]))
    if np.any(np.diff(x_src) <= 0):
        raise ValueError("x_src 必须严格递增")

    y_new = np.interp(x_new, x_src, y_src)
    left = x_new < x_src[0]
    if np.any(left):
        slope = (y_src[1] - y_src[0]) / (x_src[1] - x_src[0])
        y_new[left] = y_src[0] + slope * (x_new[left] - x_src[0])
    right = x_new > x_src[-1]
    if np.any(right):
        slope = (y_src[-1] - y_src[-2]) / (x_src[-1] - x_src[-2])
        y_new[right] = y_src[-1] + slope * (x_new[right] - x_src[-1])
    return y_new


def resample_transfer_function(
    frequency_hz: np.ndarray,
    transfer: np.ndarray,
    *,
    dt_ps: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """把复传输函数的幅度 dB 与展开相位插值到目标 rFFT 频轴。"""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    transfer = np.asarray(transfer, dtype=complex)
    if frequency_hz.ndim != 1 or transfer.ndim != 1:
        raise ValueError("frequency_hz 与 transfer 必须是一维数组")
    if frequency_hz.size != transfer.size:
        raise ValueError("frequency_hz 与 transfer 长度必须一致")
    if frequency_hz.size < 2:
        raise ValueError("至少需要两个频点")
    if dt_ps <= 0 or int(n_points) < 4:
        raise ValueError("dt_ps 必须为正数，n_points 必须不小于 4")

    target_frequency = np.fft.rfftfreq(int(n_points), d=float(dt_ps) * 1e-12)
    magnitude_db = 20.0 * np.log10(np.abs(transfer) + 1e-20)
    phase_rad = np.unwrap(np.angle(transfer))
    magnitude_target = linear_interpolate_extrapolate(
        frequency_hz, magnitude_db, target_frequency
    )
    phase_target = linear_interpolate_extrapolate(
        frequency_hz, phase_rad, target_frequency
    )
    transfer_target = 10.0 ** (magnitude_target / 20.0) * np.exp(1j * phase_target)
    return target_frequency, transfer_target


def impulse_response_from_transfer(
    frequency_hz: np.ndarray,
    transfer: np.ndarray,
    *,
    dt_ps: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """由复传输函数生成目标时间步长下的实数冲激响应。"""
    _, transfer_target = resample_transfer_function(
        frequency_hz, transfer, dt_ps=dt_ps, n_points=n_points
    )
    impulse = np.fft.irfft(transfer_target, n=int(n_points))
    time_ps = np.arange(int(n_points), dtype=float) * float(dt_ps)
    return time_ps, np.real(impulse)


def trapezoidal_pulse(
    n: int,
    dt_s: float,
    rise_time_ps: float,
    pulse_width_ps: float,
    *,
    sample_rounding: str = "continuous",
) -> np.ndarray:
    """按真实时间对归一化梯形脉冲取样，不把边沿时间取整。

    continuous（默认）：拐点为 0、rise、width、width+rise，width 为半高宽。
    width == rise 时为三角脉冲，width < rise 无效。输出保持 n 个均匀采样点，
    不插入非网格拐点；过粗采样可能漏掉整个脉冲。
    显式 nearest / floor 保留历史量化行为，仅用于兼容旧调用方。
    """
    if not np.isfinite(n) or n < 1 or int(n) != n:
        raise ValueError("n 必须为有限正整数")
    if any(not np.isfinite(value) or value <= 0 for value in (dt_s, rise_time_ps, pulse_width_ps)):
        raise ValueError("dt、上升时间和脉冲宽度必须为有限正数")
    if sample_rounding not in {"continuous", "nearest", "floor"}:
        raise ValueError("sample_rounding 必须为 'continuous'、'nearest' 或 'floor'")
    if sample_rounding == "continuous":
        if pulse_width_ps < rise_time_ps:
            raise ValueError("脉冲宽度 UI 必须大于或等于上升时间 rise")
        time_ps = np.arange(int(n), dtype=float) * (float(dt_s) * 1e12)
        pulse = np.ones(int(n), dtype=float)
        rising = time_ps < rise_time_ps
        falling = time_ps >= pulse_width_ps
        pulse[rising] = time_ps[rising] / rise_time_ps
        pulse[falling] = 1.0 - (time_ps[falling] - pulse_width_ps) / rise_time_ps
        return np.clip(pulse, 0.0, 1.0)

    convert = round if sample_rounding == "nearest" else int
    rise_n = max(1, int(convert(rise_time_ps * 1e-12 / dt_s)))
    fall_n = rise_n
    width_n = max(1, int(convert(pulse_width_ps * 1e-12 / dt_s)))
    roof_n = max(1, width_n - (rise_n + fall_n) // 2)

    shape = np.zeros(rise_n + roof_n + fall_n, dtype=float)
    shape[:rise_n] = np.linspace(0.0, 1.0, rise_n, endpoint=False)
    shape[rise_n : rise_n + roof_n] = 1.0
    shape[-fall_n:] = np.linspace(1.0, 0.0, fall_n, endpoint=False)
    pulse = np.zeros(int(n), dtype=float)
    pulse[: min(int(n), shape.size)] = shape[: int(n)]
    return pulse


def fft_convolve_prefix(x: np.ndarray, h: np.ndarray, n_out: int) -> np.ndarray:
    """使用 FFT 做线性卷积并返回前 ``n_out`` 点。"""
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    if n_out < 0:
        raise ValueError("n_out 不能为负数")
    n_conv = max(1, x.size + h.size - 1)
    n_fft = next_power_of_two(n_conv)
    y = np.fft.irfft(np.fft.rfft(x, n_fft) * np.fft.rfft(h, n_fft), n_fft)
    return np.real(y[: int(n_out)])


def pulse_response_from_transfer(
    frequency_hz: np.ndarray,
    transfer: np.ndarray,
    *,
    ui_ps: float,
    rise_time_ps: float,
    dt_ps: float,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """由复传输函数生成一个 UI 梯形输入的时域响应。"""
    time_ps, impulse = impulse_response_from_transfer(
        frequency_hz, transfer, dt_ps=dt_ps, n_points=n_points
    )
    pulse = trapezoidal_pulse(
        int(n_points),
        float(dt_ps) * 1e-12,
        float(rise_time_ps),
        float(ui_ps),
    )
    response = fft_convolve_prefix(pulse, impulse, int(n_points))
    return time_ps, response


def find_ui_center(y_data: np.ndarray, peak_idx: int | None = None) -> int:
    """用绝对峰值两侧的半高位置确定平顶脉冲中心。"""
    absolute = np.abs(np.asarray(y_data, dtype=float))
    if absolute.size == 0:
        return 0
    if peak_idx is None:
        peak_idx = int(np.argmax(absolute))
    peak_idx = int(np.clip(peak_idx, 0, absolute.size - 1))
    half = float(absolute[peak_idx]) / 2.0
    if half <= 0:
        return peak_idx
    left = np.where(absolute[:peak_idx] < half)[0]
    right = np.where(absolute[peak_idx:] < half)[0]
    rise = int(left[-1] + 1) if left.size else 0
    fall = int(peak_idx + right[0]) if right.size else absolute.size - 1
    return int((rise + fall) // 2)
