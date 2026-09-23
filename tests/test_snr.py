import math

import numpy as np
import pytest

from sipi_sparam_core.snr import (
    combine_noise,
    find_main_cursor,
    find_window_peak,
    make_cursor_times,
)


def test_cursor_methods_and_spacing():
    y = np.array([0.0, 0.4, 1.0, 1.0, 0.4, 0.0])
    assert find_main_cursor(y, "peak") == 2
    assert find_main_cursor(y, "half_height_center") in (2, 3)
    assert make_cursor_times(100.0, 20.0, 2, 2) == {
        -2: 60.0, -1: 80.0, 0: 100.0, 1: 120.0, 2: 140.0
    }


def test_cursor_validation():
    with pytest.raises(ValueError, match="未知"):
        find_main_cursor(np.ones(3), "bad")
    with pytest.raises(ValueError, match="负数"):
        make_cursor_times(0, 1, -1, 1)
    with pytest.raises(ValueError, match="有限正数"):
        make_cursor_times(0, math.inf, 0, 1)


def test_window_peak_handles_pre_causal_zero_and_boundaries():
    t = np.arange(10, dtype=float)
    y = np.array([0, 0, 1, -2, 0, 0, 0, 0, 0, 0], dtype=float)
    zero = find_window_peak(t, y, -2, -5, 0.5)
    assert zero.abs_value == 0
    peak = find_window_peak(t, y, 0, 3, 0.5)
    assert peak.signed_value == -2
    with pytest.raises(ValueError, match="超出"):
        find_window_peak(t, y, 5, 9.8, 0.5)


def test_window_peak_rejects_bad_arrays():
    with pytest.raises(ValueError):
        find_window_peak(np.array([0, 1]), np.array([1]), 0, 0, 0.1)
    with pytest.raises(ValueError):
        find_window_peak(np.array([1, 0]), np.array([1, 2]), 0, 0, 0.1)


def test_combine_noise_uses_linear_same_cursor_and_rss_across_attackers():
    direct, xtalk, total, snr = combine_noise(1.0, [0.1, -0.2], [[0.3, -0.4], [0.5]])
    assert direct == pytest.approx(0.3)
    assert xtalk == pytest.approx(math.sqrt(0.7**2 + 0.5**2))
    assert total == pytest.approx(math.hypot(0.3, xtalk))
    assert snr == pytest.approx(1 / total)
    assert math.isinf(combine_noise(1.0, [], [])[3])
