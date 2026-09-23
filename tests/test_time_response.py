import numpy as np
import pytest

from sipi_sparam_core.time_response import (
    fft_convolve_prefix,
    find_ui_center,
    impulse_response_from_transfer,
    linear_interpolate_extrapolate,
    next_power_of_two,
    pulse_response_from_transfer,
    resample_transfer_function,
    trapezoidal_pulse,
)


def test_linear_interpolation_and_extrapolation():
    result = linear_interpolate_extrapolate(
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 4.0, 6.0]),
        np.array([0.0, 1.5, 4.0]),
    )
    np.testing.assert_allclose(result, [0.0, 3.0, 8.0])


def test_interpolation_rejects_non_increasing_axis():
    with pytest.raises(ValueError, match="严格递增"):
        linear_interpolate_extrapolate(
            np.array([1.0, 1.0]), np.array([1.0, 2.0]), np.array([1.0])
        )


def test_next_power_of_two_has_minimum_four():
    assert next_power_of_two(1) == 4
    assert next_power_of_two(5) == 8


def test_flat_transfer_has_delta_impulse():
    n_points = 64
    dt_ps = 10.0
    frequency = np.fft.rfftfreq(n_points, d=dt_ps * 1e-12)
    time_ps, impulse = impulse_response_from_transfer(
        frequency,
        np.full(frequency.size, 0.5 + 0j),
        dt_ps=dt_ps,
        n_points=n_points,
    )
    assert time_ps[-1] == pytest.approx((n_points - 1) * dt_ps)
    assert impulse[0] == pytest.approx(0.5)
    np.testing.assert_allclose(impulse[1:], 0.0, atol=1e-14)


def test_resample_transfer_returns_rfft_axis():
    n_points = 32
    dt_ps = 5.0
    frequency = np.array([0.0, 100e9])
    target_frequency, target = resample_transfer_function(
        frequency,
        np.ones(2, dtype=complex),
        dt_ps=dt_ps,
        n_points=n_points,
    )
    np.testing.assert_allclose(target_frequency, np.fft.rfftfreq(n_points, dt_ps * 1e-12))
    np.testing.assert_allclose(target, 1.0)


def test_trapezoidal_pulse_and_flat_half_vtf_response():
    n_points = 128
    dt_ps = 10.0
    frequency = np.fft.rfftfreq(n_points, d=dt_ps * 1e-12)
    pulse = trapezoidal_pulse(n_points, dt_ps * 1e-12, 20.0, 80.0)
    _, response = pulse_response_from_transfer(
        frequency,
        np.full(frequency.size, 0.5 + 0j),
        ui_ps=80.0,
        rise_time_ps=20.0,
        dt_ps=dt_ps,
        n_points=n_points,
    )
    assert np.max(pulse) == pytest.approx(1.0)
    assert np.max(response) == pytest.approx(0.5)


def test_fft_convolution_matches_numpy():
    x = np.array([1.0, 2.0, 3.0])
    h = np.array([0.5, -0.25])
    np.testing.assert_allclose(
        fft_convolve_prefix(x, h, 4), np.convolve(x, h)[:4], atol=1e-14
    )


def test_find_ui_center_uses_flat_top_center():
    y = np.zeros(30)
    y[10:20] = 1.0
    assert find_ui_center(y) in (14, 15)


def test_fractional_ui_uses_exact_falling_edge_samples():
    pulse = trapezoidal_pulse(512, 12.5e-12, 25, 108.7)
    np.testing.assert_allclose(pulse[8:12], [1, 0.848, 0.348, 0], atol=1e-14)
    assert pulse.shape == (512,)


@pytest.mark.parametrize("rise,expected", [(23, [0, 12.5 / 23, 1]),
                                            (30, [0, 12.5 / 30, 25 / 30])])
def test_fractional_rise_is_not_rounded(rise, expected):
    pulse = trapezoidal_pulse(512, 12.5e-12, rise, 112.5)
    np.testing.assert_allclose(pulse[:3], expected, atol=1e-14)
    assert pulse[10] == pytest.approx(1 - 12.5 / rise)


def test_integer_aligned_pulse_retains_golden_samples():
    pulse = trapezoidal_pulse(16, 12.5e-12, 25, 112.5)
    np.testing.assert_allclose(pulse, [0, .5, 1, 1, 1, 1, 1, 1, 1, 1, .5, 0, 0, 0, 0, 0], atol=1e-14)


def test_equal_ui_and_rise_is_triangle_without_forced_roof():
    np.testing.assert_allclose(trapezoidal_pulse(6, 12.5e-12, 25, 25),
                               [0, .5, 1, .5, 0, 0], atol=1e-14)


def test_ui_shorter_than_rise_is_rejected():
    with pytest.raises(ValueError, match="脉冲宽度.*上升时间"):
        trapezoidal_pulse(64, 12.5e-12, 30, 25)


@pytest.mark.parametrize("dt,rise,ui", [(np.nan, 25, 108.7), (12.5e-12, np.inf, 108.7),
                                       (12.5e-12, 25, np.nan)])
def test_nonfinite_pulse_parameters_are_rejected(dt, rise, ui):
    with pytest.raises(ValueError, match="有限"):
        trapezoidal_pulse(64, dt, rise, ui)


def test_subsample_pulse_is_not_artificially_lengthened():
    np.testing.assert_array_equal(trapezoidal_pulse(4, 12.5e-12, 2, 3), [0, 0, 0, 0])


@pytest.mark.parametrize("mode,expected", [
    ("nearest", [0, .5, 1, 1, 1, 1, 1, 1, 1, 1, .5, 0]),
    ("floor", [0, .5, 1, 1, 1, 1, 1, 1, 1, .5, 0, 0]),
])
def test_explicit_legacy_rounding_modes_remain_compatible(mode, expected):
    np.testing.assert_allclose(trapezoidal_pulse(12, 12.5e-12, 25, 108.7, sample_rounding=mode), expected)


def test_fractional_pulse_half_gain_delayed_response_matches_analytic_pwl():
    n, dt = 512, 12.5
    frequency = np.fft.rfftfreq(n, dt * 1e-12)
    transfer = 0.5 * np.exp(-2j * np.pi * frequency * 500e-12)
    time_ps, response = pulse_response_from_transfer(
        frequency, transfer, ui_ps=108.7, rise_time_ps=30, dt_ps=dt, n_points=n)
    expected = 0.5 * np.interp(time_ps - 500, [0, 30, 108.7, 138.7], [0, 1, 1, 0], left=0, right=0)
    np.testing.assert_allclose(response, expected, atol=2e-14)
