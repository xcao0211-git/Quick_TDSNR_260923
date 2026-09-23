"""QS_domain/algorithms/impedance.py 的纯函数单元测试 — 不依赖 Qt。

覆盖：
* Touchstone 选项行参考阻抗解析（含 R 后无数值、无 R 段、小写 r、多值等边界）；
* enforce_nonzero_z0 对零阻抗网络的修正：R 后无数值时回落默认 50Ω（回归
  真实文件 '# GHz S RI R' 触发的 "无法解析参考阻抗" 崩溃）；
* 声明 R 0 等非正参考阻抗时同样回落 50Ω；
* 无零阻抗时不改动网络。
"""

import numpy as np
import pytest
import skrf as rf

from sipi_sparam_core.impedance import (
    _parse_ref_impedance,
    enforce_nonzero_z0,
    has_zero_impedance,
    parallel_rc_impedance,
)


@pytest.mark.parametrize("line, expected", [
    ("# GHz S RI R 50", 50.0),
    ("# MHz S DB R 75 75", 75.0),   # 多端口逐口阻抗，取首个
    ("# ghz s ri r 90", 90.0),      # 小写 r
    ("# Hz Y MA R 100", 100.0),
    ("# GHz S RI R 0", 0.0),        # 显式 0（调用方再回落）
    ("# GHz S RI R", None),         # R 后无数值 —— 曾触发崩溃
    ("# GHz S RI", None),           # 无 R 段
    ("# GHz S RI R foo", None),     # R 后非数值
])
def test_parse_ref_impedance(line, expected):
    assert _parse_ref_impedance(line) == expected


def test_parallel_rc_impedance_reduces_to_resistance_when_capacitance_is_zero():
    result = parallel_rc_impedance(np.array([0.0, 1e9]), 60.0, 0.0)

    np.testing.assert_allclose(result, [60.0, 60.0])


def test_parallel_rc_impedance_at_corner_frequency():
    resistance = 50.0
    capacitance = 1e-12
    corner_frequency = 1.0 / (2.0 * np.pi * resistance * capacitance)

    result = parallel_rc_impedance([corner_frequency], resistance, capacitance)

    np.testing.assert_allclose(result, [25.0 - 25.0j])


@pytest.mark.parametrize("resistance, capacitance", [(0.0, 1e-12), (-1.0, 1e-12), (50.0, -1e-12)])
def test_parallel_rc_impedance_rejects_nonphysical_values(resistance, capacitance):
    with pytest.raises(ValueError):
        parallel_rc_impedance([1e9], resistance, capacitance)


def _make_zero_z0_network(tmp_path, option_line, nports=2, nfreqs=5):
    """构造一个 z0 含 0 的网络，并写出对应 option 行的 touchstone 文件。"""
    freq = rf.Frequency(1.0, 5.0, nfreqs, 'GHz')
    s = np.zeros((nfreqs, nports, nports), dtype=complex)
    ntwk = rf.Network(frequency=freq, s=s, z0=50.0)
    ntwk.z0 = np.zeros((nfreqs, nports))  # 人为制造零阻抗
    path = tmp_path / "dut.s2p"
    path.write_text(f"{option_line}\n", encoding="utf-8")
    return ntwk, str(path)


def test_enforce_nonzero_z0_r_without_value_falls_back_50(tmp_path):
    """'# GHz S RI R'（R 后无数值）不再抛错，回落默认 50Ω。"""
    ntwk, path = _make_zero_z0_network(tmp_path, "# GHz S RI R")
    assert has_zero_impedance(ntwk)
    enforce_nonzero_z0(ntwk, path)
    assert np.all(np.array(ntwk.z0) == 50.0)


def test_enforce_nonzero_z0_reads_declared_impedance(tmp_path):
    """R 后有有效数值时按声明值修正。"""
    ntwk, path = _make_zero_z0_network(tmp_path, "# GHz S RI R 90")
    enforce_nonzero_z0(ntwk, path)
    assert np.all(np.array(ntwk.z0) == 90.0)


def test_enforce_nonzero_z0_nonpositive_declared_falls_back_50(tmp_path):
    """声明 R 0 时回落 50Ω，避免又把阻抗设回 0。"""
    ntwk, path = _make_zero_z0_network(tmp_path, "# GHz S RI R 0")
    enforce_nonzero_z0(ntwk, path)
    assert np.all(np.array(ntwk.z0) == 50.0)


def test_enforce_nonzero_z0_noop_when_no_zero(tmp_path):
    """无零阻抗端口时不读文件、不改动网络。"""
    freq = rf.Frequency(1.0, 5.0, 5, 'GHz')
    s = np.zeros((5, 2, 2), dtype=complex)
    ntwk = rf.Network(frequency=freq, s=s, z0=75.0)
    enforce_nonzero_z0(ntwk, "/nonexistent/path.s2p")  # 不应尝试打开
    assert np.all(np.array(ntwk.z0) == 75.0)
