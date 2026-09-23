"""批量 Port_family 重归一化服务测试。"""

import numpy as np
import pytest
import skrf as rf

from sipi_sparam_core.impedance import parallel_rc_impedance
from sipi_sparam_core.touchstone_io import apply_qs_s_def_patch
from sipi_sparam_core.renormalization import (
    BatchRenormalizationCase,
    build_case_filename,
    count_batch_cases,
    iter_batch_cases,
    renormalize_family_case,
    validate_family_ports,
    write_touchstone_with_z0,
)


def _network(nports=6):
    frequency = rf.Frequency(1, 3, 3, "GHz")
    s = np.zeros((3, nports, nports), dtype=complex)
    network = rf.Network(frequency=frequency, s=s, z0=50.0)
    network.name = "inside.s6p"
    network.port_names = [f"P{index}" for index in range(1, nports + 1)]
    return network


def test_case_count_and_iteration_are_cartesian_product():
    family_ids = ["family1", "family2", "family3"]
    candidates = {
        "family1": [34.0, 40.0],
        "family2": [48.0, 60.0],
        "family3": [40.0],
    }
    targets = ["family2", "family3"]
    cio = {family: 1.2 for family in family_ids}

    cases = list(iter_batch_cases(family_ids, targets, candidates, cio))

    assert count_batch_cases(targets, candidates) == 8
    assert len(cases) == 8
    assert cases[0].target_family == "family2"
    assert cases[-1].target_family == "family3"
    assert {case.resistance_ohm["family1"] for case in cases} == {34.0, 40.0}


def test_renormalize_case_keeps_family1_and_target_without_mutating_input():
    network = _network()
    original_s = network.s.copy()
    original_z0 = network.z0.copy()
    family_ports = {
        "family1": [1, 2],
        "family2": [3, 4],
        "family3": [5, 6],
    }
    case = BatchRenormalizationCase(
        target_family="family3",
        resistance_ohm={"family1": 34.0, "family2": 40.0, "family3": 60.0},
        cio_pf={"family1": 0.0, "family2": 0.0, "family3": 0.0},
    )

    result = renormalize_family_case(network, family_ports, case)

    assert result.nports == 4
    assert result.s_def == "traveling"
    assert result.port_names == ["P1", "P2", "P5", "P6"]
    np.testing.assert_allclose(result.z0, [[34.0, 34.0, 60.0, 60.0]] * 3)
    np.testing.assert_array_equal(network.s, original_s)
    np.testing.assert_array_equal(network.z0, original_z0)


def test_parallel_rc_reference_impedance_is_frequency_dependent(tmp_path):
    network = _network()
    family_ports = {
        "family1": [1, 2],
        "family2": [3, 4],
        "family3": [5, 6],
    }
    case = BatchRenormalizationCase(
        target_family="family2",
        resistance_ohm={family: 50.0 for family in family_ports},
        cio_pf={"family1": 1.0, "family2": 2.0, "family3": 3.0},
    )

    result = renormalize_family_case(network, family_ports, case)

    assert result.z0.shape == (3, 4)
    assert np.any(np.imag(result.z0) != 0)
    assert not np.allclose(result.z0[0], result.z0[-1])
    path = write_touchstone_with_z0(result, tmp_path / "rc_case.s4p")
    reloaded = rf.Network(str(path))
    # 当前 scikit-rf 能识别其自身写出的 traveling 注释；QS 元数据补丁作为
    # 跨版本/其他写入器场景下的显式兜底，重复应用不改变 S 数据。
    assert apply_qs_s_def_patch(reloaded, str(path)) is True
    assert reloaded.s_def == "traveling"
    np.testing.assert_allclose(reloaded.z0, result.z0)
    np.testing.assert_allclose(reloaded.s, result.s, rtol=1e-6, atol=1e-8)


def test_non_target_family_is_physically_terminated_by_parallel_rc():
    frequency = rf.Frequency(1, 3, 3, "GHz")
    z_matrix = np.empty((3, 3, 3), dtype=complex)
    base = np.array([[45.0, 5.0, 3.0], [5.0, 55.0, 4.0], [3.0, 4.0, 65.0]])
    for index in range(3):
        z_matrix[index] = base + 1j * (index + 1) * np.eye(3)
    network = rf.Network(frequency=frequency, z=z_matrix, z0=50.0)
    family_ports = {"family1": [1], "family2": [2], "family3": [3]}
    case = BatchRenormalizationCase(
        target_family="family2",
        resistance_ohm={"family1": 34.0, "family2": 60.0, "family3": 48.0},
        cio_pf={"family1": 1.0, "family2": 2.0, "family3": 3.0},
    )

    result = renormalize_family_case(network, family_ports, case)
    terminated_z = parallel_rc_impedance(frequency.f, 48.0, 3.0e-12)

    for index, load_z in enumerate(terminated_z):
        expected = (
            z_matrix[index, :2, :2]
            - z_matrix[index, :2, 2:3]
            @ np.linalg.inv(z_matrix[index, 2:3, 2:3] + np.array([[load_z]]))
            @ z_matrix[index, 2:3, :2]
        )
        np.testing.assert_allclose(result.z[index], expected, rtol=1e-11, atol=1e-11)


def test_precomputed_z_matrix_has_identical_result():
    network = _network()
    family_ports = {
        "family1": [1, 2],
        "family2": [3, 4],
        "family3": [5, 6],
    }
    case = BatchRenormalizationCase(
        target_family="family2",
        resistance_ohm={family: 50.0 for family in family_ports},
        cio_pf={"family1": 1.0, "family2": 2.0, "family3": 3.0},
    )

    direct = renormalize_family_case(network, family_ports, case)
    precomputed = renormalize_family_case(
        network, family_ports, case, z_matrix=network.z.copy()
    )

    np.testing.assert_allclose(precomputed.s, direct.s)
    np.testing.assert_allclose(precomputed.z0, direct.z0)


def test_family_mapping_must_cover_every_port_once():
    with pytest.raises(ValueError, match="重复"):
        validate_family_ports({"family1": [1, 2], "family2": [2, 3]}, 3)

    with pytest.raises(ValueError, match="缺少"):
        validate_family_ports({"family1": [1], "family2": [2]}, 3)


def test_filename_and_touchstone_writer(tmp_path):
    network = _network()
    case = BatchRenormalizationCase(
        target_family="family3",
        resistance_ohm={"family1": 34.0, "family2": 40.0, "family3": 60.0},
        cio_pf={"family1": 0.8, "family2": 1.2, "family3": 1.2},
    )
    name = build_case_filename(
        "inside.s6p", case, ["family1", "family2", "family3"], 4
    )
    path = write_touchstone_with_z0(network, tmp_path / "roundtrip.s6p")

    assert name == "inside_T3_R34-40-60_C0p8-1p2-1p2.s4p"
    assert path.name == "roundtrip.s6p"
    assert path.exists()
    assert path.stat().st_size > 0
    contents = path.read_text(encoding="utf-8")
    assert contents.startswith("! QS_S_DEF power\n! QS_ZREF_MODEL parallel_rc\n")
    reloaded = rf.Network(str(path))
    assert reloaded.nports == 6
    assert reloaded.f.shape == (3,)


def test_writer_rejects_complex_power_wave_network(tmp_path):
    network = _network()
    network.z0[:, 0] = 34.0 - 2.0j

    with pytest.raises(ValueError, match="traveling-wave"):
        write_touchstone_with_z0(network, tmp_path / "invalid.s6p")
