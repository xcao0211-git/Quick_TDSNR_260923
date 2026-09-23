"""touchstone_patch — PowerSI 端口名二次识别测试。"""

import numpy as np
import skrf as rf

from sipi_sparam_core.touchstone_io import (
    apply_port_name_patch,
    apply_qs_s_def_patch,
    parse_powersi_port_names,
    parse_qs_s_def,
)


def _write_header(tmp_path, header: str, suffix: str = "s2p") -> str:
    """只关心头部注释——数据段写一个最小占位即可（parse 函数不读数据）。"""
    path = tmp_path / f"sample.{suffix}"
    path.write_text(header.rstrip() + "\n# GHz S MA R 50\n", encoding="utf-8")
    return str(path)


def _make_empty_network(nports: int = 2) -> rf.Network:
    freq = rf.Frequency(1, 3, 3, "GHz")
    s = np.zeros((3, nports, nports), dtype=complex)
    return rf.Network(frequency=freq, s=s, z0=50)


class TestParsePortNames:
    def test_indexed_block_two_ports(self, tmp_path):
        header = "!Port 1=0CS-4$7N39591\n!Port2=0CS-5 $7N39597"
        path = _write_header(tmp_path, header)
        assert parse_powersi_port_names(path, 2) == [
            "0CS-4$7N39591",
            "0CS-5 $7N39597",
        ]

    def test_doublecolon_block_fallback(self, tmp_path):
        header = "!Port32_0SC_P::$7N39591\n!Port33_0SC_N::$7N39597"
        path = _write_header(tmp_path, header)
        assert parse_powersi_port_names(path, 2) == [
            "32_0SC_P::$7N39591",
            "33_0SC_N::$7N39597",
        ]

    def test_indexed_preferred_over_doublecolon(self, tmp_path):
        header = (
            "!Port32_0SC_P::$7N39591\n"
            "!Port33_0SC_N::$7N39597\n"
            "!PowerSI Version:18.0\n"
            "!Port 1=A\n"
            "!Port 2=B"
        )
        path = _write_header(tmp_path, header)
        assert parse_powersi_port_names(path, 2) == ["A", "B"]

    def test_count_mismatch_returns_none(self, tmp_path):
        header = "!Port 1=A\n!Port 2=B\n!Port 3=C"
        path = _write_header(tmp_path, header)
        assert parse_powersi_port_names(path, 2) is None

    def test_missing_index_returns_none(self, tmp_path):
        header = "!Port 1=A\n!Port 3=B"
        path = _write_header(tmp_path, header)
        assert parse_powersi_port_names(path, 2) is None

    def test_no_port_comments_returns_none(self, tmp_path):
        path = _write_header(tmp_path, "!Generated from foo.spd")
        assert parse_powersi_port_names(path, 2) is None

    def test_nonexistent_file_returns_none(self):
        assert parse_powersi_port_names("/nonexistent/path.s2p", 2) is None

    def test_zero_nports_returns_none(self, tmp_path):
        path = _write_header(tmp_path, "!Port 1=A")
        assert parse_powersi_port_names(path, 0) is None

    def test_stops_at_option_line(self, tmp_path):
        # 选项行之后的 !Port 注释不应被采纳
        path = tmp_path / "sample.s2p"
        path.write_text(
            "!Port 1=A\n!Port 2=B\n# GHz S MA R 50\n!Port 3=C\n",
            encoding="utf-8",
        )
        assert parse_powersi_port_names(str(path), 2) == ["A", "B"]


class TestApplyPatch:
    def test_skips_when_port_names_already_set(self, tmp_path):
        ntwk = _make_empty_network()
        ntwk.port_names = ["X", "Y"]
        header = "!Port 1=A\n!Port 2=B"
        path = _write_header(tmp_path, header)
        assert apply_port_name_patch(ntwk, path) is False
        assert ntwk.port_names == ["X", "Y"]

    def test_fills_when_empty(self, tmp_path):
        ntwk = _make_empty_network()
        header = "!Port 1=A\n!Port 2=B"
        path = _write_header(tmp_path, header)
        assert apply_port_name_patch(ntwk, path) is True
        assert ntwk.port_names == ["A", "B"]

    def test_returns_false_when_unparseable(self, tmp_path):
        ntwk = _make_empty_network()
        path = _write_header(tmp_path, "!Generated from foo.spd")
        assert apply_port_name_patch(ntwk, path) is False
        assert not ntwk.port_names


class TestScatteringDefinitionPatch:
    def test_parses_traveling_wave_metadata(self, tmp_path):
        path = _write_header(tmp_path, "! QS_S_DEF traveling")

        assert parse_qs_s_def(path) == "traveling"

    def test_restores_s_def_without_changing_s_data(self, tmp_path):
        network = _make_empty_network()
        original_s = network.s.copy()
        path = _write_header(tmp_path, "! QS_S_DEF traveling")

        assert network.s_def == "power"
        assert apply_qs_s_def_patch(network, path) is True
        assert network.s_def == "traveling"
        np.testing.assert_array_equal(network.s, original_s)

    def test_missing_metadata_is_left_unchanged(self, tmp_path):
        network = _make_empty_network()
        path = _write_header(tmp_path, "! ordinary comment")

        assert apply_qs_s_def_patch(network, path) is False
        assert network.s_def == "power"
