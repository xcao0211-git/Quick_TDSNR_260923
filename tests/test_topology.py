"""QS_domain/algorithms/topology_detect.py 的单元测试。"""

import numpy as np
import skrf as rf
import pytest

from sipi_sparam_core.topology import detect_topology, format_report, TopologyReport


def _freq(n_pts=201, f_start_ghz=0.05, f_stop_ghz=10.0) -> rf.Frequency:
    return rf.Frequency(f_start_ghz, f_stop_ghz, n_pts, 'ghz')


def _tline_s(freq: rf.Frequency, length_m: float, eps_r: float = 4.0,
             alpha_np_per_m: float = 0.1, z0: float = 50.0) -> np.ndarray:
    """单根理想 / 微损 50Ω 传输线（2 端口）的 S 参数序列 (Nf,2,2)。"""
    c = 3e8
    beta = 2 * np.pi * freq.f * np.sqrt(eps_r) / c
    gamma = alpha_np_per_m + 1j * beta
    s21 = np.exp(-gamma * length_m)
    s = np.zeros((len(freq.f), 2, 2), dtype=complex)
    s[:, 0, 0] = 0.0
    s[:, 1, 1] = 0.0
    s[:, 0, 1] = s21
    s[:, 1, 0] = s21
    return s


def _block_diag_2x2(blocks: list) -> np.ndarray:
    """把若干个 (Nf,2,2) S 参数块沿对角组合成 (Nf, 2K, 2K)。块间互不耦合。"""
    nf = blocks[0].shape[0]
    k = len(blocks)
    out = np.zeros((nf, 2 * k, 2 * k), dtype=complex)
    for idx, b in enumerate(blocks):
        out[:, 2 * idx:2 * idx + 2, 2 * idx:2 * idx + 2] = b
    return out


# -------------------------------------------------------------------
# 1 驱 1 — 两条独立点对点链路（4 端口）
# -------------------------------------------------------------------

class TestPointToPoint:
    def setup_method(self):
        freq = _freq()
        # 端口顺序：1=TX_a, 2=RX_a, 3=TX_b, 4=RX_b
        block_a = _tline_s(freq, length_m=0.05)
        block_b = _tline_s(freq, length_m=0.10)
        s = _block_diag_2x2([block_a, block_b])
        self.ntwk = rf.Network(frequency=freq, s=s, z0=50.0, name="p2p_4port")

    def test_two_channels_detected(self):
        rep = detect_topology(self.ntwk)
        assert isinstance(rep, TopologyReport)
        assert len(rep.channels) == 2
        for ch in rep.channels:
            assert ch.topology == "p2p"
            assert len(ch.ports) == 2

    def test_pairings_are_correct(self):
        rep = detect_topology(self.ntwk)
        pairs = sorted([tuple(sorted(ch.ports)) for ch in rep.channels])
        assert pairs == [(1, 2), (3, 4)]

    def test_pair_report_contains_low_frequency_s_and_z_values(self):
        rep = detect_topology(self.ntwk)
        first = sorted(rep.channels, key=lambda ch: ch.ports)[0]
        # Z 仍写入数据结构，供后续 PI 场景使用；文本侧目前只显示 S。
        assert first.s_value is not None
        assert first.z_value is not None
        # 默认判据为 Y，耦合按 Y 参数（西门子）显示
        text = format_report(rep)
        assert "Y1,2=" in text
        assert "|Z" not in text

    def test_no_isolated_ports(self):
        rep = detect_topology(self.ntwk)
        assert rep.isolated_ports == []


# -------------------------------------------------------------------
# 1 驱多 — 星型 4 端口（port 1 为 hub，2/3/4 为叶子）
# -------------------------------------------------------------------

def _star_s_from_y(n_branches: int, y_branch: float = 0.02, y_self: float = 0.025,
                   z0: float = 50.0) -> np.ndarray:
    """构造 (n_branches+1) 端口星型网络的 S 矩阵（单频点，频率无关）。

    Y 矩阵：hub 与每个 leaf 之间的互导纳 -y_branch；leaf 间互导纳为 0；
    leaf 对地 y_self；hub 对地 y_self。然后 S = (I - Z0 Y)(I + Z0 Y)^{-1}。
    """
    n = n_branches + 1
    Y = np.zeros((n, n), dtype=complex)
    for k in range(1, n):
        Y[0, k] = -y_branch
        Y[k, 0] = -y_branch
        Y[k, k] = y_branch + y_self
    Y[0, 0] = n_branches * y_branch + y_self
    Yn = z0 * Y
    eye = np.eye(n, dtype=complex)
    return (eye - Yn) @ np.linalg.inv(eye + Yn)


class TestMultiDrop:
    def setup_method(self):
        freq = _freq()
        nf = len(freq.f)
        n = 4
        S0 = _star_s_from_y(n_branches=n - 1)
        s = np.broadcast_to(S0, (nf, n, n)).copy()
        self.ntwk = rf.Network(frequency=freq, s=s, z0=50.0, name="star_4port")

    def test_single_multidrop_channel(self):
        rep = detect_topology(self.ntwk)
        assert len(rep.channels) == 1
        ch = rep.channels[0]
        assert ch.topology == "multi-drop"
        assert ch.tx == 1
        assert sorted(ch.rxs) == [2, 3, 4]
        assert rep.isolated_ports == []

    def test_multidrop_report_lists_all_rx_paths(self):
        rep = detect_topology(self.ntwk)
        text = format_report(rep, file_label="star.s4p")
        assert "联通簇" in text
        assert "4 端口" in text
        assert "port 1*" in text       # 中心节点带 * 标记
        assert "port 2" in text and "port 3" in text and "port 4" in text
        assert "Y1,2=" in text
        assert "Y1,3=" in text
        assert "Y1,4=" in text

    def test_report_values_follow_metric(self):
        # S 判据用 dB，Z 判据用欧姆(Ω)，标签前缀同步切换
        text_s = format_report(detect_topology(self.ntwk, metric="S"))
        assert "判据 = S 参数" in text_s
        assert "S1,2=" in text_s and "dB" in text_s
        text_z = format_report(detect_topology(self.ntwk, metric="Z"))
        assert "判据 = Z 参数" in text_z
        assert "Z1,2=" in text_z and "Ω" in text_z

    def test_multidrop_rx_s_values_populated(self):
        rep = detect_topology(self.ntwk)
        ch = rep.channels[0]
        assert len(ch.rx_s_values) == 3
        assert len(ch.rx_z_values) == 3
        assert ch.s_value == ch.rx_s_values[0]


class TestMultiDropMixedWithP2P:
    """6 端口：port 1 驱 {2,3}（星型），port 4-5 是独立 trace，port 6 孤立。"""

    def setup_method(self):
        freq = _freq()
        nf = len(freq.f)
        # 3 端口星型（hub=0, leaves=1,2）
        star3 = _star_s_from_y(n_branches=2)
        # 2 端口 trace
        line2 = _tline_s(freq, length_m=0.05)
        # 把 star3（频率无关）扩成 (nf,3,3)
        star3_f = np.broadcast_to(star3, (nf, 3, 3)).copy()
        # 1 端口孤立段：S = 0（匹配吸收）
        iso = np.zeros((nf, 1, 1), dtype=complex)
        n = 6
        s = np.zeros((nf, n, n), dtype=complex)
        s[:, 0:3, 0:3] = star3_f
        s[:, 3:5, 3:5] = line2
        s[:, 5:6, 5:6] = iso
        self.ntwk = rf.Network(frequency=freq, s=s, z0=50.0, name="mixed_6port")

    def test_mixed_topology(self):
        rep = detect_topology(self.ntwk)
        by_topo = {ch.topology: ch for ch in rep.channels}
        assert "multi-drop" in by_topo
        assert "p2p" in by_topo
        multi = by_topo["multi-drop"]
        p2p = by_topo["p2p"]
        assert multi.tx == 1
        assert sorted(multi.rxs) == [2, 3]
        assert tuple(sorted(p2p.ports)) == (4, 5)
        assert rep.isolated_ports == [6]


# -------------------------------------------------------------------
# 孤立端口
# -------------------------------------------------------------------

class TestIsolatedPort:
    def setup_method(self):
        freq = _freq()
        block = _tline_s(freq, length_m=0.05)
        nf = len(freq.f)
        s = np.zeros((nf, 3, 3), dtype=complex)
        s[:, :2, :2] = block
        # port 3 与外界全部断开：S31, S32, S13, S23 = 0；S33 = 0（匹配）
        self.ntwk = rf.Network(frequency=freq, s=s, z0=50.0, name="with_isolated")

    def test_one_channel_and_one_isolated(self):
        rep = detect_topology(self.ntwk)
        assert len(rep.channels) == 1
        assert tuple(sorted(rep.channels[0].ports)) == (1, 2)
        assert rep.isolated_ports == [3]


# -------------------------------------------------------------------
# 自适应间隙连通判据 + 扇出电平校验
# -------------------------------------------------------------------

def _net_from_sym_s(n: int, pairs: dict, f_pts: int = 11) -> rf.Network:
    """由单频对称 S（频率无关，广播到全频带）构造网络。pairs: {(i,j): value}。"""
    freq = _freq(n_pts=f_pts)
    S0 = np.zeros((n, n), dtype=complex)
    for (i, j), v in pairs.items():
        S0[i, j] = v
        S0[j, i] = v
    s = np.broadcast_to(S0, (len(freq.f), n, n)).copy()
    return rf.Network(frequency=freq, s=s, z0=50.0)


class TestGapConnectivity:
    def test_weakly_mutual_ports_are_isolated(self):
        # 1-2 强 p2p；3-4 仅在 -40 dB 互为最强、其余 -60 dB。
        # 旧"互为最强"判据会把 3-4 错配成对；间隙判据应判为孤立。
        ntwk = _net_from_sym_s(4, {
            (0, 1): 0.95, (2, 3): 0.01,
            (0, 2): 1e-3, (0, 3): 1e-3, (1, 2): 1e-3, (1, 3): 1e-3,
        })
        rep = detect_topology(ntwk, metric="S")
        assert len(rep.channels) == 1
        assert tuple(sorted(rep.channels[0].ports)) == (1, 2)
        assert rep.isolated_ports == [3, 4]

    def test_cliff_threshold_is_adjustable(self):
        # 两对不同电平 + 非零串扰：阈值低→切在串扰上方得两对 p2p；
        # 阈值高→无显著断崖→全部并成一个 net。
        ntwk = _net_from_sym_s(4, {
            (0, 1): 0.9, (2, 3): 0.2,
            (0, 2): 0.02, (0, 3): 0.02, (1, 2): 0.02, (1, 3): 0.02,
        })
        low = detect_topology(ntwk, metric="S", min_cliff_db=12.0)
        assert len(low.channels) == 2
        assert all(c.topology == "p2p" for c in low.channels)
        high = detect_topology(ntwk, metric="S", min_cliff_db=25.0)
        assert len(high.channels) == 1
        assert high.channels[0].topology == "multi-drop"


class TestFlybyJunction:
    def test_ideal_3way_junction_is_multidrop(self):
        # 三条 50Ω 线汇于一节点（低频 flyby 近似）：S_ii=-1/3, S_ij=2/3≈-3.5dB。
        # 三个 |S| 完全相等、无噪声地板 → 间隙判据须判为同一 net 的 1 驱 2，不能漏判。
        n = 3
        S0 = np.full((n, n), 2.0 / n, dtype=complex)
        np.fill_diagonal(S0, 2.0 / n - 1)
        freq = _freq(n_pts=11)
        s = np.broadcast_to(S0, (len(freq.f), n, n)).copy()
        ntwk = rf.Network(frequency=freq, s=s, z0=50.0)
        rep = detect_topology(ntwk, metric="S")
        assert len(rep.channels) == 1
        ch = rep.channels[0]
        assert ch.topology == "multi-drop"
        assert sorted(ch.rxs) == [2, 3]
        assert rep.isolated_ports == []
        # -3.5dB/支 ≈ 二劈，电平校验应一致、无告警
        assert ch.fanout_consistent is True
        assert "⚠" not in format_report(rep)


class TestFanoutLevelCheck:
    def test_ideal_split_is_consistent(self):
        # 理想 1 驱 2：每支 |S|=1/√2 → 反推 N≈2，与结构一致。
        v = 1.0 / np.sqrt(2)
        ntwk = _net_from_sym_s(3, {(0, 1): v, (0, 2): v, (1, 2): 0.05})
        rep = detect_topology(ntwk, metric="S")
        ch = rep.channels[0]
        assert ch.topology == "multi-drop"
        assert ch.fanout_consistent is True
        assert round(ch.fanout_implied) == 2
        assert "⚠" not in format_report(rep)

    def test_daisy_chain_level_mismatch_flagged(self):
        # hub 对两支都近 0 dB（|S|≈0.9）→ 反推 N≈1，与结构 2 路不符，应告警。
        ntwk = _net_from_sym_s(3, {(0, 1): 0.9, (0, 2): 0.9, (1, 2): 0.05})
        rep = detect_topology(ntwk, metric="S")
        ch = rep.channels[0]
        assert ch.fanout_consistent is False
        assert "⚠" in format_report(rep)

    def test_level_check_only_for_s_metric(self):
        v = 1.0 / np.sqrt(2)
        ntwk = _net_from_sym_s(3, {(0, 1): v, (0, 2): v, (1, 2): 0.05})
        rep = detect_topology(ntwk, metric="Y")
        assert rep.channels[0].fanout_consistent is None


# -------------------------------------------------------------------
# 边界 / 输入校验
# -------------------------------------------------------------------

class TestEdgeCases:
    def test_single_port_returns_isolated(self):
        freq = _freq(n_pts=21)
        s = np.zeros((len(freq.f), 1, 1), dtype=complex)
        ntwk = rf.Network(frequency=freq, s=s, z0=50.0)
        rep = detect_topology(ntwk)
        assert rep.channels == []
        assert rep.isolated_ports == [1]

    def test_format_report_runs(self):
        freq = _freq()
        block = _tline_s(freq, length_m=0.05)
        ntwk = rf.Network(frequency=freq, s=block, z0=50.0, name="t")
        rep = detect_topology(ntwk)
        text = format_report(rep, file_label="t.s2p")
        assert "拓扑识别" in text
        assert "t.s2p" in text
        assert "联通端口对" in text or "未识别" in text

    def test_legacy_threshold_arguments_do_not_block_pairing(self):
        freq = _freq()
        block = _tline_s(freq, length_m=0.05)
        ntwk = rf.Network(frequency=freq, s=block, z0=50.0)
        rep = detect_topology(
            ntwk,
            band_ghz=(50.0, 100.0),
            y_threshold_siemens=1e9,
            s_threshold_db=0.0,
            delay_tolerance_ns=0.0,
        )
        assert len(rep.channels) == 1

    def test_zero_coupling_ports_are_isolated(self):
        freq = _freq(n_pts=21)
        s = np.zeros((len(freq.f), 2, 2), dtype=complex)
        ntwk = rf.Network(frequency=freq, s=s, z0=50.0)
        rep = detect_topology(ntwk)
        assert rep.channels == []
        assert rep.isolated_ports == [1, 2]
