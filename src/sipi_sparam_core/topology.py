"""拓扑识别 — 用低频参数矩阵的相对量识别 1 驱 1 / 1 驱多端口对。

判别只用单个低频点，可选 |S| / |Y| / |Z| 作判据矩阵（`metric`）。核心是
"相对量判据"，不是简单取最强对端：

1. 连通判据（自适应间隙）：把全部端口对耦合按 dB 降序排列，找最大跳变（断崖）。
   连通端口（个位数 dB 量级）与非连通端口（-20~-100 dB）之间天然有几十 dB 间隙，
   切点落在间隙中央。断崖上方的端口对才算"同 net 的强耦合边"。
2. 成 net：强耦合边的连通分量即一个 net。size==2 → p2p；size>=3 → 1 驱多，
   hub 取分量内强耦合行和最大的端口（星型中心）。
3. 扇出电平校验（仅 S 判据）：理想等分功分插损 ≈ -10·log10(N) dB，故
   N ≈ 1/|S|²。用连通 |S| 电平反推 N，与结构得到的叶子数比对，不一致时
   在报告里标注（疑似级联/菊花链）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import skrf as rf


# ============================================================
# 数据类
# ============================================================

@dataclass
class ChannelInfo:
    """一条联通通道。

    S 参数互易（S_ij = S_ji），无法从矩阵判断信号方向；这里的 `tx` 仅表示
    "拓扑中心"（T 型节点 / 星型 hub），不是物理 driver。p2p 时 `tx` 取低序号端口。
    """
    ports: List[int]              # 1-based 端口号，含中心节点与全部分支节点
    tx: Optional[int] = None      # 拓扑中心节点（1-based）；p2p 时为低序号端口
    rxs: List[int] = field(default_factory=list)  # 其余分支节点（1-based，按端口号升序）
    s_value: Optional[complex] = None             # rx_s_values[0]，留作 p2p 后向兼容
    z_value: Optional[complex] = None             # rx_z_values[0]
    rx_s_values: List[complex] = field(default_factory=list)  # 与 rxs 一一对应
    rx_z_values: List[complex] = field(default_factory=list)
    rx_metric_values: List[complex] = field(default_factory=list)  # 判据矩阵取值，与 rxs 一一对应
    delays_ns: List[float] = field(default_factory=list)
    il_db: List[float] = field(default_factory=list)
    topology: str = "p2p"          # "p2p" | "multi-drop"
    fanout_expected: Optional[int] = None    # 结构得到的分支数（= len(rxs)）
    fanout_implied: Optional[float] = None   # 由 S 电平 1/|S|² 反推的平均分支数（仅 S 判据）
    fanout_consistent: Optional[bool] = None # 结构与电平是否一致（仅 S 判据）


@dataclass
class TopologyReport:
    n_ports: int
    band_ghz: Tuple[float, float]       # 兼容旧报告结构，本版不使用 mid-band
    low_freq_ghz: float                  # 用于 Y 矩阵连通性的低频
    y_threshold_siemens: float
    s_threshold_db: float
    channels: List[ChannelInfo]
    isolated_ports: List[int]            # 没有进入任何链路的孤立端口（1-based）
    metric: str = "Y"                    # 聚类判据所用参数矩阵："S" | "Y" | "Z"


# ============================================================
# 工具函数
# ============================================================

def _s_to_y(s_slice: np.ndarray, z0: np.ndarray) -> np.ndarray:
    """单频点 S → Y。s_slice (N,N), z0 (N,) 实数。返回 Y (N,N)。

    Y = sqrt(Y0) (I − S)(I + S)^{-1} sqrt(Y0)，使用归一化阻抗。
    """
    n = s_slice.shape[0]
    y0 = 1.0 / np.asarray(z0, dtype=float)
    sqrt_y0 = np.diag(np.sqrt(y0))
    eye = np.eye(n, dtype=complex)
    # 用 solve 而非 inv，避免奇异时数值崩溃
    try:
        inv_part = np.linalg.solve(eye + s_slice, eye - s_slice)
    except np.linalg.LinAlgError:
        inv_part = np.linalg.pinv(eye + s_slice) @ (eye - s_slice)
    return sqrt_y0 @ inv_part @ sqrt_y0


# 认为构成"断崖"所需的最小落差（dB）—— 连通耦合与噪声地板之间的幅值比。
# 连通(个位数 dB)与非连通之间天然落差很大，而同一 net 内各分支彼此只差几 dB，
# 故取中间值区分二者。若全矩阵找不到这么大的落差（如理想等分功分/三通节点
# |S| 全相等），说明没有噪声地板可切 → 这些端口同属一个 net，全部判为强耦合边。
# 各判据动态范围不同：S 区分度最大、Z 最小，故默认值分开给。可在对话框覆盖。
DEFAULT_CLIFF_DB = {"S": 12.0, "Y": 10.0, "Z": 6.0}
# 兼容 QS 现有对话框；新代码使用公开名称 DEFAULT_CLIFF_DB。
_DEFAULT_CLIFF_DB = DEFAULT_CLIFF_DB


def default_cliff_db(metric: str) -> float:
    """返回 S/Y/Z 判据的默认断崖阈值。"""
    metric_u = str(metric).upper()
    if metric_u not in DEFAULT_CLIFF_DB:
        raise ValueError(f"未知拓扑判据: {metric}")
    return DEFAULT_CLIFF_DB[metric_u]


def _strong_edge_mask(score_mat: np.ndarray, min_cliff_db: float) -> np.ndarray:
    """用自适应间隙把"强耦合带"从噪声地板里切出来，返回对称邻接 bool 矩阵。

    把所有上三角 |参数| 降序排列，转 dB 后找最大跳变（断崖）：
    - 断崖 ≥ `min_cliff_db`：在断崖处（两侧几何平均）切，上方为强耦合边；
    - 找不到这么大的断崖：无噪声地板，所有非零耦合同属一个 net，全部成边。
    """
    n = score_mat.shape[0]
    mask = np.zeros((n, n), dtype=bool)
    if n < 2:
        return mask
    iu = np.triu_indices(n, k=1)
    vals = np.sort(score_mat[iu])[::-1]  # 降序
    pos = vals[vals > 0]
    if pos.size == 0:
        return mask
    if pos.size == 1:
        cut = pos[0] * 1e-3
    else:
        # 存在 0 耦合时，补一个地板项代表噪声基底，让"最弱连通 → 地板"成为断崖
        seq = pos
        if pos.size < vals.size:
            seq = np.append(pos, pos[-1] * 1e-3)
        db = 20.0 * np.log10(seq)
        gaps = db[:-1] - db[1:]
        k = int(np.argmax(gaps))
        if gaps[k] < min_cliff_db:
            cut = pos[-1] * 0.5          # 无显著断崖 → 全部非零耦合都算连通
        else:
            cut = float(np.sqrt(seq[k] * seq[k + 1]))
    mask = score_mat > cut
    np.fill_diagonal(mask, False)
    return mask | mask.T


def _connected_components(mask: np.ndarray) -> List[List[int]]:
    """无向图连通分量（0-based，升序）。"""
    n = mask.shape[0]
    seen = [False] * n
    comps: List[List[int]] = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        comp: List[int] = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in np.nonzero(mask[u])[0]:
                vi = int(v)
                if not seen[vi]:
                    seen[vi] = True
                    stack.append(vi)
        comps.append(sorted(comp))
    return comps


def _cluster_by_gap(score_mat: np.ndarray, min_cliff_db: float) -> List[Tuple[int, List[int]]]:
    """基于自适应间隙的连通判据 → [(hub, sorted([leaf, ...])), ...]。

    - 强耦合边的连通分量即一个 net；单端口分量不返回（归为孤立端口）。
    - size==2 → p2p，hub 取低序号端口。
    - size>=3 → 1 驱多，hub 取分量内强耦合行和最大的端口（星型中心），其余为叶子。
    """
    mask = _strong_edge_mask(score_mat, min_cliff_db)
    clusters: List[Tuple[int, List[int]]] = []
    for comp in _connected_components(mask):
        if len(comp) < 2:
            continue
        if len(comp) == 2:
            clusters.append((min(comp), [max(comp)]))
            continue
        rowsum = {u: float(sum(score_mat[u, v] for v in comp if v != u)) for u in comp}
        hub = max(comp, key=lambda u: (rowsum[u], -u))
        leaves = sorted(v for v in comp if v != hub)
        clusters.append((hub, leaves))
    return clusters


def _format_s_db(value: Optional[complex]) -> str:
    if value is None:
        return "N/A"
    mag = abs(complex(value))
    with np.errstate(divide='ignore'):
        db = 20.0 * np.log10(mag)
    return f"{db:.2f} dB"


def _format_abs(value: Optional[complex], unit: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{abs(complex(value)):.4g}{unit}"


def _fanout_note(ch: "ChannelInfo") -> str:
    """S 判据下，结构叶子数与电平反推扇出不一致时给出提示。"""
    if ch.fanout_consistent is None or ch.fanout_implied is None:
        return ""
    if ch.fanout_consistent:
        return ""
    return (
        f" ⚠ 电平反推≈{ch.fanout_implied:.1f} 路，"
        f"与结构 {ch.fanout_expected} 路不符（疑似级联/菊花链或非等分功分）"
    )


def _format_metric_value(value: Optional[complex], metric: str) -> str:
    """按判据参数渲染耦合量：S 用 dB，Y 用西门子(S)，Z 用欧姆(Ω)。"""
    if value is None:
        return "N/A"
    if metric == "S":
        return _format_s_db(value)
    if metric == "Y":
        return _format_abs(value, " S")
    if metric == "Z":
        return _format_abs(value, " Ω")
    return _format_abs(value)


# ============================================================
# 主入口
# ============================================================

def detect_topology(
    network: rf.Network,
    low_freq_ghz: float = 0.1,
    band_ghz: Optional[Tuple[float, float]] = None,
    y_threshold_siemens: float = 5e-4,
    s_threshold_db: float = -25.0,
    delay_tolerance_ns: float = 0.1,
    fanout_tol: float = 0.5,
    metric: str = "Y",
    min_cliff_db: Optional[float] = None,
) -> TopologyReport:
    """推断端口联通关系（1 驱 1 / 1 驱多）。

    `band_ghz`、`y_threshold_siemens`、`s_threshold_db`、`delay_tolerance_ns`、
    `fanout_tol` 均保留为兼容参数，本版不参与判别。连通与否改用自适应间隙判据
    （见模块 docstring）。`metric` 选择判据矩阵（"S" | "Y" | "Z"），用相对量
    区分连通 / 非连通；S 判据还会做扇出电平校验。`min_cliff_db` 是断崖最小落差，
    留空则按判据取 `_DEFAULT_CLIFF_DB` 默认值。
    """
    _ = (band_ghz, y_threshold_siemens, s_threshold_db, delay_tolerance_ns, fanout_tol)
    metric_u = (metric or "Y").upper()
    if metric_u not in ("S", "Y", "Z"):
        metric_u = "Y"
    cliff_db = _DEFAULT_CLIFF_DB[metric_u] if min_cliff_db is None else float(min_cliff_db)
    n = network.nports
    freq = network.f
    if n < 2:
        return TopologyReport(
            n_ports=n, band_ghz=(0.0, 0.0), low_freq_ghz=0.0,
            y_threshold_siemens=y_threshold_siemens,
            s_threshold_db=s_threshold_db,
            channels=[], isolated_ports=list(range(1, n + 1)),
            metric=metric_u,
        )

    target_hz = low_freq_ghz * 1e9
    low_idx = int(np.abs(freq - target_hz).argmin())
    low_used_ghz = float(freq[low_idx] / 1e9)
    z0_at_low = np.real(np.asarray(network.z0[low_idx]))
    s_mat = network.s[low_idx]
    y_mat = _s_to_y(s_mat, z0_at_low)
    try:
        z_mat = network.z[low_idx]
    except Exception:
        z_mat = None

    if metric_u == "S":
        metric_mat = s_mat
    elif metric_u == "Z":
        if z_mat is None:
            raise ValueError("无法获取 Z 矩阵，无法按 Z 参数判定拓扑。")
        metric_mat = z_mat
    else:
        metric_mat = y_mat
    score_mat = np.array(np.abs(metric_mat), dtype=float, copy=True)
    np.fill_diagonal(score_mat, 0.0)

    clusters = _cluster_by_gap(score_mat, cliff_db)
    paired_ports: set[int] = set()
    channels: List[ChannelInfo] = []
    for d_idx, rx_idxs in clusters:
        tx_port = d_idx + 1
        rx_ports = [r + 1 for r in rx_idxs]
        paired_ports.add(tx_port)
        paired_ports.update(rx_ports)
        rx_s = [complex(s_mat[d_idx, r]) for r in rx_idxs]
        rx_z = (
            [complex(z_mat[d_idx, r]) for r in rx_idxs]
            if z_mat is not None else []
        )
        rx_metric = [complex(metric_mat[d_idx, r]) for r in rx_idxs]
        topo = "p2p" if len(rx_ports) == 1 else "multi-drop"
        channels.append(ChannelInfo(
            ports=[tx_port] + rx_ports,
            tx=tx_port,
            rxs=rx_ports,
            s_value=rx_s[0] if rx_s else None,
            z_value=rx_z[0] if rx_z else None,
            rx_s_values=rx_s,
            rx_z_values=rx_z,
            rx_metric_values=rx_metric,
            topology=topo,
        ))

    # 扇出电平校验（仅 S 判据）：等分功分 |S|≈1/√N → N≈1/|S|²。
    if metric_u == "S":
        for ch in channels:
            mags = [abs(v) for v in ch.rx_s_values if abs(v) > 0]
            if not mags:
                continue
            implied = float(np.mean([1.0 / (m * m) for m in mags]))
            ch.fanout_expected = len(ch.rxs)
            ch.fanout_implied = implied
            ch.fanout_consistent = (round(implied) == len(ch.rxs))

    isolated = [p for p in range(1, n + 1) if p not in paired_ports]

    return TopologyReport(
        n_ports=n,
        band_ghz=(0.0, 0.0),
        low_freq_ghz=low_used_ghz,
        y_threshold_siemens=y_threshold_siemens,
        s_threshold_db=s_threshold_db,
        channels=channels,
        isolated_ports=isolated,
        metric=metric_u,
    )


def format_report(report: TopologyReport, file_label: str = "") -> str:
    """把 TopologyReport 渲染成可读文本，供主窗口控制台输出。

    耦合量按本次判据 `report.metric` 显示：S 用 dB、Y 用西门子(S)、Z 用欧姆(Ω)，
    标签前缀同步为 S/Y/Z；S、Z 值仍全量保留在 ChannelInfo 里供下游使用。
    联通簇用 `[port a, port b*, ...]` 列出全部成员，`*` 标记拓扑中心节点
    （T 型 / 星型分支点，不代表信号方向）。
    """
    lines: List[str] = []
    head = "=== 拓扑识别"
    if file_label:
        head += f"：{file_label}"
    head += " ==="
    lines.append(head)
    metric_label = {"S": "S 参数", "Y": "Y 参数", "Z": "Z 参数"}.get(
        report.metric, report.metric
    )
    lines.append(
        f"端口数 = {report.n_ports}，"
        f"判据 = {metric_label}，"
        f"低频探测点 = {report.low_freq_ghz:.4f} GHz："
    )
    if not report.channels:
        lines.append("未识别到任何耦合端口对。")
    for idx, ch in enumerate(report.channels, start=1):
        all_ports = sorted(set(ch.ports))
        members = ", ".join(
            f"port {p}*" if p == ch.tx else f"port {p}"
            for p in all_ports
        )
        if ch.topology == "multi-drop" and ch.tx is not None:
            s_parts = ", ".join(
                f"{report.metric}{ch.tx},{r}={_format_metric_value(mv, report.metric)}"
                for r, mv in zip(ch.rxs, ch.rx_metric_values)
            )
            note = _fanout_note(ch)
            lines.append(
                f"联通簇 {idx}（{len(all_ports)} 端口）: "
                f"[{members}]  耦合: {s_parts};{note}"
            )
            continue

        rx = ch.rxs[0] if ch.rxs else None
        if rx is not None:
            mv = ch.rx_metric_values[0] if ch.rx_metric_values else None
            lines.append(
                f"联通端口对 {idx}: [{members}]  "
                f"耦合: {report.metric}{ch.tx},{rx}={_format_metric_value(mv, report.metric)}"
            )
    if report.isolated_ports:
        lines.append(
            "孤立端口（未与任何端口形成显著耦合）：" +
            ", ".join(str(p) for p in report.isolated_ports)
        )
    return "\n".join(lines)
