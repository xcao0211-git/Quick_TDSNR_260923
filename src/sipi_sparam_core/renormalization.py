"""批量端口族重归一化服务。

本模块只负责编排已有算法：

1. ``parallel_rc_impedance`` 计算每个 Port_family 的 R//C 参考阻抗；
2. 以 traveling-wave 定义把阻抗不变量 Z 转换到逐频点复参考阻抗；
3. 保留 family1 与目标 family，其余端口按其 R//C 参考阻抗匹配缩并。

模块不依赖 Qt，可被桌面界面、独立脚本和测试共同调用。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
import re

import numpy as np
import skrf as rf

from .impedance import parallel_rc_impedance


QS_S_DEF = "traveling"
QS_ZREF_MODEL_COMMENT = "! QS_ZREF_MODEL parallel_rc"


@dataclass(frozen=True)
class BatchRenormalizationCase:
    """一个笛卡尔积 case 的阻抗配置。"""

    target_family: str
    resistance_ohm: dict[str, float]
    cio_pf: dict[str, float]


def count_batch_cases(
    target_families: list[str],
    resistance_candidates: dict[str, list[float]],
) -> int:
    """返回目标族与各族 R 候选值笛卡尔积的 case 数。"""
    if not target_families:
        return 0
    total = len(target_families)
    for values in resistance_candidates.values():
        total *= len(values)
    return total


def iter_batch_cases(
    family_ids: list[str],
    target_families: list[str],
    resistance_candidates: dict[str, list[float]],
    cio_pf: dict[str, float],
):
    """按稳定顺序生成所有批量重归一化 case。"""
    _validate_case_inputs(family_ids, target_families, resistance_candidates, cio_pf)
    candidate_axes = [resistance_candidates[family_id] for family_id in family_ids]
    for target_family in target_families:
        for values in product(*candidate_axes):
            yield BatchRenormalizationCase(
                target_family=target_family,
                resistance_ohm=dict(zip(family_ids, values)),
                cio_pf={family_id: float(cio_pf[family_id]) for family_id in family_ids},
            )


def validate_family_ports(family_ports: dict[str, list[int]], nports: int) -> None:
    """校验映射恰好覆盖 ``1..nports``，且端口不重复。"""
    if not family_ports:
        raise ValueError("Port_family 映射不能为空")
    flattened = [port for ports in family_ports.values() for port in ports]
    if any(not isinstance(port, int) for port in flattened):
        raise ValueError("端口序号必须是整数")
    duplicates = sorted({port for port in flattened if flattened.count(port) > 1})
    expected = set(range(1, nports + 1))
    actual = set(flattened)
    missing = sorted(expected - actual)
    out_of_range = sorted(actual - expected)
    if duplicates:
        raise ValueError(f"端口序号重复: {duplicates}")
    if missing:
        raise ValueError(f"端口映射缺少: {missing}")
    if out_of_range:
        raise ValueError(f"端口序号越界: {out_of_range}")


def renormalize_family_case(
    network: rf.Network,
    family_ports: dict[str, list[int]],
    case: BatchRenormalizationCase,
    source_family: str = "family1",
    z_matrix: np.ndarray | None = None,
) -> rf.Network:
    """处理一个 case，并返回仅保留源族与目标族的 traveling-wave Network。

    输入端口号为 1-based；入参 ``network`` 不会被修改。``z_matrix`` 可由调用方
    对同一输入预先计算一次，避免笛卡尔积中重复执行 S→Z。
    """
    validate_family_ports(family_ports, network.nports)
    family_ids = list(family_ports)
    if source_family not in family_ports:
        raise ValueError(f"源 Port_family 不存在: {source_family}")
    if case.target_family not in family_ports:
        raise ValueError(f"目标 Port_family 不存在: {case.target_family}")
    if case.target_family == source_family:
        raise ValueError("目标 Port_family 不能与源 Port_family 相同")
    _validate_case_inputs(
        family_ids,
        [case.target_family],
        {family_id: [case.resistance_ohm[family_id]] for family_id in family_ids},
        case.cio_pf,
    )

    if np.any(np.asarray(network.z0) == 0):
        raise ValueError("输入 S 参数存在 0 Ω 参考阻抗，请先修正参考阻抗")

    frequency_hz = network.frequency.f
    z0_new = np.asarray(network.z0, dtype=complex).copy()
    for family_id, ports in family_ports.items():
        impedance = parallel_rc_impedance(
            frequency_hz,
            case.resistance_ohm[family_id],
            case.cio_pf[family_id] * 1e-12,
        )
        for port in ports:
            z0_new[:, port - 1] = impedance

    invariant_z = np.asarray(network.z if z_matrix is None else z_matrix, dtype=complex)
    expected_shape = (len(frequency_hz), network.nports, network.nports)
    if invariant_z.shape != expected_shape:
        raise ValueError(f"z_matrix 形状必须为 {expected_shape}，收到 {invariant_z.shape}")
    traveling = rf.Network(
        frequency=network.frequency.copy(),
        z=invariant_z,
        z0=z0_new,
        s_def=QS_S_DEF,
    )
    keep_ports_1based = family_ports[source_family] + family_ports[case.target_family]
    keep_indices = [port - 1 for port in keep_ports_1based]
    # skrf.Network.subnetwork 当前会把 s_def 重置为默认 power；这里直接切片并显式
    # 构造 traveling-wave 网络，确保写盘/重读后的 VTF 口径不发生静默变化。
    reduced = rf.Network(
        frequency=traveling.frequency.copy(),
        s=traveling.s[:, keep_indices][:, :, keep_indices],
        z0=traveling.z0[:, keep_indices],
        s_def=QS_S_DEF,
    )
    reduced.name = network.name
    reduced.comments = network.comments
    if network.port_names:
        reduced.port_names = [network.port_names[index] for index in keep_indices]
    return reduced


def build_case_filename(
    source_name: str,
    case: BatchRenormalizationCase,
    family_ids: list[str],
    output_nports: int,
) -> str:
    """生成短而可追溯的 Touchstone 文件名。"""
    stem = Path(source_name).stem
    target_index = re.sub(r"\D", "", case.target_family) or case.target_family
    r_values = "-".join(_number_token(case.resistance_ohm[item]) for item in family_ids)
    c_values = "-".join(_number_token(case.cio_pf[item]) for item in family_ids)
    return f"{stem}_T{target_index}_R{r_values}_C{c_values}.s{output_nports}p"


def write_touchstone_with_z0(network: rf.Network, output_path: str | Path) -> Path:
    """写出逐频点复 Zref，并用 QS 注释显式保存 ``s_def``。"""
    s_def = str(getattr(network, "s_def", "power")).lower()
    z0 = np.asarray(network.z0, dtype=complex)
    if np.any(np.abs(z0.imag) > 1e-15) and s_def != QS_S_DEF:
        raise ValueError(
            "复数频变 Zref 的批量输出必须使用 traveling-wave S 参数；"
            f"当前 s_def={s_def!r}"
        )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    original_name = network.name
    if not original_name:
        network.name = path.stem
    try:
        text = network.write_touchstone(return_string=True, write_z0=True)
    finally:
        network.name = original_name
    metadata = f"! QS_S_DEF {s_def}\n{QS_ZREF_MODEL_COMMENT}\n"
    path.write_text(metadata + text, encoding="utf-8")
    return path


def _validate_case_inputs(
    family_ids: list[str],
    target_families: list[str],
    resistance_candidates: dict[str, list[float]],
    cio_pf: dict[str, float],
) -> None:
    if not family_ids:
        raise ValueError("Port_family 列表不能为空")
    if len(set(family_ids)) != len(family_ids):
        raise ValueError("Port_family 名称不能重复")
    if any(target not in family_ids for target in target_families):
        raise ValueError("目标 Port_family 不在映射中")
    for family_id in family_ids:
        values = resistance_candidates.get(family_id)
        if not values:
            raise ValueError(f"{family_id} 的 R 候选值不能为空")
        if any(not np.isfinite(value) or value <= 0 for value in values):
            raise ValueError(f"{family_id} 的 R 候选值必须为有限正数")
        capacitance = cio_pf.get(family_id)
        if capacitance is None or not np.isfinite(capacitance) or capacitance < 0:
            raise ValueError(f"{family_id} 的 Cio 必须为有限非负数")


def _number_token(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")
