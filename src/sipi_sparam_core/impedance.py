"""
端口阻抗检查与修正 — 纯函数层，无 Qt 依赖。

UI 层负责弹窗询问用户，然后调用 replace_zero_impedance()。
enforce_nonzero_impedance() 作为向后兼容别名保留在 sparam_core.py。
"""

import numpy as np
import skrf as rf


def parallel_rc_impedance(frequency_hz, resistance_ohm: float,
                          capacitance_farad: float = 0.0) -> np.ndarray:
    """计算并联 ``R // C`` 的频变复阻抗。

    ``frequency_hz`` 单位为 Hz，``resistance_ohm`` 单位为 Ω，
    ``capacitance_farad`` 单位为 F：

        Z(f) = 1 / (1/R + j*2*pi*f*C)
             = R / (1 + j*2*pi*f*R*C)

    ``C=0`` 时退化为纯电阻。该函数是端口重归一化界面与独立脚本的共同实现。
    """
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    resistance_ohm = float(resistance_ohm)
    capacitance_farad = float(capacitance_farad)

    if np.any(~np.isfinite(frequency_hz)) or np.any(frequency_hz < 0):
        raise ValueError("频率必须为有限的非负数")
    if not np.isfinite(resistance_ohm) or resistance_ohm <= 0:
        raise ValueError(f"电阻必须为有限正数，收到: {resistance_ohm}")
    if not np.isfinite(capacitance_farad) or capacitance_farad < 0:
        raise ValueError(f"电容必须为有限非负数，收到: {capacitance_farad}")

    if capacitance_farad == 0:
        return np.full(frequency_hz.shape, resistance_ohm, dtype=complex)
    omega = 2.0 * np.pi * frequency_hz
    return resistance_ohm / (1.0 + 1j * omega * resistance_ohm * capacitance_farad)


def has_zero_impedance(network: rf.Network) -> bool:
    """检查网络是否存在零阻抗端口。"""
    return bool(np.any(np.array(network.z0) == 0))


def replace_zero_impedance(network: rf.Network, z0: float) -> None:
    """将网络所有端口阻抗替换为指定值。"""
    if z0 <= 0:
        raise ValueError(f"阻抗值必须为正数，收到: {z0}")
    n_ports = network.nports
    network.z0 = np.ones((len(network.f), n_ports)) * z0
    print(f"已将全部端口阻抗设置为 {z0}Ω")


def _parse_ref_impedance(option_line: str) -> float | None:
    """从 Touchstone 选项行提取参考阻抗。

    选项行格式：``# <freq_unit> <param> <format> R <z0 ...>``，参考阻抗是 ``R``
    之后的首个数值。以下情况按 Touchstone 规范视为"未指定"（返回 None，由调用方套用默认 50Ω）：

    - ``R`` 之后没有数值（如 ``# GHz S RI R``）；
    - 整行没有 ``R`` 段（如 ``# GHz S RI``）。
    """
    tokens = option_line.lstrip('#').split()
    for i, tok in enumerate(tokens):
        if tok.upper() == 'R':
            for val in tokens[i + 1:]:
                try:
                    return float(val)
                except ValueError:
                    continue
            return None
    return None


def enforce_nonzero_z0(network_ori: rf.Network, filepath: str) -> None:
    """检查并处理阻抗矩阵中的零值，从文件头行读取参考阻抗直接修正。"""
    z0_array = np.array(network_ori.z0)
    if not np.any(z0_array == 0):
        return

    print("Zc_testing")
    new_z0 = None

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                new_z0 = _parse_ref_impedance(line)
                break

    if new_z0 is None or new_z0 <= 0:
        # 未找到 '#' 头行、R 后无数值、或声明的参考阻抗本身非正 —— 按 Touchstone 规范默认 50Ω
        new_z0 = 50.0
        print("未从文件头行解析到有效参考阻抗，使用默认参考阻抗 50Ω")

    n_ports = network_ori.nports
    network_ori.z0 = np.ones((len(network_ori.f), n_ports)) * new_z0
    print(f"检测到端口参考阻抗存在0，已将全部端口阻抗设置为 {new_z0}Ω")
