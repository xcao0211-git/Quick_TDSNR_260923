"""电压传输函数（VTF）算法。

本模块复现频域分析中的 VTF 定义，并把原先散落在 UI 绘图代码中的公式集中到
纯函数层。这里采用 traveling-wave S 参数，且源端、负载端均匹配各自参考阻抗：

    H_v = V_load / V_source = 0.5 * S_out,in * sqrt(Z_load / Z_source)

其中 ``V_source`` 是戴维南源的开路电压，因此匹配源首先产生 1/2 分压。该定义下，
等阻抗、无损、完全匹配的直通通道 VTF 为 0.5，即 -6.0206 dB。
"""

from __future__ import annotations

import numpy as np


def voltage_transfer(s_out_in, z_source, z_load) -> np.ndarray:
    """由 ``S_out,in`` 计算复数电压传输函数。

    参数可以是标量，也可以是可广播的频率数组。参考阻抗必须为有限、非零值；
    上层需保证 ``s_out_in`` 与该参考阻抗采用同一种 traveling-wave 定义。
    """
    s_out_in = np.asarray(s_out_in, dtype=complex)
    z_source = np.asarray(z_source, dtype=complex)
    z_load = np.asarray(z_load, dtype=complex)

    if np.any(~np.isfinite(z_source)) or np.any(z_source == 0):
        raise ValueError("VTF 的源端参考阻抗必须为有限、非零值")
    if np.any(~np.isfinite(z_load)) or np.any(z_load == 0):
        raise ValueError("VTF 的负载端参考阻抗必须为有限、非零值")

    return 0.5 * s_out_in * np.sqrt(z_load / z_source)


def network_voltage_transfer(network, rx_port: int, tx_port: int) -> np.ndarray:
    """从 Network 的 ``S[rx,tx]`` 和逐频点 Zref 计算 traveling-wave VTF。

    端口号为 1-based。实数参考阻抗下 power/pseudo/traveling 定义等价；存在
    复数参考阻抗时必须显式标记为 ``traveling``，避免把 power-wave S 静默套入
    traveling-wave 电压换算公式。
    """
    if not 1 <= rx_port <= network.nports or not 1 <= tx_port <= network.nports:
        raise ValueError(f"VTF 端口越界: rx={rx_port}, tx={tx_port}")
    z0 = np.asarray(network.z0, dtype=complex)
    if np.any(np.abs(z0.imag) > 1e-15) and getattr(network, "s_def", "power") != "traveling":
        raise ValueError(
            "复数频变 Zref 的 VTF 必须使用 traveling-wave S 参数；"
            f"当前 s_def={getattr(network, 's_def', None)!r}"
        )
    return voltage_transfer(
        network.s[:, rx_port - 1, tx_port - 1],
        z0[:, tx_port - 1],
        z0[:, rx_port - 1],
    )


def magnitude_db(values) -> np.ndarray:
    """把复数幅值转换为 dB；零幅值按数学定义返回 ``-inf``。"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return 20.0 * np.log10(np.abs(values))


def voltage_crosstalk_sum_db(vtf_values, axis=0) -> np.ndarray:
    """计算多个攻击源的 VTF 功率和（dB）。

    ``vtf_values`` 沿 ``axis`` 每一项代表一个攻击源：

        VTF_XTSum = 10*log10(sum_k(|H_v,k|^2))

    这等价于先求各攻击源 VTF 的平方和开根（RSS），再做 ``20*log10``。
    """
    values = np.asarray(vtf_values, dtype=complex)
    power_sum = np.sum(np.abs(values) ** 2, axis=axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(power_sum)
