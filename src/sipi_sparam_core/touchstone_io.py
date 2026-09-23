"""Touchstone 补充元数据识别。

skrf 的 Touchstone 解析器只识别 HFSS/Nexxim 风格的端口名注释::

    re.match(r"! Port\\[(\\d+)\\]\\s*=\\s*(.*)$", line)

要求 ``!`` 后有空格、索引用方括号、用 ``=`` 分隔。PowerSI 导出文件
常见两组端口名注释块，二者**都不匹配**上述正则，导致 ``Network.port_names``
为空、应用层弹出"端口名缺失"兜底对话框::

    顶部块（PowerSI 全局编号，与矩阵列不一定一致）::
        !Port32_0SC_P::$7N39591
        !Port33_0SC_N::$7N39597

    底部块（索引 1..N，与矩阵列严格对应——权威映射）::
        !Port 1=0CS-4$7N39591
        !Port2=0CS-5 $7N39597

本模块在 ``rf.Network`` 加载后回填 ``port_names``：优先采用底部块；
仅当底部块缺失且顶部块条目数恰好等于 nports 时，按文件出现顺序回退。
同时识别 Quick_Sparam 写出的 ``! QS_S_DEF ...``，恢复复数参考阻抗文件的
散射波定义；恢复动作不改变 S 数值。
"""

from __future__ import annotations

import re

import skrf as rf

# 底部块：!Port {N} = {name}  —— ``!`` 后可无空格，``N`` 与 ``=`` 间可无空格
_RE_INDEXED = re.compile(r"^!\s*Port\s*(\d+)\s*=\s*(.+?)\s*$")

# 顶部块：!Port{name}::${net}  —— name 不含 ``:`` / ``=`` / 起始空白
_RE_DOUBLECOLON = re.compile(r"^!\s*Port([^\s:=][^:]*?)::\$?(\S+?)\s*$")
_RE_QS_S_DEF = re.compile(r"^!\s*QS_S_DEF\s+(power|pseudo|traveling)\s*$", re.IGNORECASE)


def _read_header_comments(file_path: str) -> list[str]:
    """读取首条非注释/非空行之前的所有 ``!`` 注释行。

    PowerSI 的两组端口块都位于 ``# ... S ... R ...`` 选项行之前，
    因此遇到非 ``!`` 行即可停止。
    """
    lines: list[str] = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as fp:
            for raw in fp:
                stripped = raw.strip()
                if not stripped:
                    continue
                if stripped.startswith("!"):
                    lines.append(stripped)
                else:
                    break
    except OSError:
        return []
    return lines


def parse_powersi_port_names(file_path: str, nports: int) -> list[str] | None:
    """从 PowerSI 注释块中解析端口名。

    Args:
        file_path: Touchstone 文件路径。
        nports: 期望的端口数；用于校验解析结果是否完整。

    Returns:
        长度为 ``nports`` 的端口名列表；任一条件不满足则返回 ``None``，
        交由上层"端口名缺失"对话框兜底。
    """
    if nports <= 0:
        return None
    comments = _read_header_comments(file_path)
    if not comments:
        return None

    indexed: dict[int, str] = {}
    for line in comments:
        m = _RE_INDEXED.match(line)
        if m:
            name = m.group(2).strip()
            if name:
                indexed[int(m.group(1))] = name
    if len(indexed) == nports and set(indexed) == set(range(1, nports + 1)):
        return [indexed[i] for i in range(1, nports + 1)]

    ordered: list[str] = []
    for line in comments:
        m = _RE_DOUBLECOLON.match(line)
        if m:
            name_part = m.group(1).strip()
            net_part = m.group(2).strip()
            if name_part:
                ordered.append(f"{name_part}::${net_part}")
    if len(ordered) == nports:
        return ordered

    return None


def parse_qs_s_def(file_path: str) -> str | None:
    """读取 Quick_Sparam 写入的散射波定义元数据。"""
    for line in _read_header_comments(file_path):
        match = _RE_QS_S_DEF.match(line)
        if match:
            return match.group(1).lower()
    return None


def apply_qs_s_def_patch(network: rf.Network, file_path: str) -> bool:
    """恢复 Touchstone 本身无法标准表达的 ``Network.s_def``。"""
    s_def = parse_qs_s_def(file_path)
    if s_def is None:
        return False
    network.s_def = s_def
    return True


def apply_port_name_patch(network: rf.Network, file_path: str) -> bool:
    """当 ``network.port_names`` 为空时尝试从 PowerSI 注释回填。

    Returns:
        ``True`` 表示成功回填；``False`` 表示无需回填或解析失败。
    """
    if network.port_names:
        return False
    names = parse_powersi_port_names(file_path, network.nports)
    if names is None:
        return False
    network.port_names = names
    return True
