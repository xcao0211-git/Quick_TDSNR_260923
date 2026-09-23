"""
纯端口/频率字符串解析器，不依赖任何 UI 框架。
调用方负责把 ValueError 转换成用户提示。
"""

import re


def parse_port_input(input_str: str, type: str = 'port') -> list:
    """
    解析端口或频率输入字符串。

    支持格式:
        1:5            → [1,2,3,4,5]
        1:2:5          → [1,3,5]  (start:step:end)
        [1,3,5]        → [1,3,5]
        1 3 5          → [1,3,5]
        1，3，5        → [1,3,5]  (全角逗号)
        1:4,16:20      → [1,2,3,4,16,17,18,19,20]  (区间+列表混合, 逗号分隔)
        1:4 16:20      → [1,2,3,4,16,17,18,19,20]  (区间+列表混合, 空格分隔)

    输入被切分为若干 token（逗号/空格分隔），每个 token 单独展开：
    含冒号者按区间展开，否则视为单个数值；最终按出现顺序拼接。

    type='port' 返回 list[int]；type='freq' 返回 list[float]

    Raises:
        ValueError: 输入格式无效或为空
    """
    if not input_str or not input_str.strip():
        raise ValueError("输入为空")

    cleaned = input_str.strip().strip('[]')
    cleaned = cleaned.replace('，', ',').replace('：', ':')
    # 去掉冒号两侧的空格，使 "1 : 5" 仍能被识别成单个区间 token
    cleaned = re.sub(r'\s*:\s*', ':', cleaned)
    convert = float if type == 'freq' else int

    tokens = cleaned.replace(',', ' ').split()
    if not tokens:
        raise ValueError("输入为空")

    result: list = []
    for token in tokens:
        if ':' in token:
            result.extend(_expand_range(token, convert, type, input_str))
        else:
            try:
                result.append(convert(token))
            except ValueError:
                raise ValueError(f"输入中包含非数字内容: '{input_str}'")
    return result


def line_port_pairs(num_port: int, arrangement: str = 'inside',
                    direction: str = '正向') -> list:
    """按「端口排布方式」把 num_port 个端口划分为各 line 的直通端口对。

    与频域分析（freq_analysis）共用同一套划分规则，是「按侧 / 按线」自动划分端口对
    的单一信息源。

    arrangement:
        'inside' / '按侧排布' → 按侧排布：line i = (i, i + N//2)
            如 16 口 → (1,9),(2,10),…,(8,16)
        'inline' / '按线排布' → 按线排布：line i = (2i-1, 2i)
            如 16 口 → (1,2),(3,4),…,(15,16)
    direction:
        '正向' / 'forward' 保持 (近端, 远端)；
        '反向' / 'reverse' 交换为 (远端, 近端)。

    返回 [(p1, p2), …]，端口号从 1 开始；num_port 为奇数时末端口不参与配对。
    """
    half = int(num_port) // 2
    arr = str(arrangement).strip().lower()
    is_inline = arr.startswith('inline') or '按线' in str(arrangement)
    if is_inline:
        pairs = [(2 * i - 1, 2 * i) for i in range(1, half + 1)]
    else:
        pairs = [(i, i + half) for i in range(1, half + 1)]
    if str(direction).strip().lower() in ('反向', 'reverse'):
        pairs = [(p2, p1) for (p1, p2) in pairs]
    return pairs


def _expand_range(token: str, convert, type: str, input_str: str) -> list:
    """把单个 'start:end' 或 'start:step:end' token 展开为数值列表。"""
    try:
        parts = list(map(convert, token.split(':')))
    except ValueError:
        raise ValueError(f"冒号分隔符中包含非数字内容: '{input_str}'")

    if len(parts) == 2:
        start, end = parts
        step = 1.0 if type == 'freq' else 1
    elif len(parts) == 3:
        start, step, end = parts
    else:
        raise ValueError("冒号格式应为 start:end 或 start:step:end")

    if type == 'freq':
        result: list = []
        current = start
        while (step > 0 and current <= end + 1e-12) or (step < 0 and current >= end - 1e-12):
            result.append(round(current, 10))
            current += step
        return result
    else:
        return list(range(int(start), int(end) + 1, int(step)))
