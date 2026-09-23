"""QS_domain/port_parser.py 的单元测试（不需要 QApplication）。"""

import pytest
from sipi_sparam_core.ports import parse_port_input, line_port_pairs


class TestParsePortInputPort:
    def test_space_separated(self):
        assert parse_port_input("1 2 3") == [1, 2, 3]

    def test_colon_range(self):
        assert parse_port_input("1:5") == [1, 2, 3, 4, 5]

    def test_colon_step(self):
        assert parse_port_input("1:2:7") == [1, 3, 5, 7]

    def test_comma_separated(self):
        assert parse_port_input("1,3,5") == [1, 3, 5]

    def test_fullwidth_comma(self):
        assert parse_port_input("1，3，5") == [1, 3, 5]

    def test_fullwidth_colon(self):
        assert parse_port_input("1：5") == [1, 2, 3, 4, 5]

    def test_brackets(self):
        assert parse_port_input("[1 2 3]") == [1, 2, 3]

    def test_single_port(self):
        assert parse_port_input("3") == [3]

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_port_input("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            parse_port_input("   ")

    def test_non_numeric_raises(self):
        with pytest.raises(ValueError):
            parse_port_input("a b c")

    def test_bad_colon_raises(self):
        with pytest.raises(ValueError):
            parse_port_input("1:2:3:4")

    def test_mixed_range_comma(self):
        assert parse_port_input("1:4,16:20") == [1, 2, 3, 4, 16, 17, 18, 19, 20]

    def test_mixed_range_space(self):
        assert parse_port_input("1:4 16:20") == [1, 2, 3, 4, 16, 17, 18, 19, 20]

    def test_mixed_range_and_single(self):
        assert parse_port_input("1:4,7,9:11") == [1, 2, 3, 4, 7, 9, 10, 11]

    def test_spaces_around_colon(self):
        assert parse_port_input("1 : 5") == [1, 2, 3, 4, 5]


class TestParsePortInputFreq:
    def test_float_space(self):
        result = parse_port_input("1.0 2.5 3.0", type='freq')
        assert result == pytest.approx([1.0, 2.5, 3.0])

    def test_freq_range(self):
        result = parse_port_input("1.0:0.5:2.0", type='freq')
        assert result == pytest.approx([1.0, 1.5, 2.0])

    def test_freq_simple_range(self):
        result = parse_port_input("0:3", type='freq')
        assert result == pytest.approx([0.0, 1.0, 2.0, 3.0])


class TestLinePortPairs:
    """按侧 / 按线 自动划分端口对（频域 / 时域共用）。"""

    def test_inside_16(self):
        assert line_port_pairs(16, "inside") == [
            (1, 9), (2, 10), (3, 11), (4, 12),
            (5, 13), (6, 14), (7, 15), (8, 16)]

    def test_inline_16(self):
        assert line_port_pairs(16, "inline") == [
            (1, 2), (3, 4), (5, 6), (7, 8),
            (9, 10), (11, 12), (13, 14), (15, 16)]

    def test_inside_4_default_arrangement(self):
        # 默认 arrangement='inside'
        assert line_port_pairs(4) == [(1, 3), (2, 4)]

    def test_direction_reverse_swaps_pair(self):
        assert line_port_pairs(4, "inside", "反向") == [(3, 1), (4, 2)]
        assert line_port_pairs(4, "inside", "reverse") == [(3, 1), (4, 2)]

    def test_direction_forward_default(self):
        assert line_port_pairs(4, "inside", "正向") == [(1, 3), (2, 4)]

    def test_chinese_arrangement_labels(self):
        assert line_port_pairs(4, "按侧排布") == [(1, 3), (2, 4)]
        assert line_port_pairs(4, "按线排布") == [(1, 2), (3, 4)]

    def test_odd_port_count_drops_last(self):
        # 5 端口 → 2 个 line，端口 5 不参与配对
        assert line_port_pairs(5, "inside") == [(1, 3), (2, 4)]
        assert line_port_pairs(5, "inline") == [(1, 2), (3, 4)]

    def test_two_port(self):
        assert line_port_pairs(2, "inside") == [(1, 2)]
        assert line_port_pairs(2, "inline") == [(1, 2)]
