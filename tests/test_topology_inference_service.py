import numpy as np
import pytest
import skrf as rf

from quick_tdsnr.services.topology_inference_service import (
    AnalysisCancelled,
    TopologyInferenceService,
)


def _parallel_three_family_network(lines: int = 4) -> rf.Network:
    nports = lines * 3
    frequency = rf.Frequency(0.1, 1.0, 3, unit="GHz")
    s = np.zeros((3, nports, nports), dtype=complex)
    for line in range(lines):
        ports = (line, line + lines, line + 2 * lines)
        for left in ports:
            for right in ports:
                if left != right:
                    s[:, left, right] = 0.6
    return rf.Network(frequency=frequency, s=s, z0=50.0)


def _single_bit_network(family_count: int) -> rf.Network:
    """构造只有一个 bit、各端口对低频耦合完全无落差的网络。"""
    frequency = rf.Frequency(0.1, 1.0, 3, unit="GHz")
    s = np.full((3, family_count, family_count), 0.4, dtype=complex)
    for index in range(family_count):
        s[:, index, index] = 0.0
    return rf.Network(frequency=frequency, s=s, z0=50.0)


def test_inference_builds_rectangular_mapping_and_requires_source_confirmation():
    network = _parallel_three_family_network()
    result = TopologyInferenceService().analyse({"demo": network})
    proposal = result.recommended_proposal
    assert proposal is not None
    assert proposal.family_count == 3
    assert proposal.line_count == 4
    assert proposal.rows == ((1, 5, 9), (2, 6, 10), (3, 7, 11), (4, 8, 12))
    assert proposal.requires_source_confirmation


@pytest.mark.parametrize("family_count", [2, 4])
def test_inference_builds_mapping_for_single_bit_without_energy_cliff(family_count):
    network = _single_bit_network(family_count)

    result = TopologyInferenceService().analyse({"single_bit": network})

    proposal = result.recommended_proposal
    assert proposal is not None
    assert proposal.rows == (tuple(range(1, family_count + 1)),)
    assert proposal.family_count == family_count
    assert proposal.line_count == 1
    assert proposal.requires_source_confirmation
    assert all(
        "只识别到一条联通簇" not in evaluation.reasons
        for evaluation in result.evaluations.values()
    )


def test_inference_rejects_invalid_parameters():
    network = _parallel_three_family_network()
    service = TopologyInferenceService()
    with pytest.raises(ValueError, match="频点"):
        service.analyse({"demo": network}, low_freq_ghz=-1.0)
    with pytest.raises(ValueError, match="断崖"):
        service.analyse({"demo": network}, min_cliff_db=0.0)
    with pytest.raises(ValueError, match="频点"):
        service.analyse({"demo": network}, low_freq_ghz=float("nan"))
    with pytest.raises(ValueError, match="断崖"):
        service.analyse({"demo": network}, min_cliff_db=float("inf"))


def test_inference_honours_cancellation():
    network = _parallel_three_family_network()
    with pytest.raises(AnalysisCancelled):
        TopologyInferenceService().analyse(
            {"demo": network}, should_cancel=lambda: True
        )
