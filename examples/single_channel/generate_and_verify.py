"""Generate an analytic single-ended 2-port and verify the existing tool pipeline.

Run from project root: python examples/single_channel/generate_and_verify.py
No application algorithms are changed. No GUI modules are imported.
"""
from pathlib import Path
import json
import os

os.environ.setdefault("SKRF_PLOT_ENV", "none")

import numpy as np
from quick_tdsnr.domain.project_models import (
    ConfirmedMapping, LineEndpoint, ProjectConfig, ProjectInput, TimeDomainSettings,
)
from quick_tdsnr.services.input_preflight_service import InputPreflightService
from quick_tdsnr.services.renormalization_job import RenormalizationJob
from quick_tdsnr.services.snr_analysis_service import SNRAnalysisService
from quick_tdsnr.services.sweep_planning_service import SweepPlanningService
from quick_tdsnr.services.topology_inference_service import TopologyInferenceService
from sipi_sparam_core.time_response import trapezoidal_pulse


def main():
    destination = Path(__file__).resolve().parent
    path = destination / "single_channel_50ohm_500ps_6dB.s2p"
    frequency = np.arange(5001, dtype=float) * 1e8
    transmission = 0.5 * np.exp(-2j * np.pi * frequency * 500e-12)

    # Independent circuit golden reference: symmetric resistive T pad + line.
    # K=2: series arms = Z0*(K-1)/(K+1); shunt = 2*Z0*K/(K*K-1).
    series = np.array([[1.0, 50.0 / 3.0], [0.0, 1.0]])
    shunt = np.array([[1.0, 0.0], [3.0 / 200.0, 1.0]])
    pad = series @ shunt @ series
    theta = 2 * np.pi * frequency * 500e-12
    line = np.empty((len(frequency), 2, 2), dtype=complex)
    line[:, 0, 0] = line[:, 1, 1] = np.cos(theta)
    line[:, 0, 1] = 50j * np.sin(theta)
    line[:, 1, 0] = 1j * np.sin(theta) / 50
    circuit = pad @ line
    a, b, c, d = (circuit[:, 0, 0], circuit[:, 0, 1],
                   circuit[:, 1, 0], circuit[:, 1, 1])
    denominator = a + b / 50 + c * 50 + d
    np.testing.assert_allclose(2 / denominator, transmission, atol=1e-13)
    np.testing.assert_allclose((a + b / 50 - c * 50 - d) / denominator, 0, atol=1e-13)
    np.testing.assert_allclose((-a + b / 50 - c * 50 + d) / denominator, 0, atol=1e-13)
    np.testing.assert_allclose(2 * (a*d - b*c) / denominator, transmission, atol=1e-13)

    # Touchstone 1.x two-port column order is S11, S21, S12, S22.
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("! Synthetic analytic benchmark, not measured PCB data.\n")
        stream.write("! 50 ohm matched reciprocal attenuator, |S21|=0.5, delay=500 ps.\n")
        stream.write("! Port[1] = TX\n! Port[2] = RX\n# Hz S RI R 50\n")
        for freq, value in zip(frequency, transmission):
            stream.write(f"{freq:.12e} 0 0 {value.real:.16e} {value.imag:.16e} "
                         f"{value.real:.16e} {value.imag:.16e} 0 0\n")

    preflight = InputPreflightService()
    batch = preflight.inspect_files([path])
    assert batch.compatible, batch.errors
    network = preflight.get_network(path)
    assert network.nports == 2 and len(network.f) == 5001
    assert tuple(network.port_names) == ("TX", "RX")
    np.testing.assert_allclose(network.s[:, 1, 0], transmission, atol=1e-14)
    np.testing.assert_allclose(network.s, network.s.transpose(0, 2, 1), atol=1e-14)
    assert float(np.linalg.svd(network.s, compute_uv=False).max()) <= 0.500000000001

    topology = TopologyInferenceService().analyse(preflight.network_map(batch), low_freq_ghz=0.1)
    assert topology.recommended_proposal is not None
    assert topology.recommended_proposal.rows == ((1, 2),)
    mapping = ConfirmedMapping(((1, 2),), ("family1", "family2"), "family1")
    config = ProjectConfig(
        inputs=(ProjectInput(str(path), batch.files[0].sha256),), mapping=mapping,
        target_families=("family2",),
        resistance_candidates={"family1": (50.0,), "family2": (50.0,)},
        cio_pf={"family1": 0.0, "family2": 0.0},
        topology_metric="S", topology_frequency_ghz=0.1,
    )
    plan = SweepPlanningService().build_plan(config)
    run = RenormalizationJob().run(config, plan, write_outputs=False, keep_networks=True)
    assert run.manifest.failed == 0 and run.manifest.in_memory == 1
    renormalized = next(iter(run.networks.values()))
    np.testing.assert_allclose(renormalized.s, network.s, atol=1e-13)

    settings = TimeDomainSettings(100.0, 20.0, 1.0, 4096,
                                  "half_height_center", 3, 20, 0.1)
    result = SNRAnalysisService().analyse_victim(
        renormalized, LineEndpoint(1, 1, 2, "Single channel"), (), settings)
    assert not result.warnings, result.warnings
    # Independent PWL golden waveform with equal 20 ps rise/fall times.
    time_ps = np.arange(4096, dtype=float)
    knots = [0.0, 20.0, 100.0, 120.0]
    levels = [0.0, 1.0, 1.0, 0.0]
    source = np.interp(time_ps, knots, levels, left=0.0, right=0.0)
    golden = 0.25 * np.interp(time_ps - 500.0, knots, levels, left=0.0, right=0.0)
    np.testing.assert_allclose(trapezoidal_pulse(4096, 1e-12, 20, 100), source, atol=1e-14)
    np.testing.assert_allclose(result.direct_waveform, golden, atol=1e-11, rtol=0)
    assert abs(result.signal - 0.25) < 1e-11
    assert result.xtalk_noise == 0 and result.direct_noise < 1e-11
    np.savetxt(destination / "reference_waveform.csv",
               np.column_stack((time_ps * 1e-12, time_ps, source, golden,
                                result.direct_waveform, result.direct_waveform - golden)),
               delimiter=",", comments="", fmt="%.16e",
               header="time_s,time_ps,source_V,analytic_rx_V,tool_rx_V,tool_minus_analytic_V")
    summary = {
        "sample": path.name, "sha256": batch.files[0].sha256,
        "frequency_points": 5001, "fmax_hz": 5e11, "df_hz": 1e8,
        "topology_recommended_metric": topology.recommended_metric,
        "mapping_rows": topology.recommended_proposal.rows,
        "topology_warnings": topology.warnings,
        "ui_ps": 100, "rise_ps": 20, "dt_ps": 1, "fft_points": 4096,
        "signal_V": result.signal, "main_time_ps": result.main_time_ps,
        "direct_noise_V": result.direct_noise, "xtalk_noise_V": result.xtalk_noise,
        "max_waveform_error_V": float(np.max(np.abs(result.direct_waveform - golden))),
        "analytic_S21_db": float(20*np.log10(0.5)),
        "analytic_VTF21_db": float(20*np.log10(0.25)),
        "hspice_executed": False,
    }
    (destination / "validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
