"""Regenerate synthetic exports using this checkout's actual excitation."""
from pathlib import Path
import numpy as np
from quick_tdsnr.domain.project_models import TimeDomainSettings, TimeWaveformResult
from quick_tdsnr.services.sample_analysis_service import SampleAnalysisService
from quick_tdsnr.services.waveform_export_service import prepare_waveforms, export_waveforms
from sipi_sparam_core.time_response import pulse_response_from_transfer


def main():
    output = Path(__file__).resolve().parent
    settings = TimeDomainSettings(100, 20, 2, 256)
    waveforms = []
    for index, gain in enumerate((0.5, -0.1), 1):
        frequency = np.linspace(0, 250e9, 257)
        time_ps, response = pulse_response_from_transfer(
            frequency, np.full(257, complex(gain)), ui_ps=100, rise_time_ps=20,
            dt_ps=2, n_points=256,
        )
        waveforms.append(TimeWaveformResult(
            f'demo-{index}', f'示例 {index}：固定增益 {gain}', 1, 2,
            SampleAnalysisService.settings_hash(settings), time_ps, response,
        ))
    batch = prepare_waveforms(waveforms, settings)
    for suffix in ('.xlsx', '.txt'):
        export_waveforms(output / ('waveform_example' + suffix), batch)


if __name__ == '__main__':
    main()
