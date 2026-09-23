from dataclasses import replace
import re
import subprocess
import sys
import time
from types import SimpleNamespace
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import numpy as np
import pytest

from quick_tdsnr.domain.project_models import TimeDomainSettings, TimeWaveformResult
from quick_tdsnr.services.sample_analysis_service import SampleAnalysisService
from sipi_sparam_core.time_response import pulse_response_from_transfer

export = pytest.importorskip("quick_tdsnr.services.waveform_export_service")


def example():
    settings = TimeDomainSettings(80.0, 20.0, 10.0, 32)
    frequency = np.linspace(0, 50e9, 33)
    time_ps, output = pulse_response_from_transfer(
        frequency, np.full(33, 0.5 + 0j), ui_ps=80.0, rise_time_ps=20.0,
        dt_ps=10.0, n_points=32,
    )
    result = TimeWaveformResult(
        "sample-1", "=样本<&>甲", 1, 2, SampleAnalysisService.settings_hash(settings),
        time_ps, output,
    )
    return settings, result


def read_sheet(archive, number):
    ns = {"m": export.XML_NS}
    root = ET.fromstring(archive.read(f"xl/worksheets/sheet{number}.xml"))
    return [[cell.find("m:is/m:t", ns).text if cell.get("t") == "inlineStr"
             else float(cell.find("m:v", ns).text)
             for cell in row] for row in root.findall("m:sheetData/m:row", ns)]


def test_export_input_matches_existing_half_gain_analytic_response():
    settings, result = example()
    batch = export.prepare_waveforms([result], settings)
    np.testing.assert_allclose(batch.outputs[0], 0.5 * batch.input_waveform, atol=1e-15)
    np.testing.assert_array_equal(batch.outputs[0], result.waveform)
    result.waveform[:] = 123
    assert np.max(batch.outputs[0]) == pytest.approx(0.5)


def test_fractional_excitation_export_and_response_share_exact_pulse(tmp_path):
    settings = TimeDomainSettings(108.7, 25, 12.5, 512)
    frequency = np.fft.rfftfreq(512, 12.5e-12)
    time_ps, output = pulse_response_from_transfer(
        frequency, np.full(frequency.size, 0.5 + 0j),
        ui_ps=108.7, rise_time_ps=25, dt_ps=12.5, n_points=512)
    result = TimeWaveformResult("fractional", "fractional", 1, 2,
        SampleAnalysisService.settings_hash(settings), time_ps, output)
    batch = export.prepare_waveforms([result], settings)
    np.testing.assert_allclose(batch.input_waveform[9:11], [.848, .348], atol=1e-14)
    np.testing.assert_allclose(batch.outputs[0], batch.input_waveform * .5, atol=1e-14)
    path = tmp_path / "fractional.txt"
    export.export_waveforms(path, batch)
    assert path.read_text().splitlines()[10].split("\t") == ["1.125E-10", "8.480E-01", "4.240E-01"]


def test_xlsx_contains_numeric_full_data_and_literal_metadata(tmp_path):
    settings, result = example()
    second = replace(result, sample_id="sample-2", waveform=-result.waveform)
    batch = export.prepare_waveforms([result, second], settings)
    path = tmp_path / "中文 波形.xlsx"
    assert export.export_waveforms(path, batch) == (path,)
    with ZipFile(path) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            ET.fromstring(archive.read(name))
        rows = read_sheet(archive, 1)
        assert rows[0] == ["time_s", "time_ps", "input_norm", "out0001_tx1_rx2", "out0002_tx1_rx2"]
        values = np.asarray(rows[1:])
        assert len(values) == settings.n_points
        np.testing.assert_allclose(values[:, 0], result.time_ps * 1e-12, rtol=1e-15)
        np.testing.assert_array_equal(values[:, 1], result.time_ps)
        np.testing.assert_allclose(values[:, 3], result.waveform, atol=1e-15)
        np.testing.assert_allclose(values[:, 4], -result.waveform, atol=1e-15)
        metadata = read_sheet(archive, 2)
        assert any("=样本<&>甲" in row for row in metadata)
        assert b"<f>" not in archive.read("xl/worksheets/sheet2.xml")


def test_xlsx_splits_both_axes_without_losing_points(tmp_path, monkeypatch):
    settings, result = example()
    monkeypatch.setattr(export, "EXCEL_ROWS", 16)
    monkeypatch.setattr(export, "EXCEL_COLUMNS", 4)
    batch = export.prepare_waveforms([result, replace(result, sample_id="second")], settings)
    path = tmp_path / "split.xlsx"
    export.export_waveforms(path, batch)
    with ZipFile(path) as archive:
        rows = [read_sheet(archive, i) for i in range(1, 7)]
        assert [len(part) - 1 for part in rows] == [15, 15, 2, 15, 15, 2]
        for group in (rows[:3], rows[3:]):
            np.testing.assert_array_equal([row[1] for part in group for row in part[1:]], result.time_ps)


@pytest.mark.parametrize("change,match", [
    ({"settings_hash": "stale"}, "参数"),
    ({"time_ps": np.arange(32)}, "时间轴"),
    ({"waveform": np.zeros(31)}, "等长"),
    ({"waveform": np.full(32, np.nan)}, "有限"),
    ({"tx_port": 0}, "端口"),
])
def test_rejects_stale_or_invalid_waveforms(change, match):
    settings, result = example()
    with pytest.raises(ValueError, match=match):
        export.prepare_waveforms([replace(result, **change)], settings)


def test_txt_seconds_only_scientific_precision_and_all_samples(tmp_path):
    settings, result = example()
    values = np.linspace(-0.87654321, 0.91234567, 32)
    values[:5] = [0, 1.23456789, -1.23456789, 1e30, 1e-120]
    result = replace(result, waveform=values)
    batch = export.prepare_waveforms([result, replace(result, waveform=-values)], settings)
    path = tmp_path / "中文 波形.TXT"
    assert export.export_waveforms(path, batch) == (path,)
    lines = path.read_text(encoding="ascii").splitlines()
    assert lines[0].split("\t") == ["time_s", "input_norm", "out0001_tx1_rx2", "out0002_tx1_rx2"]
    assert "time_ps" not in lines[0]
    assert len(lines) == 33
    fields = [line.split("\t") for line in lines[1:]]
    assert all(len(row) == 4 for row in fields)
    assert all(re.fullmatch(r"-?\d\.\d{3}E[+-]\d{2,3}", value) for row in fields for value in row)
    assert fields[0][0] == "0.000E+00"
    assert fields[1][0] == "1.000E-11"
    assert fields[1][2] == "1.235E+00"
    assert fields[2][2] == "-1.235E+00"
    numeric = np.asarray(fields, dtype=float)
    np.testing.assert_allclose(numeric[:, 0], result.time_ps * 1e-12, rtol=5e-4, atol=0)
    np.testing.assert_allclose(numeric[:, 1], batch.input_waveform, rtol=5e-4, atol=0)
    np.testing.assert_allclose(numeric[:, 2], values, rtol=5e-4, atol=0)
    np.testing.assert_allclose(numeric[:, 3], -values, rtol=5e-4, atol=0)
    assert list(tmp_path.iterdir()) == [path]  # No companion metadata file.


def test_xlsx_scientific_display_keeps_underlying_precision(tmp_path):
    settings, result = example()
    result = replace(result, waveform=np.full(32, 0.1234567890123456))
    path = tmp_path / "precision.xlsx"
    export.export_waveforms(path, export.prepare_waveforms([result], settings))
    ns = {"m": export.XML_NS}
    with ZipFile(path) as archive:
        styles = ET.fromstring(archive.read("xl/styles.xml"))
        number_format = styles.find("m:numFmts/m:numFmt", ns)
        assert number_format.get("formatCode") == "0.000E+00"
        numeric_style = styles.find("m:cellXfs", ns)[2]
        assert numeric_style.get("numFmtId") == number_format.get("numFmtId")
        for name in archive.namelist():
            if name.startswith("xl/worksheets/"):
                cells = ET.fromstring(archive.read(name)).findall(".//m:c", ns)
                assert all(cell.get("s") == "2" for cell in cells if cell.find("m:v", ns) is not None)
        assert read_sheet(archive, 1)[1][3] == 0.1234567890123456


@pytest.mark.parametrize("suffix", [".xlsx", ".txt"])
def test_failed_export_preserves_existing_file_and_cleans_temporary_files(tmp_path, monkeypatch, suffix):
    settings, result = example()
    path = tmp_path / ("existing" + suffix)
    path.write_bytes(b"previous file")
    def fail(*_args):
        raise OSError("disk full")
    monkeypatch.setattr(export, "_write_xlsx" if suffix == ".xlsx" else "_write_txt", fail)
    with pytest.raises(OSError, match="disk full"):
        export.export_waveforms(path, export.prepare_waveforms([result], settings))
    assert path.read_bytes() == b"previous file"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("suffix", [".xlsx", ".txt"])
def test_locked_destination_preserves_file_and_cleans_temporary(tmp_path, monkeypatch, suffix):
    settings, result = example()
    path = tmp_path / ("locked" + suffix)
    path.write_bytes(b"original")
    def fail(*_args):
        raise PermissionError("locked")
    monkeypatch.setattr(export.os, "replace", fail)
    with pytest.raises(PermissionError):
        export.export_waveforms(path, export.prepare_waveforms([result], settings))
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


def test_optional_module_absence_keeps_original_ui_available():
    # Simulate deletion in a fresh interpreter without touching the working file.
    code = '''
import sys
sys.modules["quick_tdsnr.services.waveform_export_service"] = None
from quick_tdsnr.ui.application import create_application
from quick_tdsnr.ui.pages.sample_workspace_page import TimePanel
from quick_tdsnr.ui.main_window import QuickTDSNRMainWindow
app = create_application(["optional-export-test"])
window = QuickTDSNRMainWindow()
assert window.execution_page.sample_workspace.time_panel._waveform_export is None
panel = TimePanel()
assert panel._waveform_export is None
assert panel.plot_button.isEnabled()
panel.set_busy(True)
panel.set_busy(False)
panel.clear()
window.close()
'''
    subprocess.run([sys.executable, "-c", code], check=True, timeout=45)


@pytest.mark.parametrize("suffix", [".xlsx", ".txt"])
def test_ui_export_invalidates_changes_and_ignores_late_results(tmp_path, monkeypatch, suffix):
    from quick_tdsnr.ui.application import create_application
    from quick_tdsnr.ui.pages.sample_workspace_page import TimePanel
    app = create_application(["export-test"])
    panel = TimePanel()
    panel.ui_edit.setText("80")
    panel.points_edit.setText("32")
    settings, result = example()
    controller = panel._waveform_export
    assert not controller.button.isEnabled()
    controller.begin((1, 2, settings))
    controller.set_busy(True)
    controller.receive([result])
    assert not controller.button.isEnabled()
    controller.set_busy(False)
    assert controller.button.isEnabled()
    path = tmp_path / ("ui" + suffix)
    def choose(*args):
        assert "*.txt" in args[-1] and "*.xlsx" in args[-1]
        assert "tr0" not in args[-1].lower()
        return str(path), f"(*{suffix})"
    monkeypatch.setattr(controller.file_dialog, "getSaveFileName", choose)
    controller.choose_export()
    deadline = time.monotonic() + 5
    while controller._exporting and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert path.exists() and not controller._exporting
    panel.dt_edit.setText("5")
    assert not controller.button.isEnabled()
    controller.receive([result])
    controller.set_busy(False)
    assert not controller.button.isEnabled()
    panel.close()


def test_main_window_worker_saves_generation_settings_and_enables_export(monkeypatch):
    from quick_tdsnr.ui.application import create_application
    from quick_tdsnr.ui.main_window import QuickTDSNRMainWindow
    app = create_application(["export-worker-test"])
    window = QuickTDSNRMainWindow()
    settings, result = example()
    panel = window.execution_page.sample_workspace.time_panel
    panel.ui_edit.setText("80")
    panel.points_edit.setText("32")
    monkeypatch.setattr(window._sample_analysis_service, "time_waveform", lambda *args, **kwargs: result)
    window._start_time_waveforms([SimpleNamespace(display_name="sample")], (1, 2, settings))
    worker = window._time_waveform_worker
    deadline = time.monotonic() + 5
    while (worker.isRunning() or not panel._waveform_export.button.isEnabled()) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert worker.wait(1000)
    app.processEvents()
    assert panel._waveform_export.button.isEnabled()
    assert panel._waveform_export._request[2] == settings
    assert panel._waveform_export._request[2] is not settings
    assert panel._plot_dialog.isVisible()
    window.execution_page.sample_workspace._emit_checked_samples()
    assert not panel._waveform_export.button.isEnabled()
    panel.clear()
    window.close()


def test_ui_cancel_and_implicit_extension_do_not_overwrite(tmp_path, monkeypatch):
    from quick_tdsnr.ui.application import create_application
    from quick_tdsnr.ui.pages.sample_workspace_page import TimePanel
    app = create_application(["export-cancel-test"])
    panel = TimePanel()
    panel.ui_edit.setText("80")
    panel.points_edit.setText("32")
    settings, result = example()
    controller = panel._waveform_export
    controller.begin((1, 2, settings))
    controller.receive([result])
    monkeypatch.setattr(controller.file_dialog, "getSaveFileName", lambda *_: ("", ""))
    controller.choose_export()
    assert controller.button.isEnabled() and not controller._exporting
    path = tmp_path / "existing.xlsx"
    path.write_bytes(b"keep this")
    monkeypatch.setattr(controller.file_dialog, "getSaveFileName", lambda *_: (str(path.with_suffix("")), "*.xlsx"))
    controller.choose_export()
    assert path.read_bytes() == b"keep this"
    assert "已存在" in panel.status_label.text()
    panel.close()


def test_removed_tr0_format_is_rejected_without_creating_files(tmp_path):
    settings, result = example()
    batch = export.prepare_waveforms([result], settings)
    with pytest.raises(ValueError, match="xlsx.*txt"):
        export.export_waveforms(tmp_path / "removed.tr0", batch)
    assert not list(tmp_path.iterdir())
    assert not hasattr(export, "_write_tr0")
