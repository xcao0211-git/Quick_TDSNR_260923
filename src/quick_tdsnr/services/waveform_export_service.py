"""Optional waveform export; standard-library file writers, no GUI imports.

Remove this file and restart to disable the export UI. Numerical routines remain
in sipi_sparam_core. Exports XLSX and tab-separated scientific-notation TXT.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import chain
import math
import os
from pathlib import Path
import re
import tempfile
from threading import Thread
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from sipi_sparam_core.time_response import trapezoidal_pulse
from quick_tdsnr.services.sample_analysis_service import SampleAnalysisService
from quick_tdsnr.services.snr_analysis_service import SNRAnalysisService


EXCEL_ROWS = 1_048_576
EXCEL_COLUMNS = 16_384
XML_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NORMALIZATION = "输入为峰值 1 的归一化源激励；输出为对应归一化响应，不是实测 Tx 端电压。"


@dataclass(frozen=True)
class ExportBatch:
    time_ps: np.ndarray
    input_waveform: np.ndarray
    outputs: tuple[np.ndarray, ...]
    signal_names: tuple[str, ...]
    metadata: dict


def prepare_waveforms(waveforms, settings) -> ExportBatch:
    """Validate generation provenance, then freeze the exact plotted samples."""
    waveforms = tuple(waveforms)
    if not waveforms:
        raise ValueError("请先生成时域波形")
    SNRAnalysisService.validate_settings(settings)
    settings_hash = SampleAnalysisService.settings_hash(settings)
    time_ps = np.arange(settings.n_points, dtype=float) * settings.dt_ps
    if not np.all(np.isfinite(time_ps)) or np.any(np.diff(time_ps) <= 0):
        raise ValueError("时间轴必须有限且严格递增")
    outputs, names, signals = [], ["input_norm"], []
    for index, result in enumerate(waveforms, 1):
        if result.settings_hash != settings_hash:
            raise ValueError("波形参数与生成记录不一致，请重新生成波形")
        if result.tx_port < 1 or result.rx_port < 1:
            raise ValueError("端口编号必须从 1 开始")
        if not np.array_equal(np.asarray(result.time_ps), time_ps):
            raise ValueError("波形时间轴与生成参数不一致，请重新生成波形")
        output = np.array(result.waveform, dtype=float, copy=True)
        if output.shape != time_ps.shape or not np.all(np.isfinite(output)):
            raise ValueError("输出波形必须是与时间轴等长的有限数值数组")
        name = f"out{index:04d}_tx{result.tx_port}_rx{result.rx_port}"
        output.setflags(write=False)
        outputs.append(output)
        names.append(name)
        signals.append({
            "signal": name, "sample_id": result.sample_id, "label": result.label,
            "tx_port": result.tx_port, "rx_port": result.rx_port,
            "settings_hash": result.settings_hash, "warnings": list(result.warnings),
        })
    pulse = trapezoidal_pulse(settings.n_points, settings.dt_ps * 1e-12,
                              settings.rise_time_ps, settings.ui_ps)
    time_ps.setflags(write=False)
    pulse.setflags(write=False)
    return ExportBatch(time_ps, pulse, tuple(outputs), tuple(names), {
        "format": "quick_tdsnr_waveform_export_v1", "time_unit": "s",
        "amplitude": "normalized", "normalization": NORMALIZATION,
        "settings": asdict(settings), "signals": signals,
    })


def _xml_text(value) -> str:
    # XML 1.0 disallows control characters even inside escaped strings.
    return escape(re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]", "", str(value))[:32767])


def _column(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _sheet(archive, number, rows, columns):
    with archive.open(f"xl/worksheets/sheet{number}.xml", "w") as stream:
        def write(text):
            stream.write(text.encode("utf-8"))
        write(f'<?xml version="1.0" encoding="UTF-8"?><worksheet xmlns="{XML_NS}">')
        write('<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" '
              'activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>')
        write(f'<cols><col min="1" max="{columns}" width="24" customWidth="1"/></cols><sheetData>')
        for row_index, row in enumerate(rows, 1):
            write(f'<row r="{row_index}">')
            for col_index, value in enumerate(row, 1):
                reference = f"{_column(col_index)}{row_index}"
                if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
                    if not math.isfinite(float(value)):
                        raise ValueError("Excel 数据不能包含 NaN 或 Infinity")
                    style = 2
                    write(f'<c r="{reference}" s="{style}"><v>{float(value):.16g}</v></c>')
                else:
                    # inlineStr deliberately prevents sample names starting '=' becoming formulas.
                    write(f'<c r="{reference}" t="inlineStr" s="{1 if row_index == 1 else 0}">'
                          f'<is><t xml:space="preserve">{_xml_text(value)}</t></is></c>')
            write('</row>')
        write('</sheetData></worksheet>')


def _write_xlsx(path: Path, batch: ExportBatch):
    outputs_per_sheet = EXCEL_COLUMNS - 3
    rows_per_sheet = EXCEL_ROWS - 1
    sheet_names = []
    with ZipFile(path, "w", ZIP_DEFLATED, allowZip64=True) as archive:
        for column_start in range(0, len(batch.outputs), outputs_per_sheet):
            output_group = batch.outputs[column_start:column_start + outputs_per_sheet]
            for row_start in range(0, len(batch.time_ps), rows_per_sheet):
                number = len(sheet_names) + 1
                sheet_names.append(f"波形数据{number}")
                header = ("time_s", "time_ps", "input_norm", *batch.signal_names[1 + column_start:1 + column_start + len(output_group)])
                end = min(row_start + rows_per_sheet, len(batch.time_ps))
                rows = ((float(batch.time_ps[i]) * 1e-12, float(batch.time_ps[i]),
                         float(batch.input_waveform[i]), *(float(output[i]) for output in output_group))
                        for i in range(row_start, end))
                _sheet(archive, number, chain((header,), rows), len(header))
        metadata_rows = [("参数", "值"), ("幅值说明", NORMALIZATION),
                         ("时间单位", "time_s：秒；time_ps：皮秒"),
                         *batch.metadata["settings"].items(),
                         ("输出信号", "样本名称", "sample_id", "Tx", "Rx", "settings_hash", "warnings")]
        metadata_rows.extend((item["signal"], item["label"], item["sample_id"], item["tx_port"],
                              item["rx_port"], item["settings_hash"], "；".join(item["warnings"]))
                             for item in batch.metadata["signals"])
        for start in range(0, len(metadata_rows), EXCEL_ROWS - 1):
            sheet_names.append("参数说明" if start == 0 else f"参数说明{start}")
            _sheet(archive, len(sheet_names), metadata_rows[start:start + EXCEL_ROWS - 1], 7)
        archive.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            + ''.join(f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
                      for i in range(1, len(sheet_names) + 1)) + '</Types>')
        archive.writestr("_rels/.rels", f'<Relationships xmlns="{REL_NS}"><Relationship Id="rId1" '
            f'Type="{OFFICE_REL}/officeDocument" Target="xl/workbook.xml"/></Relationships>')
        archive.writestr("xl/workbook.xml", f'<workbook xmlns="{XML_NS}" xmlns:r="{OFFICE_REL}"><sheets>'
            + ''.join(f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>' for i, name in enumerate(sheet_names, 1))
            + '</sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", f'<Relationships xmlns="{REL_NS}">'
            + ''.join(f'<Relationship Id="rId{i}" Type="{OFFICE_REL}/worksheet" Target="worksheets/sheet{i}.xml"/>'
                      for i in range(1, len(sheet_names) + 1))
            + f'<Relationship Id="styles" Type="{OFFICE_REL}/styles" Target="styles.xml"/></Relationships>')
        archive.writestr("xl/styles.xml", f'<styleSheet xmlns="{XML_NS}">'
            '<numFmts count="1"><numFmt numFmtId="164" formatCode="0.000E+00"/></numFmts>'
            '<fonts count="2"><font><sz val="11"/><name val="Calibri"/></font>'
            '<font><b/><color rgb="FFFFFFFF"/><sz val="11"/><name val="Calibri"/></font></fonts>'
            '<fills count="3"><fill><patternFill patternType="none"/></fill>'
            '<fill><patternFill patternType="gray125"/></fill>'
            '<fill><patternFill patternType="solid"><fgColor rgb="FF24476A"/><bgColor indexed="64"/></patternFill></fill></fills>'
            '<borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs>'
            '<cellXfs count="3"><xf fontId="0" fillId="0" borderId="0" xfId="0"/>'
            '<xf fontId="1" fillId="2" borderId="0" xfId="0" applyFill="1" applyFont="1"/>'
            '<xf fontId="0" fillId="0" borderId="0" xfId="0" numFmtId="164" applyNumberFormat="1"/></cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>')


def _write_txt(path: Path, batch: ExportBatch):
    """One seconds-only time column; all data fields have three decimal places."""
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("\t".join(("time_s", *batch.signal_names)) + "\n")
        for index, time_ps in enumerate(batch.time_ps):
            values = chain((float(time_ps) * 1e-12, float(batch.input_waveform[index])),
                           (float(output[index]) for output in batch.outputs))
            stream.write("\t".join(f"{value:.3E}" for value in values) + "\n")


def export_waveforms(path, batch: ExportBatch) -> tuple[Path, ...]:
    """Stage the file before atomically replacing its destination."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".xlsx", ".txt"}:
        raise ValueError("请选择 .xlsx 或 .txt 文件")
    temporary = []
    def stage(destination):
        fd, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
        os.close(fd)
        temporary.append(Path(name))
        return Path(name)
    try:
        staged = stage(path)
        if suffix == ".xlsx":
            _write_xlsx(staged, batch)
        else:
            _write_txt(staged, batch)
        os.replace(staged, path)
        return (path,)
    finally:
        for item in temporary:
            item.unlink(missing_ok=True)


class WaveformExportController:
    """UI adapter using injected existing widgets/signals; no Qt dependency here."""
    def __init__(self, panel, controls, button_type, file_dialog, layout_type):
        self.panel, self.file_dialog = panel, file_dialog
        self.button = button_type("导出波形…")
        self.button.setEnabled(False)
        self.button.setToolTip("导出完整输入激励和输出波形：Excel / TXT（三位小数科学计数法）")
        controls.removeWidget(panel.plot_button)
        actions = layout_type()
        actions.addWidget(panel.plot_button)
        actions.addWidget(self.button)
        controls.addLayout(actions, 5, 3)
        self.button.clicked.connect(self.choose_export)
        panel.export_completed.connect(self.finish_export)
        self._waveforms = ()
        self._request = None
        self._dirty = True
        self._computing = False
        self._exporting = False
        for edit in (panel.ui_edit, panel.rise_edit, panel.dt_edit, panel.points_edit,
                     panel.pre_edit, panel.post_edit, panel.half_ui_edit):
            edit.textChanged.connect(self.invalidate)
        panel.tx_spin.valueChanged.connect(self.invalidate)
        panel.rx_spin.valueChanged.connect(self.invalidate)
        panel.main_combo.currentIndexChanged.connect(self.invalidate)

    def invalidate(self, *_args):
        had_data = bool(self._waveforms)
        self._dirty = True
        self._waveforms = ()
        self.button.setEnabled(False)
        if had_data and not self._exporting:
            self.panel.status_label.setText("参数或样本已改变，请重新生成波形后导出")

    def begin(self, request):
        tx, rx, settings = request
        self._request = (tx, rx, replace(settings))
        self._dirty = False
        self._waveforms = ()
        self.button.setEnabled(False)

    def receive(self, waveforms):
        if self._request is not None and not self._dirty:
            self._waveforms = tuple(waveforms)
        self.set_busy(self._computing)

    def set_busy(self, busy):
        self._computing = busy
        self.button.setEnabled(bool(self._waveforms) and not self._dirty and not busy and not self._exporting)
        if self._exporting:
            self.panel.plot_button.setEnabled(False)

    def choose_export(self):
        if self._dirty or not self._waveforms or self._computing or self._exporting:
            return
        try:
            if self.panel.request() != self._request:
                self.invalidate()
                return
            path, selected = self.file_dialog.getSaveFileName(
                self.panel, "导出输入和输出时域波形", "time_waveforms",
                "Excel 工作簿 (*.xlsx);;制表符分隔文本 (*.txt)")
            if not path:
                return
            destination = Path(path)
            extension = ".txt" if "*.txt" in selected else ".xlsx"
            if not destination.suffix:
                destination = destination.with_suffix(extension)
                if destination.exists():
                    raise ValueError("目标文件已存在，请输入完整文件名后确认覆盖")
            elif destination.suffix.lower() != extension:
                raise ValueError(f"文件名扩展名与所选格式不一致，请使用 {extension}")
            batch = prepare_waveforms(self._waveforms, self._request[2])
            self._exporting = True
            self.button.setEnabled(False)
            self.panel.plot_button.setEnabled(False)
            self.panel.status_label.setText("正在导出完整采样数据…")
            def run():
                try:
                    result = (True, export_waveforms(destination, batch))
                except Exception as exc:
                    result = (False, str(exc))
                try:
                    self.panel.export_completed.emit(result)
                except RuntimeError:
                    pass  # Window was closed during the export.
            Thread(target=run, name="waveform-export", daemon=True).start()
        except Exception as exc:
            self._exporting = False
            self.panel.plot_button.setEnabled(not self._computing)
            self.set_busy(self._computing)
            self.panel.status_label.setText(f"导出失败：{exc}")

    def finish_export(self, result):
        self._exporting = False
        self.panel.plot_button.setEnabled(not self._computing)
        self.set_busy(self._computing)
        success, payload = result
        message = "导出完成：" + "；".join(str(path) for path in payload) if success else f"导出失败：{payload}"
        self.panel.status_label.setText(message)
