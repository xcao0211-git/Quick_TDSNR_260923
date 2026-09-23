from pathlib import Path


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src" / "sipi_sparam_core"


def test_core_has_no_qt_or_matplotlib_dependency():
    forbidden = ("PyQt", "PySide", "qtpy", "matplotlib")
    for path in SOURCE_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{path} contains {token}"
