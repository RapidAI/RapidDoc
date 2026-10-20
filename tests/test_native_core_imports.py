"""Check dependency isolation in a new interpreter, independent of global installs."""
from pathlib import Path
import subprocess
import sys
import textwrap
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def test_runtime_distributions_are_selected_only_by_extras():
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    forbidden = ("onnxruntime", "docvortex", "magika", "fasttext")
    assert not any(dep.lower().startswith(forbidden) for dep in project["dependencies"])
    extras = project["optional-dependencies"]
    assert "docvortex" not in extras
    assert extras["cpu"] == ["onnxruntime>=1.18.0"]
    assert extras["gpu"] == ["onnxruntime-gpu<=1.23.0"]


def test_office_and_pdf_core_run_with_external_models_blocked():
    script = textwrap.dedent('''
        import importlib.abc
        import sys
        from pathlib import Path
        from io import BytesIO
        blocked = {"docvortex", "magika", "fasttext", "onnxruntime"}
        class BlockExternal(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in blocked:
                    raise AssertionError("Unexpected dependency: " + fullname)
        sys.meta_path.insert(0, BlockExternal())
        from rapid_doc.backend.office.office_analyze import office_analyze
        from rapid_doc.model.docvortex_adapter import analyze_docvortex_layout, recover_docvortex_table
        from rapid_doc._vendor.docvortex_core._compute_backend import backend_info
        from reportlab.pdfgen.canvas import Canvas
        for path in Path("tests/fixtures/docvortex_migration").iterdir():
            if path.suffix in {".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}:
                middle, raw = office_analyze(path.read_bytes())
                assert raw and middle["pdf_info"]
        out = BytesIO()
        canvas = Canvas(out, pagesize=(200, 160))
        for x in (20, 70, 120, 170): canvas.line(x, 50, x, 140)
        for y in (50, 80, 110, 140): canvas.line(20, y, 170, y)
        for r in range(3):
            for c in range(3): canvas.drawString(25 + c * 50, 125 - r * 30, f"{r}{c}")
        canvas.save()
        data = out.getvalue()
        assert analyze_docvortex_layout(data)[0]
        result = recover_docvortex_table({"_source_pdf_bytes": data, "_source_page_index": 0}, (20, 20, 170, 110))
        assert result and "22" in result.html
        assert backend_info()["backend"] == "python"
        assert not blocked.intersection(sys.modules)
    ''')
    result = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).parents[1], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
