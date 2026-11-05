"""Inventory local Office imports, installed dependencies and existing fixtures; read-only."""
import argparse
import ast
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OFFICE = ROOT / "rapid_doc/_vendor/docvortex_core/analyzers/native/office"
DISTRIBUTIONS = {"PIL":"pillow", "docx":"python-docx", "pptx":"pypptx-with-oxml",
                 "bs4":"beautifulsoup4", "metafile_render":"metafile-render",
                 "resvg_py":"resvg_py"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    imports = {}
    sources = {}
    office_files = sorted(OFFICE.rglob("*.py"))
    # SVG rendering is used through this shared helper, not imported by the converters.
    files = office_files + [ROOT / "rapid_doc/_vendor/docvortex_core/foundation/_svg_raster.py"]
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8-sig"))):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            for name in names:
                top = name.split(".")[0]
                if top and top not in sys.stdlib_module_names:
                    imports.setdefault(top, set()).add(relative)
    packages = {}
    for top, users in sorted(imports.items()):
        distribution = DISTRIBUTIONS.get(top, top)
        installed = metadata.distribution(distribution)
        packages[top] = dict(distribution=distribution, version=installed.version,
                             license_expression=installed.metadata.get("License-Expression"),
                             project_urls=installed.metadata.get_all("Project-URL") or [],
                             requirements=installed.requires or [], sources=sorted(users))
    fixtures = []
    for base in (ROOT / "tests", ROOT / "demo/ms_office_docs"):
        for path in sorted(base.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx"}:
                fixtures.append(dict(path=path.relative_to(ROOT).as_posix(), bytes=path.stat().st_size,
                                     sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report = dict(status="preparation_only_no_conversion_parity_claim", office_modules=len(office_files),
                  office_lines=sum(len(path.read_text(encoding="utf-8-sig").splitlines()) for path in office_files),
                  direct_external_roots=packages, sources_sha256=sources, fixtures=fixtures,
                  scope="Static imports in Office plus SVG helper; requirements list exposes transitive dependencies,"
                        " not a complete dynamically observed dependency graph.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Office: {len(office_files)} modules; {len(packages)} external roots; {len(fixtures)} fixture paths")


if __name__ == "__main__":
    main()
