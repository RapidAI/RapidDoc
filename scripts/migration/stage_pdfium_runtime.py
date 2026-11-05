"""为本机开发 wheel 准备独立 PDFium 副本、版本和许可；不联网下载。"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from common import ROOT, digest, write_json
from license_bundle import compact


def main():
    """显式指定运行库及来源许可，写入可复核哈希，不修改来源文件。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--licenses", type=Path, required=True)
    parser.add_argument("--version-json", type=Path, required=True)
    args = parser.parse_args()
    if not args.licenses.is_dir() or not list(args.licenses.rglob("*.txt")):
        parser.error("A non-empty license directory is required")
    version = json.loads(args.version_json.read_text(encoding="utf-8"))
    names = {"win32": "rapiddoc_pdfium.dll", "linux": "librapiddoc_pdfium.so", "darwin": "librapiddoc_pdfium.dylib"}
    if sys.platform not in names:
        parser.error("Unsupported runtime packaging platform")
    target = ROOT / "bindings/python/python/rapid_doc_rust/runtime"
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.library, target / names[sys.platform])
    shutil.copytree(args.licenses, target / "licenses", dirs_exist_ok=True)
    compact(target / "licenses")
    manifest = {"platform": sys.platform, "library": names[sys.platform], "sha256": digest(args.library.read_bytes()),
                "version": version, "license_files": {p.relative_to(target).as_posix(): digest(p.read_bytes()) for p in sorted((target / "licenses").rglob("*")) if p.is_file()},
                "source": "Local pypdfium2 distribution; explicit development staging; no automatic download"}
    write_json(target / "manifest.json", manifest)
    write_json(ROOT / "packaging/pdfium-runtime-manifest.json", manifest)
    print(f"Staged runtime with licenses: {target}")


if __name__ == "__main__":
    main()
