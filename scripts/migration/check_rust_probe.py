"""在独立 Rust 进程中对照 PDF 字符基线，禁止混用 Python/PDFium 句柄。"""

import argparse
import json
import subprocess
from pathlib import Path
from common import ROOT, digest, load_manifest, write_json


def main():
    """验证三个 PDF 的页数、原始文字和字符数，差异以非零状态返回。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--pdfium", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "build/pdfium-probe-report.json")
    args = parser.parse_args()
    cases = [case for case in load_manifest(ROOT / "tests/migration/corpus_manifest.json") if case.get("capture") == "pdf-facts"]
    if not cases:
        raise ValueError("PDF probe corpus is empty")
    results = []
    for case in cases:
        process = subprocess.run([str(args.binary.resolve()), str(args.pdfium.resolve()), str(ROOT / case["input"])], capture_output=True, text=True, encoding="utf-8", check=True, timeout=60)
        pages = json.loads(process.stdout)
        expected = json.loads((ROOT / "tests/migration/golden" / case["expected"]).read_text(encoding="utf-8"))["pages"]
        if len(pages) != len(expected):
            raise AssertionError(f"Page count differs: {case['id']}")
        for actual, golden in zip(pages, expected):
            assert actual["page_index"] == golden["page_index"]
            assert actual["char_count"] == len(golden["chars"]), case["id"]
            assert actual["text"] == "".join(char["char"] for char in golden["chars"]), case["id"]
        results.append({"id": case["id"], "pages": len(pages), "text_and_count_match": True})
    write_json(args.output, {"pdfium_sha256": digest(args.pdfium.read_bytes()), "binary_sha256": digest(args.binary.read_bytes()),
               "scope": "Text and count only; geometry/path/image parity still pending", "cases": results})
    print(f"Rust PDFium probe: {len(results)} documents match text and character counts")


if __name__ == "__main__":
    main()
