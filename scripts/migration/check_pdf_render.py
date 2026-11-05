"""独立进程验证已安装 Rust wheel 的页面像素，不比较 PNG 压缩字节。"""

import argparse
import json
import subprocess
from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image
from common import ROOT, digest, write_json
from generate_geometry_corpus import generate


def main():
    """同版本 PDFium 与相同 DPI，逐像素要求完全相等。"""
    parser=argparse.ArgumentParser()
    parser.add_argument("--python",type=Path,required=True)
    parser.add_argument("--existing-corpus",action="store_true",help="只读现有冻结PDF，不重新生成语料")
    args=parser.parse_args()
    if not args.existing_corpus:generate()
    subprocess.run([str(args.python.resolve()),"-I",str(ROOT/"bindings/python/tests/render_install.py"),str(ROOT)],check=True,timeout=120)
    cases=json.loads((ROOT/"build/render-installed/manifest.json").read_text(encoding="utf-8"))
    assert len(cases)==36
    results=[]
    for case in cases:
        with pdfium.PdfDocument(case["input"]) as document:
            page=document[case["page"]]
            try:
                bitmap=page.render(scale=case["dpi"]/72.)
                try:expected=bitmap.to_pil().convert("RGBA")
                finally:bitmap.close()
            finally:page.close()
        with Image.open(case["png"]) as image:
            assert image.mode==case['mode']
            actual=image.convert("RGBA")
        assert actual.size==expected.size==(case["width"],case["height"]),case["id"]
        assert actual.tobytes()==expected.tobytes(),case["id"]
        results.append({"id":case["id"],"rgba_sha256":digest(actual.tobytes()),"width":actual.width,"height":actual.height})
    write_json(ROOT/"build/pdf-render-report.json",{"scope":"White background, page annotations, no user form values; identical RGBA pixels, 72/144 DPI","cases":results})
    print(f"PDF raster differential: 18 RGBA + 18 RGB cases exactly match old rendered pixels")


if __name__=="__main__":main()
