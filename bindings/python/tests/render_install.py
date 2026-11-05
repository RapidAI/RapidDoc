"""在无 Pillow 的安装环境导出页面渲染供宿主做像素差分。"""

import json
from pathlib import Path
import sys
from rapid_doc_rust import render_pdf_page, RapidDocError


def main():
    """导出不同 DPI/旋转/扫描页，并检查资源上限和错误后的恢复。"""
    root = Path(sys.argv[1]).resolve()
    output = root / "build/render-installed"
    output.mkdir(parents=True, exist_ok=True)
    cases = [
        ("native", root / "tests/migration/inputs/pdf_native.pdf", 0),
        ("scan", root / "tests/migration/inputs/pdf_scan.pdf", 0),
    ]
    cases += [
        (f"mixed-{index}", root / "tests/migration/inputs/pdf_mixed.pdf", index)
        for index in range(3)
    ]
    cases += [
        (f"forms-{rotation}", root / f"build/migration/geometry/forms-rot{rotation}.pdf", 0)
        for rotation in (0, 90, 180, 270)
    ]
    records = []
    for name, path, index in cases:
        for dpi in (72.0, 144.0):
            for mode in ["RGBA", "RGB"]:
                page = render_pdf_page(path, page_index=index, dpi=dpi, color_mode=mode)
                assert page.png.startswith(b"\x89PNG\r\n\x1a\n")
                case_id = f"{name}-{int(dpi)}-{mode}"
                target = output / f"{case_id}.png"
                target.write_bytes(page.png)
                records.append(
                    {
                        "id": case_id,
                        "input": str(path),
                        "page": index,
                        "dpi": dpi,
                        "mode": mode,
                        "width": page.width,
                        "height": page.height,
                        "png": str(target),
                    }
                )
    native = root / "tests/migration/inputs/pdf_native.pdf"
    for expected, options in [
        ("resource_limit", {"max_pixels": 10}),
        ("invalid_input", {"dpi": float("inf")}),
        ("invalid_input", {"page_index": 100}),
    ]:
        try:
            render_pdf_page(native, **options)
        except RapidDocError as error:
            assert error.code == expected, (error.code, expected)
        else:
            raise AssertionError(expected)
    assert render_pdf_page(native, dpi=72.0).width == 400
    (output / "manifest.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Installed wheel: {len(records)} rendered page/DPI cases, limits and error recovery passed"
    )


if __name__ == "__main__":
    main()
