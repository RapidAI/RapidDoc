from pathlib import Path

import indra_ocr
markdown = ""
conv = indra_ocr.load(threads=4, math=False)          # mode="auto" | "ocr" | "text"
path = "../demo/pdfs/03.pdf"
for i, page in conv.convert(path):
    # print(page.source, page.timings["total_ms"])
    # print(page.markdown)                  # headings, lists, tables as HTML
    # page.blocks                           # [{"label", "bbox", "text"}] in reading order
    # page.lines                            # [Line(box, text, conf)] in reading order

    markdown += page.markdown
# 输出 Markdown 文件路径
md_path = Path("031.md")

# 写入 Markdown
md_path.write_text(markdown, encoding="utf-8")
#