from io import BytesIO
# from IPython.display import display_html

from img2table.document import PDF

# PaddleOCR
# from img2table.ocr import PaddleOCR
# paddle_ocr = PaddleOCR(lang="en", kw={"use_dilation": True})

# pdf = PDF(src="D:\\file\\text-pdf\\示例1-论文模板.pdf")
pdf = PDF(src="D:\\file\\text-pdf\\比亚迪财报.pdf")

# Extract tables
extracted_tables = pdf.extract_tables(ocr=None,
                                      implicit_rows=False,
                                      borderless_tables=False,
                                      min_confidence=50)

print(extracted_tables)

for page, tables in extracted_tables.items():
    for idx, table in enumerate(tables):
        print(table.html_repr(title=f"Page {page + 1} - Extracted table n°{idx + 1}"))