import pypdfium2 as pdfium

# 打开 PDF 文件
pdf = pdfium.PdfDocument("D:\\file\\text-pdf\\比亚迪财报.pdf")

# 遍历每一页
for i in range(len(pdf)):
    page = pdf[i]

    # 加载文本
    textpage = page.get_textpage()

    # 获取整页文本
    text = textpage.get_text_range()
    print(f"第 {i + 1} 页内容：\n{text}\n{'-' * 40}")
