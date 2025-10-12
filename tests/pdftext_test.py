from pdftext.extraction import plain_text_output, dictionary_output

PDF_PATH = "D:\\file\\text-pdf\\示例1-论文模板.pdf"
text0 = plain_text_output(PDF_PATH, sort=False, hyphens=False) # Optional arguments explained above
print(text0)

text = dictionary_output(PDF_PATH, sort=False,  keep_chars=False) # Optional arguments explained above

print(text)
