from pdftext.extraction import plain_text_output, dictionary_output

PDF_PATH = r"D:\file\text-pdf\img\default - 副本.pdf"
text0 = plain_text_output(PDF_PATH, sort=False, hyphens=False) # Optional arguments explained above
print(text0)

# text = dictionary_output(PDF_PATH, sort=False,  keep_chars=False) # Optional arguments explained above
#
# print(text)
