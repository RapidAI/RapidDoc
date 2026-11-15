from pdftext.extraction import plain_text_output, dictionary_output

PDF_PATH = r"D:\file\text-pdf\images\song_ultra_origin_origin.pdf"
text0 = plain_text_output(PDF_PATH, sort=False, hyphens=False) # Optional arguments explained above
print(text0)

# text = dictionary_output(PDF_PATH, sort=False,  keep_chars=False) # Optional arguments explained above
#
# print(text)
