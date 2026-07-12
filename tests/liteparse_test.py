# pip install liteparse
from pathlib import Path

from liteparse import LiteParse

parser = LiteParse(
ocr_enabled=False,
    output_format="markdown",   # "json" | "text" | "markdown"
    image_mode="placeholder",   # "placeholder" | "off" | "embed"
    extract_links=True,         # render [text](url) link syntax (default: True)
)
result = parser.parse(r'D:\file\text-pdf\1a4ceacef7adb0b6a0100a2dee6e1abc_origin.pdf')
print(result.text)  # rendered Markdown

# 输出 Markdown 文件路径
md_path = Path("1a4ceacef7adb0b6a0100a2dee6e1abc_origin.md")

# 写入 Markdown
md_path.write_text(result.text, encoding="utf-8")
