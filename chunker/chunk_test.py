import json
from chunker.text_splitters import MarkdownTextSplitter, num_tokens_from_string
from chunker.get_bbox_page_fast import get_bbox_for_chunk, get_blocks_from_middle
from bs4 import BeautifulSoup

def clean_text_for_embedding(text: str) -> str:
    """
    清洗文本用于向量化：
    - 移除所有 <img> 标签
    - HTML 表格 → 纯文本表格（| 分隔）
    - 移除其他 HTML 标签
    - 压缩所有空白为单个空格
    """
    if not text:
        return ""

    soup = BeautifulSoup(text, "html.parser")

    # 移除所有 <img> 标签
    for img in soup.find_all("img"):
        img.decompose()  # 直接删除

    # 处理 HTML 表格
    tables = soup.find_all("table")
    for table in tables:
        rows_text = []

        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all(["td", "th"])
            cols_text = [" ".join(col.get_text(separator=" ").split()) for col in cols]
            row_text = " | ".join(cols_text)
            rows_text.append(row_text)

        table_text = "\n".join(rows_text)

        # 用纯文本表格替换 HTML 表格
        table.replace_with(table_text)

    # 移除所有剩余 HTML 标签
    text = soup.get_text(separator=" ")

    # 压缩所有空白，包括换行、tab
    text = " ".join(text.split())

    return text.strip()

if __name__ == '__main__':


    with open(r'D:\CodeProjects\doc\RapidAI\RapidDoc\output\ea6c0a89-dd49-4d72-b8c0-4e774d24d9dc\auto\ea6c0a89-dd49-4d72-b8c0-4e774d24d9dc.md', 'r', encoding='utf-8') as f:
        markdown_document = f.read()

    with open(r'D:\CodeProjects\doc\RapidAI\RapidDoc\output\ea6c0a89-dd49-4d72-b8c0-4e774d24d9dc\auto\ea6c0a89-dd49-4d72-b8c0-4e774d24d9dc_middle.json', 'r', encoding='utf-8') as f:
        middle_json_content = json.load(f)

    # 分块
    text_splitter = MarkdownTextSplitter(
        chunk_token_num=1024, min_chunk_tokens=200
    )
    chunk_list = text_splitter.split_text(markdown_document)

    max_tokens = 0
    max_chunk = None

    for chunk in chunk_list:
        tokens = num_tokens_from_string(chunk)
        if tokens > max_tokens:
            max_tokens = tokens
            max_chunk = chunk

    max_chunk_txt = clean_text_for_embedding(max_chunk)
    txt_tokens = num_tokens_from_string(max_chunk_txt)
    # 定位分块原始位置
    block_list = get_blocks_from_middle(middle_json_content)
    matched_global_indices = set()
    for i, chunk in enumerate(chunk_list):
        position_int_temp = get_bbox_for_chunk(chunk.strip(), block_list, matched_global_indices)
        print(position_int_temp)
