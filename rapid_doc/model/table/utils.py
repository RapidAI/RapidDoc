from bs4 import BeautifulSoup
from loguru import logger
import re

def count_table_cells_physical(html_code):
    """更精确地计算表格单元格数量（考虑合并单元格）"""
    if not html_code:
        return 0
    try:
        soup = BeautifulSoup(html_code, 'html.parser')
        total = 0
        for cell in soup.find_all(['td', 'th']):
            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))
            total += rowspan * colspan
        return total
    except Exception:
        return 0


def count_text_matches(ocr_result, html_code):
    """统计OCR文字在HTML中出现的次数（去除空白）"""
    if not html_code:
        return 0
    html_text = re.sub(r'\s+', '', html_code)
    count = 0
    for text in ocr_result[1]:
        text = re.sub(r'\s+', '', text)
        if text and text in html_text:
            count += 1
    return count


def select_best_table_model(ocr_result, wired_html_code, wireless_html_code):
    try:
        wired_len = count_table_cells_physical(wired_html_code)
        wireless_len = count_table_cells_physical(wireless_html_code)
        gap_of_len = wireless_len - wired_len

        # 计算 OCR 匹配量
        wireless_text_count = count_text_matches(ocr_result, wireless_html_code)
        wired_text_count = count_text_matches(ocr_result, wired_html_code)

        # 计算空/非空单元格数量
        wireless_soup = BeautifulSoup(wireless_html_code or "", 'html.parser')
        wired_soup = BeautifulSoup(wired_html_code or "", 'html.parser')
        wireless_blank_count = sum(1 for c in wireless_soup.find_all(['td', 'th']) if not c.text.strip())
        wired_blank_count = sum(1 for c in wired_soup.find_all(['td', 'th']) if not c.text.strip())
        wireless_non_blank_count = wireless_len - wireless_blank_count
        wired_non_blank_count = wired_len - wired_blank_count

        switch_flag = False
        if wireless_non_blank_count > wired_non_blank_count:
            wired_table_scale = round(wired_non_blank_count ** 0.5)
            wired_scale_plus_2_cols = wired_non_blank_count + (wired_table_scale * 2)
            wired_scale_squared_plus_2_rows = wired_table_scale * (wired_table_scale + 2)
            if (wireless_non_blank_count + 3) >= max(wired_scale_plus_2_cols, wired_scale_squared_plus_2_rows):
                switch_flag = True

        prefer_wireless = (
            switch_flag
            or (0 <= gap_of_len <= 5 and wired_len <= round(wireless_len * 0.75))
            or (gap_of_len == 0 and wired_len <= 4)
            or (wired_text_count <= wireless_text_count * 0.6 and wireless_text_count >= 10)
        )

        return wireless_html_code if prefer_wireless else wired_html_code

    except Exception as e:
        logger.warning(f"[select_best_table_model] Exception: {e}")
        return wireless_html_code or ""

