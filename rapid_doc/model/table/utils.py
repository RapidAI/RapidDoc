from bs4 import BeautifulSoup
from loguru import logger
from html import unescape
import difflib


# def select_best_table_model(ocr_result, wired_html_code, wireless_html_code):
#
#     try:
#         wired_len = count_table_cells_physical(wired_html_code)
#         wireless_len = count_table_cells_physical(wireless_html_code)
#         # 计算两种模型检测的单元格数量差异
#         gap_of_len = wireless_len - wired_len
#         # logger.debug(f"wired table cell bboxes: {wired_len}, wireless table cell bboxes: {wireless_len}")
#
#         # 使用OCR结果计算两种模型填入的文字数量
#         wireless_text_count = 0
#         wired_text_count = 0
#         for ocr_res in ocr_result:
#             if ocr_res[1] in wireless_html_code:
#                 wireless_text_count += 1
#             if ocr_res[1] in wired_html_code:
#                 wired_text_count += 1
#         # logger.debug(f"wireless table ocr text count: {wireless_text_count}, wired table ocr text count: {wired_text_count}")
#
#         # 使用HTML解析器计算空单元格数量
#         wireless_soup = BeautifulSoup(wireless_html_code, 'html.parser') if wireless_html_code else BeautifulSoup("", 'html.parser')
#         wired_soup = BeautifulSoup(wired_html_code, 'html.parser') if wired_html_code else BeautifulSoup("", 'html.parser')
#         # 计算空单元格数量(没有文本内容或只有空白字符)
#         wireless_blank_count = sum(1 for cell in wireless_soup.find_all(['td', 'th']) if not cell.text.strip())
#         wired_blank_count = sum(1 for cell in wired_soup.find_all(['td', 'th']) if not cell.text.strip())
#         # logger.debug(f"wireless table blank cell count: {wireless_blank_count}, wired table blank cell count: {wired_blank_count}")
#
#         # 计算非空单元格数量
#         wireless_non_blank_count = wireless_len - wireless_blank_count
#         wired_non_blank_count = wired_len - wired_blank_count
#         # 无线表非空格数量大于有线表非空格数量时，才考虑切换
#         switch_flag = False
#         if wireless_non_blank_count > wired_non_blank_count:
#             # 假设非空表格是接近正方表，使用非空单元格数量开平方作为表格规模的估计
#             wired_table_scale = round(wired_non_blank_count ** 0.5)
#             # logger.debug(f"wireless non-blank cell count: {wireless_non_blank_count}, wired non-blank cell count: {wired_non_blank_count}, wired table scale: {wired_table_scale}")
#             # 如果无线表非空格的数量比有线表多一列或以上，需要切换到无线表
#             wired_scale_plus_2_cols = wired_non_blank_count + (wired_table_scale * 2)
#             wired_scale_squared_plus_2_rows = wired_table_scale * (wired_table_scale + 2)
#             if (wireless_non_blank_count + 3) >= max(wired_scale_plus_2_cols, wired_scale_squared_plus_2_rows):
#                 switch_flag = True
#
#         # 判断是否使用无线表格模型的结果
#         if (
#             switch_flag
#             or (0 <= gap_of_len <= 5 and wired_len <= round(wireless_len * 0.75))  # 两者相差不大但有线模型结果较少
#             or (gap_of_len == 0 and wired_len <= 4)  # 单元格数量完全相等且总量小于等于4
#             or (wired_text_count <= wireless_text_count * 0.6 and  wireless_text_count >=10) # 有线模型填入的文字明显少于无线模型
#         ):
#             # logger.debug("fall back to wireless table model")
#             html_code = wireless_html_code
#         else:
#             html_code = wired_html_code
#
#         return html_code
#     except Exception as e:
#         logger.warning(e)
#         return wireless_html_code
#
# def count_table_cells_physical(html_code):
#     """计算表格的物理单元格数量（合并单元格算一个）"""
#     if not html_code:
#         return 0
#
#     # 简单计数td和th标签的数量
#     html_lower = html_code.lower()
#     td_count = html_lower.count('<td')
#     th_count = html_lower.count('<th')
#     return td_count + th_count

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
        # 解析失败时，退回简单计数
        html_lower = html_code.lower()
        return html_lower.count('<td') + html_lower.count('<th')


def clean_text(s):
    """去除空格、HTML转义符、控制字符"""
    if not s:
        return ""
    return unescape(s).strip().replace('\xa0', '').replace('\n', '').replace('\r', '')


def count_ocr_matches(ocr_result, html_code, fuzzy=False):
    """统计 OCR 文本在 HTML 中的匹配次数"""
    if not ocr_result or not html_code:
        return 0

    html_text = clean_text(BeautifulSoup(html_code, 'html.parser').get_text())
    count = 0
    for ocr_res in ocr_result:
        text = clean_text(ocr_res[1]) if len(ocr_res) > 1 else ""
        if not text:
            continue
        if fuzzy:
            # 模糊匹配相似度 > 0.8
            matcher = difflib.SequenceMatcher(None, text, html_text)
            if matcher.quick_ratio() > 0.8:
                count += 1
        else:
            if text in html_text:
                count += 1
    return count


def count_blank_cells(html_code):
    """统计空白单元格数量（排除仅含图片/换行）"""
    if not html_code:
        return 0
    soup = BeautifulSoup(html_code, 'html.parser')
    blank_count = 0
    for cell in soup.find_all(['td', 'th']):
        text = clean_text(cell.get_text())
        if not text and not cell.find(['img', 'input', 'svg', 'br']):
            blank_count += 1
    return blank_count


def select_best_table_model(ocr_result, wired_html_code, wireless_html_code):
    """增强版：更精确、更稳健的表格模型选择逻辑"""
    try:
        # ===== Step 1: 基础统计 =====
        wired_len = count_table_cells_physical(wired_html_code)
        wireless_len = count_table_cells_physical(wireless_html_code)
        gap_of_len = wireless_len - wired_len

        # ===== Step 2: OCR 匹配情况 =====
        wired_text_count = count_ocr_matches(ocr_result, wired_html_code)
        wireless_text_count = count_ocr_matches(ocr_result, wireless_html_code)

        # ===== Step 3: 空单元格与非空单元格 =====
        wired_blank_count = count_blank_cells(wired_html_code)
        wireless_blank_count = count_blank_cells(wireless_html_code)

        wired_non_blank = max(wired_len - wired_blank_count, 0)
        wireless_non_blank = max(wireless_len - wireless_blank_count, 0)

        # ===== Step 4: 打分模型 =====
        # 定义加权系数，可根据数据集调参
        weight_non_blank = 0.5
        weight_ocr_match = 0.4
        weight_blank_penalty = 0.2

        # 综合得分（无线 - 有线）
        score = (
            (wireless_non_blank - wired_non_blank) * weight_non_blank +
            (wireless_text_count - wired_text_count) * weight_ocr_match -
            (wireless_blank_count - wired_blank_count) * weight_blank_penalty
        )

        # ===== Step 5: 辅助启发条件 =====
        if wireless_non_blank > wired_non_blank * 1.2 and wireless_text_count >= wired_text_count * 1.1:
            score += 2  # 加强信号：无线明显更“充实”
        if 0 <= gap_of_len <= 5 and wired_len <= round(wireless_len * 0.75):
            score += 1
        if wired_len == wireless_len and wired_len <= 4:
            score += 0.5

        # ===== Step 6: 最终决策 =====
        if score > 0:
            chosen = wireless_html_code
            chosen_label = "wireless"
        else:
            chosen = wired_html_code
            chosen_label = "wired"

        logger.debug(
            f"[select_best_table_model] score={score:.2f}, "
            f"wired(len={wired_len}, text={wired_text_count}, blank={wired_blank_count}) "
            f"vs wireless(len={wireless_len}, text={wireless_text_count}, blank={wireless_blank_count}) "
            f"=> choose {chosen_label}"
        )
        return chosen or wireless_html_code or wired_html_code or ""

    except Exception as e:
        logger.warning(f"select_best_table_model failed: {e}")
        # 回退策略：选择结构更完整的表格
        try:
            wired_len = count_table_cells_physical(wired_html_code)
            wireless_len = count_table_cells_physical(wireless_html_code)
            return wireless_html_code if wireless_len >= wired_len else wired_html_code
        except Exception:
            return wireless_html_code or wired_html_code or ""
