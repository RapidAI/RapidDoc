import numpy as np


def compute_inter(rec1, rec2):
    """
    computing intersection over rec2_area
    Args:
        rec1 (list): (x1, y1, x2, y2)
        rec2 (list): (x1, y1, x2, y2)
    Returns:
        float: Intersection over rec2_area
    """
    x1_1, y1_1, x2_1, y2_1 = map(float, rec1)
    x1_2, y1_2, x2_2, y2_2 = map(float, rec2)
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    inter_width = max(0, x_right - x_left)
    inter_height = max(0, y_bottom - y_top)
    inter_area = inter_width * inter_height
    rec2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    if rec2_area == 0:
        return 0
    iou = inter_area / rec2_area
    return iou


def match_table_and_ocr(cell_box_list, ocr_dt_boxes, table_cells_flag, row_start_index):
    """
    match table and ocr

    Args:
        cell_box_list (list): bbox for table cell, 2 points, [left, top, right, bottom]
        ocr_dt_boxes (list): bbox for ocr, 2 points, [left, top, right, bottom]

    Returns:
        dict: matched dict, key is table index, value is ocr index
    """
    all_matched = []
    for k in range(len(table_cells_flag) - 1):
        matched = {}
        for i, table_box in enumerate(
            cell_box_list[table_cells_flag[k] : table_cells_flag[k + 1]]
        ):
            if len(table_box) == 8:
                table_box = [
                    np.min(table_box[0::2]),
                    np.min(table_box[1::2]),
                    np.max(table_box[0::2]),
                    np.max(table_box[1::2]),
                ]
            for j, ocr_box in enumerate(np.array(ocr_dt_boxes)):
                if compute_inter(table_box, ocr_box) > 0.7:
                    if i not in matched.keys():
                        matched[i] = [j]
                    else:
                        matched[i].append(j)
        real_len = max(matched.keys()) + 1 if len(matched) != 0 else 0
        if table_cells_flag[k + 1] < row_start_index[k + 1]:
            for s in range(row_start_index[k + 1] - table_cells_flag[k + 1]):
                matched[real_len + s] = []
        elif table_cells_flag[k + 1] > row_start_index[k + 1]:
            for s in range(table_cells_flag[k + 1] - row_start_index[k + 1]):
                matched[real_len - 1].append(matched[real_len + s])
        all_matched.append(matched)
    return all_matched


def get_html_result(
    all_matched_index: dict, ocr_contents: dict, pred_structures: list, table_cells_flag
) -> str:
    """
    Generates HTML content based on the matched index, OCR contents, and predicted structures.

    Args:
        matched_index (dict): A dictionary containing matched indices.
        ocr_contents (dict): A dictionary of OCR contents.
        pred_structures (list): A list of predicted HTML structures.

    Returns:
        str: Generated HTML content as a string.
    """
    pred_html = []
    td_index = 0
    td_count = 0
    matched_list_index = 0
    head_structure = pred_structures[0:3]
    html = "".join(head_structure)
    table_structure = pred_structures[3:-3]
    for tag in table_structure:
        matched_index = all_matched_index[matched_list_index]
        if "</td>" in tag:
            if "<td></td>" == tag:
                pred_html.extend("<td>")
            if td_index in matched_index.keys():
                if len(matched_index[td_index]) == 0:
                    continue
                b_with = False
                if (
                    "<b>" in ocr_contents[matched_index[td_index][0]]
                    and len(matched_index[td_index]) > 1
                ):
                    b_with = True
                    pred_html.extend("<b>")
                for i, td_index_index in enumerate(matched_index[td_index]):
                    content = ocr_contents[td_index_index]
                    if len(matched_index[td_index]) > 1:
                        if len(content) == 0:
                            continue
                        if content[0] == " ":
                            content = content[1:]
                        if "<b>" in content:
                            content = content[3:]
                        if "</b>" in content:
                            content = content[:-4]
                        if len(content) == 0:
                            continue
                        if i != len(matched_index[td_index]) - 1 and " " != content[-1]:
                            content += " "
                    pred_html.extend(content)
                if b_with:
                    pred_html.extend("</b>")
            if "<td></td>" == tag:
                pred_html.append("</td>")
            else:
                pred_html.append(tag)
            td_index += 1
            td_count += 1
            if (
                td_count >= table_cells_flag[matched_list_index + 1]
                and matched_list_index < len(all_matched_index) - 1
            ):
                matched_list_index += 1
                td_index = 0
        else:
            pred_html.append(tag)
    html += "".join(pred_html)
    end_structure = pred_structures[-3:]
    html += "".join(end_structure)
    return html


# def sort_table_cells_boxes(boxes):
#     """
#     Sort the input list of bounding boxes.
#
#     Args:
#         boxes (list of lists): The input list of bounding boxes, where each bounding box is formatted as [x1, y1, x2, y2].
#
#     Returns:
#         sorted_boxes (list of lists): The list of bounding boxes sorted.
#     """
#
#     boxes_sorted_by_y = sorted(boxes, key=lambda box: box[1])
#     rows = []
#     current_row = []
#     current_y = None
#     tolerance = 10
#     for box in boxes_sorted_by_y:
#         x1, y1, x2, y2 = box
#         if current_y is None:
#             current_row.append(box)
#             current_y = y1
#         else:
#             if abs(y1 - current_y) <= tolerance:
#                 current_row.append(box)
#             else:
#                 current_row.sort(key=lambda x: x[0])
#                 rows.append(current_row)
#                 current_row = [box]
#                 current_y = y1
#     if current_row:
#         current_row.sort(key=lambda x: x[0])
#         rows.append(current_row)
#     sorted_boxes = []
#     flag = [0]
#     for i in range(len(rows)):
#         sorted_boxes.extend(rows[i])
#         if i < len(rows):
#             flag.append(flag[i] + len(rows[i]))
#     return sorted_boxes, flag


def sort_table_cells_boxes(boxes, overlap_threshold=0.5):
    """
    对表格单元格的检测框进行排序 (更鲁棒版本)

    参数:
        boxes (list of lists): 输入的检测框列表，
                               每个检测框格式为 [x1, y1, x2, y2]
        overlap_threshold (float): 判断是否同行的垂直重叠率阈值，默认 0.5

    返回:
        sorted_boxes (list of lists): 按行优先、列次序排序后的检测框列表
        flag (list): 每行起始索引的标记列表，用于区分行
    """

    def is_same_row(box1, box2):
        _, y1a, _, y2a = box1
        _, y1b, _, y2b = box2
        # 计算上下边界的重叠
        overlap = max(0, min(y2a, y2b) - max(y1a, y1b))
        min_height = min(y2a - y1a, y2b - y1b)
        if min_height <= 0:
            return False
        return overlap / min_height >= overlap_threshold

    # 1. 先按 y1 排序
    boxes_sorted_by_y = sorted(boxes, key=lambda box: box[1])

    rows = []
    current_row = [boxes_sorted_by_y[0]]

    # 2. 分行
    for box in boxes_sorted_by_y[1:]:
        if is_same_row(current_row[-1], box):
            current_row.append(box)
        else:
            # 当前行结束，保存
            current_row.sort(key=lambda x: x[0])
            rows.append(current_row)
            current_row = [box]
    if current_row:
        current_row.sort(key=lambda x: x[0])
        rows.append(current_row)

    # 3. 拼接结果 & flag
    sorted_boxes = []
    flag = [0]
    for row in rows:
        sorted_boxes.extend(row)
        flag.append(flag[-1] + len(row))

    return sorted_boxes, flag

def find_row_start_index(html_list):
    """
    find the index of the first cell in each row

    Args:
        html_list (list): list for html results

    Returns:
        row_start_indices (list): list for the index of the first cell in each row
    """
    # Initialize an empty list to store the indices of row start positions
    row_start_indices = []
    # Variable to track the current index in the flattened HTML content
    current_index = 0
    # Flag to check if we are inside a table row
    inside_row = False
    # Iterate through the HTML tags
    for keyword in html_list:
        # If a new row starts, set the inside_row flag to True
        if keyword == "<tr>":
            inside_row = True
        # If we encounter a closing row tag, set the inside_row flag to False
        elif keyword == "</tr>":
            inside_row = False
        # If we encounter a cell and we are inside a row
        elif (keyword == "<td></td>" or keyword == "</td>") and inside_row:
            # Append the current index as the starting index of the row
            row_start_indices.append(current_index)
            # Set the flag to ensure we only record the first cell of the current row
            inside_row = False
        # Increment the current index if we encounter a cell regardless of being inside a row or not
        if keyword == "<td></td>" or keyword == "</td>":
            current_index += 1
    # Return the computed starting indices of each row
    return row_start_indices


def map_and_get_max(table_cells_flag, row_start_index):
    """
    Retrieve table recognition result from cropped image info, table structure prediction, and overall OCR result.

    Args:
        table_cells_flag (list): List of the flags representing the end of each row of the table cells detection results.
        row_start_index (list): List of the flags representing the end of each row of the table structure predicted results.

    Returns:
        max_values: List of the process results.
    """

    max_values = []
    i = 0
    max_value = None
    for j in range(len(row_start_index)):
        while i < len(table_cells_flag) and table_cells_flag[i] <= row_start_index[j]:
            if max_value is None or table_cells_flag[i] > max_value:
                max_value = table_cells_flag[i]
            i += 1
        max_values.append(max_value if max_value is not None else row_start_index[j])
    return max_values


def get_table_recognition_res(
    table_structure_result,
    table_cells_result,
    ocr_dt_boxes,
    ocr_texts_res,
):
    table_cells_result, table_cells_flag = sort_table_cells_boxes(table_cells_result)
    row_start_index = find_row_start_index(table_structure_result)
    table_cells_flag = map_and_get_max(table_cells_flag, row_start_index)
    table_cells_flag.append(len(table_cells_result))
    row_start_index.append(len(table_cells_result))
    matched_index = match_table_and_ocr(
        table_cells_result, ocr_dt_boxes, table_cells_flag, table_cells_flag
    )
    # ocr_texts_res = [text for text, score in ocr_texts_res]

    pred_html = get_html_result(
        matched_index, ocr_texts_res, table_structure_result, row_start_index
    )

    return pred_html, table_cells_result



def convert_points_to_boxes(dt_polys: list) -> np.ndarray:
    """
    Converts a list of polygons to a numpy array of bounding boxes.

    Args:
        dt_polys (list): A list of polygons, where each polygon is represented
                        as a list of (x, y) points.

    Returns:
        np.ndarray: A numpy array of bounding boxes, where each box is represented
                    as [left, top, right, bottom].
                    If the input list is empty, returns an empty numpy array.
    """

    if len(dt_polys) > 0:
        dt_polys_tmp = dt_polys.copy()
        dt_polys_tmp = np.array(dt_polys_tmp)
        boxes_left = np.min(dt_polys_tmp[:, :, 0], axis=1)
        boxes_right = np.max(dt_polys_tmp[:, :, 0], axis=1)
        boxes_top = np.min(dt_polys_tmp[:, :, 1], axis=1)
        boxes_bottom = np.max(dt_polys_tmp[:, :, 1], axis=1)
        dt_boxes = np.array([boxes_left, boxes_top, boxes_right, boxes_bottom])
        dt_boxes = dt_boxes.T
    else:
        dt_boxes = np.array([])
    return dt_boxes
