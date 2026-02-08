# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html
import itertools
import math
import re
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from pydantic import BaseModel, computed_field, model_validator

from rapid_doc.utils.ocr_utils import points_to_bbox


def paint_token(image, box, token_str):
    """
    Fill a rectangular area in the image with a white background and write the given token string.

    Args:
        image (np.ndarray): Image to paint on.
        box (tuple): (x1, y1, x2, y2) coordinates of rectangle.
        token_str (str): Token string to write.

    Returns:
        np.ndarray: Modified image.
    """
    import cv2

    def get_optimal_font_scale(text, fontFace, square_size, fill_ratio=0.9):
        # the scale is greater than 0.2 and less than 10,
        # suitable for square_size is greater than 30 and less than 1000
        left, right = 0.2, 10
        optimal_scale = left
        # search the optimal font scale
        while right - left > 1e-2:
            mid = (left + right) / 2
            (w, h), _ = cv2.getTextSize(text, fontFace, mid, thickness=1)
            if w < square_size * fill_ratio and h < square_size * fill_ratio:
                optimal_scale = mid
                left = mid
            else:
                right = mid
        return optimal_scale, w, h

    x1, y1, x2, y2 = [int(v) for v in box]
    box_w = x2 - x1
    box_h = y2 - y1

    img = image.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=-1)

    # automatically set scale and thickness according to length of the shortest side
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness_scale_ratio = 4
    font_scale, text_w, text_h = get_optimal_font_scale(
        token_str, font, min(box_w, box_h), fill_ratio=0.9
    )
    font_thickness = max(1, math.floor(font_scale * thickness_scale_ratio))

    # calculate center coordinates of the patinting text
    text_x = x1 + (box_w - text_w) // 2
    text_y = y1 + (box_h + text_h) // 2

    cv2.putText(
        img,
        token_str,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        font_thickness,
        lineType=cv2.LINE_AA,
    )
    return img


def tokenize_figure_of_table(table_block_img, figures):
    """
    Replace figures in a table area with tokens, return new image and token map.

    Args:
        table_block_img (np.ndarray): Table image.
        figures (List[Dict]): List of figure dicts (must contain 'coordinate', 'path').

    Returns:
        Tuple[np.ndarray, Dict[str, str], List[str]]:
            - New table image,
            - Token-to-img HTML map,
            - List of figure paths dropped.
    """

    def gen_random_map(num):
        exclude_digits = {"0", "1", "9"}
        seq = []
        i = 0
        while len(seq) < num:
            if not (set(str(i)) & exclude_digits):
                seq.append(i)
            i += 1
        return seq

    import random

    random.seed(1024)
    token_map = {}
    drop_idxes = []
    random_map = gen_random_map(len(figures))
    random.shuffle(random_map)
    for figure_id, figure in enumerate(figures):
        drop_idxes.append(figure_id)
        draw_box = points_to_bbox(figure["ocr_bbox"])
        # the figure is too small to can't be tokenized and recognized when shortest length is less than 25
        # if min(figure_x_max - figure_x_min, figure_y_max - figure_y_min) < 25:
        #     continue
        token_str = "[F" + str(random_map[figure_id]) + "]"
        table_block_img = paint_token(table_block_img, draw_box, token_str)
        # token_map[token_str] = f'<img src="{figure["path"]}" >'
        token_map[token_str] = figure["uuid"]
    drop_figures = [f["uuid"] for i, f in enumerate(figures) if i in drop_idxes]
    return table_block_img, token_map, drop_figures


def untokenize_figure_of_table(table_res_str, figure_token_map, image_path_to_obj_map):
    """
    Replace tokens in a string with their HTML image equivalents.

    Args:
        table_res_str (str): Table string with tokens.
        figure_token_map (dict): Mapping from tokens to HTML img tags.

    Returns:
        str: Untokenized string.
    """

    def repl(match):
        token_id = match.group(1)
        token = f"[F{token_id}]"
        img_path = figure_token_map.get(token, match.group(0))
        img_block = image_path_to_obj_map.get(img_path, None)
        if img_block is None:
            return match.group(0)
        else:
            img_tags = []
            img_tags.append(
                '<img src="{}" alt="Image"" />'.format(
                    img_path.replace("-\n", "").replace("\n", " ")
                ),
            )
            image_info = "\n".join(img_tags)
            if img_block.content != "":
                ocr_content = img_block.content
                image_info += "\n\n" + ocr_content + "\n\n"
            return image_info

    pattern = r"\[F(\d+)\]"
    return re.sub(pattern, repl, table_res_str)


class TableCell(BaseModel):
    """
    TableCell represents a single cell in a table.

    Attributes:
        row_span (int): Number of rows spanned.
        col_span (int): Number of columns spanned.
        start_row_offset_idx (int): Start row index.
        end_row_offset_idx (int): End row index (exclusive).
        start_col_offset_idx (int): Start column index.
        end_col_offset_idx (int): End column index (exclusive).
        text (str): Cell text content.
        column_header (bool): Whether this cell is a column header.
        row_header (bool): Whether this cell is a row header.
        row_section (bool): Whether this cell is a row section.
    """

    row_span: int = 1
    col_span: int = 1
    start_row_offset_idx: int
    end_row_offset_idx: int
    start_col_offset_idx: int
    end_col_offset_idx: int
    text: str
    column_header: bool = False
    row_header: bool = False
    row_section: bool = False

    @model_validator(mode="before")
    @classmethod
    def from_dict_format(cls, data: Any) -> Any:
        """
        Create TableCell from dict, extracting 'text' property correctly.

        Args:
            data (Any): Input data.

        Returns:
            Any: TableCell-compatible dict.
        """
        if isinstance(data, Dict):
            if "text" in data:
                return data
            text = data["bbox"].get("token", "")
            if not len(text):
                text_cells = data.pop("text_cell_bboxes", None)
                if text_cells:
                    for el in text_cells:
                        text += el["token"] + " "
                text = text.strip()
            data["text"] = text
        return data


class TableData(BaseModel):
    """
    TableData holds a table's cells, row and column counts, and provides a grid property.

    Attributes:
        table_cells (List[TableCell]): List of table cells.
        num_rows (int): Number of rows.
        num_cols (int): Number of columns.
    """

    table_cells: List[TableCell] = []
    num_rows: int = 0
    num_cols: int = 0

    @computed_field
    @property
    def grid(self) -> List[List[TableCell]]:
        """
        Returns a 2D grid of TableCell objects for the table.

        Returns:
            List[List[TableCell]]: Table as 2D grid.
        """
        table_data = [
            [
                TableCell(
                    text="",
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )
                for j in range(self.num_cols)
            ]
            for i in range(self.num_rows)
        ]
        for cell in self.table_cells:
            for i in range(
                min(cell.start_row_offset_idx, self.num_rows),
                min(cell.end_row_offset_idx, self.num_rows),
            ):
                for j in range(
                    min(cell.start_col_offset_idx, self.num_cols),
                    min(cell.end_col_offset_idx, self.num_cols),
                ):
                    table_data[i][j] = cell
        return table_data


# OTSL tag constants
OTSL_NL = "<nl>"
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"

NON_CAPTURING_TAG_GROUP = "(?:<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>)"
OTSL_FIND_PATTERN = re.compile(
    f"{NON_CAPTURING_TAG_GROUP}.*?(?={NON_CAPTURING_TAG_GROUP}|$)", flags=re.DOTALL
)


def otsl_extract_tokens_and_text(s: str):
    """
    Extract OTSL tags and text parts from the input string.

    Args:
        s (str): OTSL string.

    Returns:
        Tuple[List[str], List[str]]: (tokens, text_parts)
    """
    pattern = (
        r"("
        + r"|".join([OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL])
        + r")"
    )
    tokens = re.findall(pattern, s)
    text_parts = re.split(pattern, s)
    text_parts = [token for token in text_parts if token.strip()]
    return tokens, text_parts


def otsl_parse_texts(texts, tokens):
    """
    Parse OTSL text and tags into TableCell objects and tag structure.

    Args:
        texts (List[str]): List of tokens and text.
        tokens (List[str]): List of OTSL tags.

    Returns:
        Tuple[List[TableCell], List[List[str]]]: (table_cells, split_row_tokens)
    """
    split_word = OTSL_NL
    split_row_tokens = [
        list(y)
        for x, y in itertools.groupby(tokens, lambda z: z == split_word)
        if not x
    ]
    table_cells = []
    r_idx = 0
    c_idx = 0

    # Ensure matrix completeness
    if split_row_tokens:
        max_cols = max(len(row) for row in split_row_tokens)
        for row in split_row_tokens:
            while len(row) < max_cols:
                row.append(OTSL_ECEL)
        new_texts = []
        text_idx = 0
        for row in split_row_tokens:
            for token in row:
                new_texts.append(token)
                if text_idx < len(texts) and texts[text_idx] == token:
                    text_idx += 1
                    if text_idx < len(texts) and texts[text_idx] not in [
                        OTSL_NL,
                        OTSL_FCEL,
                        OTSL_ECEL,
                        OTSL_LCEL,
                        OTSL_UCEL,
                        OTSL_XCEL,
                    ]:
                        new_texts.append(texts[text_idx])
                        text_idx += 1
            new_texts.append(OTSL_NL)
            if text_idx < len(texts) and texts[text_idx] == OTSL_NL:
                text_idx += 1
        texts = new_texts

    def count_right(tokens, c_idx, r_idx, which_tokens):
        span = 0
        c_idx_iter = c_idx
        while tokens[r_idx][c_idx_iter] in which_tokens:
            c_idx_iter += 1
            span += 1
            if c_idx_iter >= len(tokens[r_idx]):
                return span
        return span

    def count_down(tokens, c_idx, r_idx, which_tokens):
        span = 0
        r_idx_iter = r_idx
        while tokens[r_idx_iter][c_idx] in which_tokens:
            r_idx_iter += 1
            span += 1
            if r_idx_iter >= len(tokens):
                return span
        return span

    for i, text in enumerate(texts):
        cell_text = ""
        if text in [OTSL_FCEL, OTSL_ECEL]:
            row_span = 1
            col_span = 1
            right_offset = 1
            if text != OTSL_ECEL:
                cell_text = texts[i + 1]
                right_offset = 2

            next_right_cell = (
                texts[i + right_offset] if i + right_offset < len(texts) else ""
            )
            next_bottom_cell = ""
            if r_idx + 1 < len(split_row_tokens):
                if c_idx < len(split_row_tokens[r_idx + 1]):
                    next_bottom_cell = split_row_tokens[r_idx + 1][c_idx]

            if next_right_cell in [OTSL_LCEL, OTSL_XCEL]:
                col_span += count_right(
                    split_row_tokens, c_idx + 1, r_idx, [OTSL_LCEL, OTSL_XCEL]
                )
            if next_bottom_cell in [OTSL_UCEL, OTSL_XCEL]:
                row_span += count_down(
                    split_row_tokens, c_idx, r_idx + 1, [OTSL_UCEL, OTSL_XCEL]
                )

            table_cells.append(
                TableCell(
                    text=cell_text.strip(),
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=r_idx,
                    end_row_offset_idx=r_idx + row_span,
                    start_col_offset_idx=c_idx,
                    end_col_offset_idx=c_idx + col_span,
                )
            )
        if text in [OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]:
            c_idx += 1
        if text == OTSL_NL:
            r_idx += 1
            c_idx = 0
    return table_cells, split_row_tokens


def export_to_html(table_data: TableData):
    """
    Export TableData to HTML table.

    Args:
        table_data (TableData): TableData object.

    Returns:
        str: HTML string.
    """
    nrows = table_data.num_rows
    ncols = table_data.num_cols
    if len(table_data.table_cells) == 0:
        return ""
    body = ""
    grid = table_data.grid
    for i in range(nrows):
        body += "<tr>"
        for j in range(ncols):
            cell: TableCell = grid[i][j]
            rowspan, rowstart = (cell.row_span, cell.start_row_offset_idx)
            colspan, colstart = (cell.col_span, cell.start_col_offset_idx)
            if rowstart != i or colstart != j:
                continue
            content = html.escape(cell.text.strip())
            celltag = "th" if cell.column_header else "td"
            opening_tag = f"{celltag}"
            if rowspan > 1:
                opening_tag += f' rowspan="{rowspan}"'
            if colspan > 1:
                opening_tag += f' colspan="{colspan}"'
            body += f"<{opening_tag}>{content}</{celltag}>"
        body += "</tr>"
    body = f"<table>{body}</table>"
    return body


def otsl_pad_to_sqr_v2(otsl_str: str) -> str:
    """
    Pad OTSL string to a square (rectangular) format, ensuring each row has equal number of cells.

    Args:
        otsl_str (str): OTSL string.

    Returns:
        str: Padded OTSL string.
    """
    assert isinstance(otsl_str, str)
    otsl_str = otsl_str.strip()
    if OTSL_NL not in otsl_str:
        return otsl_str + OTSL_NL
    lines = otsl_str.split(OTSL_NL)
    row_data = []
    for line in lines:
        if not line:
            continue
        raw_cells = OTSL_FIND_PATTERN.findall(line)
        if not raw_cells:
            continue
        total_len = len(raw_cells)
        min_len = 0
        for i, cell_str in enumerate(raw_cells):
            if cell_str.startswith(OTSL_FCEL):
                min_len = i + 1
        row_data.append(
            {"raw_cells": raw_cells, "total_len": total_len, "min_len": min_len}
        )
    if not row_data:
        return OTSL_NL
    global_min_width = max(row["min_len"] for row in row_data) if row_data else 0
    max_total_len = max(row["total_len"] for row in row_data) if row_data else 0
    search_start = global_min_width
    search_end = max(global_min_width, max_total_len)
    min_total_cost = float("inf")
    optimal_width = search_end

    for width in range(search_start, search_end + 1):
        current_total_cost = sum(abs(row["total_len"] - width) for row in row_data)
        if current_total_cost < min_total_cost:
            min_total_cost = current_total_cost
            optimal_width = width

    repaired_lines = []
    for row in row_data:
        cells = row["raw_cells"]
        current_len = len(cells)
        if current_len > optimal_width:
            new_cells = cells[:optimal_width]
        else:
            padding = [OTSL_ECEL] * (optimal_width - current_len)
            new_cells = cells + padding
        repaired_lines.append("".join(new_cells))
    return OTSL_NL.join(repaired_lines) + OTSL_NL


def convert_otsl_to_html(otsl_content: str):
    """
    Convert OTSL-v1.0 string to HTML. Only 6 tags allowed: <fcel>, <ecel>, <nl>, <lcel>, <ucel>, <xcel>.

    Args:
        otsl_content (str): OTSL string.

    Returns:
        str: HTML table.
    """
    otsl_content = otsl_pad_to_sqr_v2(otsl_content)
    tokens, mixed_texts = otsl_extract_tokens_and_text(otsl_content)
    table_cells, split_row_tokens = otsl_parse_texts(mixed_texts, tokens)
    table_data = TableData(
        num_rows=len(split_row_tokens),
        num_cols=(max(len(row) for row in split_row_tokens) if split_row_tokens else 0),
        table_cells=table_cells,
    )
    return export_to_html(table_data)


def crop_margin(img):
    import cv2

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    max_val = gray.max()
    min_val = gray.min()

    if max_val == min_val:
        return img

    data = (gray - min_val) / (max_val - min_val) * 255
    data = data.astype(np.uint8)

    _, binary = cv2.threshold(data, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)

    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y : y + h, x : x + w]

    return cropped