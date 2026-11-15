import logging
import traceback
import cv2
import numpy as np
from typing import List, Dict, Any
from loguru import logger
from .main import TSRUnetStructurer
from .table_recover import TableRecover
from rapid_doc.model.table.utils import normalize_table_ocr_text
from rapid_doc.utils.span_pre_proc import calculate_contrast
from .utils.utils_table_recover import (
    match_ocr_cell,
    plot_html_table,
    box_4_2_poly_to_box_4_1,
    sorted_ocr_boxes,
    gather_ocr_list_by_row,
)

class UnetTableRecognition:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.table_structure = TSRUnetStructurer(self.cfg)
        self.table_recover = TableRecover()
        self.ocr_engine = None

    def __call__(
        self, ori_imgs: List[np.ndarray], ocr_results,
        *, token_cells=None, evidence_grids=None, verified_cell_splits=None, protected_token_count=0,
        **kwargs,
    ):
        need_ocr = True
        col_threshold = 15
        row_threshold = 10
        if kwargs:
            need_ocr = kwargs.get("need_ocr", True)
            col_threshold = kwargs.get("col_threshold", 15)
            row_threshold = kwargs.get("row_threshold", 10)

        pred_htmls, cell_bboxes, logic_points_list = [], [], []
        for i, img in enumerate(ori_imgs):
            if verified_cell_splits is not None:
                verified_cell_splits.append(0)
            assignments = {} if token_cells is not None else None
            if token_cells is not None:
                token_cells.append(None)
            if evidence_grids is not None:
                evidence_grids.append(None)

            ocr_result = list(
                zip(ocr_results[i][0], ocr_results[i][1], ocr_results[i][2])
            )
            polygons, rotated_polygons = self.table_structure(img, **kwargs)
            if polygons is None:
                pred_htmls.append("")
                cell_bboxes.append([])
                logic_points_list.append([])
                # logging.warning("polygons is None.")
                continue

            try:
                table_res, logi_points = self.table_recover(
                    rotated_polygons, row_threshold, col_threshold
                )
                # 将坐标由逆时针转为顺时针方向，后续处理与无线表格对齐
                polygons[:, 1, :], polygons[:, 3, :] = (
                    polygons[:, 3, :].copy(),
                    polygons[:, 1, :].copy(),
                )
                if not need_ocr:
                    sorted_polygons, idx_list = sorted_ocr_boxes(
                        [box_4_2_poly_to_box_4_1(box) for box in polygons]
                    )
                    pred_htmls.append("")
                    cell_bboxes.append(sorted_polygons)
                    logic_points_list.append(logi_points[idx_list])
                    continue

                recovered = self.recognize_cross_cell_text(
                    img, polygons, logi_points, ocr_result, protected_token_count=protected_token_count
                )
                original_ids = [i for i in range(len(ocr_result)) if i not in recovered]
                remaining = [ocr_result[i] for i in original_ids]
                cell_box_det_map, not_match_orc_boxes = match_ocr_cell(remaining, polygons, token_cells=assignments)
                if recovered:
                    if assignments is not None:
                        assignments = {original_ids[i]: cells for i, cells in assignments.items()}
                    for token_id, cells in recovered.items():
                        for cell, text in cells.items():
                            cell_box_det_map.setdefault(cell, []).extend(text)
                        if assignments is not None:
                            assignments[token_id] = tuple(cells)
                if self.ocr_engine is not None and not_match_orc_boxes:
                    self.recover_joined_cells(
                        img, polygons, logi_points, cell_box_det_map, not_match_orc_boxes,
                        token_ids={id(token): idx for idx, token in enumerate(ocr_result)} if assignments is not None else None,
                        token_cells=assignments,
                    )
                # 转换为中间格式，修正识别框坐标,将物理识别框，逻辑识别框，ocr识别框整合为dict，方便后续处理
                t_rec_ocr_list = self.transform_res(cell_box_det_map, polygons, logi_points)
                # 将每个单元格中的ocr识别结果排序和同行合并，输出的html能完整保留文字的换行格式
                t_rec_ocr_list = self.sort_and_gather_ocr_res(t_rec_ocr_list)

                cell_box_det_map = {
                    t_box_ocr["cell_idx"]: [
                        ocr_box_and_text[1] for ocr_box_and_text in t_box_ocr["t_ocr_res"]
                    ]
                    for t_box_ocr in t_rec_ocr_list
                }
                emitted = {} if assignments is not None else None
                pred_html = plot_html_table(logi_points, cell_box_det_map, polygons, emitted_cells=emitted)
                if verified_cell_splits is not None:
                    verified_cell_splits[-1] = sum(len(cells) - 1 for cells in recovered.values())
                polygons = np.array(polygons).reshape(-1, 8)
                logi_points = np.array(logi_points)
                if emitted:
                    # HTML 会裁掉边缘噪声格，证据网格与文字归属使用实际输出的单元格。
                    indices = {cell: index for index, cell in enumerate(emitted)}
                    token_cells[-1] = {
                        token: tuple(indices[cell] for cell in cells if cell in indices)
                        for token, cells in assignments.items()
                    }
                    if evidence_grids is not None:
                        evidence_grids[-1] = (polygons[list(emitted)], np.asarray(list(emitted.values())))

                pred_htmls.append(pred_html)
                cell_bboxes.append(polygons)
                logic_points_list.append(logi_points)

            except Exception:
                logging.warning(traceback.format_exc())
                pred_htmls.append("")
                cell_bboxes.append([])
                logic_points_list.append([])

        return pred_htmls, cell_bboxes, logic_points_list

    def recognize_cross_cell_text(self, image, polygons, logic_points, tokens, *, protected_token_count=0):
        if self.ocr_engine is None or not tokens or len(polygons) < 2:
            return {}

        from rapid_doc.model.table.table_evidence import _raster_lines

        lower, upper = polygons.min(axis=1), polygons.max(axis=1)
        logic = np.asarray(logic_points)
        rects = np.asarray([np.stack((np.min(box, axis=0), np.max(box, axis=0))) for box, _, _ in tokens])
        em = max(2., float(np.median(rects[:, 1, 1] - rects[:, 0, 1])))
        candidates = {}
        lines = None
        for token_id, ((left, top), (right, bottom)) in enumerate(rects):
            if token_id >= len(tokens) - protected_token_count:
                break
            height = bottom - top
            if height <= 0 or not tokens[token_id][1].strip():
                continue
            overlap = np.minimum(upper, [right, bottom]) - np.maximum(lower, [left, top])
            cells = np.flatnonzero((overlap[:, 0] >= .3 * height) & (overlap[:, 1] >= .75 * height))
            if len(cells) < 2:
                continue
            cells = cells[np.argsort(logic[cells, 2])]
            rows, cols = logic[cells, :2], logic[cells, 2:]
            if not (np.all(rows == rows[0, 0]) and np.all(np.diff(cols[:, 0]) == 1)
                    and np.all(cols[:, 0] == cols[:, 1])):
                continue
            if lines is None:
                lines = [(pos, lo, hi) for axis, pos, lo, hi in _raster_lines(image, em) if axis == 0]
            # 模型网格与原图同时确认分隔线，避免把同一个合并单元格内的文字拆开。
            boundaries = (upper[cells[:-1], 0] + lower[cells[1:], 0]) / 2
            if not all(any(abs(pos - edge) <= .45 * em and left < pos < right
                           and lo <= top + .2 * height and hi >= bottom - .2 * height
                           for pos, lo, hi in lines) for edge in boundaries):
                continue
            candidates[token_id] = tuple(map(int, cells))

        if not candidates:
            return {}
        # 一个格子还有独立 OCR 框时，保留原匹配，避免把整格重识别结果叠加进去。
        owners = {}
        for i, ((left, top), (right, bottom)) in enumerate(rects):
            center = [(left + right) / 2, (top + bottom) / 2]
            cells = candidates.get(i)
            if cells is None:
                cells = np.flatnonzero(np.all((lower < center) & (center < upper), axis=1))
            for cell in cells:
                owners.setdefault(cell, set()).add(i)
        candidates = {i: cells for i, cells in candidates.items() if all(len(owners[c]) == 1 for c in cells)}
        if not candidates:
            return {}
        cells = sorted({cell for group in candidates.values() for cell in group})
        recognized = self.fill_blank_rec(image, polygons[cells], {})
        recognized = {cell: recognized[i] for i, cell in enumerate(cells)}
        # 每个待拆框都必须在格线两侧得到独立识别，且保留原识别的非空白字符。
        return {
            i: {cell: recognized[cell] for cell in group}
            for i, group in candidates.items()
            if all(recognized[cell][0][1] for cell in group)
            and "".join("".join(recognized[cell][0][1].split()) for cell in group)
            == "".join(tokens[i][1].split())
        }

    def recover_joined_cells(self, img, polygons, logic_points, matched, unmatched, *, token_ids=None, token_cells=None):
        lower = polygons.min(axis=1)
        upper = polygons.max(axis=1)
        logic = np.asarray(logic_points)
        for token in unmatched:
            box, text, score = token
            joined = "".join(str(text).split())
            if score < 0.9 or len(joined) < 2:
                continue
            left, top = np.min(box, axis=0)
            right, bottom = np.max(box, axis=0)
            if bottom <= top or right - left < 2 * (bottom - top):
                continue
            overlap = np.minimum(upper, [right, bottom]) - np.maximum(lower, [left, top])
            indices = np.flatnonzero(
                (overlap[:, 0] > 0.5 * (upper[:, 0] - lower[:, 0]))
                & (overlap[:, 1] >= 0.75 * (bottom - top))
            )
            if not 2 <= len(indices) <= len(joined) or any(i in matched for i in indices):
                continue
            indices = indices[np.argsort(logic[indices, 2])]
            rows = logic[indices, :2]
            cols = logic[indices, 2:]
            if not (np.all(rows == rows[0, 0]) and np.all(cols[:, 0] == cols[:, 1])
                    and np.all(np.diff(cols[:, 0]) == 1)):
                continue
            recovered = self.fill_blank_rec(img, polygons[indices], {})
            texts = ["".join(recovered[i][0][1].split()) for i in range(len(indices))]
            if (len(indices) >= 4 and len(joined) == len(indices)
                    and texts.count("") == 1
                    and all(not text or text == char for text, char in zip(texts, joined))):
                # The other cells independently confirm the character alignment.
                missing = texts.index("")
                recovered[missing] = [[polygons[indices[missing]], joined[missing], score]]
                texts[missing] = joined[missing]
            if all(texts) and "".join(texts) == joined:
                matched.update((int(idx), recovered[i]) for i, idx in enumerate(indices))
                if token_cells is not None:
                    token_cells[token_ids[id(token)]] = tuple(map(int, indices))

    def transform_res(
        self,
        cell_box_det_map: Dict[int, List[any]],
        polygons: np.ndarray,
        logi_points: List[np.ndarray],
    ) -> List[Dict[str, any]]:
        res = []
        for i in range(len(polygons)):
            ocr_res_list = cell_box_det_map.get(i)
            if not ocr_res_list:
                continue
            xmin = min([ocr_box[0][0][0] for ocr_box in ocr_res_list])
            ymin = min([ocr_box[0][0][1] for ocr_box in ocr_res_list])
            xmax = max([ocr_box[0][2][0] for ocr_box in ocr_res_list])
            ymax = max([ocr_box[0][2][1] for ocr_box in ocr_res_list])
            dict_res = {
                # 物理 cell 的原始下标，用于按完整结构回填空单元格
                "cell_idx": i,
                # xmin,xmax,ymin,ymax
                "t_box": [xmin, ymin, xmax, ymax],
                # row_start,row_end,col_start,col_end
                "t_logic_box": logi_points[i].tolist(),
                # [[xmin,xmax,ymin,ymax], text]
                "t_ocr_res": [
                    [box_4_2_poly_to_box_4_1(ocr_det[0]), ocr_det[1]]
                    for ocr_det in ocr_res_list
                ],
            }
            res.append(dict_res)
        return res

    def sort_and_gather_ocr_res(self, res):
        for i, dict_res in enumerate(res):
            _, sorted_idx = sorted_ocr_boxes(
                [ocr_det[0] for ocr_det in dict_res["t_ocr_res"]], threhold=0.3
            )
            dict_res["t_ocr_res"] = [dict_res["t_ocr_res"][i] for i in sorted_idx]
            dict_res["t_ocr_res"] = gather_ocr_list_by_row(
                dict_res["t_ocr_res"], threhold=0.3
            )
        return res

    def fill_blank_rec(
        self,
        img: np.ndarray,
        sorted_polygons: np.ndarray,
        cell_box_map: Dict[int, List[str]],
    ) -> Dict[int, List[Any]]:
        """找到poly对应为空的框，尝试将直接将poly框直接送到识别中"""
        if self.ocr_engine is None:
            for i in range(sorted_polygons.shape[0]):
                if not cell_box_map.get(i):
                    cell_box_map[i] = [[sorted_polygons[i], "", 1]]
            return cell_box_map

        bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_crop_info_list = []
        img_crop_list = []
        for i in range(sorted_polygons.shape[0]):
            if cell_box_map.get(i):
                continue
            box = sorted_polygons[i]
            x1, y1 = int(box[0][0]) + 1, int(box[0][1]) + 1
            x2, y2 = int(box[2][0]) - 1, int(box[2][1]) - 1
            if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
                cell_box_map[i] = [[box, "", 1]]
                continue
            if (x2 - x1) / max(y2 - y1, 1) > 20 or (y2 - y1) / max(x2 - x1, 1) > 20:
                cell_box_map[i] = [[box, "", 1]]
                continue

            img_crop = bgr_img[y1:y2, x1:x2]
            if calculate_contrast(img_crop, img_mode='bgr') <= 0.17:
                cell_box_map[i] = [[box, "", 0.1]]
                continue

            img_crop_list.append(img_crop)
            img_crop_info_list.append((i, box))

        if not img_crop_list:
            return cell_box_map

        try:
            ocr_result = self.ocr_engine.ocr(img_crop_list, det=False)
        except Exception as exc:
            logger.warning(f"Blank table cell OCR-rec failed: {exc}")
            for i, box in img_crop_info_list:
                cell_box_map.setdefault(i, [[box, "", 1]])
            return cell_box_map

        if not ocr_result or not isinstance(ocr_result, list) or len(ocr_result) == 0:
            for i, box in img_crop_info_list:
                cell_box_map.setdefault(i, [[box, "", 1]])
            return cell_box_map

        ocr_res_list = ocr_result[0]
        if not isinstance(ocr_res_list, list) or len(ocr_res_list) != len(img_crop_list):
            for i, box in img_crop_info_list:
                cell_box_map.setdefault(i, [[box, "", 1]])
            return cell_box_map

        for (i, box), ocr_res in zip(img_crop_info_list, ocr_res_list):
            ocr_text, ocr_score = ocr_res
            if self._should_drop_blank_cell_rec_result(ocr_text, ocr_score):
                cell_box_map[i] = [[box, "", 0.1]]
                continue
            cell_box_map[i] = [[box, normalize_table_ocr_text(ocr_text), ocr_score]]
        return cell_box_map

    @staticmethod
    def _should_drop_blank_cell_rec_result(text: str, score) -> bool:
        try:
            if float(score) < 0.6:
                return True
        except (TypeError, ValueError):
            return True

        normalized_text = "" if text is None else str(text).strip()
        return not normalized_text
