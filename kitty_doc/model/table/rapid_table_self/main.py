# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
import argparse
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from .wired_table_rec.main import WiredTableInput, WiredTableRecognition

# from .table_matcher.table_match_pipeline import TableMatchPipeline
from .model_processor.main import ModelProcessor
from .table_matcher import TableMatch
from .utils import (
    LoadImage,
    Logger,
    ModelType,
    RapidTableInput,
    RapidTableOutput,
    get_boxes_recs,
    import_package,
)

logger = Logger(logger_name=__name__).get_log()
root_dir = Path(__file__).resolve().parent


class RapidTable:
    def __init__(self, cfg: Optional[RapidTableInput] = None):
        if cfg is None:
            cfg = RapidTableInput()

        if not cfg.model_dir_or_path and cfg.model_type != ModelType.UNET:
            cfg.model_dir_or_path = ModelProcessor.get_model_path(cfg.model_type)

        self.cfg = cfg
        self.table_structure = self._init_table_structer()

        self.ocr_engine = None
        if cfg.use_ocr:
            self.ocr_engine = self._init_ocr_engine(self.cfg.ocr_params)

        self.table_matcher = TableMatch()
        # self.table_matcher_pipeline = TableMatchPipeline()
        self.load_img = LoadImage()

    def _init_ocr_engine(self, params: Dict[Any, Any]):
        rapidocr_ = import_package("rapidocr")
        if rapidocr_ is None:
            logger.warning("rapidocr package is not installed, only table rec")
            return None

        if not params:
            return rapidocr_.RapidOCR()
        return rapidocr_.RapidOCR(params=params)

    def _init_table_structer(self):
        if self.cfg.model_type == ModelType.UNITABLE:
            from .table_structure.unitable import UniTableStructure
            return UniTableStructure(asdict(self.cfg))

        if self.cfg.model_type == ModelType.UNET:
            wired_input = WiredTableInput(model_path=self.cfg.model_dir_or_path, use_cuda=self.cfg.engine_cfg.get('use_cuda', False))
            return WiredTableRecognition(wired_input)

        from .table_structure.pp_structure import PPTableStructurer

        return PPTableStructurer(asdict(self.cfg))

    def __call__(
        self,
        img_content: Union[str, np.ndarray, bytes, Path],
        ocr_results: Optional[Tuple[np.ndarray, Tuple[str], Tuple[float]]] = None,
        cell_results: Optional[Tuple[List[List[float]], Tuple[float]]] = None,
    ) -> RapidTableOutput:
        s = time.perf_counter()

        img = self.load_img(img_content)
        if self.cfg.model_type == ModelType.UNET:
            ocr_results = list(
                zip(ocr_results[0], ocr_results[1], ocr_results[2])
            )
            table_results = self.table_structure(img, ocr_result=ocr_results)
            return RapidTableOutput(img, table_results.pred_html, table_results.cell_bboxes,
                                    table_results.logic_points.tolist() if table_results.logic_points is not None else None,
                                    table_results.elapse)

        dt_boxes, rec_res = self.get_ocr_results(img, ocr_results)
        pred_structures, cell_bboxes, logic_points = self.get_table_rec_results(img)
        if cell_results is not None:
            cell_results, _ = cell_results
            cell_results1, cells_flags = self.sort_table_cells_boxes(cell_results)
            cell_bboxes = self.convert_to_four_point_coordinates(cell_results1)
        pred_html = self.get_table_matcher(
            pred_structures, cell_bboxes, dt_boxes, rec_res
        )
        # if cell_results is None:
        #     pred_html = self.get_table_matcher(
        #         pred_structures, cell_bboxes, dt_boxes, rec_res
        #     )
        # else:
        #     pred_html, cell_bboxes = self.table_matcher_pipeline(
        #         pred_structures, cell_bboxes, dt_boxes, rec_res, cell_results
        #     )
        #     cell_bboxes = self.convert_to_four_point_coordinates(cell_bboxes)

        elapse = time.perf_counter() - s
        return RapidTableOutput(img, pred_html, cell_bboxes, logic_points, elapse)

    def convert_to_four_point_coordinates(self, boxes: List[List[float]]) -> np.ndarray:
        """
        将二维坐标 [x_min, y_min, x_max, y_max] 转换为四点坐标
        格式：[x1, y1, x2, y2, x3, y3, x4, y4]
        """
        result = []
        for x_min, y_min, x_max, y_max in boxes:
            result.append([
                x_min, y_min,  # 左上角 (Top-Left)
                x_max, y_min,  # 右上角 (Top-Right)
                x_max, y_max,  # 右下角 (Bottom-Right)
                x_min, y_max  # 左下角 (Bottom-Left)
            ])
        return np.array(result)

    def sort_table_cells_boxes(self, boxes, overlap_threshold=0.5):
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

    def get_ocr_results(
        self, img: np.ndarray, ocr_results: Tuple[np.ndarray, Tuple[str], Tuple[float]]
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if ocr_results is not None:
            return get_boxes_recs(ocr_results, img.shape[:2])

        if not self.cfg.use_ocr:
            return None, None

        ori_ocr_res = self.ocr_engine(img)
        if ori_ocr_res.boxes is None:
            logger.warning("OCR Result is empty")
            return None, None

        ocr_results = [ori_ocr_res.boxes, ori_ocr_res.txts, ori_ocr_res.scores]
        return get_boxes_recs(ocr_results, img.shape[:2])

    def get_table_rec_results(self, img: np.ndarray):
        pred_structures, cell_bboxes, _ = self.table_structure(img)
        logic_points = self.table_matcher.decode_logic_points(pred_structures)
        return pred_structures, cell_bboxes, logic_points

    def get_table_matcher(self, pred_structures, cell_bboxes, dt_boxes, rec_res):
        if dt_boxes is None and rec_res is None:
            return None

        return self.table_matcher(pred_structures, cell_bboxes, dt_boxes, rec_res)


def parse_args(arg_list: Optional[List[str]] = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str, help="the image path or URL of the table")
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        default=ModelType.SLANETPLUS.value,
        choices=[v.value for v in ModelType],
        help="Supported table rec models",
    )
    parser.add_argument(
        "-v",
        "--vis",
        action="store_true",
        default=False,
        help="Wheter to visualize the layout results.",
    )
    args = parser.parse_args(arg_list)
    return args


def main(arg_list: Optional[List[str]] = None):
    args = parse_args(arg_list)
    img_path = args.img_path

    input_args = RapidTableInput(model_type=ModelType(args.model_type))
    table_engine = RapidTable(input_args)

    if table_engine.ocr_engine is None:
        raise ValueError("ocr engine is None")

    ori_ocr_res = table_engine.ocr_engine(img_path)
    ocr_results = [ori_ocr_res.boxes, ori_ocr_res.txts, ori_ocr_res.scores]
    table_results = table_engine(img_path, ocr_results=ocr_results)
    print(table_results.pred_html)

    if args.vis:
        save_dir = Path(img_path).resolve().parent
        table_results.vis(save_dir, save_name=Path(img_path).stem)


if __name__ == "__main__":
    main()
