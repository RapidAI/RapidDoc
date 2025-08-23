import html

import cv2
import numpy as np
from loguru import logger

from kitty_doc.model.table.rapid_table_self.table_cls import TableCls
from kitty_doc.model.table.rapid_table_self import ModelType, RapidTable, RapidTableInput
from kitty_doc.model.layout.rapid_layout_self import RapidLayoutInput, RapidLayout, ModelType as LayoutModelType
from kitty_doc.utils.config_reader import get_device


def escape_html(input_string):
    """Escape HTML Entities."""
    return html.escape(input_string)


class RapidTableModel(object):
    def __init__(self, ocr_engine, table_config=None):
        if table_config is None:
            table_config = {}
        device = get_device()
        self.table_pipeline = table_config.get("table_pipeline", False)
        self.model_type = table_config.get("model_type", ModelType.SLANETPLUS)
        self.ocr_engine = ocr_engine

        if self.table_pipeline:
            # 采用“表格分类+表格结构识别+单元格检测”多模型串联组网方案
            self.table_cls = TableCls()
            wired_cell_args = RapidLayoutInput(model_type=LayoutModelType.RT_DETR_L_WIRED_TABLE_CELL_DET,
                                               model_dir_or_path=table_config.get("wired_cell.model_dir_or_path"))
            self.wired_table_cell = RapidLayout(cfg=wired_cell_args)
            wireless_cell_args = RapidLayoutInput(model_type=LayoutModelType.RT_DETR_L_WIRELESS_TABLE_CELL_DET,
                             model_dir_or_path=table_config.get("wireless_cell.model_dir_or_path"))
            self.wireless_table_cell = RapidLayout(cfg=wireless_cell_args)

        if self.model_type == ModelType.SLANEXT_WIRED_WIRELESS:
            wired_input_args = RapidTableInput(model_type=ModelType.SLANEXT_WIRED, use_ocr=False,
                                               model_dir_or_path=table_config.get("wired_table.model_dir_or_path"))
            self.wired_table_model = RapidTable(wired_input_args)
            wireless_input_args = RapidTableInput(model_type=ModelType.SLANEXT_WIRELESS, use_ocr=False,
                                                  model_dir_or_path=table_config.get("wireless_table.model_dir_or_path"))
            self.wireless_table_model = RapidTable(wireless_input_args)
        else:
            input_args = RapidTableInput(model_type=self.model_type, use_ocr=False,
                                         model_dir_or_path=table_config.get("model_dir_or_path"))
            self.table_model = RapidTable(input_args)

    def predict(self, image):
        bgr_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

        # First check the overall image aspect ratio (height/width)
        img_height, img_width = bgr_image.shape[:2]
        img_aspect_ratio = img_height / img_width if img_width > 0 else 1.0
        img_is_portrait = img_aspect_ratio > 1.2

        if img_is_portrait:

            det_res = self.ocr_engine.ocr(bgr_image, rec=False)[0]
            # Check if table is rotated by analyzing text box aspect ratios
            is_rotated = False
            if det_res:
                vertical_count = 0

                for box_ocr_res in det_res:
                    p1, p2, p3, p4 = box_ocr_res

                    # Calculate width and height
                    width = p3[0] - p1[0]
                    height = p3[1] - p1[1]

                    aspect_ratio = width / height if height > 0 else 1.0

                    # Count vertical vs horizontal text boxes
                    if aspect_ratio < 0.8:  # Taller than wide - vertical text
                        vertical_count += 1
                    # elif aspect_ratio > 1.2:  # Wider than tall - horizontal text
                    #     horizontal_count += 1

                # If we have more vertical text boxes than horizontal ones,
                # and vertical ones are significant, table might be rotated
                if vertical_count >= len(det_res) * 0.3:
                    is_rotated = True

                # logger.debug(f"Text orientation analysis: vertical={vertical_count}, det_res={len(det_res)}, rotated={is_rotated}")

            # Rotate image if necessary
            if is_rotated:
                # logger.debug("Table appears to be in portrait orientation, rotating 90 degrees clockwise")
                image = cv2.rotate(np.asarray(image), cv2.ROTATE_90_CLOCKWISE)
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Continue with OCR on potentially rotated image
        ocr_result = self.ocr_engine.ocr(bgr_image)[0]
        if ocr_result:
            # ocr_result = [[item[0], escape_html(item[1][0]), item[1][1]] for item in ocr_result if
            #           len(item) == 2 and isinstance(item[1], tuple)]
            ocr_result = [list(x) for x in zip(*[[item[0], item[1][0], item[1][1]] for item in ocr_result])]
        else:
            ocr_result = None


        if ocr_result:
            try:
                image = np.asarray(image)
                if not self.table_pipeline:
                    table_results = self.table_model(image, ocr_result)
                else:
                    cls, elasp = self.table_cls(image)
                    if cls == "wired":
                        cell_res = self.wired_table_model([image])
                        model_runner = (
                            self.wired_table_model if self.model_type == ModelType.SLANEXT_WIRED_WIRELESS
                            else self.table_model
                        )
                    else:  # wireless
                        cell_res = self.wireless_table_cell([image])
                        model_runner = (
                            self.wireless_table_model if self.model_type == ModelType.SLANEXT_WIRED_WIRELESS
                            else self.table_model
                        )
                    table_results = model_runner(image, ocr_result, cell_results=cell_res[0].boxes)

                html_code = table_results.pred_html
                table_cell_bboxes = table_results.cell_bboxes
                logic_points = table_results.logic_points
                elapse = table_results.elapse
                return html_code, table_cell_bboxes, logic_points, elapse
            except Exception as e:
                logger.exception(e)

        return None, None, None, None
