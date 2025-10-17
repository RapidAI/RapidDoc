from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .logger import Logger
from .utils import save_img

logger = Logger(logger_name=__name__).get_log()

PP_DOCLAYOUT_PLUS_L_Threshold = {
    0: 0.3,   # paragraph_title
    1: 0.5,   # image
    2: 0.4,   # text
    3: 0.5,   # number
    4: 0.5,   # abstract
    5: 0.5,   # content
    6: 0.5,   # figure_table_chart_title
    7: 0.3,   # formula
    8: 0.5,   # table
    9: 0.5,   # reference
    10: 0.5,  # doc_title
    11: 0.5,  # footnote
    12: 0.5,  # header
    13: 0.5,  # algorithm
    14: 0.5,  # footer
    15: 0.45, # seal
    16: 0.5,  # chart
    17: 0.5,  # formula_number
    18: 0.5,  # aside_text
    19: 0.5,  # reference_content
}

PP_DOCLAYOUT_L_Threshold = {
    0: 0.3,    # paragraph_title
    1: 0.5,    # image
    2: 0.4,    # text
    3: 0.5,    # number
    4: 0.5,    # abstract
    5: 0.5,    # content
    6: 0.5,    # figure_title (默认值)
    7: 0.3,    # formula
    8: 0.5,    # table
    9: 0.5,    # table_title (默认值)
    10: 0.5,   # reference
    11: 0.5,   # doc_title
    12: 0.5,   # footnote
    13: 0.5,   # header
    14: 0.5,   # algorithm
    15: 0.5,   # footer
    16: 0.45,  # seal
    17: 0.5,   # chart_title (默认值)
    18: 0.5,   # chart
    19: 0.5,   # formula_number
    20: 0.5,   # header_image (默认值)
    21: 0.5,   # footer_image (默认值)
    22: 0.5    # aside_text
}

class ModelType(Enum):
    PP_DOCLAYOUT_PLUS_L = "pp_doclayout_plus_l"
    PP_DOCLAYOUT_L = "pp_doclayout_l"
    PP_DOCLAYOUT_M = "pp_doclayout_m"
    PP_DOCLAYOUT_S = "pp_doclayout_s"
    RT_DETR_L_WIRED_TABLE_CELL_DET = "rt_detr_l_wired_table_cell_det"
    RT_DETR_L_WIRELESS_TABLE_CELL_DET = "rt_detr_l_wireless_table_cell_det"


class EngineType(Enum):
    ONNXRUNTIME = "onnxruntime"
    # OPENVINO = "openvino"


@dataclass
class RapidLayoutInput:
    model_type: ModelType = ModelType.PP_DOCLAYOUT_L
    model_dir_or_path: Union[str, Path, None] = None

    engine_type: EngineType = EngineType.ONNXRUNTIME
    engine_cfg: dict = field(default_factory=dict)

    conf_thresh: Union[float, dict] = 0.5
    iou_thresh: float = 0.5


@dataclass
class RapidLayoutOutput:
    img: Optional[np.ndarray] = None
    boxes: Optional[List[List[float]]] = None
    class_names: Optional[List[str]] = None
    scores: Optional[List[float]] = None
    elapse: Optional[float] = None

    def vis(self, save_path: Union[str, Path, None] = None) -> Optional[np.ndarray]:
        if self.img is None or self.boxes is None:
            logger.warning("No image or boxes to visualize.")
            return None

        from .vis_res import VisLayout

        vis_img = VisLayout.draw_detections(
            self.img,
            np.array(self.boxes),
            np.array(self.scores),
            np.array(self.class_names),
        )
        if save_path is not None and vis_img is not None:
            save_img(save_path, vis_img)
            logger.info(f"Visualization saved as {save_path}")

        return vis_img
