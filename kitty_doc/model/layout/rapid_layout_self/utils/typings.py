from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .logger import Logger
from .utils import save_img

logger = Logger(logger_name=__name__).get_log()


class ModelType(Enum):
    PP_DOCLAYOUT_S = "pp_doclayout_s"
    PP_DOCLAYOUT_M = "pp_doclayout_m"
    PP_DOCLAYOUT_L = "pp_doclayout_l"
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

    conf_thresh: float = 0.5
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
