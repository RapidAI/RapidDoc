# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np

from .utils import mkdir
from .vis import VisTable


class EngineType(Enum):
    ONNXRUNTIME = "onnxruntime"
    TORCH = "torch"


class ModelType(Enum):
    SLANETPLUS = "slanet_plus"
    UNITABLE = "unitable"
    SLANEXT_WIRED = "slanext_wired"
    SLANEXT_WIRELESS = "slanext_wireless"
    UNET = "unet" # 有线表格unet
    UNET_SLANET_PLUS = "unet_slanet_plus"  # 有线表格使用unet，无线表格使用slanet_plus
    UNET_UNITABLE = "unet_unitable"  # 有线表格使用unet，无线表格使用unitable
    SLANEXT = "slanext" # 有线表格使用slanext_wired，无线表格使用slanext_wireless


@dataclass
class RapidTableInput:
    model_type: Optional[ModelType] = ModelType.SLANETPLUS
    model_dir_or_path: Union[str, Path, None, Dict[str, str]] = None

    engine_type: Optional[EngineType] = None
    engine_cfg: dict = field(default_factory=dict)

    use_ocr: bool = True
    ocr_params: dict = field(default_factory=dict)


@dataclass
class RapidTableOutput:
    img: Optional[np.ndarray] = None
    pred_html: Optional[str] = None
    cell_bboxes: Optional[np.ndarray] = None
    logic_points: Optional[np.ndarray] = None
    elapse: Optional[float] = None

    def vis(
        self, save_dir: Union[str, Path, None] = None, save_name: Optional[str] = None
    ) -> np.ndarray:
        vis = VisTable()

        mkdir(save_dir)
        save_html_path = Path(save_dir) / f"{save_name}.html"
        save_drawed_path = Path(save_dir) / f"{save_name}_vis.jpg"
        save_logic_points_path = Path(save_dir) / f"{save_name}_col_row_vis.jpg"

        vis_img = vis(
            self.img,
            self.pred_html,
            self.cell_bboxes,
            self.logic_points,
            save_html_path,
            save_drawed_path,
            save_logic_points_path,
        )
        return vis_img
