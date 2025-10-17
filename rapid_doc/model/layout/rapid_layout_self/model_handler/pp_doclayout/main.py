import time
from typing import List, Union

import numpy as np

from ...inference_engine.base import InferSession
from ...utils.typings import RapidLayoutOutput
from ..base import BaseModelHandler
from .post_process import PPPostProcess
from .pre_process import PPPreProcess
from ..utils import ModelType

class PPDocLayoutModelHandler(BaseModelHandler):
    def __init__(self, labels, conf_thres: Union[float, dict], iou_thres, session: InferSession, model_type: ModelType):
        if model_type == ModelType.PP_DOCLAYOUT_PLUS_L:
            target_size = (800, 800)
        elif model_type == ModelType.PP_DOCLAYOUT_S:
            target_size = (480, 480)
        else:
            # PP_DOCLAYOUT_L、PP_DOCLAYOUT_M、RT_DETR_L_WIRED_TABLE_CELL_DET、RT_DETR_L_WIRELESS_TABLE_CELL_DET
            target_size = (640, 640)
        self.img_size = target_size
        self.pp_preprocess = PPPreProcess(img_size=self.img_size, model_type=model_type)
        self.pp_postprocess = PPPostProcess(labels, conf_thres, iou_thres)

        self.session = session

    def __call__(self, ori_img_list: List[np.ndarray]) -> List[RapidLayoutOutput]:
        s1 = time.perf_counter()
        # 1、前置处理
        img_inputs = []
        scale_factor_inputs = []
        for ori_img in ori_img_list:
            ori_img_shape = ori_img.shape[:2]
            img = self.preprocess(ori_img)
            scale_factor = [  # [w_scale, h_scale]
                self.img_size[0] / ori_img_shape[0],
                self.img_size[1] / ori_img_shape[1],
            ]
            img_inputs.append(img)
            scale_factor_inputs.append(scale_factor)
        img_inputs = np.concatenate(img_inputs, axis=0) # 拼接 batch
        scale_factor_inputs = np.array(scale_factor_inputs, np.float32)
        # 2、推理
        batch_preds = self.session(img_inputs, scale_factor_inputs)
        # 3、后处理
        batch_outputs = self._format_output(batch_preds)
        result_list = []
        for i, output in enumerate(batch_outputs):
            ori_img_shape = ori_img_list[i].shape[:2]
            datas = self.pp_postprocess(output["boxes"],[ori_img_shape[1], ori_img_shape[0]])
            if datas:
                boxes, scores, class_names = zip(*[(d["coordinate"], d["score"], d["label"]) for d in datas])
            else:
                boxes, scores, class_names = [], [], []
            elapse = time.perf_counter() - s1
            result = RapidLayoutOutput(img=ori_img_list[i], boxes=boxes,
                                       class_names=class_names, scores=scores, elapse=elapse)
            result_list.append(result)
        return result_list

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return self.pp_preprocess(image)

    def postprocess(self, ori_img_shape, img, preds):
        return self.pp_postprocess(ori_img_shape, img, preds)

    def _format_output(self, pred):
        """
        Transform batch outputs into a list of single image output.

        Args:
            pred (Sequence[Any]): The input predictions, which can be either a list of 3 or 4 elements.
                - When len(pred) == 4, it is expected to be in the format [boxes, class_ids, scores, masks],
                  compatible with SOLOv2 output.
                - When len(pred) == 3, it is expected to be in the format [boxes, box_nums, masks],
                  compatible with Instance Segmentation output.

        Returns:
            List[dict]: A list of dictionaries, each containing either 'class_id' and 'masks' (for SOLOv2),
                or 'boxes' and 'masks' (for Instance Segmentation), or just 'boxes' if no masks are provided.
        """
        box_idx_start = 0
        pred_box = []

        if len(pred) == 4:
            # Adapt to SOLOv2
            pred_class_id = []
            pred_mask = []
            pred_class_id.append([pred[1], pred[2]])
            pred_mask.append(pred[3])
            return [
                {
                    "class_id": np.array(pred_class_id[i]),
                    "masks": np.array(pred_mask[i]),
                }
                for i in range(len(pred_class_id))
            ]

        if len(pred) == 3:
            # Adapt to Instance Segmentation
            pred_mask = []
        for idx in range(len(pred[1])):
            np_boxes_num = pred[1][idx]
            box_idx_end = box_idx_start + np_boxes_num
            np_boxes = pred[0][box_idx_start:box_idx_end]
            pred_box.append(np_boxes)
            if len(pred) == 3:
                np_masks = pred[2][box_idx_start:box_idx_end]
                pred_mask.append(np_masks)
            box_idx_start = box_idx_end

        if len(pred) == 3:
            return [
                {"boxes": np.array(pred_box[i]), "masks": np.array(pred_mask[i])}
                for i in range(len(pred_box))
            ]
        else:
            return [{"boxes": np.array(res)} for res in pred_box]
