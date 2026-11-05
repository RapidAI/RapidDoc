import cv2
import numpy as np
from copy import deepcopy
from html import unescape
from loguru import logger

from rapid_doc.model.table.rapid_table_self.table_cls import TableCls
from rapid_doc.model.table.rapid_table_self import ModelType, RapidTable, RapidTableInput, EngineType
from rapid_doc.model.table.utils import (
    normalize_table_html_cell_text,
    normalize_table_ocr_text,
    select_best_table_model,
)
from rapid_doc.utils.boxbase import is_in
from rapid_doc.utils.config_reader import get_device
from rapid_doc.utils.ocr_utils import points_to_bbox, bbox_to_points


class RapidTableModel(object):
    def __init__(self, ocr_engine, table_config=None):
        if table_config is None:
            table_config = {}
        device = get_device()
        engine_cfg = None
        if device.startswith('cuda'):
            device_id = int(device.split(':')[1]) if ':' in device else 0  # GPU 编号
            engine_cfg = {'use_cuda': True, "cuda_ep_cfg.device_id": device_id}
        elif device.startswith('npu'):
            device_id = int(device.split(':')[1]) if ':' in device else 0  # npu 编号
            engine_cfg = {'use_cann': True, "cann_ep_cfg.device_id": device_id}
        engine_cfg = engine_cfg or {}
        # Override individual options while preserving the selected device.
        if table_config.get('engine_cfg'):
            engine_cfg.update(table_config['engine_cfg'])
        self.use_compare_table = table_config.get('use_compare_table') if table_config else False
        self.cls_conf_threshold = table_config.get('cls_conf_threshold', 0.9) if table_config else 0.9
        self.enable_blank_cell_rec = table_config.get('enable_blank_cell_rec', False) if table_config else False
        self.model_type = table_config.get("model_type", ModelType.UNET_SLANET_PLUS)
        if isinstance(self.model_type, str):
            self.model_type = ModelType('slanet_plus' if self.model_type == 'slanet-plus' else self.model_type)
        self.fallback_model_type = getattr(table_config.get('fallback_model_type'), 'value', table_config.get('fallback_model_type'))
        if self.fallback_model_type == 'slanet-plus':
            self.fallback_model_type = 'slanet_plus'
        self.fallback_to_text = table_config.get('fallback_to_text', False)
        if type(self.fallback_to_text) is not bool:
            raise ValueError('fallback_to_text must be bool')
        self.fallback_text_loss_threshold = table_config.get('fallback_text_loss_threshold', .2)
        loss = self.fallback_text_loss_threshold
        if isinstance(loss, bool) or not isinstance(loss, (int, float)) or not 0 <= loss <= 1:
            raise ValueError('fallback_text_loss_threshold must be between 0 and 1')
        supported = ('unet_slanet_plus', 'slanet_plus', 'slanet-plus', 'unet', 'table_transformer')
        if self.fallback_model_type is not None and self.fallback_model_type not in supported:
            raise ValueError(f'Unsupported fallback_model_type: {self.fallback_model_type}')
        self._fallback_config = dict(table_config)
        self._fallback_model = None
        self.ocr_engine = ocr_engine

        # 获取用户传入的 engine_type
        self.engine_type = table_config.get('engine_type') if table_config else None
        # # CPU 上优先使用 OpenVINO（如果可用且用户未指定 engine_type）
        # if device.startswith('cpu') and check_openvino() and not self.engine_type:
        #     self.engine_type = EngineType.OPENVINO

        if self.model_type == ModelType.UNET_SLANET_PLUS:
            cls_input_args = RapidTableInput(model_type=table_config.get("cls.model_type", ModelType.PADDLE_Q_CLS), engine_type=self.engine_type,
                                            model_dir_or_path=table_config.get("cls.model_dir_or_path"),
                                            engine_cfg=engine_cfg, use_ocr=False)
            self.table_cls = TableCls(cls_input_args)
            wired_input_args = RapidTableInput(model_type=ModelType.UNET, engine_type=self.engine_type,
                                               model_dir_or_path=table_config.get("unet.model_dir_or_path"),
                                               engine_cfg=engine_cfg, use_ocr=False)
            self.wired_table_model = RapidTable(wired_input_args)
            self._attach_wired_ocr_engine(self.wired_table_model)
            wireless_input_args = RapidTableInput(model_type=ModelType.SLANETPLUS, engine_type=self.engine_type,
                                                  model_dir_or_path=table_config.get("slanet_plus.model_dir_or_path"),
                                                  engine_cfg=engine_cfg, use_ocr=False)
            self.wireless_table_model = RapidTable(wireless_input_args)
        elif self.model_type == ModelType.UNET_UNITABLE:
            cls_input_args = RapidTableInput(model_type=table_config.get("cls.model_type", ModelType.PADDLE_Q_CLS), engine_type=self.engine_type,
                                            model_dir_or_path=table_config.get("cls.model_dir_or_path"),
                                            engine_cfg=engine_cfg, use_ocr=False)
            self.table_cls = TableCls(cls_input_args)
            wired_input_args = RapidTableInput(model_type=ModelType.UNET, engine_type=self.engine_type,
                                               model_dir_or_path=table_config.get("unet.model_dir_or_path"),
                                               engine_cfg=engine_cfg, use_ocr=False)
            self.wired_table_model = RapidTable(wired_input_args)
            self._attach_wired_ocr_engine(self.wired_table_model)
            wireless_input_args = RapidTableInput(model_type=ModelType.UNITABLE, engine_type=EngineType.TORCH,
                                                  model_dir_or_path=table_config.get("unitable.model_dir_or_path"),
                                                  engine_cfg=engine_cfg, use_ocr=False)
            self.wireless_table_model = RapidTable(wireless_input_args)
        elif self.model_type == ModelType.UNET_SLANET1M:
            cls_input_args = RapidTableInput(model_type=table_config.get("cls.model_type", ModelType.PADDLE_Q_CLS), engine_type=self.engine_type,
                                            model_dir_or_path=table_config.get("cls.model_dir_or_path"),
                                            engine_cfg=engine_cfg, use_ocr=False)
            self.table_cls = TableCls(cls_input_args)
            wired_input_args = RapidTableInput(model_type=ModelType.UNET, engine_type=self.engine_type,
                                               model_dir_or_path=table_config.get("unet.model_dir_or_path"),
                                               engine_cfg=engine_cfg, use_ocr=False)
            self.wired_table_model = RapidTable(wired_input_args)
            self._attach_wired_ocr_engine(self.wired_table_model)
            wireless_input_args = RapidTableInput(model_type=ModelType.SLANET1M, engine_type=self.engine_type,
                                                  model_dir_or_path=table_config.get("slanet_1m.model_dir_or_path"),
                                                  engine_cfg=engine_cfg, use_ocr=False)
            self.wireless_table_model = RapidTable(wireless_input_args)
        else:
            if self.model_type == ModelType.UNITABLE:
                self.engine_type = EngineType.TORCH
            input_args = RapidTableInput(model_type=self.model_type, engine_type=self.engine_type,
                                         model_dir_or_path=table_config.get("model_dir_or_path"),
                                         engine_cfg=engine_cfg, use_ocr=False)
            self.table_model = RapidTable(input_args)
            if self.model_type == ModelType.UNET:
                self._attach_wired_ocr_engine(self.table_model)

    def _attach_wired_ocr_engine(self, table_model):
        if not self.enable_blank_cell_rec:
            return
        table_structure = getattr(table_model, "table_structure", None)
        if table_structure is not None and hasattr(table_structure, "ocr_engine"):
            table_structure.ocr_engine = self.ocr_engine

    def rule_table_class(self, image):
        """Return wired/wireless eligibility without running OCR.

        Single-model configurations preserve their existing routing: UNET is
        inherently wired; all other single models skip rule parsing.
        """
        combined = [ModelType.UNET_SLANET_PLUS, ModelType.UNET_SLANET1M, ModelType.UNET_UNITABLE]
        if self.model_type == ModelType.UNET:
            return "wired", 1.0
        if self.model_type not in combined:
            return None, 1.0
        result, scores, _ = self.table_cls([cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)], return_scores=True)
        return (result[0] if result else None, scores[0] if scores else 1.0)

    def batch_predict(self, images: list, ocr_result=None, fill_image_res=None, mfd_res=None, skip_text_in_image=True, skip_table_orientation=None, table_class=None, table_class_score=1.0) -> list[str]:
        results = []
        for image in images:
            res = self.predict(
                image=image,
                ocr_result=ocr_result,
                fill_image_res=fill_image_res,
                mfd_res=mfd_res,
                skip_text_in_image=skip_text_in_image,
                skip_table_orientation=skip_table_orientation,
                table_class=table_class,
                table_class_score=table_class_score,
            )
            results.append(res)
        return results

    def predict(self, image, ocr_result=None, fill_image_res=None, mfd_res=None, skip_text_in_image=True, skip_table_orientation=None, table_class=None, table_class_score=1.0):
        bgr_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        if (self.model_type == ModelType.TABLE_TRANSFORMER or self.fallback_model_type is not None or self.fallback_to_text) and ocr_result is not None:
            # Formula/image placeholders must not mutate externally supplied OCR.
            ocr_result = deepcopy(ocr_result)
            ocr_result = [list(part) for part in ocr_result]

        if skip_table_orientation is None:
            skip_table_orientation = ocr_result is not None

        # First check the overall image aspect ratio (height/width)
        img_height, img_width = bgr_image.shape[:2]
        img_aspect_ratio = img_height / img_width if img_width > 0 else 1.0
        img_is_portrait = img_aspect_ratio > 1.2

        if img_is_portrait and not skip_table_orientation:

            from rapid_doc.model.orientation.rapid_orientation_model import RapidOrientationModel
            if not hasattr(self, '_orientation_model'):
                self._orientation_model = RapidOrientationModel()
            label = self._orientation_model.predict(np.asarray(image), table_mode=True)
            if label in ('90', '270'):
                code = cv2.ROTATE_90_CLOCKWISE if label == '270' else cv2.ROTATE_90_COUNTERCLOCKWISE
                image = cv2.rotate(np.asarray(image), code)
                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Continue with OCR on potentially rotated image
        if not ocr_result:
            ocr_result = self.ocr_engine.ocr(bgr_image, mfd_res=mfd_res)[0]
            if ocr_result:
                ocr_result = [list(x) for x in zip(*[
                    [item[0], normalize_table_ocr_text(item[1][0]), item[1][1]]
                    for item in ocr_result
                ])]
            else:
                ocr_result = None

        if not ocr_result:
            if self.model_type == ModelType.TABLE_TRANSFORMER:
                ocr_result = [[], [], []]
            else:
                return None
        # 把图片结果，添加到ocr_result里。uuid作为占位符，后面保存图片时替换
        if fill_image_res:
            for fill_image in fill_image_res:
                bbox = points_to_bbox(fill_image['ocr_bbox'])
                cv2.rectangle(bgr_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), thickness=-1) # 填白图像区域，防止表格识别被影响
                ocr_result[0].append(fill_image['ocr_bbox'])
                ocr_result[1].append(fill_image['uuid'])
                ocr_result[2].append(1)
                if skip_text_in_image:
                    # 找出所有 OCR 框在图片框内的下标
                    delete_indices = []
                    for idx, ocr in enumerate(ocr_result[0][:-1]):  # 排除刚添加的图片框自身
                        if is_in(points_to_bbox(ocr), points_to_bbox(fill_image['ocr_bbox'])):
                            delete_indices.append(idx)
                    # 按逆序删除，防止下标错位
                    for idx in sorted(delete_indices, reverse=True):
                        del ocr_result[0][idx]
                        del ocr_result[1][idx]
                        del ocr_result[2][idx]
        # 表格内的公式填充
        if mfd_res:
            from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import inline_left_delimiter, inline_right_delimiter

            for mfd in mfd_res:
                if mfd.get('latex'):
                    ocr_result[1].append(
                        normalize_table_ocr_text(
                            f"{inline_left_delimiter}{mfd['latex']}{inline_right_delimiter}"
                        )
                    )
                elif mfd.get('checkbox'):
                    ocr_result[1].append(normalize_table_ocr_text(mfd['checkbox']))
                else:
                    continue
                ocr_result[0].append(bbox_to_points(mfd['bbox']))
                ocr_result[2].append(1)

        main = self._structure(bgr_image, ocr_result, table_class, table_class_score)
        if self.fallback_model_type is None and not self.fallback_to_text:
            return main
        from .fallback import quality, plain_html
        excluded = [i['uuid'] for i in (fill_image_res or [])]
        if mfd_res:
            from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import inline_left_delimiter, inline_right_delimiter
            excluded += [normalize_table_ocr_text(f"{inline_left_delimiter}{f['latex']}{inline_right_delimiter}") if f.get('latex') else normalize_table_ocr_text(f['checkbox']) for f in mfd_res if f.get('latex') or f.get('checkbox')]
        first = quality(main, ocr_result, excluded)
        if not first.needs_fallback(self.fallback_text_loss_threshold):
            return main
        if self.fallback_model_type is not None and self.fallback_model_type != getattr(self.model_type, 'value', self.model_type):
            if self._fallback_model is None:
                config = dict(self._fallback_config)
                config.pop('fallback_model_type', None)
                config['model_type'] = self.fallback_model_type
                config['model_dir_or_path'] = config.pop('fallback_model_dir_or_path', None)
                self._fallback_model = RapidTableModel(self.ocr_engine, config)
            alternative = self._fallback_model._structure(bgr_image, ocr_result)
            second = quality(alternative, ocr_result, excluded)
            if second.valid and second.retained > first.retained and not second.needs_fallback(self.fallback_text_loss_threshold):
                logger.info('Table fallback selected: {} (retained {}/{}, main {})', self.fallback_model_type, second.retained, second.total, first.retained)
                return alternative
        if self.fallback_to_text:
            logger.warning('Table text recovery: retained {}/{}; emitting existing OCR text', first.retained, first.total)
            return plain_html(ocr_result)
        return main

    def _structure(self, bgr_image, ocr_result, table_class=None, table_class_score=1.0):
        cls = table_class
        cls_score = table_class_score

        """使用 rapid_table_self 识别"""
        try:
            if self.model_type == ModelType.TABLE_TRANSFORMER:
                # The PDF/OCR pipeline escapes text once. XML serialization will
                # escape it again, so restore raw text only at this boundary.
                ocr_result = [ocr_result[0], [unescape(str(text)) for text in ocr_result[1]], ocr_result[2]]
            ocr_result = [ocr_result]
            bgr_image = [bgr_image]

            if self.model_type in [ModelType.UNET_SLANET_PLUS, ModelType.UNET_SLANET1M, ModelType.UNET_UNITABLE]:
                if not cls:
                    cls, cls_scores, elasp = self.table_cls(bgr_image, return_scores=True)
                    cls = cls[0]
                    cls_score = cls_scores[0] if cls_scores else 1.0
                should_compare = self.use_compare_table or cls_score < self.cls_conf_threshold
                if cls == "wired":
                    wired_pred = self.wired_table_model(bgr_image, ocr_result).pred_htmls
                    wired_html_code = wired_pred[0] if len(wired_pred) > 0 else None
                    if should_compare:
                        wireless_pred = self.wireless_table_model(bgr_image, ocr_result).pred_htmls
                        wireless_html_code = wireless_pred[0] if len(wireless_pred) > 0 else None
                        html_code = select_best_table_model(ocr_result[0], wired_html_code, wireless_html_code)
                    else:
                        html_code = wired_html_code
                else:  # wireless
                    html = self.wireless_table_model(bgr_image, ocr_result).pred_htmls
                    wireless_html_code = html[0] if len(html) > 0 else None
                    if should_compare:
                        wired_pred = self.wired_table_model(bgr_image, ocr_result).pred_htmls
                        wired_html_code = wired_pred[0] if len(wired_pred) > 0 else None
                        html_code = select_best_table_model(ocr_result[0], wired_html_code, wireless_html_code)
                    else:
                        html_code = wireless_html_code
            else:
                html_code = self.table_model(bgr_image, ocr_result).pred_htmls[0]
            return normalize_table_html_cell_text(html_code)
        except Exception as e:
            logger.exception(e)
            return None
