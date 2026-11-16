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


class _TableMarkup(str):
    """Internal HTML result carrying the corresponding model grid."""

    def __new__(cls, html, prediction):
        result = super().__new__(cls, html)
        result.grid = (prediction.cell_bboxes[0], prediction.logic_points[0]) if (
            len(prediction.cell_bboxes) and len(prediction.logic_points)
        ) else None
        assignments = getattr(prediction, 'token_cells', ())
        result.token_cells = assignments[0] if assignments else None
        grids = getattr(prediction, 'evidence_grids', ())
        result.evidence_grid = grids[0] if grids and grids[0] is not None else result.grid
        return result


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
        supported = ('unet_slanet_plus', 'slanet_plus', 'slanet-plus', 'unet', 'table_transformer', 'img2table')
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

        if self.model_type == ModelType.IMG2TABLE:
            from .img2table_backend import Img2TableBackend
            self.img2table_model = Img2TableBackend(table_config.get('img2table', {}))
        elif self.model_type == ModelType.UNET_SLANET_PLUS:
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

    def predict(self, image, ocr_result=None, fill_image_res=None, mfd_res=None, skip_text_in_image=True, skip_table_orientation=None, table_class=None, table_class_score=1.0, *, vector_lines=(), _allow_table_split=True):
        bgr_image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        if (self.model_type in (ModelType.TABLE_TRANSFORMER, ModelType.IMG2TABLE) or self.fallback_model_type is not None or self.fallback_to_text) and ocr_result is not None:
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
        protected_token_count = 0
        # 把图片结果，添加到ocr_result里。uuid作为占位符，后面保存图片时替换
        if fill_image_res:
            for fill_image in fill_image_res:
                bbox = points_to_bbox(fill_image['ocr_bbox'])
                cv2.rectangle(bgr_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), thickness=-1) # 填白图像区域，防止表格识别被影响
                ocr_result[0].append(fill_image['ocr_bbox'])
                ocr_result[1].append(fill_image['uuid'])
                ocr_result[2].append(1)
                protected_token_count += 1
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
                protected_token_count += 1

        protected = {'protected_token_count': protected_token_count} if protected_token_count else {}
        main = self._structure(bgr_image, ocr_result, table_class, table_class_score, **protected)
        main_grid = getattr(main, 'grid', None)
        main_evidence_grid = getattr(main, 'evidence_grid', main_grid)
        main_tokens = getattr(main, 'token_cells', None)
        if isinstance(main, _TableMarkup):
            main = str(main)
        if self.fallback_model_type is None and not self.fallback_to_text:
            return main
        from .fallback import (grid_evidence, join_aligned_row_bands, keeps_candidate_numbers, quality, plain_html, repair_local_grid,
                               restore_internal_rows, restore_trailing_row, structure_evidence)
        excluded = [i['uuid'] for i in (fill_image_res or [])]
        if mfd_res:
            from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import inline_left_delimiter, inline_right_delimiter
            excluded += [normalize_table_ocr_text(f"{inline_left_delimiter}{f['latex']}{inline_right_delimiter}") if f.get('latex') else normalize_table_ocr_text(f['checkbox']) for f in mfd_res if f.get('latex') or f.get('checkbox')]
        first = quality(main, ocr_result, excluded)
        if not first.valid:
            repaired = repair_local_grid(main)
            if repaired is not None:
                # 修正 HTML 中的空占位。候选的原始像素几何仍用于核对原图。
                main = repaired
                first = quality(main, ocr_result, excluded)
        text_loss = first.needs_fallback(self.fallback_text_loss_threshold)
        has_backup = (self.fallback_model_type is not None
                      and self.fallback_model_type != getattr(self.model_type, 'value', self.model_type))
        evidence = structure_evidence(main, ocr_result, excluded) if has_backup else None
        grid = grid_evidence(main_grid, ocr_result, excluded) if has_backup else None
        grid_suspicious = grid is not None and len(grid.conflicts) >= 2
        source = None
        if has_backup and main_evidence_grid is not None:
            from .table_evidence import TableEvidence

            source = TableEvidence(bgr_image, ocr_result, excluded, vector_lines)
        primary_score = source.score(main_evidence_grid, main_tokens) if source is not None else None
        source_suspicious = primary_score is not None and primary_score.suspicious
        if (_allow_table_split and not excluded and primary_score and primary_score.suspicious
                and source is not None):
            repaired_sections = self._recognize_stacked_sections(
                bgr_image, ocr_result, source, vector_lines, first, primary_score)
            if repaired_sections is not None:
                return repaired_sections
        ruled_rows = source.ruled_empty_rows(main_evidence_grid, evidence.empty_indices) if source is not None else set()
        structure_suspicious = evidence and evidence.suspicious_with_ruling(ruled_rows)
        if not text_loss and not structure_suspicious and not grid_suspicious and not source_suspicious:
            if has_backup:
                logger.debug('Table fallback skipped: retained {}/{}, row mismatch {}, split lines {}, merged values {}, consecutive empty rows {}, ruled blank rows {}, evidence {}',
                             first.retained, first.total, evidence.row_misalignment,
                             evidence.split_lines, evidence.merged_values, evidence.empty_row_run, len(ruled_rows),
                             primary_score.summary() if primary_score else None)
            return main
        if has_backup:
            if self._fallback_model is None:
                config = dict(self._fallback_config)
                config.pop('fallback_model_type', None)
                config['model_type'] = self.fallback_model_type
                config['model_dir_or_path'] = config.pop('fallback_model_dir_or_path', None)
                self._fallback_model = RapidTableModel(self.ocr_engine, config)
                self._fallback_model._collect_evidence = True
            alternative = self._fallback_model._structure(bgr_image, ocr_result)
            second = quality(alternative, ocr_result, excluded)
            backup = structure_evidence(alternative, ocr_result, excluded)
            alternative_grid = grid_evidence(getattr(alternative, 'grid', None), ocr_result, excluded)
            comparable = grid is not None and alternative_grid is not None
            gains, losses = alternative_grid.compare(grid) if comparable else (0, 0)
            better_grid = (gains - losses >= 3 and gains >= 2 * losses
                           and backup.row_count <= evidence.row_count + 1)
            more_consistent_rows = (losses == 0 and backup.row_count < evidence.row_count
                                    and backup.full_width_rows < evidence.full_width_rows
                                    and backup.row_misalignment < evidence.row_misalignment)
            retention_margin = .05 if better_grid and losses == 0 and gains >= 5 else .03
            candidate_grid = getattr(alternative, 'evidence_grid', getattr(alternative, 'grid', None))
            candidate_tokens = getattr(alternative, 'token_cells', None)
            candidate_score = source.score(candidate_grid, candidate_tokens) if primary_score else None
            if candidate_score and candidate_score.exact and primary_score.exact:
                # 公式占位已包含在共享 OCR 中。图片占位不能通过文字带重建。
                if second.valid and not fill_image_res and not candidate_score.missing:
                    band_source = (TableEvidence(bgr_image, ocr_result, vector_lines=vector_lines)
                                          if excluded else source)
                    joined = join_aligned_row_bands(
                        alternative, candidate_grid, candidate_tokens, band_source, ocr_result)
                    if (joined is not None
                            and quality(joined, ocr_result, excluded).retained >= max(first.retained, second.retained)
                            and quality(joined, ocr_result).retained >= quality(alternative, ocr_result).retained):
                        logger.info('Table fallback selected: {} (joined aligned row bands)',
                                    self.fallback_model_type)
                        return joined
                lost = candidate_score.missing - primary_score.missing
                restored = restore_trailing_row(
                    main, alternative, main_evidence_grid, candidate_grid, main_tokens,
                    candidate_tokens, source, lost,
                )
                if restored is None and not first.valid:
                    restored = restore_internal_rows(
                        alternative, candidate_grid, candidate_tokens, source, lost, ocr_result)
                if restored is not None:
                    html, restored_grid, restored_tokens = restored
                    restored_quality = quality(html, ocr_result, excluded)
                    restored_score = source.score(restored_grid, restored_tokens)
                    if (restored_quality.valid and restored_quality.retained >= max(first.retained, second.retained)
                            and restored_score and restored_score.missing <= primary_score.missing):
                        alternative, candidate_grid, candidate_tokens = restored
                        second, candidate_score = restored_quality, restored_score
                        backup = structure_evidence(alternative, ocr_result, excluded)
            more_text = text_loss and second.can_replace(first, self.fallback_text_loss_threshold, self.fallback_to_text)
            delta = candidate_score.improvement_over(primary_score) if candidate_score else (0., 0., 0.)
            text_gain, line_gain, gap_gain = delta
            # 框与 OCR 的覆盖差异也可能来自两模型的坐标习惯。只有实际文字实例恢复、
            # 或框线与间距的合计改善足够明显时，才用新证据扩大切换范围。
            recovered = (primary_score.missing - candidate_score.missing
                         if candidate_score and candidate_score.exact and primary_score.exact else set())
            geometry_gain = line_gain + gap_gain
            net_gain = .5 * text_gain + geometry_gain
            # 空白填写格本身也是结构。缩减行数若超出没有框线支撑的空行数，保留主网格。
            protected_empty_rows = (first.valid and bool(ruled_rows) and evidence.row_count - backup.row_count
                                    > evidence.empty_rows - len(ruled_rows))
            # 旧的拆行、串格和近似归属检查只能提供局部证据，不能独自支持大幅缩表。
            loses_ruling = candidate_score is not None and candidate_score.loses_supported_lines(primary_score, line_gain)
            protected_structure = (loses_ruling
                                   and (backup.row_count <= .75 * evidence.row_count
                                        or comparable and alternative_grid.width <= .75 * grid.width)
                                   and not recovered and gap_gain < .5)
            # OCR 文字相同、间距也未改善时，旧的 HTML 行错位推断不能压过可见框线退步。
            ruling_disagrees = (candidate_score is not None and line_gain <= -.15
                                and gap_gain <= 0 and text_gain <= 0 and gains <= losses)
            removes_noise_rows = backup.removes_empty_rows(evidence, ruled_rows) and not loses_ruling
            # 只有确认在清理无支撑空行，并有独立文字归属改善时，才放宽新增列限制。
            cleanup_supports_columns = removes_noise_rows and not ruled_rows and text_gain >= .65 and gap_gain >= 0
            # 行结构改善本身不能证明增列合理，须有独立间距改善且框线未退步。
            consistent_layout = more_consistent_rows and line_gain >= 0 and gap_gain >= .35 and text_gain >= 0
            unsupported_extra_columns = (comparable and alternative_grid.width > grid.width and gains < 3
                                         and not (cleanup_supports_columns or consistent_layout))
            conflicts_with_geometry = comparable and (losses - gains >= 4 or unsupported_extra_columns)
            better_structure = (second.valid and first.valid and not protected_structure and not ruling_disagrees
                                and (better_grid or (not conflicts_with_geometry
                                     and (backup.improves(evidence) or removes_noise_rows)))
                                and not second.needs_fallback(self.fallback_text_loss_threshold)
                                and second.retained >= first.retained - first.total * retention_margin)
            # 同一视觉行对齐可以提示错位，但不能单独证明增加逻辑行是正确的。
            row_gains = candidate_score.separator_gains(primary_score, 1) if candidate_score else 0
            col_gains = candidate_score.separator_gains(primary_score, 0) if candidate_score else 0
            row_lines = candidate_score.line_improvement(primary_score, 1) if candidate_score else 0.
            col_lines = candidate_score.line_improvement(primary_score, 0) if candidate_score else 0.
            extra_rows_supported = (backup.row_count <= evidence.row_count or row_lines >= .35
                                    or row_gains >= 2 or len(recovered) >= 2)
            extra_columns_supported = not unsupported_extra_columns or col_gains >= 3 or col_lines >= .55
            # 旧的 OCR 框重叠归属属于近似值。真实归属与原图框线同时改善时，
            # 允许两份相互独立的证据纠正旧检查；任一层明显退步时保留原保护。
            corroborated = (candidate_score is not None and candidate_score.exact and primary_score.exact
                            and line_gain >= .35 and gap_gain >= .15 and text_gain >= 0)
            geometry_supported = losses - gains < 4 or corroborated
            local_line_gain = candidate_score.local_line_improvement(primary_score) if candidate_score else 0.
            # 在行列规模一致时，局部线段可以确认边界位置改善；同时检查整表的
            # 框线、间距和文字，避免只修好一处就接受其他位置的明显退步。
            local_boundary_supported = (comparable and alternative_grid.width == grid.width
                                        and backup.row_count == evidence.row_count
                                        and local_line_gain >= .5 and line_gain > 0
                                        and gap_gain >= -.2 and text_gain >= .5)
            # 相同规模下，分散的空行也可能是错归造成的。用实际 OCR 归属
            # 交叉验证，并检查相邻文本的行列关系和原图线段是否退步。
            restores_row_content = (candidate_score is not None and candidate_score.exact
                                    and primary_score.exact and comparable
                                    and alternative_grid.width == grid.width
                                    and backup.row_count == evidence.row_count
                                    and evidence.empty_rows - backup.empty_rows >= 2
                                    and evidence.empty_rows >= evidence.row_count / 4
                                    and second.retained >= first.retained
                                    and text_gain >= .65 and line_gain >= 0 and gap_gain >= 0)
            better_evidence = (net_gain >= .65 and second.valid
                               and (geometry_gain >= .5 or local_boundary_supported
                                    or len(recovered) >= 2 and geometry_gain >= 0)
                               and not second.needs_fallback(self.fallback_text_loss_threshold)
                               and second.retained >= first.retained - first.total * .03
                               and extra_rows_supported and extra_columns_supported
                               and geometry_supported)
            better_evidence |= second.valid and restores_row_content and losses <= gains
            keeps_numbers = keeps_candidate_numbers(
                evidence.numbers, backup.numbers, ocr_result, primary_score, candidate_score
            )
            logger.debug('Table fallback evidence: main {}, alternative {}, delta {}, trigger {}',
                         primary_score.summary() if primary_score else None,
                         candidate_score.summary() if candidate_score else None,
                         tuple(round(v, 3) for v in delta), source_suspicious)
            repairs_grid = not first.valid and second.valid and (
                not second.needs_fallback(self.fallback_text_loss_threshold) or more_text
            )
            unsupported_splits = 0
            if (first.valid and not more_text and backup.row_count > evidence.row_count
                    and candidate_score and candidate_score.exact and primary_score.exact):
                unsupported_splits = source.unsupported_row_splits(
                    main_evidence_grid, candidate_grid, main_tokens, candidate_tokens)
                if unsupported_splits:
                    logger.debug('Table fallback has {} unsupported row splits', unsupported_splits)
            if ((more_text or better_structure or better_evidence or repairs_grid) and keeps_numbers
                    and not protected_empty_rows and not unsupported_splits):
                reason = 'text'
                if better_evidence:
                    reason = 'evidence'
                elif better_structure:
                    reason = 'grid' if better_grid else 'structure'
                elif repairs_grid:
                    reason = 'valid grid'
                logger.info('Table fallback selected: {} ({}, retained {}/{}, main {})',
                            self.fallback_model_type, reason,
                            second.retained, second.total, first.retained)
                return str(alternative) if isinstance(alternative, _TableMarkup) else alternative
            logger.debug('Table fallback candidate rejected: {} valid={} retained={}/{} row mismatch={}→{} empty rows={}→{} keeps numbers={} grid gains={} losses={} ruled blank rows={} protected empty rows={} protected structure={}',
                         self.fallback_model_type, second.valid, second.retained, second.total,
                         evidence.row_misalignment, backup.row_misalignment,
                         evidence.empty_row_run, backup.empty_row_run, keeps_numbers, gains, losses,
                         len(ruled_rows), protected_empty_rows, protected_structure)
        if self.fallback_to_text and text_loss:
            logger.warning('Table text recovery: retained {}/{}; emitting existing OCR text', first.retained, first.total)
            return plain_html(ocr_result)
        return main

    def _recognize_stacked_sections(self, image, rows, source, vector_lines, primary, primary_score):
        from .fallback import quality

        height, width = image.shape[:2]
        cut = source.stacked_table_boundary(rows, height, width)
        if cut is None:
            return None
        sections = []
        retained = 0
        for top, bottom in ((0, cut), (cut, height)):
            local = [[], [], []]
            for points, text, confidence in zip(*rows):
                box = points_to_bbox(points)
                if top <= (box[1] + box[3]) / 2 < bottom:
                    if box[1] < top or box[3] > bottom:
                        return None
                    local[0].append([[x, y - top] for x, y in points])
                    local[1].append(text)
                    local[2].append(confidence)
            lines = [(a, p - top, lo, hi) if a == 1 else
                     (a, p, max(top, lo) - top, min(bottom, hi) - top)
                     for a, p, lo, hi in vector_lines
                     if (top <= p <= bottom if a == 1 else max(top, lo) < min(bottom, hi))]
            if not local[1]:
                return None
            html = self.predict(cv2.cvtColor(image[top:bottom], cv2.COLOR_BGR2RGB), local,
                                skip_table_orientation=True, vector_lines=lines, _allow_table_split=False)
            result = quality(html, local)
            if not result.valid or result.needs_fallback(self.fallback_text_loss_threshold):
                return None
            retained += result.retained
            sections.append(html[html.find('<table'):html.rfind('</table>') + len('</table>')])
        # 拆分会产生两个独立结果，文字完整且主候选存在明显归属问题时才提交。
        if (retained < primary.retained or any(not section for section in sections)
                or (retained == primary.retained and sum(primary_score.text.values()) < 2)):
            return None
        logger.debug('Split stacked tables at crop row {} with retained OCR {}/{}', cut, retained, primary.total)
        return '\n\n'.join(sections)

    def _structure(self, bgr_image, ocr_result, table_class=None, table_class_score=1.0, *, protected_token_count=0):
        cls = table_class
        cls_score = table_class_score

        """使用 rapid_table_self 识别"""
        try:
            if self.model_type == ModelType.IMG2TABLE:
                return normalize_table_html_cell_text(self.img2table_model.predict(bgr_image, ocr_result))
            if self.model_type == ModelType.TABLE_TRANSFORMER:
                # The PDF/OCR pipeline escapes text once. XML serialization will
                # escape it again, so restore raw text only at this boundary.
                ocr_result = [ocr_result[0], [unescape(str(text)) for text in ocr_result[1]], ocr_result[2]]
            ocr_result = [ocr_result]
            bgr_image = [bgr_image]
            kwargs = {'collect_evidence': True} if (
                self.fallback_model_type is not None or getattr(self, '_collect_evidence', False)
            ) else {}
            if protected_token_count:
                kwargs['protected_token_count'] = protected_token_count

            if self.model_type in [ModelType.UNET_SLANET_PLUS, ModelType.UNET_SLANET1M, ModelType.UNET_UNITABLE]:
                if not cls:
                    cls, cls_scores, elasp = self.table_cls(bgr_image, return_scores=True)
                    cls = cls[0]
                    cls_score = cls_scores[0] if cls_scores else 1.0
                should_compare = self.use_compare_table or cls_score < self.cls_conf_threshold
                models = {"wired": self.wired_table_model, "wireless": self.wireless_table_model}
                route = "wired" if cls == "wired" else "wireless"
                prediction = models[route](bgr_image, ocr_result, **kwargs)
                html_code = prediction.pred_htmls[0] if prediction.pred_htmls else None
                if should_compare:
                    other_route = "wireless" if route == "wired" else "wired"
                    other = models[other_route](bgr_image, ocr_result, **kwargs)
                    other_html = other.pred_htmls[0] if other.pred_htmls else None
                    options = {route: html_code, other_route: other_html}
                    selected = select_best_table_model(ocr_result[0], options["wired"], options["wireless"])
                    wired = prediction if route == "wired" else other
                    wireless = other if route == "wired" else prediction
                    if self._prefer_verified_wired(wired, wireless, options, ocr_result[0]):
                        selected = options["wired"]
                    if selected != html_code:
                        prediction = other
                    html_code = selected
            else:
                prediction = self.table_model(bgr_image, ocr_result, **kwargs)
                html_code = prediction.pred_htmls[0]
            html_code = normalize_table_html_cell_text(html_code)
            return _TableMarkup(html_code, prediction) if html_code else html_code
        except Exception as e:
            logger.exception(e)
            return None

    @staticmethod
    def _prefer_verified_wired(wired, wireless, options, rows):
        splits = sum(getattr(wired, 'verified_cell_splits', ()))
        if not splits or not options["wired"] or not options["wireless"]:
            return False
        from .fallback import quality

        wired_quality = quality(options["wired"], rows)
        wireless_quality = quality(options["wireless"], rows)
        if not wired_quality.valid:
            return False
        # 同一逻辑网格内，用经原图确认且重新读到文字的单元格衡量改善。
        # 跨格 OCR 的原字符数本身可能偏多，不能把它当成绝对真值。
        if not len(wired.logic_points) or not len(wireless.logic_points):
            return False
        wired_grid = sorted(map(tuple, wired.logic_points[0]))
        wireless_grid = sorted(map(tuple, wireless.logic_points[0]))
        loss = wireless_quality.retained - wired_quality.retained
        return (bool(wired_grid) and wired_grid == wireless_grid and loss <= splits / 3
                and loss <= .08 * wired_quality.total)
