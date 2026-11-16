# Copyright (c) RapidAI. All rights reserved.
"""
批量分析模块
"""
import copy
from typing import List, Dict
from collections import defaultdict

import cv2
import numpy as np
from loguru import logger
from tqdm import tqdm

from .layout_recovery import _has_foreground_ink
from .model_init import AtomModelSingleton
from .model_list import AtomicModel
from ...model.table.utils import normalize_table_ocr_text
from ...model.table.rule_table import build_rule_table, build_track_table
from ...model.docvortex_adapter import recover_docvortex_table, use_docvortex_table_rules, use_docvortex_ocr_table_rules
from ...utils.bbox_utils import normalize_to_int_bbox
from ...utils.boxbase import rotate_table_image
from ...utils.enum_class import CategoryId
from ...utils.model_utils import crop_img
from ...utils.ocr_utils import (
    merge_det_boxes, update_det_boxes, sorted_boxes, get_rotate_crop_image,
    get_adjusted_mfdetrec_res, get_ocr_result_list, OcrConfidence, get_ocr_result_list_table
)
from ...utils.span_pre_proc import (
    txt_spans_extract, txt_spans_bbox_extract,
    txt_most_angle_extract_table, extract_table_fill_image
)


# =================================== OCR-det ===================================
def _native_row_gaps(image, native_boxes, useful_list):
    pad_x, pad_y, x0, y0, x1, y1, _, _ = useful_list
    bands = sorted((min(p[1] for p in box), max(p[1] for p in box)) for box in native_boxes)
    line_height = float(np.median([bottom - top for top, bottom in bands]))
    if line_height <= 0:
        return []
    # 只检查高度足够容纳另一行的空档。同行内的缺字继续由原生文本负责。
    end = pad_y + y1 - y0
    cursor = pad_y
    for top, bottom in (*bands, (end, end)):
        top = min(top, end)
        if top - cursor >= .8 * line_height and _has_foreground_ink(
            image, (pad_x, cursor, pad_x + x1 - x0, top)
        ):
            return bands
        cursor = max(cursor, bottom)
    return []


def _without_native_rows(boxes, native_bands):
    missing = []
    for box in boxes:
        top, bottom = min(p[1] for p in box), max(p[1] for p in box)
        if bottom > top and not any(
            min(bottom, end) - max(top, start) > .2 * min(bottom - top, end - start)
            for start, end in native_bands
        ):
            missing.append(box)
    return missing


def _extract_text_from_pdf(
        ocr_res_all_page: List[Dict],
        pdf_dict_list: List[Dict],
        scale_list: List[float],
        *,
        recover_missing_rows=False,
):
    """从 PDF 中提取文本框"""
    ocr_res_grouped = {}
    for x in ocr_res_all_page:
        ocr_res_grouped.setdefault(x["page_idx"], []).append(x)

    total_texts = sum(len(texts) for texts in ocr_res_grouped.values())

    with tqdm(total=total_texts, desc="PDF-det Predict") as pbar:
        for page_idx, text_list in ocr_res_grouped.items():
            page_dict = pdf_dict_list[page_idx] if text_list else {}
            scale = scale_list[page_idx] if text_list else 1.0

            for ocr_res_dict in text_list:
                if ocr_res_dict['ocr_enable']:
                    continue
                if page_dict.get("rotate_label") in ["90", "180", "270"]:
                    ocr_res_dict['ocr_enable'] = True
                    continue

                for res in ocr_res_dict['ocr_res_list']:
                    if res.get('native_text'):
                        ocr_res_dict['layout_res'].append({
                            'poly': res['poly'],
                            'category_id': CategoryId.OcrText,
                            'score': 1.0,
                            'text': res['native_text'],
                            'docvortex_native_text': True,
                            'original_label': res.get('original_label'),
                            'original_order': res.get('original_order'),
                        })
                        continue
                    new_image, useful_list = crop_img(
                        res, ocr_res_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                    )

                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                        ocr_res_dict['single_page_mfdetrec_res'] + ocr_res_dict['checkbox_res'],
                        useful_list
                    )

                    bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                    ocr_res = txt_spans_bbox_extract(
                        page_dict, res, mfd_res=adjusted_mfdetrec_res,
                        scale=scale, useful_list=useful_list
                    )

                    if ocr_res:
                        if recover_missing_rows:
                            native_bands = _native_row_gaps(bgr_image, ocr_res, useful_list)
                            if native_bands:
                                res['_native_ocr_bands'] = native_bands
                        ocr_result_list = get_ocr_result_list(
                            ocr_res, useful_list, ocr_res_dict['ocr_enable'],
                            bgr_image, ocr_res_dict['lang'],
                            res['original_label'], res['original_order']
                        )
                        ocr_res_dict['layout_res'].extend(ocr_result_list)

                pbar.update(1)

def _apply_mask_boxes_to_image(
    bgr_image: np.ndarray,
    mask_boxes: list[dict] | None,
) -> np.ndarray:
    if not mask_boxes:
        return bgr_image

    masked_image = bgr_image.copy()
    image_h, image_w = masked_image.shape[:2]
    for mask_box in mask_boxes:
        bbox = mask_box.get("bbox")
        if bbox is None:
            continue

        int_bbox = normalize_to_int_bbox(bbox, image_size=(image_h, image_w))
        if int_bbox is None:
            continue

        x0, y0, x1, y1 = int_bbox
        masked_image[y0:y1, x0:x1] = 255

    return masked_image

def _run_ocr_det_batch(
        ocr_res_all_page: List[Dict],
        atom_model_manager: AtomModelSingleton,
        ocr_config
):
    """批量 OCR 检测"""
    # OCR 配置
    use_det_mode = ocr_config.get("use_det_mode", "auto")
    ocr_det_base_batch_size = ocr_config.get("Det.rec_batch_num", 1)

    all_cropped_info = []

    for ocr_res_dict in ocr_res_all_page:
        for res in ocr_res_dict['ocr_res_list']:
            ocr_enable = ocr_res_dict['ocr_enable']
            native_bands = res.pop('_native_ocr_bands', [])

            if not ocr_res_dict['ocr_enable']:
                if res.get('need_ocr_det') or native_bands:
                    ocr_enable = True
                elif use_det_mode == 'txt' or (use_det_mode != 'ocr' and not res.get('need_ocr_det')):
                    continue

            res.pop('need_ocr_det', None)

            new_image, useful_list = crop_img(
                res, ocr_res_dict['np_img'], crop_paste_x=50, crop_paste_y=50
            )

            adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                ocr_res_dict['single_page_mfdetrec_res'] + ocr_res_dict['checkbox_res'],
                useful_list
            )

            bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
            det_image = _apply_mask_boxes_to_image(
                bgr_image,
                adjusted_mfdetrec_res,
            )

            all_cropped_info.append((
                bgr_image, det_image, useful_list, ocr_res_dict,
                adjusted_mfdetrec_res, ocr_res_dict['lang'], res, ocr_enable, native_bands
            ))

    if not all_cropped_info:
        return

    # 按语言分组
    lang_groups = defaultdict(list)
    for info in all_cropped_info:
        lang_groups[info[5]].append(info)

    RESOLUTION_GROUP_STRIDE = 64

    for lang, lang_crop_list in lang_groups.items():
        if not lang_crop_list:
            continue

        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            lang=lang,
            ocr_config=ocr_config,
        )

        # 按分辨率分组
        resolution_groups = defaultdict(list)
        for info in lang_crop_list:
            cropped_img = info[1]
            h, w = cropped_img.shape[:2]
            # 直接计算目标尺寸并用作分组键
            target_h = ((h + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
            target_w = ((w + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
            group_key = (target_h, target_w)
            resolution_groups[group_key].append(info)

        # 对每个分辨率组进行批处理
        for (target_h, target_w), group_crops in tqdm(resolution_groups.items(), desc=f"OCR-det {lang}"):
            batch_images = []
            for info in group_crops:
                img = info[1] # _det_image
                h, w = img.shape[:2]
                padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
                padded_img[:h, :w] = img
                batch_images.append(padded_img)

            det_batch_size = min(len(batch_images), ocr_det_base_batch_size)
            batch_results = ocr_model.det_batch_predict(batch_images, det_batch_size)

            for info, (dt_boxes, _) in zip(group_crops, batch_results):
                bgr_image, _det_image, useful_list, ocr_res_dict, adjusted_mfdetrec_res, _lang, res, ocr_enable, native_bands = info

                if dt_boxes is not None and len(dt_boxes) > 0:
                    dt_boxes_sorted = sorted_boxes(dt_boxes)
                    dt_boxes_merged = merge_det_boxes(dt_boxes_sorted) if dt_boxes_sorted else []

                    dt_boxes_final = (
                        update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
                        if dt_boxes_merged and adjusted_mfdetrec_res
                        else dt_boxes_merged
                    )
                    if native_bands:
                        dt_boxes_final = _without_native_rows(dt_boxes_final, native_bands)

                    if dt_boxes_final:
                        ocr_res = [box.tolist() if hasattr(box, 'tolist') else box for box in dt_boxes_final]
                        ocr_result_list = get_ocr_result_list(
                            ocr_res, useful_list, ocr_enable, bgr_image,
                            _lang, res['original_label'], res['original_order']
                        )
                        ocr_res_dict['layout_res'].extend(ocr_result_list)


# =================================== OCR-rec ===================================
def _run_ocr_rec_postprocess(images_layout_res: List[List[Dict]], ocr_config):
    """OCR rec 后处理"""
    atom_model_manager = AtomModelSingleton()

    need_ocr_by_lang = {}
    img_crop_by_lang = {}

    for layout_res in images_layout_res:
        for item in layout_res:
            if item['category_id'] == CategoryId.OcrText:
                if 'np_img' in item and 'lang' in item:
                    lang = item['lang']

                    if lang not in need_ocr_by_lang:
                        need_ocr_by_lang[lang] = []
                        img_crop_by_lang[lang] = []

                    need_ocr_by_lang[lang].append(item)
                    img_crop_by_lang[lang].append(item['np_img'])

                    item.pop('np_img')
                    item.pop('lang')

    if not img_crop_by_lang:
        return

    for lang, img_crop_list in img_crop_by_lang.items():
        if not img_crop_list:
            continue

        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name=AtomicModel.OCR,
            lang=lang,
            ocr_config=ocr_config,
        )

        try:
            ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]
        except Exception as exc:
            logger.warning(f'OCR-rec batch failed, retry one by one: {exc}')
            ocr_res_list = []
            safe_items = []
            for item, img_crop in zip(need_ocr_by_lang[lang], img_crop_list):
                try:
                    one_res = ocr_model.ocr([img_crop], det=False, tqdm_enable=False)[0]
                except Exception as one_exc:
                    logger.warning(f'skip failed OCR-rec crop: {one_exc}')
                    item['text'] = ''
                    item['score'] = 0.0
                    item['category_id'] = CategoryId.LowScoreText
                    continue
                if not one_res:
                    item['text'] = ''
                    item['score'] = 0.0
                    item['category_id'] = CategoryId.LowScoreText
                    continue
                ocr_res_list.append(one_res[0])
                safe_items.append(item)
            need_ocr_by_lang[lang] = safe_items

        assert len(ocr_res_list) == len(need_ocr_by_lang[lang])

        for item, (ocr_text, ocr_score) in zip(need_ocr_by_lang[lang], ocr_res_list):
            item['text'] = ocr_text
            item['score'] = float(f"{ocr_score:.3f}")

            if ocr_score < OcrConfidence.min_confidence:
                item['category_id'] = CategoryId.LowScoreText
            else:
                # 特殊字符过滤
                bbox = [item['poly'][0], item['poly'][1], item['poly'][4], item['poly'][5]]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]

                special_texts = ['（204号', '（20', '（2', '（2号', '（20号', '号', '（204']
                if ocr_text in special_texts and ocr_score < 0.8 and width < height:
                    item['category_id'] = CategoryId.LowScoreText

# =================================== OCR-rec ===================================
def _detect_table_text(
        ocr_model,
        det_image: np.ndarray,
        mfd_res,
        retry_padding: int = 8,
        retry_short_side: int = 32,
):
    """Detect table text, retrying boundary-clipped thin crops with white padding."""
    det_res = ocr_model.ocr(det_image, mfd_res=mfd_res, rec=False)[0]
    if det_res is not None and len(det_res) > 0:
        return det_res

    height, width = det_image.shape[:2]
    padding = max(0, int(retry_padding))
    if padding == 0 or min(height, width) >= int(retry_short_side):
        return det_res

    padded_image = cv2.copyMakeBorder(
        det_image,
        padding, padding, padding, padding,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    # Formula regions are already masked in det_image. Omitting mfd_res here
    # avoids mixing original-image coordinates with the padded retry image.
    padded_res = ocr_model.ocr(padded_image, mfd_res=None, rec=False)[0]
    if padded_res is None or len(padded_res) == 0:
        return det_res

    restored = np.asarray(padded_res, dtype=np.float32).copy()
    restored[..., 0] = np.clip(restored[..., 0] - padding, 0, max(0, width - 1))
    restored[..., 1] = np.clip(restored[..., 1] - padding, 0, max(0, height - 1))
    logger.debug(
        f'table OCR-det recovered {len(restored)} box(es) with {padding}px edge padding '
        f'for image shape {det_image.shape}'
    )
    return restored.tolist()


def _recover_rotated_table_boxes(before, after, source_shape, angle):
    """Replace a merged line only when the old boxes align with nearby columns."""
    if not len(before) or not len(after):
        return after
    height, width = source_shape[:2]
    x, y = np.asarray(before, dtype=np.float32).transpose(2, 0, 1)
    if angle == "270":
        projected = np.stack((height - 1 - y, x), axis=-1)[:, [3, 0, 1, 2]]
    else:
        projected = np.stack((y, width - 1 - x), axis=-1)[:, [1, 2, 3, 0]]
    lo, hi = projected.min(axis=1), projected.max(axis=1)
    centers = (lo + hi) / 2
    sizes = hi - lo
    recovered = []
    for box in after:
        left, top = np.min(box, axis=0)
        right, bottom = np.max(box, axis=0)
        w, h = right - left, bottom - top
        if h <= 0 or w < 4 * h:
            recovered.append(box)
            continue
        overlap_y = np.minimum(hi[:, 1], bottom) - np.maximum(lo[:, 1], top)
        ids = np.flatnonzero(
            (centers[:, 0] >= left) & (centers[:, 0] <= right)
            & (overlap_y >= 0.65 * np.maximum(h, sizes[:, 1]))
            & (sizes[:, 0] > 0) & (sizes[:, 0] <= w / 2)
        )
        ids = ids[np.argsort(centers[ids, 0])]
        if (len(ids) < 3 or hi[ids[-1], 0] - lo[ids[0], 0] < 0.8 * w
                or np.any(np.diff(centers[ids, 0]) < 0.7 * np.maximum(sizes[ids[:-1], 0], sizes[ids[1:], 0]))):
            recovered.append(box)
            continue
        # Adjacent rows provide evidence that these are columns rather than
        # fragments of a sentence in a single merged cell.
        center_y = (top + bottom) / 2
        dy = np.abs(centers[:, 1] - center_y)
        peers = centers[(dy >= h) & (dy <= 4 * h), 0]
        aligned = np.any(
            np.abs(centers[ids, 0, None] - peers) <= 0.3 * sizes[ids, 0, None], axis=1,
        )
        if np.count_nonzero(aligned) >= 0.75 * len(ids):
            recovered.extend(projected[ids].tolist())
        else:
            recovered.append(box)
    return recovered


def _verify_table_direction(ocr_model, image, boxes, model_label):
    """Compare the same elongated text crops in both sideways directions."""
    candidates = []
    for box in boxes:
        points = np.asarray(box, dtype=np.float32)
        width = max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3]))
        height = max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2]))
        if width >= 2 and 1.4 * width <= height <= 10 * width:
            candidates.append((height / width, points))
    crops = [get_rotate_crop_image(image, points, rotate_ratio=1.4)
             for _, points in sorted(candidates, key=lambda item: item[0], reverse=True)[:4]]
    # get_rotate_crop_image rotates tall crops counterclockwise.
    crops = [crop for crop in crops if crop.shape[1] > crop.shape[0]]
    if len(crops) < 2:
        return "0"
    flipped = [cv2.rotate(crop, cv2.ROTATE_180) for crop in crops]
    try:
        results = ocr_model.ocr(crops + flipped, det=False)[0]
    except Exception as exc:
        logger.debug(f"Table orientation OCR verification skipped: {exc}")
        return "0"

    def evidence(result):
        text, score = result[:2]
        return score if sum(char.isalpha() for char in text) >= 2 else 0

    left_votes = right_votes = 0
    numeric_agreement = set()
    numeric_conflict = False
    for left, right in zip(results[:len(crops)], results[len(crops):]):
        lscore, rscore = evidence(left), evidence(right)
        left_votes += lscore >= .8 and lscore - rscore >= .2
        right_votes += rscore >= .8 and rscore - lscore >= .2
        # On numeric tables, use repeated score differences for the same
        # recognized number as corroboration of the model's direction.
        number = "".join(left[0].split())
        if len(number) >= 4 and number.isdecimal() and number == "".join(right[0].split()):
            preferred, opposite = (left, right) if model_label == "90" else (right, left)
            if preferred[1] >= .98 and preferred[1] - opposite[1] >= .15:
                numeric_agreement.add(number)
            numeric_conflict |= opposite[1] >= .98 and opposite[1] - preferred[1] >= .15
    if left_votes >= 2 and not right_votes:
        return "90"
    if right_votes >= 2 and not left_votes:
        return "270"
    if not left_votes and not right_votes and len(numeric_agreement) >= 3 and not numeric_conflict:
        return model_label
    return "0"


def _get_table_ocr_model(atom_model_manager, lang, ocr_config):
    config = copy.deepcopy(ocr_config) if ocr_config is not None else None
    if config is not None:
        config.pop("custom_model", None)
    return atom_model_manager.get_atom_model(
        atom_model_name=AtomicModel.OCR,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        lang=lang,
        ocr_config=config,
        enable_merge_det_boxes=False,
    )


def _probe_table_direction(table, page, scale, manager, ocr_model, table_config, masks):
    """Shared direction probe; its boxes can be reused after formula inference."""
    image = cv2.cvtColor(table['table_img'], cv2.COLOR_RGB2BGR)
    det_image = _apply_mask_boxes_to_image(image, masks) if masks else image
    boxes = _detect_table_text(
        ocr_model, det_image, masks,
        retry_padding=int(table_config.get('ocr_det_retry_padding', 8)),
        retry_short_side=int(table_config.get('ocr_det_retry_short_side', 32)),
    )
    label, angles = '0', []
    if page.get('rotate_label') not in ['90', '180', '270']:
        label, angles = txt_most_angle_extract_table(page, table, scale=scale)
    if not angles:
        model = manager.get_atom_model(atom_model_name=AtomicModel.ImgOrientationCls)
        label = model.predict(
            image, boxes, table_mode=True,
            verify_direction=lambda candidate: _verify_table_direction(ocr_model, image, boxes, candidate),
        )
    return boxes, label


def _process_single_table(
        table_res_dict: Dict,
        page_dict: Dict,
        scale: float,
        atom_model_manager: AtomModelSingleton,
        table_config,
        ocr_config,
):
    """处理单个表格"""
    # 表格配置
    table_force_ocr = table_config.get("force_ocr", False)
    skip_text_in_image = table_config.get("skip_text_in_image", True)
    ocr_rule_candidate = use_docvortex_ocr_table_rules(table_config)
    table_use_word_box = table_config.get("use_word_box", False)
    table_formula_enable = table_config.get("table_formula_enable", True)
    table_image_enable = table_config.get("table_image_enable", True)
    table_extract_original_image = table_config.get("extract_original_image", False)
    use_rule_table = table_config.get("use_rule_table", True)
    rule_table_engine = table_config.get("rule_table_engine", "docvortex")
    rule_table_score_threshold = float(table_config.get("rule_table_score_threshold", 0.90))
    ocr_det_retry_padding = int(table_config.get("ocr_det_retry_padding", 8))
    ocr_det_retry_short_side = int(table_config.get("ocr_det_retry_short_side", 32))


    _lang = table_res_dict['lang']
    useful_list = table_res_dict['useful_list']

    adjusted_mfdetrec_res = None
    if table_formula_enable:
        adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
            table_res_dict['single_page_mfdetrec_res'] + table_res_dict['checkbox_res'],
            useful_list, return_text=True
        )

    table_class = None
    table_class_score = 1.0
    pdf_not_rotate = page_dict.get("rotate_label") not in ["90", "180", "270"]
    if (
        use_docvortex_table_rules(table_config) and not table_res_dict['ocr_enable']
        and pdf_not_rotate and '_source_pdf_bytes' in page_dict
    ):
        poly = table_res_dict['table_res']['poly']
        table_bbox = (poly[0] / scale, poly[1] / scale, poly[4] / scale, poly[5] / scale)
        table_bbox_scaled = (poly[0], poly[1], poly[4], poly[5])
        has_embedded_formula = any(
            item.get('bbox') and _box_center_in(table_bbox_scaled, item['bbox'])
            for item in table_res_dict['single_page_mfdetrec_res'] + table_res_dict['checkbox_res']
        )
        has_native_image = any(
            img.get('bbox') and _boxes_overlap(table_bbox, img['bbox'])
            for img in page_dict.get('ori_image_list', [])
        )
        if not has_embedded_formula and not has_native_image:
            try:
                native_result = recover_docvortex_table(page_dict, table_bbox)
                if (native_result is not None and native_result.html.strip()
                        and native_result.confidence >= rule_table_score_threshold):
                    table_res_dict['table_res'].pop('layout_image_list', None)
                    table_res_dict['table_res']['html'] = native_result.html
                    table_res_dict['table_res']['rule_table_score'] = native_result.confidence
                    table_res_dict['table_res']['table_parse_method'] = 'docvortex_native'
                    return
            except ImportError:
                raise
            except Exception as exc:
                logger.warning(f'DocVortex table failed, fallback to model: {exc}')

    # The model is initialized only after the native route declines this table.
    table_model = None if ocr_rule_candidate else atom_model_manager.get_atom_model(
        atom_model_name='table', lang=_lang, ocr_config=ocr_config,
        table_config=table_config,
    )
    native_rule_candidate = (
        use_rule_table and rule_table_engine == "legacy"
        and not table_force_ocr
        and not table_res_dict['ocr_enable']
        and pdf_not_rotate
        and hasattr(table_model, 'rule_table_class')
    )
    if native_rule_candidate:
        try:
            table_class, table_class_score = table_model.rule_table_class(table_res_dict['table_img'])
            if table_class == "wired":
                poly = table_res_dict['table_res']['poly']
                table_bbox = [poly[0] / scale, poly[1] / scale, poly[4] / scale, poly[5] / scale]
                table_bbox_scaled = [poly[0], poly[1], poly[4], poly[5]]
                has_embedded_formula = any(
                    item.get('bbox') and _box_center_in(table_bbox_scaled, item['bbox'])
                    for item in table_res_dict['single_page_mfdetrec_res'] + table_res_dict['checkbox_res']
                )
                # Native images make cell semantics ambiguous. Skip even when
                # table_image_enable is disabled, since this is a rule-parser
                # safety decision rather than an output feature switch.
                has_native_image = any(
                    img.get('bbox') and _boxes_overlap(table_bbox, img['bbox'])
                    for img in page_dict.get('ori_image_list', [])
                )
                if not has_native_image and not has_embedded_formula:
                    rule_result = (
                        build_rule_table(page_dict, table_bbox)
                        or build_track_table(page_dict, table_bbox)
                    )
                    if rule_result and rule_result.score >= rule_table_score_threshold:
                        table_res_dict['table_res'].pop('layout_image_list', None)
                        table_res_dict['table_res']['html'] = rule_result.html
                        table_res_dict['table_res']['rule_table_score'] = rule_result.score
                        table_res_dict['table_res']['table_parse_method'] = 'pdfium_rule'
                        return
        except Exception as exc:
            logger.warning(f'PDFium rule table failed, fallback to model: {exc}')

    ocr_model = _get_table_ocr_model(atom_model_manager, _lang, ocr_config)

    # 获取表格文本框
    bgr_image = cv2.cvtColor(table_res_dict["table_img"], cv2.COLOR_RGB2BGR)
    probe = table_res_dict.pop('_formula_direction_probe', None)
    if probe is None:
        probe = _probe_table_direction(
            table_res_dict, page_dict, scale, atom_model_manager,
            ocr_model, table_config, adjusted_mfdetrec_res,
        )
    det_res, rotate_label = probe
    if rotate_label in ["90", "270"]:
        original_boxes = det_res
        original_shape = bgr_image.shape
        rotate_table_image(table_res_dict, rotate_label)
        from .table_formula import rotate_table_hint_boxes
        adjusted_mfdetrec_res = rotate_table_hint_boxes(
            adjusted_mfdetrec_res, original_shape, rotate_label
        )
        # 旋转后的表格需要重新获取文本框
        bgr_image = cv2.cvtColor(table_res_dict["table_img"], cv2.COLOR_RGB2BGR)
        det_image = (
            _apply_mask_boxes_to_image(bgr_image, adjusted_mfdetrec_res)
            if adjusted_mfdetrec_res
            else bgr_image
        )
        det_res = _detect_table_text(
            ocr_model,
            det_image,
            adjusted_mfdetrec_res,
            retry_padding=ocr_det_retry_padding,
            retry_short_side=ocr_det_retry_short_side,
        )
        if original_boxes is not None and det_res is not None:
            det_res = _recover_rotated_table_boxes(original_boxes, det_res, original_shape, rotate_label)

    ocr_result = []

    # 尝试从 PDF 提取文本
    if (not table_force_ocr and not table_res_dict['ocr_enable'] and rotate_label == "0" and pdf_not_rotate):
        ocr_result = _extract_table_text_from_pdf(
            table_res_dict, page_dict, scale, det_res, useful_list, table_use_word_box
        )

    # 如果提取失败，使用 OCR
    if not ocr_result and det_res:
        ocr_result = _run_table_ocr(ocr_model, bgr_image, det_res, table_use_word_box)

    fill_image_res = []
    if table_image_enable:
        if not pdf_not_rotate:
            table_extract_original_image = False
        fill_image_res = extract_table_fill_image(
            page_dict, table_res_dict, scale, table_extract_original_image
        )

    # Known formulas and embedded layout images remain on the model route even
    # if their output flags are disabled. A full scanned-page image is not an
    # embedded cell image, so ori_image_list alone is not an exclusion here.
    known_formula = known_image = False
    if ocr_rule_candidate:
        poly = table_res_dict['table_res']['poly']
        table_bbox_scaled = (poly[0], poly[1], poly[4], poly[5])
        known_formula = any(
            item.get('bbox') and _box_center_in(table_bbox_scaled, item['bbox'])
            for item in table_res_dict['single_page_mfdetrec_res'] + table_res_dict['checkbox_res']
        )
        known_image = bool(table_res_dict['table_res'].get('layout_image_list'))
    if ocr_rule_candidate and not (known_formula or known_image or adjusted_mfdetrec_res or fill_image_res):
        from ...model.table.docvortex_ocr import recover_ocr_table, save_ocr_diagnostics

        report = {}
        try:
            rule_result = recover_ocr_table(table_res_dict['table_img'], ocr_result, diagnostics=report)
            if rule_result is not None and rule_result.html.strip() and rule_result.confidence >= rule_table_score_threshold:
                table_res_dict['table_res'].pop('layout_image_list', None)
                table_res_dict['table_res'].update(
                    html=rule_result.html, rule_table_score=rule_result.confidence,
                    table_parse_method='docvortex_ocr', rule_table_source=rule_result.source)
                return
            if rule_result is not None:
                report.update(reason='below_rule_table_score_threshold', accepted=False)
            logger.debug(f'DocVortex OCR table declined: {report.get("reason")}; fallback to model')
        except Exception as exc:
            report.update(reason='adapter_error', error=str(exc), accepted=False)
            logger.warning(f'DocVortex OCR table failed, fallback to model: {exc}')
        finally:
            debug_dir = table_config.get('rule_table_debug_dir')
            if debug_dir:
                try:
                    save_ocr_diagnostics(debug_dir, table_res_dict['table_img'], report)
                except Exception as exc:
                    logger.warning(f'DocVortex OCR diagnostic save failed: {exc}')

    if table_model is None:
        table_model = atom_model_manager.get_atom_model(
            atom_model_name='table', lang=_lang, ocr_config=ocr_config,
            table_config=table_config,
        )

    table_res_dict['table_res'].pop('layout_image_list', None)

    predict_kwargs = {'skip_table_orientation': True}
    if hasattr(table_model, 'rule_table_class'):
        predict_kwargs.update(
            table_class=table_class,
            table_class_score=table_class_score,
        )
        if table_config.get('fallback_model_type') and pdf_not_rotate and page_dict.get('vector_lines'):
            from rapid_doc.model.table.table_evidence import crop_vector_lines

            predict_kwargs['vector_lines'] = crop_vector_lines(
                page_dict['vector_lines'], useful_list, scale, rotate_label,
            )
    html_code = table_model.predict(
        table_res_dict['table_img'], ocr_result,
        fill_image_res, adjusted_mfdetrec_res,
        skip_text_in_image,
        **predict_kwargs,
    )

    from rapid_doc.model.table.fallback import TextFallback
    text_fallback = isinstance(html_code, TextFallback)
    if text_fallback or html_code and '<table>' in html_code and '</table>' in html_code:
        start_index = html_code.find('<table>')
        end_index = html_code.rfind('</table>') + len('</table>')
        table_res_dict['table_res']['html'] = str(html_code) if text_fallback else html_code[start_index:end_index]
        if text_fallback:
            table_res_dict['table_res']['text_fallback'] = True

        # 保存公式和图片位置信息
        formula_boxes = [
            t["bbox"] for t in table_res_dict['single_page_mfdetrec_res'] + table_res_dict['checkbox_res']
            if "bbox" in t
        ]
        if formula_boxes:
            table_res_dict['table_res']['formula_boxes'] = [
                [int(coord / scale) for coord in bbox] for bbox in formula_boxes
            ]

        img_boxes = [t["ori_bbox"] for t in fill_image_res if "bbox" in t]
        if img_boxes:
            table_res_dict['table_res']['img_boxes'] = [
                [int(coord / scale) for coord in bbox] for bbox in img_boxes
            ]
    else:
        logger.warning('table recognition processing fails')


def _boxes_overlap(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _box_center_in(container, item) -> bool:
    cx, cy = (item[0] + item[2]) / 2, (item[1] + item[3]) / 2
    return container[0] <= cx <= container[2] and container[1] <= cy <= container[3]

def _extract_table_text_from_pdf(
        table_res_dict: Dict,
        page_dict: Dict,
        scale: float,
        det_res: List,
        useful_list: List,
        table_use_word_box
) -> List:
    """从 PDF 中提取表格文本"""
    if not det_res:
        return []

    try:
        ocr_spans = get_ocr_result_list_table(det_res, useful_list, scale)
        poly = table_res_dict['table_res']['poly']
        table_bboxes = [[
            int(poly[0] / scale), int(poly[1] / scale),
            int(poly[4] / scale), int(poly[5] / scale),
            None, None, None, 'text', None, None, None, None, 1
        ]]

        txt_spans_extract(
            page_dict, ocr_spans, table_res_dict['table_img'], scale,
            table_bboxes, [], return_word_box=table_use_word_box,
            useful_list=table_res_dict['useful_list']
        )

        if table_use_word_box:
            filtered = [
                (w[2], normalize_table_ocr_text(w[0]), w[1])
                for item in ocr_spans
                for group in [item.get('word_result')]
                if group
                for w in group
                if w and w[2] != ""
            ]
        else:
            filtered = [
                [item['ori_bbox'], normalize_table_ocr_text(item['content']), item['score']]
                for item in ocr_spans if item.get('content')
            ]

        return [list(x) for x in zip(*filtered)] if filtered else []
    except Exception:
        logger.warning('table ocr_result get from pdf error')
        return []

def _run_table_ocr(
        ocr_model,
        bgr_image: np.ndarray,
        det_res: List,
        table_use_word_box,
) -> List:
    """执行表格 OCR"""
    rec_img_list = []
    for dt_box in det_res:
        rec_img_list.append({
            "cropped_img": get_rotate_crop_image(bgr_image, np.asarray(dt_box, dtype=np.float32)),
            "dt_box": np.asarray(dt_box, dtype=np.float32),
        })

    cropped_img_list = [item["cropped_img"] for item in rec_img_list]
    ocr_res_list = None
    try:
        ocr_res_list = ocr_model.ocr(
            cropped_img_list, det=False, tqdm_enable=False,
            return_word_box=table_use_word_box, ori_img=bgr_image, dt_boxes=det_res
        )[0]
    except Exception as exc:
        if table_use_word_box:
            logger.warning(f'table OCR word-box recognition failed, retry line-level OCR: {exc}')
            table_use_word_box = False
            try:
                ocr_res_list = ocr_model.ocr(
                    cropped_img_list, det=False, tqdm_enable=False,
                    return_word_box=False, ori_img=bgr_image, dt_boxes=det_res
                )[0]
            except Exception as retry_exc:
                logger.warning(f'table OCR batch recognition failed, retry one by one: {retry_exc}')
        else:
            logger.warning(f'table OCR batch recognition failed, retry one by one: {exc}')

    if ocr_res_list is None:
        ocr_res_list = []
        rec_img_list_safe = []
        for img_dict in rec_img_list:
            try:
                one_res = ocr_model.ocr(
                    [img_dict["cropped_img"]], det=False, tqdm_enable=False,
                    return_word_box=False, ori_img=bgr_image, dt_boxes=[img_dict["dt_box"]]
                )[0]
            except Exception as exc:
                logger.warning(f'skip failed table OCR crop: {exc}')
                continue
            if not one_res:
                continue
            ocr_res_list.append(one_res[0])
            rec_img_list_safe.append(img_dict)
        rec_img_list = rec_img_list_safe

    ocr_result = []
    for img_dict, ocr_res in zip(rec_img_list, ocr_res_list):
        if table_use_word_box:
            ocr_result.extend([
                [word_result[2], normalize_table_ocr_text(word_result[0]), word_result[1]]
                for word_result in ocr_res[2]
            ])
        else:
            ocr_result.append([img_dict["dt_box"], normalize_table_ocr_text(ocr_res[0]), ocr_res[1]])

    return [list(x) for x in zip(*ocr_result)] if ocr_result else []
