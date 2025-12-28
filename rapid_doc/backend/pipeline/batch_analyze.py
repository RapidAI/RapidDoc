import time
from PIL import Image
from typing import List, Tuple

import cv2
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
import numpy as np

from .model_init import AtomModelSingleton
from .model_list import AtomicModel
from ...utils.boxbase import is_in, rotate_image
from ...utils.checkbox_det_cls import checkbox_predict
from ...utils.config_reader import get_formula_enable, get_table_enable
from ...utils.enum_class import CategoryId
from ...utils.model_utils import crop_img, get_res_list_from_layout_res, clean_vram
from ...utils.ocr_utils import merge_det_boxes, update_det_boxes, sorted_boxes, get_rotate_crop_image
from ...utils.ocr_utils import get_adjusted_mfdetrec_res, get_ocr_result_list, OcrConfidence, get_ocr_result_list_table
from ...utils.span_pre_proc import txt_spans_extract, txt_spans_bbox_extract, \
    txt_most_angle_extract_table, extract_table_fill_image, txt_in_ori_image


# YOLO_LAYOUT_BASE_BATCH_SIZE = 8
# MFR_BASE_BATCH_SIZE = 16
# OCR_DET_BASE_BATCH_SIZE = 16

# LAYOUT_BASE_BATCH_SIZE = 1
# FORMULA_BASE_BATCH_SIZE = 1
# OCR_DET_BASE_BATCH_SIZE = 16


class BatchAnalyze:
    def __init__(self, model_manager, batch_ratio: int, formula_enable, table_enable, enable_ocr_det_batch: bool = False,
                layout_config=None,
                ocr_config=None,
                formula_config=None,
                table_config=None,
                checkbox_config=None):
        self.batch_ratio = batch_ratio
        self.formula_enable = get_formula_enable(formula_enable)
        self.formula_level = formula_config.get("formula_level", 0) if formula_config else 0
        self.table_enable = get_table_enable(table_enable)
        self.checkbox_enable = checkbox_config.get("checkbox_enable", False) if checkbox_config else False
        self.layout_config = layout_config
        self.ocr_config = ocr_config
        self.formula_config = formula_config
        self.model_manager = model_manager
        self.enable_ocr_det_batch = ocr_config.get("Det.rec_batch_num", 1) > 1 if ocr_config else False
        self.ocr_det_base_batch_size = ocr_config.get("Det.rec_batch_num", 1) if ocr_config else 1 #16
        self.layout_base_batch_size = layout_config.get("batch_num", 1) if layout_config else 1 #8
        self.formula_base_batch_size = formula_config.get("batch_num", 1) if formula_config else 1 #16
        self.use_det_mode = ocr_config.get("use_det_mode", 'auto') if ocr_config else 'auto'
        self.table_config = table_config
        self.table_force_ocr = table_config.get("force_ocr", False) if table_config else False
        self.skip_text_in_image = table_config.get("skip_text_in_image", True) if table_config else True
        self.use_img2table = table_config.get("use_img2table", False) if table_config else False
        self.table_use_word_box = table_config.get("use_word_box", True) if table_config else True
        self.table_formula_enable = table_config.get("table_formula_enable", True) if table_config else True
        self.table_image_enable = table_config.get("table_image_enable", True) if table_config else True
        self.table_extract_original_image = table_config.get("extract_original_image", False) if table_config else False

    def __call__(self, images_with_extra_info: List[Tuple[Image.Image, float, bool, str, dict]]) -> list:
        if len(images_with_extra_info) == 0:
            return []

        images_layout_res = []

        self.model = self.model_manager.get_model(
            lang=None,
            formula_enable=self.formula_enable,
            table_enable=self.table_enable,
            layout_config=self.layout_config,
            ocr_config=self.ocr_config,
            formula_config=self.formula_config,
            table_config=self.table_config,
        )
        atom_model_manager = AtomModelSingleton()

        pdf_dict_list = [pdf_dict for _, _, _, _, pdf_dict in images_with_extra_info]
        np_images = [cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) for image, _, _, _, _ in images_with_extra_info]
        scale_list = [scale for _, scale, _, _, _ in images_with_extra_info]


        # 版面识别
        images_layout_res += self.model.layout_model.batch_predict(
            np_images, self.layout_base_batch_size
        )

        if self.use_det_mode == 'txt':
            images_layout_res = remove_layout_in_ori_images(images_layout_res, pdf_dict_list, scale_list)

        # formula_level：公式识别等级，默认为0，全识别。
        # 如果公式识别等级为 1（只保留行间公式），过滤掉 category_id == 13（inline_formula）
        if self.formula_enable and self.formula_level == 1:
            images_layout_res = [
                [item for item in page if item["category_id"] != 13]
                for page in images_layout_res
            ]

        ocr_res_list_all_page = []
        table_res_list_all_page = []
        latex_res_list_all_page = []
        for index in range(len(np_images)):
            _, _, ocr_enable, _lang, _ = images_with_extra_info[index]
            layout_res = images_layout_res[index]
            np_img = np_images[index]

            ocr_res_list, table_res_list, single_page_mfdetrec_res = (
                get_res_list_from_layout_res(layout_res, np_img)
            )

            # 复选框检测
            checkbox_res = []
            if self.checkbox_enable:
                checkbox_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
                checkbox_res = checkbox_predict(checkbox_img)
                for res in checkbox_res:
                    poly = [res['bbox'][0], res['bbox'][1], res['bbox'][2], res['bbox'][1],
                            res['bbox'][2], res['bbox'][3], res['bbox'][0], res['bbox'][3]]
                    layout_res.append({'bbox': res['bbox'], 'poly': poly, 'category_id': CategoryId.CheckBox,
                                       'checkbox': res['text'], 'score': 0.9})

            ocr_res_list_all_page.append({'ocr_res_list':ocr_res_list,
                                          'lang':_lang,
                                          'ocr_enable':ocr_enable,
                                          'np_img':np_img,
                                          'single_page_mfdetrec_res':single_page_mfdetrec_res,
                                          'checkbox_res': checkbox_res,
                                          'layout_res':layout_res,
                                          'page_idx': index,
                                          })

            for table_res in table_res_list:
                table_img, useful_list = crop_img(table_res, np_img)
                table_res_list_all_page.append({'table_res':table_res,
                                                'lang':_lang,
                                                'table_img':table_img,
                                                'single_page_mfdetrec_res': single_page_mfdetrec_res,
                                                'checkbox_res': checkbox_res,
                                                'useful_list': useful_list,
                                                'ocr_enable': ocr_enable,
                                                'page_idx': index,
                                              })
            for latex_res in single_page_mfdetrec_res:
                latex_img, _ = crop_img(latex_res, np_img)
                latex_res_list_all_page.append({'latex_res': latex_res,
                                                'lang': _lang,
                                                'latex_img': latex_img,
                                                })

        if self.formula_enable:
            # 公式检测
            latex_imgs = [d['latex_img'] for d in latex_res_list_all_page]
            latex_results = self.model.formula_model.batch_predict(latex_imgs, batch_size=self.formula_base_batch_size)
            for d, res in zip(latex_res_list_all_page, latex_results):
                if res:
                    d['latex_res']['latex'] = res
                else:
                    logger.warning('latex recognition processing fails, not get latex return')

        # 清理显存
        clean_vram(self.model.device, vram_threshold=8)

        if self.use_det_mode != 'ocr':
            # 分页分组
            ocr_res_list_grouped_page = {}
            for x in ocr_res_list_all_page:
                ocr_res_list_grouped_page.setdefault(x["page_idx"], []).append(x)
            # 计算总数
            total_texts = sum(len(texts) for texts in ocr_res_list_grouped_page.values())
            with tqdm(total=total_texts, desc="PDF-det Predict") as pbar:
                for page_idx, text_list in ocr_res_list_grouped_page.items():
                    if text_list:
                        page_dict = pdf_dict_list[page_idx]
                        scale = scale_list[page_idx]
                    for ocr_res_list_dict in text_list:
                        _lang = ocr_res_list_dict['lang']
                        if ocr_res_list_dict['ocr_enable']:
                            # 需要进行ocr的这里跳过
                            continue
                        # 从pdf中获取文本行点位
                        for res in ocr_res_list_dict['ocr_res_list']:
                            # res 点位信息
                            new_image, useful_list = crop_img(
                                res, ocr_res_list_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                            )
                            # 跳过公式和复选框
                            adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                                ocr_res_list_dict['single_page_mfdetrec_res'] + ocr_res_list_dict['checkbox_res'],
                                useful_list
                            )
                            # PDF-det
                            bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                            ocr_res = txt_spans_bbox_extract(page_dict, res, mfd_res=adjusted_mfdetrec_res, scale=scale, useful_list=useful_list) # 从pdf中获取文本行点位
                            # Integration results
                            if ocr_res:
                                ocr_result_list = get_ocr_result_list(
                                    ocr_res, useful_list, ocr_res_list_dict['ocr_enable'], bgr_image, _lang, res['original_label'], res['original_order']
                                )

                                ocr_res_list_dict['layout_res'].extend(ocr_result_list)
                        pbar.update(1)  # 每处理一个更新一次


        # OCR检测处理
        if self.enable_ocr_det_batch:
            # 批处理模式 - 按语言和分辨率分组
            # 收集所有需要OCR检测的裁剪图像
            all_cropped_images_info = []

            for ocr_res_list_dict in ocr_res_list_all_page:
                _lang = ocr_res_list_dict['lang']
                for res in ocr_res_list_dict['ocr_res_list']:
                    # 仅当整页OCR未启用时，判断是否跳过
                    ocr_enable = ocr_res_list_dict['ocr_enable']
                    if not ocr_res_list_dict['ocr_enable']:
                        if res.get('need_ocr_det'):
                            ocr_enable = True
                        if (
                                self.use_det_mode == 'txt' or
                                (self.use_det_mode != 'ocr' and not ocr_res_list_dict['ocr_enable'] and not res.get('need_ocr_det'))
                        ):
                            # 从 PDF 中直接提取文本框，无需 OCR，且已提取到框，则跳过
                            continue
                    res.pop('need_ocr_det', None)
                    new_image, useful_list = crop_img(
                        res, ocr_res_list_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                    )
                    # OCR检测，跳过公式和复选框
                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                        ocr_res_list_dict['single_page_mfdetrec_res'] + ocr_res_list_dict['checkbox_res'], useful_list
                    )

                    # BGR转换
                    bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

                    all_cropped_images_info.append((
                        bgr_image, useful_list, ocr_res_list_dict, res, adjusted_mfdetrec_res, _lang, ocr_enable
                    ))

            # 按语言分组
            lang_groups = defaultdict(list)
            for crop_info in all_cropped_images_info:
                lang = crop_info[5]
                lang_groups[lang].append(crop_info)

            # 对每种语言按分辨率分组并批处理
            for lang, lang_crop_list in lang_groups.items():
                if not lang_crop_list:
                    continue

                # logger.info(f"Processing OCR detection for language {lang} with {len(lang_crop_list)} images")

                # 获取OCR模型
                ocr_model = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR,
                    det_db_box_thresh=0.3,
                    lang=lang,
                    ocr_config=self.ocr_config,
                )

                # 按分辨率分组并同时完成padding
                # RESOLUTION_GROUP_STRIDE = 32
                RESOLUTION_GROUP_STRIDE = 64

                resolution_groups = defaultdict(list)
                for crop_info in lang_crop_list:
                    cropped_img = crop_info[0]
                    h, w = cropped_img.shape[:2]
                    # 直接计算目标尺寸并用作分组键
                    target_h = ((h + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                    target_w = ((w + RESOLUTION_GROUP_STRIDE - 1) // RESOLUTION_GROUP_STRIDE) * RESOLUTION_GROUP_STRIDE
                    group_key = (target_h, target_w)
                    resolution_groups[group_key].append(crop_info)

                # 对每个分辨率组进行批处理
                for (target_h, target_w), group_crops in tqdm(resolution_groups.items(), desc=f"OCR-det {lang}"):
                    # 对所有图像进行padding到统一尺寸
                    batch_images = []
                    for crop_info in group_crops:
                        img = crop_info[0]
                        h, w = img.shape[:2]
                        # 创建目标尺寸的白色背景
                        padded_img = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
                        padded_img[:h, :w] = img
                        batch_images.append(padded_img)

                    # 批处理检测
                    det_batch_size = min(len(batch_images), self.batch_ratio * self.ocr_det_base_batch_size)
                    # batch_results = ocr_model.text_detector.batch_predict(batch_images, det_batch_size)
                    batch_results = ocr_model.det_batch_predict(batch_images, det_batch_size)

                    # 处理批处理结果
                    for crop_info, (dt_boxes, _) in zip(group_crops, batch_results):
                        bgr_image, useful_list, ocr_res_list_dict, res, adjusted_mfdetrec_res, _lang, ocr_enable = crop_info

                        if dt_boxes is not None and len(dt_boxes) > 0:
                            # 处理检测框
                            dt_boxes_sorted = sorted_boxes(dt_boxes)
                            dt_boxes_merged = merge_det_boxes(dt_boxes_sorted) if dt_boxes_sorted else []

                            # 根据公式位置更新检测框
                            dt_boxes_final = (update_det_boxes(dt_boxes_merged, adjusted_mfdetrec_res)
                                              if dt_boxes_merged and adjusted_mfdetrec_res
                                              else dt_boxes_merged)

                            if dt_boxes_final:
                                ocr_res = [box.tolist() if hasattr(box, 'tolist') else box for box in dt_boxes_final]
                                ocr_result_list = get_ocr_result_list(
                                    ocr_res, useful_list, ocr_enable, bgr_image, _lang, res['original_label'], res['original_order']
                                )

                                ocr_res_list_dict['layout_res'].extend(ocr_result_list)
        else:
            # 原始单张处理模式
            for ocr_res_list_dict in tqdm(ocr_res_list_all_page, desc="OCR-det Predict"):
                # Process each area that requires OCR processing
                _lang = ocr_res_list_dict['lang']
                # Get OCR results for this language's images
                ocr_model = atom_model_manager.get_atom_model(
                    atom_model_name=AtomicModel.OCR,
                    ocr_show_log=False,
                    det_db_box_thresh=0.3,
                    lang=_lang,
                    ocr_config=self.ocr_config,
                )
                for res in ocr_res_list_dict['ocr_res_list']:
                    # 仅当整页OCR未启用时，判断是否跳过
                    ocr_enable = ocr_res_list_dict['ocr_enable']
                    if not ocr_res_list_dict['ocr_enable']:
                        if res.get('need_ocr_det'):
                            ocr_enable = True
                        if (
                                self.use_det_mode == 'txt' or
                                (self.use_det_mode != 'ocr' and not ocr_res_list_dict['ocr_enable'] and not res.get('need_ocr_det'))
                        ):
                            # 从 PDF 中直接提取文本框，无需 OCR，且已提取到框，则跳过
                            continue
                    res.pop('need_ocr_det', None)
                    new_image, useful_list = crop_img(
                        res, ocr_res_list_dict['np_img'], crop_paste_x=50, crop_paste_y=50
                    )
                    # OCR检测，跳过公式和复选框
                    adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                        ocr_res_list_dict['single_page_mfdetrec_res'] + ocr_res_list_dict['checkbox_res'], useful_list
                    )
                    # OCR-det
                    bgr_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
                    ocr_res = ocr_model.ocr(
                        bgr_image, mfd_res=adjusted_mfdetrec_res, rec=False
                    )[0]

                    # Integration results
                    if ocr_res:
                        ocr_result_list = get_ocr_result_list(
                            ocr_res, useful_list, ocr_enable, bgr_image, _lang, res['original_label'], res['original_order']
                        )
                        ocr_res_list_dict['layout_res'].extend(ocr_result_list)

        # 表格识别 table recognition
        if self.table_enable:
            # 分页分组
            table_res_list_grouped_page = {}
            for x in table_res_list_all_page:
                table_res_list_grouped_page.setdefault(x["page_idx"], []).append(x)
            # 计算总表格数
            total_tables = sum(len(tables) for tables in table_res_list_grouped_page.values())
            with tqdm(total=total_tables, desc="Table Predict") as pbar:
                for page_idx, table_list in table_res_list_grouped_page.items():
                    page_dict = pdf_dict_list[page_idx]
                    scale = scale_list[page_idx]
                    for table_res_dict in table_list:
                        _lang = table_res_dict['lang']
                        useful_list = table_res_dict['useful_list']
                        # OCR检测，跳过公式和复选框
                        adjusted_mfdetrec_res = None
                        if self.table_formula_enable:
                            adjusted_mfdetrec_res = get_adjusted_mfdetrec_res(
                                table_res_dict['single_page_mfdetrec_res'] + table_res_dict['checkbox_res'],
                                useful_list, return_text=True
                            )

                        ocr_result = []
                        # 进行 OCR-det 识别文字框
                        ocr_model = atom_model_manager.get_atom_model(
                            atom_model_name=AtomicModel.OCR,
                            det_db_box_thresh=0.5,
                            det_db_unclip_ratio=1.6,
                            lang=_lang,
                            ocr_config=self.ocr_config,
                            enable_merge_det_boxes=False,
                        )
                        # 判断表格里文字是否旋转
                        most_angle = txt_most_angle_extract_table(page_dict, table_res_dict, scale=scale)  # 从pdf中获取文本行 角度投票
                        if most_angle in [90, 270]:
                            rotate_image(table_res_dict, most_angle)
                        bgr_image = cv2.cvtColor(table_res_dict["table_img"], cv2.COLOR_RGB2BGR)
                        det_res = ocr_model.ocr(bgr_image, mfd_res=adjusted_mfdetrec_res, rec=False)[0]
                        if not self.table_force_ocr and not table_res_dict['ocr_enable'] and most_angle == 0:
                            if det_res:
                                try:
                                    ocr_spans = get_ocr_result_list_table(det_res, useful_list, scale)
                                    poly = table_res_dict['table_res']['poly']
                                    table_bboxes = [[int(poly[0] / scale), int(poly[1] / scale), int(poly[4] / scale), int(poly[5] / scale)
                                                        , None, None, None, 'text', None, None, None, None, 1]]
                                    # 从pdf中提取表格的文本
                                    txt_spans_extract(page_dict, ocr_spans, table_res_dict['table_img'], scale, table_bboxes, []
                                                      , return_word_box=self.table_use_word_box, useful_list=table_res_dict['useful_list'])
                                    if self.table_use_word_box:
                                        filtered = [
                                            (w[2], w[0], w[1])
                                            for item in ocr_spans
                                            for group in [item.get('word_result')]
                                            if group
                                            for w in group
                                            if w and w[2] != ""
                                        ]
                                    else:
                                        filtered = [
                                            [item['ori_bbox'], item['content'], item['score']]
                                            for item in ocr_spans if item.get('content')
                                        ]
                                    ocr_result = [list(x) for x in zip(*filtered)] if filtered else []
                                except:
                                    logger.warning('table ocr_result get from pdf error')
                        # 进行 OCR-rec 识别文字框
                        if not ocr_result and det_res:
                            rec_img_list = []
                            for dt_box in det_res:
                                rec_img_list.append(
                                    {
                                        "cropped_img": get_rotate_crop_image(
                                            bgr_image, np.asarray(dt_box, dtype=np.float32)
                                        ),
                                        "dt_box": np.asarray(dt_box, dtype=np.float32),
                                    }
                                )
                            cropped_img_list = [item["cropped_img"] for item in rec_img_list]
                            ocr_res_list = ocr_model.ocr(cropped_img_list, det=False, tqdm_enable=False
                                                         , return_word_box=self.table_use_word_box, ori_img=bgr_image, dt_boxes=det_res)[0]
                            for img_dict, ocr_res in zip(rec_img_list, ocr_res_list):
                                if self.table_use_word_box:
                                    ocr_result.extend([[word_result[2], word_result[0], word_result[1]] for word_result in ocr_res[2]])
                                else:
                                    ocr_result.append([img_dict["dt_box"], ocr_res[0], ocr_res[1]])
                            ocr_result = [list(x) for x in zip(*ocr_result)]

                        table_model = atom_model_manager.get_atom_model(
                            atom_model_name='table',
                            lang=_lang,
                            ocr_config=self.ocr_config,
                            table_config=self.table_config,
                        )
                        # 从pdf里提取表格里的图片
                        fill_image_res = []
                        if self.table_image_enable:
                            fill_image_res = extract_table_fill_image(page_dict, table_res_dict, scale, self.table_extract_original_image)
                        table_res_dict['table_res'].pop('layout_image_list', None)
                        html_code = table_model.predict(table_res_dict['table_img'], ocr_result,
                                                        fill_image_res, adjusted_mfdetrec_res, self.skip_text_in_image, self.use_img2table)
                        # 判断是否返回正常
                        if html_code:
                            # 检查html_code是否包含'<table>'和'</table>'
                            if '<table>' in html_code and '</table>' in html_code:
                                # 选用<table>到</table>的内容，放入table_res_dict['table_res']['html']
                                start_index = html_code.find('<table>')
                                end_index = html_code.rfind('</table>') + len('</table>')
                                table_res_dict['table_res']['html'] = html_code[start_index:end_index]
                                # 把图片里的公式和图片放到span，方便layout画图
                                latex_boxes = [t["bbox"] for t in table_res_dict['single_page_mfdetrec_res'] + table_res_dict['checkbox_res'] if "bbox" in t]
                                if latex_boxes:
                                    table_res_dict['table_res']['latex_boxes'] = [[int(coord / scale) for coord in bbox] for bbox in latex_boxes]
                                img_boxes = [t["ori_bbox"] for t in fill_image_res if "bbox" in t]
                                if img_boxes:
                                    table_res_dict['table_res']['img_boxes'] = [[int(coord / scale) for coord in bbox] for bbox in img_boxes]
                            else:
                                logger.warning(
                                    'table recognition processing fails, not found expected HTML table end'
                                )
                        else:
                            logger.warning(
                                'table recognition processing fails, not get html return'
                            )
                        pbar.update(1)  # 每处理一个表格更新一次

        # 清理显存
        clean_vram(self.model.device, vram_threshold=8)

        # OCR rec
        # Create dictionaries to store items by language
        need_ocr_lists_by_lang = {}  # Dict of lists for each language
        img_crop_lists_by_lang = {}  # Dict of lists for each language

        for layout_res in images_layout_res:
            for layout_res_item in layout_res:
                if layout_res_item['category_id'] in [15]:
                    if 'np_img' in layout_res_item and 'lang' in layout_res_item:
                        lang = layout_res_item['lang']

                        # Initialize lists for this language if not exist
                        if lang not in need_ocr_lists_by_lang:
                            need_ocr_lists_by_lang[lang] = []
                            img_crop_lists_by_lang[lang] = []

                        # Add to the appropriate language-specific lists
                        need_ocr_lists_by_lang[lang].append(layout_res_item)
                        img_crop_lists_by_lang[lang].append(layout_res_item['np_img'])

                        # Remove the fields after adding to lists
                        layout_res_item.pop('np_img')
                        layout_res_item.pop('lang')

        if len(img_crop_lists_by_lang) > 0:

            # Process OCR by language
            total_processed = 0

            # Process each language separately
            for lang, img_crop_list in img_crop_lists_by_lang.items():
                if len(img_crop_list) > 0:
                    start = time.perf_counter()
                    # Get OCR results for this language's images

                    ocr_model = atom_model_manager.get_atom_model(
                        atom_model_name=AtomicModel.OCR,
                        det_db_box_thresh=0.3,
                        lang=lang,
                        ocr_config=self.ocr_config,
                    )
                    ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]
                    # Verify we have matching counts
                    assert len(ocr_res_list) == len(
                        need_ocr_lists_by_lang[lang]), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_lists_by_lang[lang])} for lang: {lang}'

                    # Process OCR results for this language
                    for index, layout_res_item in enumerate(need_ocr_lists_by_lang[lang]):
                        ocr_text, ocr_score = ocr_res_list[index]
                        layout_res_item['text'] = ocr_text
                        layout_res_item['score'] = float(f"{ocr_score:.3f}")
                        if ocr_score < OcrConfidence.min_confidence:
                            layout_res_item['category_id'] = 16
                        else:
                            layout_res_bbox = [layout_res_item['poly'][0], layout_res_item['poly'][1],
                                               layout_res_item['poly'][4], layout_res_item['poly'][5]]
                            layout_res_width = layout_res_bbox[2] - layout_res_bbox[0]
                            layout_res_height = layout_res_bbox[3] - layout_res_bbox[1]
                            if ocr_text in ['（204号', '（20', '（2', '（2号', '（20号', '号', '（204'] and ocr_score < 0.8 and layout_res_width < layout_res_height:
                                layout_res_item['category_id'] = 16

                    total_processed += len(img_crop_list)

        return images_layout_res


def remove_layout_in_ori_images(images_layout_res, pdf_dict_list, scale_list):
    """
    移除落在原始图片区域内的 layout_res，
    对于确实替换掉内容的图片区域，添加一条新的 'image' 类型检测结果。
    若图片区域中含有文本（背景图），则跳过该图片。
    """
    for index, layout_res in enumerate(images_layout_res):
        ori_image_list = pdf_dict_list[index].get('ori_image_list')
        scale = scale_list[index]

        if not ori_image_list:
            continue

        # ✅ 过滤掉包含文本的“背景图”区域
        valid_ori_images = []
        for ori in ori_image_list:
            if txt_in_ori_image(pdf_dict_list[index], ori['bbox']):
                # 含文本，不视为纯图片，跳过
                continue
            valid_ori_images.append(ori)

        if not valid_ori_images:
            # 没有纯图片，跳过
            images_layout_res[index] = layout_res
            continue

        # ✅ 计算所有原图 bbox（已缩放）
        scaled_ori_bboxes = [
            [
                ori['bbox'][0] * scale,
                ori['bbox'][1] * scale,
                ori['bbox'][2] * scale,
                ori['bbox'][3] * scale
            ]
            for ori in valid_ori_images
        ]

        filtered_layout_res = []
        replaced_ori_bboxes = set()  # 记录被用来替换的图片区域下标

        for res in layout_res:
            # 保留 category_id==2 的区域
            if res['category_id'] == 2:
                filtered_layout_res.append(res)
                continue

            x1, y1, x2, y2 = res['poly'][0], res['poly'][1], res['poly'][4], res['poly'][5]
            res_bbox = [int(x1), int(y1), int(x2), int(y2)]

            # 检查是否落入某个图片区域
            matched_idx = None
            for idx, ori_bbox in enumerate(scaled_ori_bboxes):
                if is_in(res_bbox, ori_bbox):
                    matched_idx = idx
                    break

            if matched_idx is not None:
                replaced_ori_bboxes.add(matched_idx)  # 记录该图片区域确实发生替换
                continue  # 删除该 layout_res
            filtered_layout_res.append(res)

        # ✅ 只添加被替换掉的图片区域
        for idx in replaced_ori_bboxes:
            xmin, ymin, xmax, ymax = map(int, scaled_ori_bboxes[idx])
            image_res = {
                "category_id": 3,
                "original_label": "image",
                "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                "score": 1.0,
            }
            filtered_layout_res.append(image_res)

        images_layout_res[index] = filtered_layout_res

    return images_layout_res
