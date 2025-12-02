# Copyright (c) Opendatalab. All rights reserved.
import os
import time

from tqdm import tqdm

from rapid_doc.backend.utils import cross_page_table_merge
from rapid_doc.utils.config_reader import get_device, get_formula_enable
from rapid_doc.backend.pipeline.model_init import AtomModelSingleton
from rapid_doc.backend.pipeline.para_split import para_split
from rapid_doc.utils.block_pre_proc import prepare_block_bboxes, process_groups
from rapid_doc.utils.block_sort import sort_blocks_by_bbox
from rapid_doc.utils.cut_image import cut_image_and_table
from rapid_doc.utils.enum_class import ContentType
from rapid_doc.utils.model_utils import clean_memory
from rapid_doc.backend.pipeline.pipeline_magic_model import MagicModel
from rapid_doc.utils.ocr_utils import OcrConfidence
from rapid_doc.utils.pdf_image_tools import save_table_fill_image
from rapid_doc.utils.span_block_fix import fill_spans_in_blocks, fix_discarded_block, fix_block_spans
from rapid_doc.utils.span_pre_proc import remove_outside_spans, remove_overlaps_low_confidence_spans, \
    remove_overlaps_min_spans, txt_spans_extract
from rapid_doc.version import __version__
from rapid_doc.utils.hash_utils import bytes_md5


def page_model_info_to_page_info(page_model_info, image_dict, page_dict, image_writer, page_index, ocr_enable=False, formula_enabled=True, image_config=None):
    scale = image_dict["scale"]
    page_pil_img = image_dict["img_pil"]
    # page_img_md5 = str_md5(image_dict["img_base64"])
    page_img_md5 = bytes_md5(page_pil_img.tobytes())
    page_w, page_h = map(int, page_dict['size'])
    magic_model = MagicModel(page_model_info, scale)
    extract_original_image = image_config.get("extract_original_image", False) if image_config else False
    extract_original_image_iou_thresh = image_config.get("extract_original_image_iou_thresh", 0.9) if image_config else 0.9
    """保存表格里的图片"""
    save_table_fill_image(page_model_info['layout_dets'], page_dict.get('table_fill_image_list', []), page_img_md5, page_index, image_writer)

    """从magic_model对象中获取后面会用到的区块信息"""
    discarded_blocks = magic_model.get_discarded()
    text_blocks = magic_model.get_text_blocks()
    title_blocks = magic_model.get_title_blocks()
    inline_equations, interline_equations, interline_equation_blocks = magic_model.get_equations()

    img_groups = magic_model.get_imgs()
    table_groups = magic_model.get_tables()

    """对image和table的区块分组"""
    img_body_blocks, img_caption_blocks, img_footnote_blocks, maybe_text_image_blocks = process_groups(
        img_groups, 'image_body', 'image_caption_list', 'image_footnote_list'
    )

    table_body_blocks, table_caption_blocks, table_footnote_blocks, _ = process_groups(
        table_groups, 'table_body', 'table_caption_list', 'table_footnote_list'
    )

    """获取所有的spans信息"""
    spans = magic_model.get_all_spans()

    """某些图可能是文本块，通过简单的规则判断一下"""
    if len(maybe_text_image_blocks) > 0:
        for block in maybe_text_image_blocks:
            should_add_to_text_blocks = False
            # 相信版面结果，图片就是图片，不再尝试转为文本块
            # if ocr_enable and block.get('original_label') != 'chart':
            #     # 找到与当前block重叠的text spans
            #     span_in_block_list = [
            #         span for span in spans
            #         if span['type'] == 'text' and
            #            calculate_overlap_area_in_bbox1_area_ratio(span['bbox'], block['bbox']) > 0.7
            #     ]
            #
            #     if len(span_in_block_list) > 0:
            #         # 计算spans总面积
            #         spans_area = sum(
            #             (span['bbox'][2] - span['bbox'][0]) * (span['bbox'][3] - span['bbox'][1])
            #             for span in span_in_block_list
            #         )
            #
            #         # 计算block面积
            #         block_area = (block['bbox'][2] - block['bbox'][0]) * (block['bbox'][3] - block['bbox'][1])
            #
            #         # 判断是否符合文本图条件
            #         if block_area > 0 and spans_area / block_area > 0.25:
            #             should_add_to_text_blocks = True

            # 根据条件决定添加到哪个列表
            if should_add_to_text_blocks:
                block.pop('group_id', None)  # 移除group_id
                text_blocks.append(block)
            else:
                img_body_blocks.append(block)


    """将所有区块的bbox整理到一起"""
    if formula_enabled:
        interline_equation_blocks = []

    if len(interline_equation_blocks) > 0:

        for block in interline_equation_blocks:
            spans.append({
                "type": ContentType.INTERLINE_EQUATION,
                'score': block['score'],
                "bbox": block['bbox'],
                "content": "",
            })

        all_bboxes, all_discarded_blocks, footnote_blocks = prepare_block_bboxes(
            img_body_blocks, img_caption_blocks, img_footnote_blocks,
            table_body_blocks, table_caption_blocks, table_footnote_blocks,
            discarded_blocks,
            text_blocks,
            title_blocks,
            interline_equation_blocks,
            page_w,
            page_h,
        )
    else:
        all_bboxes, all_discarded_blocks, footnote_blocks = prepare_block_bboxes(
            img_body_blocks, img_caption_blocks, img_footnote_blocks,
            table_body_blocks, table_caption_blocks, table_footnote_blocks,
            discarded_blocks,
            text_blocks,
            title_blocks,
            interline_equations,
            page_w,
            page_h,
        )

    """在删除重复span之前，应该通过image_body和table_body的block过滤一下image和table的span"""
    """顺便删除大水印并保留abandon的span"""
    spans = remove_outside_spans(spans, all_bboxes, all_discarded_blocks)

    """删除重叠spans中置信度较低的那些"""
    spans, dropped_spans_by_confidence = remove_overlaps_low_confidence_spans(spans)
    """删除重叠spans中较小的那些"""
    spans, dropped_spans_by_span_overlap = remove_overlaps_min_spans(spans)

    """根据parse_mode，构造spans，主要是文本类的字符填充"""
    if ocr_enable:
        pass
    else:
        """使用新版本的混合ocr方案."""
        spans = txt_spans_extract(page_dict, spans, page_pil_img, scale, all_bboxes, all_discarded_blocks)

    """先处理不需要排版的discarded_blocks"""
    discarded_block_with_spans, spans = fill_spans_in_blocks(
        all_discarded_blocks, spans, 0.4
    )
    fix_discarded_blocks = fix_discarded_block(discarded_block_with_spans)

    """如果当前页面没有有效的bbox则跳过"""
    if len(all_bboxes) == 0 and len(fix_discarded_blocks) == 0:
        return None

    """对image/table/interline_equation截图"""
    for span in spans:
        if span['type'] in [ContentType.IMAGE, ContentType.TABLE, ContentType.INTERLINE_EQUATION]:
            span = cut_image_and_table(
                span, page_dict['ori_image_list'], extract_original_image, extract_original_image_iou_thresh, page_pil_img, page_img_md5, page_index, image_writer, scale=scale
            )

    """span填充进block"""
    block_with_spans, spans = fill_spans_in_blocks(all_bboxes, spans, 0.5)

    """对block进行fix操作"""
    fix_blocks = fix_block_spans(block_with_spans)

    """对block进行排序"""
    sorted_blocks = sort_blocks_by_bbox(fix_blocks, page_w, page_h, footnote_blocks, page_pil_img)

    """构造page_info"""
    page_info = make_page_info_dict(sorted_blocks, page_index, page_w, page_h, fix_discarded_blocks)

    return page_info


def result_to_middle_json(model_list, images_list, page_dict_list, image_writer, lang=None, ocr_enable=False, formula_enabled=True, ocr_config=None, image_config=None):
    middle_json = {"pdf_info": [], "_backend":"pipeline", "_version_name": __version__}
    formula_enabled = get_formula_enable(formula_enabled)
    for page_index, page_model_info in tqdm(enumerate(model_list), total=len(model_list), desc="Processing pages"):
        page_dict = page_dict_list[page_index]
        image_dict = images_list[page_index]
        page_info = page_model_info_to_page_info(
            page_model_info, image_dict, page_dict, image_writer, page_index, ocr_enable=ocr_enable, formula_enabled=formula_enabled, image_config=image_config
        )
        if page_info is None:
            page_w, page_h = map(int, page_dict['size'])
            page_info = make_page_info_dict([], page_index, page_w, page_h, [])
        middle_json["pdf_info"].append(page_info)

    """后置ocr处理"""
    need_ocr_list = []
    img_crop_list = []
    text_block_list = []
    for page_info in middle_json["pdf_info"]:
        for block in page_info['preproc_blocks']:
            if block['type'] in ['table', 'image']:
                for sub_block in block['blocks']:
                    if sub_block['type'] in ['image_caption', 'image_footnote', 'table_caption', 'table_footnote']:
                        text_block_list.append(sub_block)
            elif block['type'] in ['text', 'title']:
                text_block_list.append(block)
        for block in page_info['discarded_blocks']:
            text_block_list.append(block)
    for block in text_block_list:
        for line in block['lines']:
            for span in line['spans']:
                if 'np_img' in span:
                    need_ocr_list.append(span)
                    img_crop_list.append(span['np_img'])
                    span.pop('np_img')
    if len(img_crop_list) > 0:
        # start = time.perf_counter()
        atom_model_manager = AtomModelSingleton()
        ocr_model = atom_model_manager.get_atom_model(
            atom_model_name='ocr',
            det_db_box_thresh=0.3,
            lang=lang,
            ocr_config=ocr_config,
        )
        ocr_res_list = ocr_model.ocr(img_crop_list, det=False, tqdm_enable=True)[0]
        assert len(ocr_res_list) == len(
            need_ocr_list), f'ocr_res_list: {len(ocr_res_list)}, need_ocr_list: {len(need_ocr_list)}'
        for index, span in enumerate(need_ocr_list):
            ocr_text, ocr_score = ocr_res_list[index]
            if ocr_score > OcrConfidence.min_confidence:
                span['content'] = ocr_text
                span['score'] = float(f"{ocr_score:.3f}")
            else:
                span['content'] = ''
                span['score'] = 0.0
        end = time.perf_counter()
        # print(f"img_crop_list Predict 耗时: {end - start:.3f} 秒, 共 {len(img_crop_list)} 张图")

    """分段"""
    para_split(middle_json["pdf_info"])

    """表格跨页合并"""
    cross_page_table_merge(middle_json["pdf_info"])

    """清理内存"""
    # pdf_doc.close()
    if os.getenv('MINERU_DONOT_CLEAN_MEM') is None and len(model_list) >= 10:
        clean_memory(get_device())

    return middle_json


def make_page_info_dict(blocks, page_id, page_w, page_h, discarded_blocks):
    return_dict = {
        'preproc_blocks': blocks,
        'page_idx': page_id,
        'page_size': [page_w, page_h],
        'discarded_blocks': discarded_blocks,
    }
    return return_dict