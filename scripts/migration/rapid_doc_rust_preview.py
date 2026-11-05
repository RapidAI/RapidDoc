"""Development-only migration checkpoints; excluded from release wheels."""
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from rapid_doc_rust._host_context import host_call as _host_call, ACTIVE as _HOST_PAGE
from rapid_doc_rust._native import VERSION as __version__, RapidDocError, parse_pdf_text_json, recover_pdf_table_json, render_pdf_page_bytes, resolve_page_range, validate_rect
from rapid_doc_rust._native import OnnxSession as _OnnxSession
from rapid_doc_rust._native import ocr_png_json
from rapid_doc_rust._native import parse_pdf_ocr_json
from rapid_doc_rust._native import match_table_json
from rapid_doc_rust._native import recover_table_polygons_json
from rapid_doc_rust._native import connect_table_lines_json
from rapid_doc_rust._native import unet_line_geometry_json
from rapid_doc_rust._native import match_wired_table_json
from rapid_doc_rust._native import wired_matches_json
from rapid_doc_rust._native import select_table_candidate_json
from rapid_doc_rust._native import _PageImage, _NativeTextPage, _NativeTablePage
from rapid_doc_rust._native import merge_text_lines, organize_page_text_json
from rapid_doc_rust._native import split_paragraphs_json
from rapid_doc_rust._native import sort_model_blocks_json
from rapid_doc_rust._native import legacy_pipeline_json
from rapid_doc_rust._native import assemble_model_page_assets_bytes
from rapid_doc_rust._native import replace_inline_image_assets
from rapid_doc_rust._native import materialize_native_table_media_page_json, materialize_native_table_media_pdf_json
from rapid_doc_rust._native import fill_native_text_bytes
from rapid_doc_rust._native import deduplicate_native_characters_json, read_pdf_native_text_geometry_json
from rapid_doc_rust._native import read_pdf_character_primitives_json
from rapid_doc_rust._native import read_pdf_vector_geometry_json
from rapid_doc_rust._native import classify_pdf_font_name_bytes, pdfium_font_runtime_info_json
from rapid_doc_rust._native import read_pdf_link_annotations_json
from rapid_doc_rust._native import read_pdf_image_infos_json
from rapid_doc_rust._native import read_pdf_text_paint_json
from rapid_doc_rust._native import native_glyph_weight_json, native_pdf_document_json, read_pdf_visible_text_geometry_json, native_pdf_table_page_json
from rapid_doc_rust._native import read_native_pdf_page_json
from rapid_doc_rust._native import extract_original_pdf_images_bytes
from rapid_doc_rust._native import prepare_model_crops_bytes
from rapid_doc_rust._native import prepare_model_detection_bytes
from rapid_doc_rust._native import prepare_model_recognition_bytes
from rapid_doc_rust._native import validate_formula_png as _validate_formula_png
from rapid_doc_rust._native import prepare_formula_input_bytes
from rapid_doc_rust._native import FormulaDecoder as _FormulaDecoder, repair_formula_latex, fix_formula_text
from rapid_doc_rust._native import prepare_table_detection_bytes, pad_table_detection_bytes, prepare_table_recognition_bytes, prepare_table_direction_bytes
from rapid_doc_rust._native import extract_table_native_text_json
from rapid_doc_rust._native import prepare_table_model_payload_bytes
from rapid_doc_rust._document import PdfTextOutput, _read_bytes, _prepare_pdf, parse_pdf_text
from native_pdf_preview import convert_native_pdf_layout, native_pdf_layout_fallback_reason, merge_native_pdf_supplements, native_pdf_table_text_angle, NativePdfTableRecoverer
from rapid_doc_rust._compat import RapidDoc, RapidDocOutput
from rapid_doc_rust._diagnostics import OcrPageOutput,OnnxSession,RecoveredTable,RenderedPage,_complete_wired_table,_wired_request,match_table,match_wired_table,ocr_pdf_page,ocr_png,recognize_table_png,recognize_wired_table_png,recover_pdf_table,render_model_pages,render_pdf_page

"""Development-only migration checkpoints; never included in the release wheel."""


def organize_table_detection_boxes(boxes,*,masks=()):
    """旧表格OCR只排序/按公式切框，不合并框；坐标按NumPy float32处理。"""
    return json.loads(legacy_pipeline_json('table_detection_boxes',json.dumps(dict(boxes=boxes,masks=masks),allow_nan=False)))


def apply_table_ocr_results(boxes,results):
    """旧_run_table_ocr行级[框,文字,分数]列格式，文字按旧规则归一化及HTML转义；不筛分数。"""
    return json.loads(legacy_pipeline_json('apply_table_ocr',json.dumps(dict(boxes=boxes,results=results),allow_nan=False)))


def apply_table_prediction(table,html_code,*,formulas=(),fill_images=(),scale=1.0):
    """旧模型表格回填，返回副本；formulas为整页公式+checkbox原列表，不在此过滤区域。"""
    return json.loads(legacy_pipeline_json('apply_table_prediction',json.dumps(dict(table=table,html_code=html_code,formulas=formulas,fill_images=fill_images,scale=scale),allow_nan=False)))


def recognize_table_text_png(source,*,detector,recognizer=None,masks=(),batch_size=6,retry_padding=8,retry_short_side=32):
    """已选定表格方向后的旧行级OCR阶段；不包含原生文本、方向选择或表格结构模型。

    返回ocr_result为旧模型的HTML转义三列格式，不是现有预览matcher的原文line字典。
    masks须为表裁图坐标。失败显式传播，不做旧批失败逐图重试或word-box回退。
    """
    if not isinstance(batch_size,int)or isinstance(batch_size,bool)or not 1<=batch_size<=64:raise ValueError('batch_size must be 1..64')
    data=_read_bytes(source,64_000_000)
    boxes,retried,_=_detect_table_text_png(data,detector,masks,retry_padding,retry_short_side)
    if not boxes:return dict(boxes=[],ocr_result=[],retried=retried)
    result=_recognize_table_rows(data,boxes,recognizer,batch_size)
    return dict(boxes=boxes,ocr_result=result,retried=retried)


def _recognize_table_rows(data,boxes,recognizer,batch_size):
    if not callable(getattr(recognizer,'recognize_batch_png',None)):raise ValueError('table recognizer is required')
    crops=prepare_table_recognition_bytes(data,boxes)
    rows=recognizer.recognize_batch_png(crops,batch_size=batch_size)
    result=apply_table_ocr_results(boxes,[(r['text'],r['confidence'])for r in rows])
    return result


def _detect_table_text_png(data,detector,masks,retry_padding,retry_short_side):
    for value,bound in [(retry_padding,1024),(retry_short_side,65536)]:
        if not isinstance(value,int)or isinstance(value,bool)or not 0<=value<=bound:raise ValueError('invalid table retry setting')
    if not callable(getattr(detector,'detect_table_png',None)):raise ValueError('table detector is required')
    organize_table_detection_boxes([],masks=masks)
    metadata,masked=prepare_table_detection_bytes(data,json.dumps(dict(masks=masks),allow_nan=False));metadata=json.loads(metadata)
    boxes=organize_table_detection_boxes([r['points']for r in detector.detect_table_png(masked)],masks=masks)
    retried=False
    if not boxes and retry_padding>0 and min(metadata['width'],metadata['height'])<retry_short_side:
        retried=True;padded=pad_table_detection_bytes(masked,retry_padding)
        retry=organize_table_detection_boxes([r['points']for r in detector.detect_table_png(padded)])
        if retry:
            boxes=json.loads(legacy_pipeline_json('restore_table_detection',json.dumps(dict(boxes=retry,padding=retry_padding,**metadata),allow_nan=False)))
    return boxes,retried,metadata


def plan_table_direction(page,table,*,width,height,boxes=(),scale=1.0):
    """旧原生文字方向投票及方向模型调用门槛；span bbox使用普通JSON四坐标。"""
    return json.loads(legacy_pipeline_json('plan_table_direction',json.dumps(dict(page=page,table=table,width=width,height=height,boxes=boxes,scale=scale),allow_nan=False)))


def prepare_oriented_table_text_png(source,*,page,table,detector,orientation=None,masks=(),scale=1.0,retry_padding=8,retry_short_side=32):
    """旧表格初次检测→原生角度/方向模型→90/270旋转重新检测；返回最终RGB PNG和框。

    180度保持图片；旋转后沿用原masks坐标。无原生角度且达到旧门槛才要求分类器。
    不包含原生表格捷径、原生文字填充、OCR识别或结构模型。
    """
    # 固定页面快照，失败不改调用方JSON。预检发生在检测推理前。
    page,table=json.loads(json.dumps([page,table],allow_nan=False))
    data=_read_bytes(source,64_000_000)
    metadata,_=prepare_table_direction_bytes(data,'0',False);metadata=json.loads(metadata)
    plan_table_direction(page,table,**metadata,scale=scale)
    boxes,retried,_=_detect_table_text_png(data,detector,masks,retry_padding,retry_short_side)
    plan=plan_table_direction(page,table,boxes=boxes,**metadata,scale=scale)
    used=plan.pop('needs_classifier');label=plan['rotate_label']
    if used:
        if not callable(getattr(orientation,'classify_orientation_png',None)):raise ValueError('table orientation classifier is required')
        # 旧_process_single_table传的是BGR ndarray，现有显式分类API接收RGB；转换一次保留旧模型输入。
        _,classifier_png=prepare_table_direction_bytes(data,'0',True)
        prediction=orientation.classify_orientation_png(classifier_png)
        label=json.loads(legacy_pipeline_json('select_table_orientation',json.dumps(prediction,allow_nan=False)))
    retries=[retried]
    # 重新编码也移除tRNS/方向元数据，旧ndarray路径不使用这些元数据。
    _,data=prepare_table_direction_bytes(data,label,False)
    if label in ('90','270'):
        boxes,retried,_=_detect_table_text_png(data,detector,masks,retry_padding,retry_short_side)
        retries.append(retried)
    return dict(plan,png=data,boxes=boxes,rotate_label=label,classifier_used=used,retries=retries)


def extract_table_native_text_png(source,boxes,*,native_page,table,useful_list,scale=1.0,word_boxes=False):
    """旧PDF表格文字填充；返回已HTML转义的三列，word_boxes=True返回逐字符框。

    source为表格RGB裁图；保留旧稀疏文字/对比度和全局bbox裁图语义。无结果返回[]。
    无效输入显式报错，不复现旧except吞掉任意异常。
    """
    if type(word_boxes)is not bool:raise ValueError('word_boxes must be bool')
    request=dict(boxes=boxes,native_page=native_page,table=table,useful_list=useful_list,scale=scale,word_boxes=word_boxes)
    return json.loads(extract_table_native_text_json(_read_bytes(source,64_000_000),json.dumps(request,allow_nan=False)))


def recognize_table_text_stage_png(source,*,native_page,table,useful_list,detector,recognizer=None,orientation=None,masks=(),scale=1.0,force_ocr=False,ocr_enable=False,batch_size=6,retry_padding=8,retry_short_side=32):
    """旧表格方向准备→原生文字优先→整体为空时行级OCR；返回最终PNG/框和三列文字。

    force_ocr/ocr_enable或页面/表格方向禁止原生文字时直接OCR。部分原生结果不会补OCR。
    不包含原生表格结构捷径、OCR word-box、公式图片插入和结构模型。
    """
    if type(force_ocr)is not bool or type(ocr_enable)is not bool:raise ValueError('OCR flags must be bool')
    if type(batch_size)is not int or not 1<=batch_size<=64:raise ValueError('batch_size must be 1..64')
    native_page,table,useful_list,masks=json.loads(json.dumps([native_page,table,useful_list,masks],allow_nan=False))
    # 校验坐标映射，所有配置错误先于模型执行。
    legacy_pipeline_json('table_native_spans',json.dumps(dict(boxes=[],useful_list=useful_list,scale=scale),allow_nan=False))
    data=_read_bytes(source,64_000_000)
    prepared=prepare_oriented_table_text_png(data,page=native_page,table=table,detector=detector,orientation=orientation,masks=masks,scale=scale,retry_padding=retry_padding,retry_short_side=retry_short_side)
    rows=[];text_source='none'
    if prepared['boxes']and not force_ocr and not ocr_enable and prepared['rotate_label']=='0'and prepared['pdf_not_rotate']:
        rows=extract_table_native_text_png(prepared['png'],prepared['boxes'],native_page=native_page,table=table,useful_list=useful_list,scale=scale)
        if rows:text_source='native'
    if not rows and prepared['boxes']:
        rows=_recognize_table_rows(prepared['png'],prepared['boxes'],recognizer,batch_size);text_source='ocr'
    return dict(prepared,ocr_result=rows,text_source=text_source)


def prepare_table_hints(page,table,*,useful_list,formulas=(),checkboxes=(),scale=1.0,formula_enabled=True,image_enabled=True,extract_original_image=False):
    """旧公式/复选框坐标调整和图片选择；返回page/table副本及formulas/fill_images。

    page.ori_image_list和table.layout_image_list为JSON元数据，图片像素由调用方按uuid另持有。
    页面rotate_label为90/180/270时按旧流程禁止选PDF原图。保留旧累计填图列表。
    """
    request=dict(page=page,table=table,useful_list=useful_list,formulas=formulas,checkboxes=checkboxes,scale=scale,formula_enabled=formula_enabled,image_enabled=image_enabled,extract_original_image=extract_original_image)
    return json.loads(legacy_pipeline_json('plan_table_hints',json.dumps(request,allow_nan=False)))


def prepare_table_model_payload_png(source,ocr_result,*,fill_images=(),formulas=(),skip_text_in_image=True,inline_delimiters=('$','$')):
    """旧RapidTableModel的图片涂白/占位符、图片内文字删除、公式插入；不运行结构模型。

    ocr_result为已转义三列，已有文字不再转义。空[]/None返回needs_ocr=True，须先完成旧全图OCR分支；
    [[],[],[]]保持旧truthy语义，可添加提示。返回{png,ocr_result,needs_ocr}，PNG保持RGB供下一阶段转BGR。
    """
    if not isinstance(inline_delimiters,(list,tuple))or len(inline_delimiters)!=2:raise ValueError('inline_delimiters must contain two strings')
    request=dict(ocr_result=ocr_result,fill_images=fill_images,formulas=formulas,skip_text_in_image=skip_text_in_image,inline_left=inline_delimiters[0],inline_right=inline_delimiters[1])
    metadata,data=prepare_table_model_payload_bytes(_read_bytes(source,64_000_000),json.dumps(request,allow_nan=False))
    return dict(json.loads(metadata),png=data)


def recognize_table_full_ocr_png(source,*,detector,recognizer=None,masks=(),batch_size=6):
    """旧predict的det+rec回退：原图检测→排序/公式切框→识别→0.5过滤→旧转义三列。

    无检测结果不要求识别器；无白边重试、无检测前遮罩、无宽3过滤。所有框被masks切掉时
    显式报错，保留旧直接识别器收到空列表会失败的行为，不冒充正常无检测。
    """
    if type(batch_size)is not int or not 1<=batch_size<=64:raise ValueError('batch_size must be 1..64')
    if masks is None:masks=()
    if not callable(getattr(detector,'detect_table_png',None)):raise ValueError('table detector is required')
    organize_table_detection_boxes([],masks=masks)
    _,data=prepare_table_direction_bytes(_read_bytes(source,64_000_000),'0',False)
    raw=[r['points']for r in detector.detect_table_png(data)]
    if not raw:return dict(boxes=[],ocr_result=[])
    boxes=organize_table_detection_boxes(raw,masks=masks)
    if not boxes:raise ValueError('legacy full OCR cannot recognize an empty list after formula cutting')
    if not callable(getattr(recognizer,'recognize_batch_png',None)):raise ValueError('table recognizer is required')
    crops=prepare_table_recognition_bytes(data,boxes)
    results=recognizer.recognize_batch_png(crops,batch_size=batch_size)
    rows=json.loads(legacy_pipeline_json('apply_table_full_ocr',json.dumps(dict(boxes=boxes,results=[(r['text'],r['confidence'])for r in results]),allow_nan=False)))
    return dict(boxes=rows[0]if rows else[],ocr_result=rows)


def prepare_table_model_input_png(source,ocr_result,*,detector=None,recognizer=None,fill_images=(),formulas=(),skip_text_in_image=True,inline_delimiters=('$','$'),batch_size=6):
    """已定方向的旧模型输入：空OCR执行全图回退，仍空时can_infer=False并跳过所有提示。

    原有三列（包括三空列）直接沿用，不重复识别；之后复用旧图片/公式payload阶段。
    不执行结构网络、直接predict的自动90度方向分支或旧异常重试。
    """
    if type(batch_size)is not int or not 1<=batch_size<=64:raise ValueError('batch_size must be 1..64')
    ocr_result,fill_images,formulas=json.loads(json.dumps([ocr_result,fill_images,formulas],allow_nan=False))
    data=_read_bytes(source,64_000_000);used=not bool(ocr_result)
    if used:
        rows=recognize_table_full_ocr_png(data,detector=detector,recognizer=recognizer,masks=formulas or (),batch_size=batch_size)
        ocr_result=rows['ocr_result']
        if not ocr_result:
            _,data=prepare_table_direction_bytes(data,'0',False)
            return dict(png=data,ocr_result=None,needs_ocr=False,can_infer=False,full_ocr_used=True)
    payload=prepare_table_model_payload_png(data,ocr_result,fill_images=fill_images,formulas=formulas,skip_text_in_image=skip_text_in_image,inline_delimiters=inline_delimiters)
    return dict(payload,can_infer=True,full_ocr_used=used)


def match_legacy_slanet_table(structure,ocr_result,*,width,height):
    """旧SLANet的OCR格式转换/过滤、IoU-distance匹配、原顺序raw HTML及逻辑点。

    structure来自SLANet结构解码，OCR须为已转义三列。保留旧文本拼接，不再转义/重排。
    raw_html尚未经过RapidTableModel的最终CJK/BeautifulSoup归一化；不是完整表格API。
    """
    request=dict(structure=structure,ocr_result=ocr_result,width=width,height=height)
    return json.loads(legacy_pipeline_json('legacy_table_match',json.dumps(request,allow_nan=False)))


def recognize_legacy_slanet_raw_png(source,ocr_result,*,structure_model=None,variant='slanet-plus',detector=None,recognizer=None,fill_images=(),formulas=(),skip_text_in_image=True,inline_delimiters=('$','$'),batch_size=6):
    """已定方向的旧模型输入→SLANet/1m结构网络→旧matcher raw HTML，不做最终HTML归一化。

    全图OCR回退后仍空时不调用结构模型；不支持UNET/分类器/双模型仲裁。
    """
    if variant not in ('slanet-plus','slanet-1m'):raise ValueError('invalid legacy SLANet variant')
    payload=prepare_table_model_input_png(source,ocr_result,detector=detector,recognizer=recognizer,fill_images=fill_images,formulas=formulas,skip_text_in_image=skip_text_in_image,inline_delimiters=inline_delimiters,batch_size=batch_size)
    if not payload['can_infer']:return dict(raw_html=None,structure=None,match=None,full_ocr_used=payload['full_ocr_used'])
    if not callable(getattr(structure_model,'table_structure_legacy_png',None)):raise ValueError('legacy SLANet structure model is required')
    metadata,_=prepare_table_direction_bytes(payload['png'],'0',False);metadata=json.loads(metadata)
    structure=structure_model.table_structure_legacy_png(payload['png'],variant=variant)
    result=match_legacy_slanet_table(structure,payload['ocr_result'],**metadata)
    return dict(raw_html=result['raw_html'],structure=structure,match=result,full_ocr_used=payload['full_ocr_used'])


def normalize_legacy_table_html(html_code):
    """旧表格最终HTML/CJK空格归一化；无直接单元格文字变化时保留输入原串。"""
    return json.loads(legacy_pipeline_json('normalize_legacy_table_html',json.dumps(html_code,allow_nan=False)))


def recognize_legacy_slanet_table_png(source,ocr_result,**options):
    """已定方向的SLANet模型输入→结构推理→旧matcher→最终HTML归一化。"""
    result=recognize_legacy_slanet_raw_png(source,ocr_result,**options)
    return dict(result,html=normalize_legacy_table_html(result['raw_html']))


def match_legacy_unet_table(cells,ocr_result):
    """旧UNET几何匹配/跨格拆分/同行归并及原HTML；文字须已按旧表格规则转义。

    cells为table_cells_png输出；保持旧默认无空格识别器配置，不对输入文字再次转义。
    """
    if not isinstance(ocr_result,(list,tuple))or len(ocr_result)!=3:raise ValueError('UNET OCR must have three columns')
    if any(not isinstance(c,(list,tuple))for c in ocr_result)or len({len(c)for c in ocr_result})!=1:raise ValueError('UNET OCR column lengths must match')
    if len(ocr_result[0])>1000:raise ValueError('UNET OCR exceeds budget')
    lines=[dict(points=p,text=t,confidence=s)for p,t,s in zip(*ocr_result)]
    request=_wired_request(cells,lines);request['legacy_text']=True
    result=json.loads(match_wired_table_json(json.dumps(request,allow_nan=False)))
    return dict(raw_html=result['html'],matches=result['matches'])


def recognize_legacy_unet_table_png(source,ocr_result,*,structure_model=None,detector=None,recognizer=None,
                                    fill_images=(),formulas=(),skip_text_in_image=True,inline_delimiters=('$','$'),batch_size=6):
    """已定方向的旧模型输入→UNET格框/逻辑格→旧文字匹配→最终HTML。

    对应旧enable_blank_cell_rec=False默认配置；无格框raw_html=''，空OCR跳过结构模型。
    旧异常吞掉后返回空表的行为改为显式报错；不调用预览OCR或安全HTML替代旧文字。
    """
    payload=prepare_table_model_input_png(source,ocr_result,detector=detector,recognizer=recognizer,
        fill_images=fill_images,formulas=formulas,skip_text_in_image=skip_text_in_image,inline_delimiters=inline_delimiters,batch_size=batch_size)
    if not payload['can_infer']:return dict(html=None,raw_html=None,cells=None,matches=None,full_ocr_used=payload['full_ocr_used'])
    if not callable(getattr(structure_model,'table_cells_png',None)):raise ValueError('legacy UNET structure model is required')
    # 旧RapidTable未向UNET传kwargs，TSRUnetStructurer中rotated_fix实际为True。
    cells=structure_model.table_cells_png(payload['png'],rotated_fix=True)
    if not cells['polygons']:return dict(html='',raw_html='',cells=cells,matches=None,full_ocr_used=payload['full_ocr_used'])
    result=match_legacy_unet_table(cells,payload['ocr_result'])
    return dict(result,html=normalize_legacy_table_html(result['raw_html']),cells=cells,full_ocr_used=payload['full_ocr_used'])


def select_legacy_table_candidate(ocr_texts,*,wired_html,wireless_html):
    """旧表格候选原HTML计数/文本/命中与优先级；OCR文字已经转义，不再转义。"""
    return json.loads(legacy_pipeline_json('select_legacy_table',json.dumps(dict(texts=list(ocr_texts),wired_html=wired_html,wireless_html=wireless_html),allow_nan=False)))


def plan_native_table_rule(table,*,scale=1.,config=None,rotate_label=None,ocr_enable=False,has_source_pdf=True,formulas=(),checkboxes=(),images=()):
    """旧DocVortex捷径在初始化表格模型前的守卫；图片框为PDF point，公式框为页面像素。"""
    request=dict(table=table,scale=scale,config={}if config is None else config,rotate_label=rotate_label,ocr_enable=ocr_enable,
                 has_source_pdf=has_source_pdf,formulas=formulas,checkboxes=checkboxes,images=images)
    return json.loads(legacy_pipeline_json('plan_native_table_rule',json.dumps(request,allow_nan=False)))


def accept_native_table_rule(table,result,*,threshold=.9):
    """旧非空HTML/置信度门槛及回填，返回副本；不检查<table>标签，不运行原生恢复。"""
    return json.loads(legacy_pipeline_json('accept_native_table_rule',json.dumps(dict(table=table,result=result,threshold=threshold),allow_nan=False)))


def select_native_table_candidates(candidates,*,remove_undercounted=True):
    """旧已生成原生候选的欠分割过滤、来源阈值和拓扑仲裁；返回原索引，不生成候选。"""
    return json.loads(legacy_pipeline_json('select_native_table_candidates',json.dumps(dict(candidates=list(candidates),remove_undercounted=remove_undercounted),allow_nan=False)))


def prepare_native_table_glyphs(chars,*,table_bbox,angle=0):
    """旧DocVortex字符筛选/归一化及局部旋转字形，输入为普通JSON字符，bbox为PDF point。

    char为str/None，bbox/loose_bbox为四坐标；不是pdftext分组字符或Rust PDF提取适配器。
    """
    request=dict(chars=list(chars),table_bbox=table_bbox,angle=angle)
    return json.loads(legacy_pipeline_json('native_table_pending',json.dumps(request,allow_nan=False)))


def build_native_table_text(chars,*,table_bbox,angle=0):
    """旧字形→视觉行/token；保留spacing_bbox与原char_idx，空字形返回None，不生成表格候选。"""
    request=dict(chars=list(chars),table_bbox=table_bbox,angle=angle)
    return json.loads(legacy_pipeline_json('build_native_table_text',json.dumps(request,allow_nan=False)))


def merge_native_inline_clusters(lines, *, page_size, table_bboxes=()):
    """旧二维簇完整合并；返回36字段的lines投影和原输入位置groups，不含LineItem私有缓存。"""
    request=dict(lines=lines,page_size=page_size,table_bboxes=list(table_bboxes))
    return json.loads(legacy_pipeline_json('merge_native_inline_clusters',json.dumps(request,allow_nan=False)))


def classify_native_table_script_cell(lines, *, page_size, tight_bboxes=None, origins=None, fraction_members=()):
    """cell视觉行→真实二维合并→脚本判定→词首抑制→同源角色冲突聚合，返回来源ID角色。"""
    request=dict(lines=lines,page_size=page_size,tight_bboxes=tight_bboxes or {},origins=origins or {},fraction_members=list(fraction_members))
    return json.loads(legacy_pipeline_json('classify_native_table_script_cell',json.dumps(request,allow_nan=False)))


def infer_native_text_lanes(line_geometry, *, local_page_width, median_height, recalculate_intervals=True):
    """旧完整栏带推断；输入(line JSON, 已转正bbox)，返回各栏及原输入位置line_indices。"""
    if not isinstance(recalculate_intervals,bool):raise ValueError('recalculate_intervals must be bool')
    request=dict(line_geometry=line_geometry,local_page_width=local_page_width,median_height=median_height,recalculate_intervals=recalculate_intervals)
    return json.loads(legacy_pipeline_json('infer_native_text_lanes',json.dumps(request,allow_nan=False)))


def classify_native_table_script_line(line, *, page_size, tight_bboxes=None, origins=None, fraction_members=()):
    """旧表格专用单行：实际脚本分类、偏移词首抑制及同源冲突聚合；不执行二维合行。"""
    request=dict(line=line,page_size=page_size,tight_bboxes=tight_bboxes or {},origins=origins or {},fraction_members=list(fraction_members))
    return json.loads(legacy_pipeline_json('classify_native_table_script_line',json.dumps(request,allow_nan=False)))


def measure_native_line_typography(line, *, page_size):
    """旧原生行排版度量，返回字体覆盖率、有效字高和行首/句尾证据。"""
    return json.loads(legacy_pipeline_json('measure_native_line_typography',json.dumps(dict(line=line,page_size=page_size),allow_nan=False)))


def prepare_native_table_script_cells(table_input, result, *, tight_bboxes=None):
    """旧表格脚本准备：来源字形、视觉行、网格线排除及分式来源保护；尚不合并二维行。"""
    request=dict(table_input=table_input,result=result,tight_bboxes=tight_bboxes or {})
    return json.loads(legacy_pipeline_json('prepare_native_table_script_cells',json.dumps(request,allow_nan=False)))


def refine_native_script_tokens(chars, roles, *, tight_bboxes=None, origins=None, formula_region=False, ordinary_native_roles=False):
    """按旧规则精炼给定几何角色，不使用新数学token策略。"""
    if not isinstance(formula_region,bool)or not isinstance(ordinary_native_roles,bool):raise ValueError('formula/native flags must be bool')
    request=dict(chars=chars,roles=roles,tight_bboxes=tight_bboxes or {},origins=origins or {},formula_region=formula_region,ordinary_native_roles=ordinary_native_roles)
    return json.loads(legacy_pipeline_json('refine_native_script_tokens',json.dumps(request,allow_nan=False)))


def classify_native_script_runs(chars, *, tight_bboxes=None, origins=None, memberships=None):
    """按公式区域连续分段实际执行几何分类和token精炼，返回角色/正文计数/区域标记。"""
    request=dict(chars=chars,tight_bboxes=tight_bboxes or {},origins=origins or {})
    if memberships is not None:request['memberships']=memberships
    return json.loads(legacy_pipeline_json('classify_native_script_runs',json.dumps(request,allow_nan=False)))


def classify_native_script_line(line, *, page_size, tight_bboxes=None, origins=None, fraction_members=()):
    """旧单行完整角色路径：来源排序、方向变换、区域分段、结构补证和分式抑制。"""
    request=dict(line=line,page_size=page_size,tight_bboxes=tight_bboxes or {},origins=origins or {},fraction_members=list(fraction_members))
    return json.loads(legacy_pipeline_json('classify_native_script_line',json.dumps(request,allow_nan=False)))


def native_fraction_member_sources(chars, drawing_lines, *, page_size, angle=0, tight_bboxes=None, prepared_rules=None):
    """按旧分数线几何返回需保护的char_idx来源ID，不是chars列表位置。"""
    request=dict(chars=chars,drawing_lines=drawing_lines,page_size=page_size,angle=angle,tight_bboxes=tight_bboxes or {},prepared_rules=prepared_rules)
    return json.loads(legacy_pipeline_json('native_fraction_member_sources',json.dumps(request,allow_nan=False)))


def classify_native_script_roles(chars, *, tight_bboxes=None, origins=None, protected_body_indices=()):
    """旧基础loose/tight/origin脚本分类；不执行inline token精炼或整表自动样式。"""
    request=dict(chars=chars,tight_bboxes=tight_bboxes or {},origins=origins or {},protected_body_indices=list(protected_body_indices))
    return json.loads(legacy_pipeline_json('classify_native_script_roles',json.dumps(request,allow_nan=False)))


def native_numeric_script_indices(chars, *, tight_bboxes=None, origins=None, protected_body_indices=(), roles=None):
    """旧数值上标补充阶段，默认以全body角色为参考。"""
    request=dict(chars=chars,tight_bboxes=tight_bboxes or {},origins=origins or {},protected_body_indices=list(protected_body_indices))
    if roles is not None:request['roles']=roles
    return json.loads(legacy_pipeline_json('native_numeric_script_indices',json.dumps(request,allow_nan=False)))


def prepare_native_table_block(source, candidate, *, failed_annotations=(), calibrated_char_bboxes=None):
    """旧 PDF 表格块准备；返回候选/投影成员/表体框/校准框/字符，不提交认领。"""
    request = dict(source=source, candidate=candidate, failed_annotations=list(failed_annotations), calibrated_char_bboxes=calibrated_char_bboxes)
    return json.loads(legacy_pipeline_json('prepare_native_table_block', json.dumps(request, allow_nan=False)))


def prepare_native_script_lines(lines, *, page_size, tight_bboxes=None, origins=None, all_chars=(), drawing_lines=()):
    """原生视觉行→紧凑脚本证据，复用角色分类/分式保护，供最终输出投影。"""
    request=dict(lines=lines,page_size=page_size,tight_bboxes=tight_bboxes or {},origins=origins or {},all_chars=all_chars,drawing_lines=drawing_lines)
    return json.loads(legacy_pipeline_json('prepare_native_script_lines',json.dumps(request,allow_nan=False)))


def trace_native_inline_evidence(blocks, *, page_size, style_lines=(), link_lines=(), script_lines=()):
    """完整行内投影及脚本物化诊断，返回blocks和materialized_ranges。"""
    request=dict(blocks=blocks,page_size=page_size,style_lines=style_lines,link_lines=link_lines,script_lines=script_lines)
    return json.loads(legacy_pipeline_json('trace_native_inline_evidence',json.dumps(request,allow_nan=False)))


def apply_native_inline_evidence(blocks, *, page_size, style_lines=(), link_lines=(), script_lines=None):
    """按旧几何/文本规则将字体与外链证据投影为独立InlineSpan输出块。"""
    request=dict(blocks=blocks,page_size=page_size,style_lines=style_lines,link_lines=link_lines,script_lines=script_lines)
    return json.loads(legacy_pipeline_json('apply_native_inline_evidence',json.dumps(request,allow_nan=False)))


def detect_native_inline_evidence(lines, *, drawing_lines=(), annotations=()):
    """旧字体/绘图样式与链接注解检测，输出独立style_lines/link_lines半开区间。"""
    request=dict(lines=lines,drawing_lines=drawing_lines,annotations=annotations)
    return json.loads(legacy_pipeline_json('detect_native_inline_evidence',json.dumps(request,allow_nan=False)))


def _physical_table_result(operation, request):
    result=json.loads(legacy_pipeline_json(operation,json.dumps(request,allow_nan=False)))
    for candidate in result.get('candidates',[]):
        for annotation in candidate.get('annotations',[]):
            annotation['line_bboxes']={int(k):b for k,b in annotation['line_bboxes']}
    return result


def inspect_native_rule_table_geometry(source, *, angle=0, median_height=None):
    """从原始页面行与绘图线自动生成局部表格行、连通网格、列轨和表题种子。"""
    return _physical_table_result('inspect_native_rule_table_geometry',dict(source=source,angle=angle,median_height=median_height))


def expand_native_connected_table_grids(source, candidates, *, angle=0, median_height=None, excluded_bboxes=()):
    """以实际连通网格扩展已有候选，补全成员并保持表题和表注排除。"""
    import copy
    candidates=copy.deepcopy(candidates)
    for candidate in candidates:
        for annotation in candidate.get('annotations',[]):
            annotation['line_bboxes']=list(annotation['line_bboxes'].items())
    return _physical_table_result('expand_native_connected_table_grids',dict(source=source,candidates=candidates,angle=angle,median_height=median_height,excluded_bboxes=excluded_bboxes))


def detect_native_closed_rule_tables(source, *, angle=0, median_height=None, excluded_bboxes=()):
    """闭合线网格完整候选分支，自动聚行、表题、表注、稀疏表及相邻页续表证明。"""
    return _physical_table_result('detect_native_closed_rule_tables',dict(source=source,angle=angle,median_height=median_height,excluded_bboxes=excluded_bboxes))


def detect_native_physical_table_candidates(source, *, excluded_bboxes=()):
    """编排填充网格、铺片网格及各方向闭合线网格；普通横线等页面分支尚未接入。"""
    return _physical_table_result('detect_native_physical_table_candidates',dict(source=source,excluded_bboxes=excluded_bboxes))


def detect_native_rule_table_candidates(source, *, angle=0, median_height=None, excluded_bboxes=()):
    """完整普通横线区间候选：自动选择核心、列统计、评分、注释及连通扩框。"""
    return _physical_table_result('detect_native_rule_table_candidates',dict(source=source,angle=angle,median_height=median_height,excluded_bboxes=excluded_bboxes))


def detect_native_supplemental_tables(source, *, kind, excluded_bboxes=()):
    """旧完整候选产生器；支持aligned_numeric/captioned_mixed及既有结构kind。"""
    return _physical_table_result('detect_native_supplemental_tables',dict(source=source,kind=kind,excluded_bboxes=excluded_bboxes))


def detect_native_structural_table_candidates(source, *, excluded_bboxes=()):
    """全部候选产生分支及替换编排，返回整页后处理之前的候选。"""
    return _physical_table_result('detect_native_structural_table_candidates',dict(source=source,excluded_bboxes=excluded_bboxes))


def detect_native_table_candidates(source, *, excluded_bboxes=()):
    """完整旧表格候选检测，含表题分组、尾段裁剪、Form 排除及续表标题收尾。"""
    return _physical_table_result('detect_native_table_candidates',dict(source=source,excluded_bboxes=excluded_bboxes))


def apply_native_table_page_policies(source, candidates, *, operation='all'):
    """对正式候选执行旧整页后处理；operation 可单独选择 join/trim/forms/externalize/bind。"""
    import copy
    candidates=copy.deepcopy(candidates)
    for candidate in candidates:
        for annotation in candidate.get('annotations',[]):
            annotation['line_bboxes']=[[int(k),v] for k,v in annotation['line_bboxes'].items()]
    return _physical_table_result('apply_native_table_page_policies',dict(source=source,candidates=candidates,operation=operation))


def trailing_native_table_grid(source, prepared):
    """从已组装页末表最后物理行提取下一页续表列轨；prepared 含 fixed_blocks 和 remaining_lines。"""
    return json.loads(legacy_pipeline_json('trailing_native_table_grid',json.dumps(dict(source=source,prepared=prepared),allow_nan=False)))


def expand_native_rule_table_candidate(rows, lines, rule_bboxes, core_row_indices, *, page_size, median_height, angle=0, caption_source_index=None, has_possible_table_notes=True):
    """旧规则核心、表题和自动表注组装为正式候选，并还原方向坐标；不生成上游规则候选。"""
    result=json.loads(legacy_pipeline_json('expand_native_rule_table_candidate',json.dumps(dict(rows=rows,lines=lines,rule_bboxes=rule_bboxes,core_row_indices=core_row_indices,page_size=page_size,median_height=median_height,angle=angle,caption_source_index=caption_source_index,has_possible_table_notes=has_possible_table_notes),allow_nan=False)))
    for annotation in result['candidate']['annotations']:
        annotation['line_bboxes']={int(k):v for k,v in annotation['line_bboxes']}
    return result


def collect_native_table_footnotes(rows, lines, rule_bbox, core_line_indices, *, page_size, median_height, angle=0):
    """旧自动表注收集，包含显式注释及有表内引用证明的辅助标记；返回独立行和注释。"""
    result=json.loads(legacy_pipeline_json('collect_native_table_footnotes',json.dumps(dict(rows=rows,lines=lines,rule_bbox=rule_bbox,core_line_indices=core_line_indices,page_size=page_size,median_height=median_height,angle=angle),allow_nan=False)))
    if result['annotation'] is not None:
        result['annotation']['line_bboxes']={int(k):v for k,v in result['annotation']['line_bboxes']}
    return result


def prepare_native_rule_table_annotations(rows, rule_bboxes, core_row_indices, *, median_height, caption_source_index=None, footnote_row_indices=()):
    """旧规则表核心/表题续行/注释记录准备；表注行显式给出，尚不自动识别。"""
    result=json.loads(legacy_pipeline_json('prepare_native_rule_table_annotations',json.dumps(dict(rows=rows,rule_bboxes=rule_bboxes,core_row_indices=core_row_indices,median_height=median_height,caption_source_index=caption_source_index,footnote_row_indices=footnote_row_indices),allow_nan=False)))
    for annotation in result['annotations']:
        annotation['line_bboxes']={int(k):v for k,v in annotation['line_bboxes']}
    return result


def exclude_native_prose_spanning_rule_drafts(source, candidates):
    """旧跨正文草稿排除；消费table/draft上游投影，返回独立候选及拒绝原数组位置。"""
    return json.loads(legacy_pipeline_json('exclude_native_prose_spanning_rule_drafts',json.dumps(dict(source=source,candidates=candidates),allow_nan=False)))


def merge_native_table_candidates(candidates):
    """旧已物化候选合并与注释冲突消解，返回独立候选；不执行draft物化。"""
    encoded=[]
    for candidate in candidates:
        candidate=dict(candidate)
        annotations=[]
        for annotation in candidate['annotations']:
            annotation=dict(annotation)
            annotation['line_bboxes']=[[int(k),v] for k,v in annotation['line_bboxes'].items()]
            annotations.append(annotation)
        candidate['annotations']=annotations
        encoded.append(candidate)
    result=json.loads(legacy_pipeline_json('merge_native_table_candidates',json.dumps(dict(candidates=encoded),allow_nan=False)))
    for candidate in result['candidates']:
        for annotation in candidate['annotations']:
            annotation['line_bboxes']={int(k):v for k,v in annotation['line_bboxes']}
    return result


def prefer_complete_native_table_replacements(base, recovered, replacements):
    """旧完整覆盖替换仲裁，返回独立base/recovered/replacements；候选产生器仍为上游阶段。"""
    return json.loads(legacy_pipeline_json('prefer_complete_native_table_replacements',json.dumps(dict(base=base,recovered=recovered,replacements=replacements),allow_nan=False)))


def inspect_native_closed_grids(source, candidates, *, allow_two_by_two=False):
    """旧闭合网格证明，消费显式候选及真实源线/文字/Form，返回原候选顺序的closed_grid_flags。"""
    return json.loads(legacy_pipeline_json('inspect_native_closed_grids',json.dumps(dict(source=source,candidates=candidates,allow_two_by_two=allow_two_by_two),allow_nan=False)))


def detect_native_filled_grid_tables(path_infos, *, page_size, excluded_bboxes=()):
    """旧外框/最大填充单元格/水平行带表格候选，不依赖CJK字框或单矩形计数。"""
    return json.loads(legacy_pipeline_json('detect_native_filled_grid_tables',json.dumps(dict(path_infos=path_infos,page_size=page_size,excluded_bboxes=excluded_bboxes),allow_nan=False)))


def protect_native_graphic_captions(lines, candidate_bboxes, *, median_height):
    """旧英文编号图题及续行保护，返回独立lines和protected_source_indices，保留caption语义修改。"""
    return json.loads(legacy_pipeline_json('protect_native_graphic_captions',json.dumps(dict(lines=lines,candidate_bboxes=candidate_bboxes,median_height=median_height),allow_nan=False)))


def detect_native_bar_graphics(source, *, em):
    """旧完整原生柱图候选：表格排除、柱组/图题、两轮扩框和真实网格拒绝，返回独立source。"""
    return json.loads(legacy_pipeline_json('detect_native_bar_graphics',json.dumps(dict(source=source,em=em),allow_nan=False)))


def prepare_native_bar_groups(source, *, em):
    """旧柱图前段：真实表框排除、柱体/共同零轴/间隔簇和数字标签证据，尚不执行图体扩展。"""
    return json.loads(legacy_pipeline_json('prepare_native_bar_groups',json.dumps(dict(source=source,em=em),allow_nan=False)))


def detect_native_tiled_tables(source, *, excluded_bboxes=()):
    """旧相接填充单元格表格候选，包含真实CJK尺度、表头补线和相邻题名转移。"""
    return json.loads(legacy_pipeline_json('detect_native_tiled_tables',json.dumps(dict(source=source,excluded_bboxes=excluded_bboxes),allow_nan=False)))


def prepare_native_bar_geometry(path_infos, drawing_lines, *, em, table_bboxes, axis_queries=()):
    """旧柱体基线/准入与零轴补边。table_bboxes为明确上游证据，尚非完整柱图检测。"""
    return json.loads(legacy_pipeline_json('prepare_native_bar_geometry',json.dumps(dict(path_infos=path_infos,drawing_lines=drawing_lines,em=em,table_bboxes=table_bboxes,axis_queries=axis_queries),allow_nan=False)))


def extract_native_index_blocks(lines, *, page_size, container_bboxes=(), require_heading=False, require_explicit_heading=False):
    """旧目录完整提取器，返回独立行副本、目录块、认领与剩余行。"""
    return json.loads(legacy_pipeline_json('extract_native_index_blocks',json.dumps(dict(lines=lines,page_size=page_size,container_bboxes=container_bboxes,require_heading=require_heading,require_explicit_heading=require_explicit_heading),allow_nan=False)))


def select_native_form_regions(source):
    """旧Form选择/收紧。来源映射使用bbox/source_indices条目列表；不提取PDF Form结构。"""
    return json.loads(legacy_pipeline_json('select_native_form_regions',json.dumps(dict(source=source),allow_nan=False)))


def prepare_native_container_table_page(source, *, excluded_bboxes=(), prior_fixed_blocks=(), prior_claimed_line_indices=(), table_header_separator_bboxes=(), tight_bboxes=None, origins=None, calibrated_char_bboxes=None):
    """目录预认领→Form选择→代码/表格。Form/图像块物化和后续文字公式尚未接入。"""
    return json.loads(legacy_pipeline_json('prepare_native_container_table_page',json.dumps(dict(source=source,excluded_bboxes=excluded_bboxes,prior_fixed_blocks=prior_fixed_blocks,prior_claimed_line_indices=prior_claimed_line_indices,table_header_separator_bboxes=table_header_separator_bboxes,tight_bboxes=tight_bboxes or {},origins=origins or {},calibrated_char_bboxes=calibrated_char_bboxes),allow_nan=False)))


def build_native_code_blocks(source, *, kind='filled', excluded_bboxes=(), claimed_line_indices=()):
    """旧完整代码容器；filled 为底色等宽代码，ruled 包含横线清单及算法标题清单。"""
    return json.loads(legacy_pipeline_json('build_native_code_blocks',json.dumps(dict(source=source,kind=kind,excluded_bboxes=excluded_bboxes,claimed_line_indices=claimed_line_indices),allow_nan=False)))


def prepare_native_code_table_page(source, *, excluded_bboxes=(), prior_fixed_blocks=(), prior_claimed_line_indices=(), table_header_separator_bboxes=(), tight_bboxes=None, origins=None, calibrated_char_bboxes=None):
    """旧页准备顺序：横线/算法代码→表格提交→底色代码；统一行认领及续表轨道。"""
    return json.loads(legacy_pipeline_json('prepare_native_code_table_page',json.dumps(dict(source=source,excluded_bboxes=excluded_bboxes,prior_fixed_blocks=prior_fixed_blocks,prior_claimed_line_indices=prior_claimed_line_indices,table_header_separator_bboxes=table_header_separator_bboxes,tight_bboxes=tight_bboxes or {},origins=origins or {},calibrated_char_bboxes=calibrated_char_bboxes),allow_nan=False)))


def prepare_native_table_page(source, *, excluded_bboxes=(), prior_fixed_blocks=(), prior_claimed_line_indices=(), table_header_separator_bboxes=(), tight_bboxes=None, origins=None, calibrated_char_bboxes=None):
    """正式页准备中的表格事务链：自动图形避让/检测/恢复/认领/数值标记及末表轨道。"""
    return json.loads(legacy_pipeline_json('prepare_native_table_page',json.dumps(dict(source=source,excluded_bboxes=excluded_bboxes,prior_fixed_blocks=prior_fixed_blocks,prior_claimed_line_indices=prior_claimed_line_indices,table_header_separator_bboxes=table_header_separator_bboxes,tight_bboxes=tight_bboxes or {},origins=origins or {},calibrated_char_bboxes=calibrated_char_bboxes),allow_nan=False)))


def detect_native_strong_graphics(source):
    """完整旧强图形检测与网格/包含/重叠仲裁；返回独立源投影及最终图框。"""
    return json.loads(legacy_pipeline_json('detect_native_strong_graphics',json.dumps(dict(source=source),allow_nan=False)))


def detect_native_diagram_components(source, *, em):
    """旧完整原生diagram组件：自动表格屏障、路径/图片连接、正文拒绝及时间轴认领。"""
    return json.loads(legacy_pipeline_json('detect_native_diagram_components',json.dumps(dict(source=source,em=em),allow_nan=False)))


def detect_native_raster_diagrams(source, *, em, frame_bboxes=()):
    """旧栅格标签连接图候选与普通文字框证据，实际网格保护在候选内部执行。"""
    return json.loads(legacy_pipeline_json('detect_native_raster_diagrams',json.dumps(dict(source=source,em=em,frame_bboxes=frame_bboxes),allow_nan=False)))


def detect_native_concentric_graphics(source, *, em):
    """旧编号图题/百分比/同心轮廓和引线证据，候选仍需共用图形网格仲裁。"""
    return json.loads(legacy_pipeline_json('detect_native_concentric_graphics',json.dumps(dict(source=source,em=em),allow_nan=False)))


def arbitrate_native_graphic_candidates(source, candidate_bboxes, *, median_height):
    """旧有字多列网格排除及图形包含/重叠仲裁；候选由调用方原顺序提供。"""
    return json.loads(legacy_pipeline_json('arbitrate_native_graphic_candidates',json.dumps(dict(source=source,candidate_bboxes=candidate_bboxes,median_height=median_height),allow_nan=False)))


def inspect_native_graphic_grids(source, regions, *, median_height):
    """从实际绘图/文字构造旧多列有字网格，返回每个区域的表格排除证据。"""
    return json.loads(legacy_pipeline_json('inspect_native_graphic_grids',json.dumps(dict(source=source,regions=regions,median_height=median_height),allow_nan=False)))


def detect_native_isolated_graphics(source):
    """旧独立矢量图体及小栅格证据，返回原路径分组、矢量框和小栅格原位置。"""
    return json.loads(legacy_pipeline_json('detect_native_isolated_graphics',json.dumps(dict(source=source),allow_nan=False)))


def detect_native_drawing_graphics(drawing_lines, path_infos, *, page_size, median_height):
    """旧绘图线组件统计和复杂路径证明图形候选，返回独立组件位置/框和候选。"""
    return json.loads(legacy_pipeline_json('detect_native_drawing_graphics',json.dumps(dict(drawing_lines=drawing_lines,path_infos=path_infos,page_size=page_size,median_height=median_height),allow_nan=False)))


def detect_native_path_graphics(path_infos, *, page_size, median_height):
    """旧复杂路径容器/坐标轴候选，保留原顺序，尚不做图形与表格仲裁。"""
    return json.loads(legacy_pipeline_json('detect_native_path_graphics',json.dumps(dict(path_infos=path_infos,page_size=page_size,median_height=median_height),allow_nan=False)))


def detect_native_stacked_graphics(source, *, em, fallback_em):
    """旧CJK图形尺度证据与堆叠柱候选，em为既有检测尺度，不在接口中重排/认领图体。"""
    return json.loads(legacy_pipeline_json('detect_native_stacked_graphics',json.dumps(dict(source=source,em=em,fallback_em=fallback_em),allow_nan=False)))


def group_native_raster_chart_descriptions(source):
    """旧刻度/图片图体证明和说明续行归组，返回独立source、图体框和刻度行位置。"""
    return json.loads(legacy_pipeline_json('group_native_raster_chart_descriptions',json.dumps(dict(source=source),allow_nan=False)))


def prepare_native_text_noise(source, *, visual_bboxes=()):
    """旧短数字噪声候选和按需颜色索引，source包含lines/chars，身份按数组位置。"""
    return json.loads(legacy_pipeline_json('prepare_native_text_noise',json.dumps(dict(source=source,visual_bboxes=visual_bboxes),allow_nan=False)))


def apply_native_text_noise(source, *, paint=None, visual_bboxes=()):
    """按旧浅灰/深色或白色正文对照清理来源号，返回独立source和诊断索引。"""
    return json.loads(legacy_pipeline_json('apply_native_text_noise',json.dumps(dict(source=source,paint=paint,visual_bboxes=visual_bboxes),allow_nan=False)))


def filter_pdf_native_text_noise(source, pdf, *, page_index=0, pdfium_library=None, password=None, visual_bboxes=()):
    """先判候选，仅有候选时读取实际PDF填色；无候选时不访问pdf。"""
    evidence=prepare_native_text_noise(source,visual_bboxes=visual_bboxes)
    paint=read_pdf_text_paint(pdf,evidence['requested_char_indices'],page_index=page_index,pdfium_library=pdfium_library,password=password) if evidence['candidates'] else None
    return apply_native_text_noise(source,paint=paint,visual_bboxes=visual_bboxes)


def prepare_native_image_regions(pages):
    """根据已提取图片证据执行旧跨页水印/背景/标题白边过滤，返回独立区域和排序指纹。"""
    return json.loads(legacy_pipeline_json('prepare_native_image_regions',json.dumps(dict(pages=pages),allow_nan=False)))


def prepare_native_source_lines(lines, *, page_size, drawing_lines=(), image_infos=()):
    """旧页源的局部段落/薄栅格图片线/装饰文字规则，返回独立lines和drawing_lines。"""
    request=dict(lines=lines,page_size=page_size,drawing_lines=drawing_lines,image_infos=image_infos)
    return json.loads(legacy_pipeline_json('prepare_native_source_lines',json.dumps(request,allow_nan=False)))


def build_native_visual_lines(chars, *, page_size, page_rotation=0, supported_angles=(0.,90.,270.)):
    """旧原生字符分行、视觉run和上下标合并；输出独立LineItem普通JSON。"""
    if isinstance(page_rotation,bool) or not isinstance(page_rotation,int):
        raise ValueError('page_rotation must be an integer')
    request=dict(chars=chars,page_size=page_size,page_rotation=page_rotation,supported_angles=supported_angles)
    return json.loads(legacy_pipeline_json('build_native_visual_lines',json.dumps(request,allow_nan=False)))


def materialize_native_table_media_page(blocks, page_png=None, *, page_size=None):
    """单页PNG→表内媒体；默认单位框，page_size指定时按旧千分刻度转换point框。"""
    request=json.dumps(dict(blocks=blocks,page_size=page_size),allow_nan=False)
    return json.loads(materialize_native_table_media_page_json(request,page_png))


def materialize_native_table_media(pages, source, *, page_sizes=None, pdfium_library=None, password=None, max_pixels=16_000_000):
    """每个有媒体的PDF页渲染一次；生成PNG data URI，普通页内失败回退文字。"""
    if isinstance(max_pixels,bool) or not isinstance(max_pixels,int) or max_pixels<=0:
        raise ValueError('max_pixels must be a positive integer')
    request=json.dumps(dict(pages=pages,page_sizes=page_sizes),allow_nan=False)
    if any(block.get('_table_images') for page in pages for block in page):
        library,data=_prepare_pdf(source,pdfium_library,None)
    else:
        library,data='',b''
    return json.loads(materialize_native_table_media_pdf_json(request,library,data,password,max_pixels))


def materialize_native_table_blocks(source, candidates, *, tight_bboxes=None, origins=None, calibrated_char_bboxes=None):
    """原生恢复/媒体/投影及原子块提交，返回独立块、来源认领和候选/行副作用。"""
    request = dict(source=source, candidates=list(candidates), tight_bboxes=tight_bboxes, origins=origins, calibrated_char_bboxes=calibrated_char_bboxes)
    return json.loads(legacy_pipeline_json('materialize_native_table_blocks', json.dumps(request, allow_nan=False)))


def prepare_native_table_annotations(source, candidate, *, calibrated_char_bboxes=None):
    """块级准备及实际注释物化；输出外置/失败集合、表体框和投影成员。"""
    request = dict(source=source, candidate=candidate, calibrated_char_bboxes=calibrated_char_bboxes)
    return json.loads(legacy_pipeline_json('prepare_native_table_annotations', json.dumps(request, allow_nan=False)))


def restore_native_front_matter_panel(source, bbox):
    """仅供结构恢复失败路径调用：返回 restore 和保留标题副作用的独立 lines。"""
    return json.loads(legacy_pipeline_json('restore_native_front_matter_panel', json.dumps(dict(source=source, bbox=bbox), allow_nan=False)))


class NativeTableRecoveryError(RapidDocError):
    """结构恢复失败；HTML 物化失败仍使用 RapidDocError。"""

    def __init__(self, message, *, code):
        super().__init__(message)
        self.code = code
        self.stage = "recovery"


def render_native_table_html_with_scripts(result, table_input, *, tight_bboxes=None, origins=None, preserve_font_styles=False):
    """中立表格结果 → 语义 HTML；保留旧 Python 的原 HTML 回退条件。"""
    if type(preserve_font_styles) is not bool:
        raise TypeError("preserve_font_styles must be bool")
    request = dict(result=result, table_input=table_input, tight_bboxes=tight_bboxes, origins=origins, preserve_font_styles=preserve_font_styles)
    return json.loads(legacy_pipeline_json('render_native_table_html_with_scripts', json.dumps(request, allow_nan=False)))


def recover_native_table_result(table_input, *, tight_bboxes=None, origins=None, preserve_font_styles=False):
    """字符/绘图 → 结构恢复和语义 HTML，返回 None 或 (content, result)。"""
    if type(preserve_font_styles) is not bool:
        raise TypeError("preserve_font_styles must be bool")
    request = dict(table_input=table_input, tight_bboxes=tight_bboxes, origins=origins, preserve_font_styles=preserve_font_styles)
    value = json.loads(legacy_pipeline_json('recover_native_table_result', json.dumps(request, allow_nan=False)))
    if value is None:
        return None
    if 'recovery_error' in value:
        error = value['recovery_error']
        raise NativeTableRecoveryError(error['message'], code=error['code'])
    return value['content'], value['result']


def render_native_table_styled_cell(cell, glyphs, *, median_height, roles=None, chars_by_source=None, tight_bboxes=None, origins=None, preserve_font_styles=False, possible_space_indices=None):
    """按给定来源角色渲染旧单元格HTML片段；不会自动生成角色。"""
    if not isinstance(preserve_font_styles,bool):raise ValueError('preserve_font_styles must be bool')
    request=dict(cell=cell,glyphs=glyphs,median_height=median_height,roles=roles or {},chars_by_source=chars_by_source,tight_bboxes=tight_bboxes,origins=origins,preserve_font_styles=preserve_font_styles,possible_space_indices=None if possible_space_indices is None else list(possible_space_indices))
    return json.loads(legacy_pipeline_json('render_native_table_styled_cell',json.dumps(request,allow_nan=False)))


def build_native_table_cell_text(glyphs,*,median_height):
    """旧单元格视觉行细分、跨行连接和逐片段来源；glyphs来自build_native_table_text。

    返回{text,parts,groups}，parts中的None表示算法插入的间隔，原始连字符不删除。
    """
    request=dict(glyphs=list(glyphs),median_height=median_height)
    return json.loads(legacy_pipeline_json('build_native_table_cell_text',json.dumps(request,allow_nan=False)))


def prepare_native_table_rule_evidence(table_input,*,snap_tolerance,join_gap,evidence_halo=0.,include_drawing=True,include_rectangles=True):
    """旧物理线/矩形→局部片段→合并线→raw/supported轨道；不是完整候选或PDF提取器。

    table_input含table_bbox、数值angle、drawing_lines和rectangles；坐标为PDF point。
    """
    request=dict(table_input=table_input,snap_tolerance=snap_tolerance,join_gap=join_gap,evidence_halo=evidence_halo,
                 include_drawing=include_drawing,include_rectangles=include_rectangles)
    return json.loads(legacy_pipeline_json('prepare_native_table_rule_evidence',json.dumps(request,allow_nan=False)))


def canonicalize_native_table_tracks(positions,*,glyph_centers,threshold,extent,evidence_halo,minimum_track_count,
                                     rules=None,orientation='',separator_extent=0.,snap_tolerance=0.,
                                     preserve_double_boundary=False,collapse_narrow_bands=True,collapse_leading_edge=True):
    """旧外框吸附/窄带折叠与alias冲突；返回tracks、alias_conflict、outer_collapse_count、unique。

    positions使用上一步浮点轨道，rules使用其合并线；不自动选择最佳假设或生成最终单元格。
    """
    request=dict(positions=list(positions),glyph_centers=list(glyph_centers),threshold=threshold,extent=extent,
                 evidence_halo=evidence_halo,minimum_track_count=minimum_track_count,rules=rules,orientation=orientation,
                 separator_extent=separator_extent,snap_tolerance=snap_tolerance,preserve_double_boundary=preserve_double_boundary,
                 collapse_narrow_bands=collapse_narrow_bands,collapse_leading_edge=collapse_leading_edge)
    return json.loads(legacy_pipeline_json('canonicalize_native_table_tracks',json.dumps(request,allow_nan=False)))


def build_native_table_vector_candidate(tracks,text,*,evidence_label='line_grid',diagnostics=None):
    """旧规范轨道→隔断连通/物理证据→文字落格/评分和普通HTML。

    tracks是旧_VectorTracks的普通JSON字段，须已通过轨道生成/规范化阶段；本接口不完成
    从PDF提取、完整轨道假设构建/选择或语义上下标渲染。返回topology/candidate/diagnostics/html。
    """
    request=dict(tracks=tracks,text=text,evidence_label=evidence_label)
    if diagnostics is not None:request['diagnostics']=diagnostics
    return json.loads(legacy_pipeline_json('build_native_table_vector_candidate',json.dumps(request,allow_nan=False)))


def generate_native_table_vector_candidates(table_input,text):
    """给定PDF绘图/矩形原语和旧NativeTableText，执行完整旧矢量候选生成。

    返回candidates/diagnostics/html（同序）；包括轨道硬门、有限raw/supported假设、line/rect候选，
    不读取PDF，不包含其他文本/稀疏候选、完整引擎仲裁或语义上下标物化。
    """
    return json.loads(legacy_pipeline_json('generate_native_table_vector_candidates',json.dumps(dict(table_input=table_input,text=text),allow_nan=False)))


def inspect_native_table_vector_tracks(table_input,text,*,include_drawing=True,include_rectangles=False,prune_unsupported_horizontal=False):
    """旧完整轨道生成阶段及拒绝诊断；不预先删除装饰竖线，不构建最终候选。"""
    request=dict(table_input=table_input,text=text,include_drawing=include_drawing,include_rectangles=include_rectangles,
                 prune_unsupported_horizontal=prune_unsupported_horizontal)
    return json.loads(legacy_pipeline_json('inspect_native_table_vector_tracks',json.dumps(request,allow_nan=False)))


def generate_native_table_text_candidates(table_input,text):
    """旧少线sparse_grid、无线text_grid和两列key_value候选，返回candidates/diagnostics/html。

    输入给定PDF原语与NativeTableText；含续行/分组表头逻辑，不是完整原生引擎或PDF读取器。
    """
    return json.loads(legacy_pipeline_json('generate_native_table_text_candidates',json.dumps(dict(table_input=table_input,text=text),allow_nan=False)))


def inspect_native_table_engine_policy(table_input,text,*,vector_attempts=(),candidates=()):
    """旧输入几何/相交原语门与矢量诊断守卫；只返回事实/索引，不执行完整_evaluate。"""
    request=dict(table_input=table_input,text=text,vector_attempts=list(vector_attempts),candidates=list(candidates))
    return json.loads(legacy_pipeline_json('inspect_native_table_engine_policy',json.dumps(request,allow_nan=False)))


def regroup_native_table_numeric_text(text):
    """旧数值文字中心重分行：保留字形/来源/spacing，按较小净空阈值重建token。"""
    return json.loads(legacy_pipeline_json('regroup_native_table_numeric_text',json.dumps(dict(text=text),allow_nan=False)))


def build_native_table_aligned_numeric_candidate(table_input,text,*,regroup=True):
    """旧数值对齐候选（默认先重分行），返回text/numeric_groups/candidate/html。

    包含重复金额右缘、表头净空/跨度及逻辑表体；无候选为None。输入是给定页面原语，
    不读取PDF、不执行完整_evaluate或语义上下标。regroup=False时row_index须与数组位置一致。
    """
    request=dict(table_input=table_input,text=text,regroup=regroup)
    return json.loads(legacy_pipeline_json('build_native_table_aligned_numeric_candidate',json.dumps(request,allow_nan=False)))


def generate_native_table_rule_band_candidates(table_input,text):
    """旧三线表换行表头/同步描述列rowspan，返回candidates/diagnostics/html。

    仅给定PDF原语/NativeTableText；普通单行表头三线表仍交其他旧路线，不自动放宽门槛。
    """
    return json.loads(legacy_pipeline_json('generate_native_table_rule_band_candidates',json.dumps(dict(table_input=table_input,text=text),allow_nan=False)))


def inspect_native_table_sparse_columns(table_input,text,*,header_bottom=0.):
    """旧多行稀疏共享列假设/墨迹占用/空白走廊诊断；不执行完整多行候选。"""
    request=dict(table_input=table_input,text=text,header_bottom=header_bottom)
    return json.loads(legacy_pipeline_json('inspect_native_table_sparse_columns',json.dumps(request,allow_nan=False)))


def generate_native_table_sparse_hybrid_candidates(table_input,text):
    """旧物理线/矩形与文字对齐融合，返回candidates/diagnostics/html。

    包含两层表头、物理边界跨token及100%捕获/顺序/零歧义发布门。
    给定页面原语/NativeTableText；不读取PDF、不执行完整候选引擎。
    """
    return json.loads(legacy_pipeline_json('generate_native_table_sparse_hybrid_candidates',json.dumps(dict(table_input=table_input,text=text),allow_nan=False)))


def generate_native_table_multiline_candidates(table_input,text):
    """旧多行稀疏主分支及整行分组带回退，返回candidates/diagnostics/html。

    table_input的chars用于回退标题的原始来源/显式空白及紧字框投影复核。
    给定页面原语/NativeTableText，不打开PDF、不执行完整候选引擎。
    """
    return json.loads(legacy_pipeline_json('generate_native_table_multiline_candidates',json.dumps(dict(table_input=table_input,text=text),allow_nan=False)))


def project_native_table_spatial_text(chars,*,table_bbox,angle=0):
    """旧PDF表格空间投影：给定原始字符JSON，保留等宽网格内部空白。"""
    request=dict(chars=list(chars),table_bbox=table_bbox,angle=angle)
    return json.loads(legacy_pipeline_json('project_native_table_spatial_text',json.dumps(request,allow_nan=False)))


def build_native_table_banded_numeric_candidate(table_input,text):
    """旧分带数值候选，返回candidate/diagnostics/html；拒绝时candidate/html为None。

    table_input.chars用于各格原始来源投影；表体投影后须保持非空白文字一致。
    给定页面原语/NativeTableText，不读PDF，不执行完整候选引擎。
    """
    return json.loads(legacy_pipeline_json('build_native_table_banded_numeric_candidate',json.dumps(dict(table_input=table_input,text=text),allow_nan=False)))


def evaluate_native_table_primitives(table_input):
    """原始字符/线/矩形→完整旧引擎，返回文字、生成/保留候选、选中项与拒绝原因。

    输入须是NativeTableInput普通JSON；文字在Rust内构建，不接受预制候选或选中项。
    不读取PDF、不包含适配器的语义上下标HTML。
    """
    return json.loads(legacy_pipeline_json('evaluate_native_table_primitives',json.dumps(dict(table_input=table_input),allow_nan=False)))


def recover_native_table_primitives(table_input):
    """完整旧引擎的NativeTableResult普通JSON；未接纳返回None，资源/输入错误传播。"""
    return json.loads(legacy_pipeline_json('recover_native_table_primitives',json.dumps(dict(table_input=table_input),allow_nan=False)))


def diagnose_native_table_primitives(table_input):
    """重放完整旧引擎及各路线诊断；共享工作预算，不作为生产结果写入文档。"""
    return json.loads(legacy_pipeline_json('diagnose_native_table_primitives',json.dumps(dict(table_input=table_input),allow_nan=False)))


def inspect_native_table_candidate(text,*,rows,cols,specs,allow_single_row=False,allow_single_column=False,use_grid_index=False):
    """旧完整网格校验、可选原子轨道索引与逐字归属诊断；不发现物理网格。"""
    request=dict(text=text,rows=rows,cols=cols,specs=list(specs),allow_single_row=allow_single_row,allow_single_column=allow_single_column,use_grid_index=use_grid_index)
    return json.loads(legacy_pipeline_json('inspect_native_table_candidate',json.dumps(request,allow_nan=False)))


def build_native_table_candidate(text,*,source,rows,cols,specs,structure_support,row_stability,column_stability,
                                 issues=(),allow_single_row=False,allow_single_column=False,require_atomic_tokens=False,
                                 use_grid_index=False,diagnostics=None):
    """旧给定网格→字符落格→单元格文字/来源→质量分和普通HTML。

    返回{candidate,diagnostics,html}，被旧门槛拒绝时candidate/html为None。结构/行/列质量须由
    上游候选算法显式提供，不自行发现网格或捏造结构证据；最终来源置信门槛仍需候选仲裁。
    """
    request=dict(text=text,source=source,rows=rows,cols=cols,specs=list(specs),structure_support=structure_support,
                 row_stability=row_stability,column_stability=column_stability,issues=list(issues),allow_single_row=allow_single_row,
                 allow_single_column=allow_single_column,require_atomic_tokens=require_atomic_tokens,use_grid_index=use_grid_index)
    if diagnostics is not None:request['diagnostics']=diagnostics
    return json.loads(legacy_pipeline_json('build_native_table_candidate',json.dumps(request,allow_nan=False)))


def render_native_table_candidate(candidate):
    """旧普通原生候选HTML：保留跨行/列、空行和quote=False转义；不含上下标等语义物化。"""
    return json.loads(legacy_pipeline_json('render_native_table_candidate',json.dumps(candidate,allow_nan=False)))


def try_native_table_rule(source,table,*,recoverer=None,page_index=0,pdfium_library=None,password=None,
                          score_threshold=.9,**options):
    """可选原生恢复器适配点；守卫通过才读source并调用recoverer.recover_table。

    恢复器接收PDF bytes及page_index/bbox/pdfium_library/password关键字，返回{html,confidence}或None。
    不内置原生候选生成；没有恢复器返回backend_unavailable。ImportError传播，其余恢复异常返回
    error_type诊断并交模型回退；不关闭调用方流，不修改输入table。
    """
    if type(page_index)is not int or not 0<=page_index<10000:raise ValueError('invalid page_index')
    if isinstance(score_threshold,bool)or not isinstance(score_threshold,(int,float))or not math.isfinite(score_threshold)or not 0<=score_threshold<=1:raise ValueError('invalid native table score threshold')
    table=json.loads(json.dumps(table,allow_nan=False));plan=plan_native_table_rule(table,**options)
    if not plan['attempt']:return dict(accepted=False,table=table,diagnostic=dict(method='model',reason=plan['reason']))
    if recoverer is None:return dict(accepted=False,table=table,diagnostic=dict(method='model',reason='backend_unavailable'))
    if not callable(getattr(recoverer,'recover_table',None)):raise TypeError('native recoverer must implement recover_table')
    # 读取/输入错误不是旧恢复调用内部异常，不在下方回退范围内。
    library,data=_prepare_pdf(source,pdfium_library,None)
    try:
        result=recoverer.recover_table(data,page_index=page_index,bbox=list(plan['table_bbox']),pdfium_library=library,password=password)
        accepted=accept_native_table_rule(table,result,threshold=score_threshold)
    except ImportError:raise
    except Exception as exc:
        return dict(accepted=False,table=table,diagnostic=dict(method='model',reason='recovery_error',error_type=type(exc).__name__))
    return dict(accepted,diagnostic=dict(method='docvortex_native'if accepted['accepted']else'model',reason='accepted'if accepted['accepted']else'not_accepted'))


def recognize_legacy_table_auto_png(source,ocr_result,*,classifier=None,wired_model=None,wireless_model=None,
                                    detector=None,recognizer=None,secondary_classifier=None,classifier_output_type=None,
                                    wireless_variant='slanet-plus',use_compare_table=False,cls_conf_threshold=.9,
                                    fill_images=(),formulas=(),skip_text_in_image=True,inline_delimiters=('$','$'),batch_size=6):
    """旧UNET+SLANet默认无空格补识别配置：共用模型输入→分类→必要候选→旧选优→归一化。

    仅执行被选中的模型；低于0.9或use_compare_table=True才执行两者。异常显式传播。
    """
    if type(use_compare_table)is not bool:raise ValueError('use_compare_table must be bool')
    if isinstance(cls_conf_threshold,bool)or not isinstance(cls_conf_threshold,(int,float))or not math.isfinite(cls_conf_threshold)or not 0<=cls_conf_threshold<=1:raise ValueError('invalid classifier confidence threshold')
    if wireless_variant not in ('slanet-plus','slanet-1m'):raise ValueError('invalid legacy SLANet variant')
    if classifier_output_type not in (None,'probabilities','logits'):raise ValueError('invalid classifier output type')
    payload=prepare_table_model_input_png(source,ocr_result,detector=detector,recognizer=recognizer,fill_images=fill_images,
        formulas=formulas,skip_text_in_image=skip_text_in_image,inline_delimiters=inline_delimiters,batch_size=batch_size)
    if not payload['can_infer']:return dict(html=None,route=None,classification=None,candidates={},selection=None,full_ocr_used=payload['full_ocr_used'])
    if not callable(getattr(classifier,'classify_table_png',None)):raise ValueError('table classifier is required')
    classification=classifier.classify_table_png(payload['png'],output_type=classifier_output_type)
    if secondary_classifier is not None:
        classification=_merge_table_classifications(classification,secondary_classifier.classify_table_png(payload['png'],variant='qanything'))
    label=classification['label'];score=classification['score']
    if label not in ('wired','wireless')or isinstance(score,bool)or not isinstance(score,(int,float))or not math.isfinite(score)or not 0<=score<=1:raise ValueError('invalid table classifier result')
    compare=use_compare_table or score<cls_conf_threshold
    # 保持旧模型执行先后：先分类命中的候选，必要时再执行另一个。
    required=[label]+(['wireless'if label=='wired'else'wired']if compare else[])
    if any((wired_model if r=='wired'else wireless_model)is None for r in required):raise ValueError('required legacy table model is missing')
    candidates={}
    for route in required:
        if route=='wired':result=recognize_legacy_unet_table_png(payload['png'],payload['ocr_result'],structure_model=wired_model)
        else:result=recognize_legacy_slanet_raw_png(payload['png'],payload['ocr_result'],structure_model=wireless_model,variant=wireless_variant)
        candidates[route]=result['raw_html']
    selection=select_legacy_table_candidate(payload['ocr_result'][1],wired_html=candidates['wired'],wireless_html=candidates['wireless'])if compare else None
    route=selection['route']if compare else label
    # 单候选UNET的CJK归一化幂等；双候选只对最终原HTML执行归一化。
    return dict(html=normalize_legacy_table_html(candidates[route]),route=route,classification=classification,candidates=candidates,
                selection=selection,full_ocr_used=payload['full_ocr_used'])


class FormulaDecoder:
    """旧BPE/ByteLevel公式token解码、LaTeX修复及ftfy6.3.1默认处理；Unicode16 NFC。"""
    def __init__(self,character):
        self._native=_FormulaDecoder(character if isinstance(character,str)else json.dumps(character,allow_nan=False))
    def decode(self,tokens):
        return self._native.decode(tokens)


def prepare_formula_input(source,*,height=384,width=384):
    """旧PP-FormulaNet Plus输入；返回shape/裁边元数据、RGB像素和小端float32 NCHW字节。

    M/S默认384×384，L用768×768。仅预处理，不运行或解码公式模型。
    Pillow像素逐字节对照；浮点灰度与旧OpenCV IPP允许≤1e-6绝对误差。
    """
    if any(not isinstance(n,int)or isinstance(n,bool)or not 2<=n<=2048 for n in [height,width]):raise ValueError('formula dimensions must be integers in 2..2048')
    metadata,rgb,tensor=prepare_formula_input_bytes(_read_bytes(source,64_000_000),height,width)
    return dict(json.loads(metadata),rgb=rgb,tensor=tensor)


def _formula_snapshots(plans):
    return json.loads(json.dumps([{k:v for k,v in p.items()if k not in ('images','page_image')}for p in plans],allow_nan=False))


def apply_model_formula_results(plans,results):
    """跨页旧公式非空回填；字符串不改写，None/空串保留原latex；结果数须相等。"""
    plans=list(plans)
    if isinstance(results,(str,bytes,dict)):raise TypeError('formula results must be an iterable of str/None')
    request=dict(plans=_formula_snapshots(plans),results=list(results))
    snapshots=json.loads(legacy_pipeline_json('apply_model_formulas',json.dumps(request,allow_nan=False)))
    for source,result in zip(plans,snapshots):
        if 'images'in source:result['images']=dict(source['images'])
        if 'page_image'in source:result['page_image']=source['page_image']
    return snapshots


def execute_model_formulas(plans,*,recognizer=None,batch_size=1):
    """按旧页序统一调用recognizer.recognize_formula_batch_png(PNG列表,batch_size=...)。

    适配器返回等长str/None列表，须自行实现旧公式模型前后处理；不是OCR识别器。
    所有输入先读取验证，失败不回填原plans。可传入OnnxSession执行PP-FormulaNet。
    """
    if not isinstance(batch_size,int)or isinstance(batch_size,bool)or not 1<=batch_size<=64:raise ValueError('batch_size must be 1..64')
    plans=list(plans);snapshots=_formula_snapshots(plans)
    jobs=json.loads(legacy_pipeline_json('plan_model_formulas',json.dumps(dict(plans=snapshots),allow_nan=False)))['jobs']
    if jobs and not callable(getattr(recognizer,'recognize_formula_batch_png',None)):raise ValueError('formula recognizer adapter is required')
    inputs=[];cache={};streams={};total=0;counts={};pixels=0
    for job in jobs:
        page,key=job['page'],job['source'];identity=(page,key)
        if identity not in cache:
            source=plans[page]['images'][key]
            if callable(getattr(source,'read',None)):
                if id(source)not in streams:streams[id(source)]=_read_bytes(source,64_000_000)
                cache[identity]=streams[id(source)]
            else:cache[identity]=_read_bytes(source,64_000_000)
        data=cache[identity];total+=len(data)
        if total>128_000_000:raise ValueError('formula input bytes exceed budget')
        if identity not in counts:counts[identity]=_validate_formula_png(data)
        pixels+=counts[identity]
        if pixels>32_000_000:raise ValueError('formula pixels exceed budget')
        inputs.append(data)
    values=recognizer.recognize_formula_batch_png(inputs,batch_size=batch_size)if jobs else[]
    # 回填之前验证全部结果；识别器异常/长度错误均不修改调用方模型。
    result=apply_model_formula_results(snapshots,values)
    for page,p in enumerate(result):
        if 'images'in plans[page]:
            p['images']=dict(plans[page]['images'])
            p['images'].update({key:data for (i,key),data in cache.items()if i==page})
        if 'page_image'in plans[page]:p['page_image']=plans[page]['page_image']
    return result


def organize_model_detection_boxes(boxes,*,masks=()):
    """旧pipeline sorted/merge/update框，输入检测点先按F32处理；不运行模型。"""
    return json.loads(legacy_pipeline_json('model_detection_boxes',json.dumps(dict(boxes=boxes,masks=masks),allow_nan=False)))


def prepare_model_recognition(plan,detections,*,lang=None,page_index=0):
    """整理det结果并从未遮罩crop_source生成旧pipeline识别裁图。"""
    snapshot=json.loads(json.dumps({k:v for k,v in plan.items()if k not in ('images','page_image')},allow_nan=False))
    request=dict(plan=dict(model=snapshot['model'],detect=snapshot['detect']),detections=detections,lang=lang,page_index=page_index)
    text=json.dumps(request,allow_nan=False)
    numeric=json.loads(legacy_pipeline_json('plan_model_recognition',text))
    sources=[];cache={};total=0
    for crop in numeric['crop_attempts']:
        key=crop['source']
        if not isinstance(key,str)or not 1<=len(key.encode('utf-8'))<=256:raise ValueError('invalid crop source ID')
        if key not in cache:cache[key]=_read_bytes(plan['images'][key],64_000_000)
        total+=len(cache[key])
        if total>128_000_000:raise ValueError('recognition crop input bytes exceed budget')
        sources.append(cache[key])
    result,images=prepare_model_recognition_bytes(text,sources)
    snapshot.update(json.loads(result));snapshot['images']=dict(plan['images'])|dict(images)
    if 'page_image'in plan:snapshot['page_image']=plan['page_image']
    return snapshot


def apply_model_ocr_recognition(model,results):
    """旧模型级_rec_postprocess，严格<0.5与窄竖向特定文字<0.8标记LowScoreText。"""
    return json.loads(legacy_pipeline_json('apply_model_recognition',json.dumps(dict(model=model,results=results),allow_nan=False)))


def execute_model_ocr(plan,*,detector=None,recognizer=None,batch_size=6,det_batch_size=1,lang=None,page_index=0):
    """执行已准备任务的旧默认det批量1/rec批量6路径；不调用公式或表格模型。"""
    result=_prepare_model_ocr_recognition(plan,detector=detector,recognizer=recognizer,batch_size=batch_size,det_batch_size=det_batch_size,lang=lang,page_index=page_index)
    targets=result['recognition']
    values=recognizer.recognize_batch_png([result['images'][t['source']]for t in targets],batch_size=batch_size)if targets else[]
    result['model']=apply_model_ocr_recognition(result['model'],[(r['text'],r['confidence'])for r in values])
    return result


def _prepare_model_ocr_recognition(plan,*,detector=None,recognizer=None,batch_size=6,det_batch_size=1,lang=None,page_index=0):
    if det_batch_size!=1 or isinstance(det_batch_size,bool):raise ValueError('currently det_batch_size must be 1')
    if not isinstance(batch_size,int)or isinstance(batch_size,bool)or not 1<=batch_size<=64:raise ValueError('batch_size must be 1..64')
    snapshot=json.loads(json.dumps({k:v for k,v in plan.items()if k not in ('images','page_image')},allow_nan=False))
    jobs=snapshot['detect']
    if jobs and detector is None:raise ValueError('detector is required')
    if any(t['ocr_enable']for t in jobs)and recognizer is None:raise ValueError('recognizer is required')
    if any(d.get('category_id')==15 and 'np_img'in d and 'lang'in d for d in snapshot['model']['layout_dets']):raise ValueError('model already contains recognition crops')
    legacy_pipeline_json('plan_model_recognition',json.dumps(dict(plan=dict(model=snapshot['model'],detect=jobs),detections=[[]for _ in jobs],lang=lang,page_index=page_index),allow_nan=False))
    inputs=[];cache={};total=0;streams={}
    for task in jobs:
        for key in [task['source'],task['crop_source']]:
            if not isinstance(key,str)or not 1<=len(key.encode('utf-8'))<=256:raise ValueError('invalid OCR image ID')
            if key not in cache:
                source=plan['images'][key]
                if callable(getattr(source,'read',None)):
                    identity=id(source)
                    if identity not in streams:streams[identity]=_read_bytes(source,64_000_000)
                    cache[key]=streams[identity]
                else:cache[key]=_read_bytes(source,64_000_000)
        key=task['source']
        total+=len(cache[key])
        if total>128_000_000:raise ValueError('detector bytes exceed budget')
        inputs.append(cache[key])
    detections=[[d['points']for d in detector.detect_model_png(data)]for data in inputs]
    working=dict(snapshot,images=dict(plan['images'])|cache)
    if 'page_image'in plan:working['page_image']=plan['page_image']
    result=prepare_model_recognition(working,detections,lang=lang,page_index=page_index)
    targets=result['recognition']
    if targets and recognizer is None:raise ValueError('tilted detections require recognizer')
    return result


def derive_native_ocr_boxes(native_page,region,*,useful_list,scale=1.0,masks=()):
    """旧txt_spans_bbox_extract；masks为已变换到裁图坐标的公式/复选框bbox。"""
    request=dict(native_page=native_page,region=region,useful_list=useful_list,scale=scale,masks=masks)
    return json.loads(legacy_pipeline_json('native_ocr_boxes',json.dumps(request,allow_nan=False)))


def prepare_model_ocr(plan,*,native_page=None,scale=None,ocr_enable=False,use_det_mode='auto',page_index=0):
    """追加旧原生OCR框并准备遮罩/64像素补白检测输入；尚不执行检测或识别模型。"""
    if not isinstance(plan,dict)or not isinstance(plan.get('images'),dict):raise TypeError('plan must contain an images mapping')
    if native_page is None:native_page=plan.get('native_page')
    if native_page is None:
        if not ocr_enable and use_det_mode!='ocr':raise ValueError('native_page is required for native OCR planning')
        native_page=dict(blocks=[])
    if scale is None:scale=plan.get('scale',1.0)
    snapshot={k:v for k,v in plan.items()if k not in ('images','page_image')}
    snapshot=json.loads(json.dumps(snapshot,allow_nan=False))
    request=dict(plan=dict(model=snapshot['model'],ocr=snapshot['ocr']),native_page=native_page,scale=scale,ocr_enable=ocr_enable,use_det_mode=use_det_mode)
    native=json.loads(legacy_pipeline_json('plan_native_ocr',json.dumps(request,allow_nan=False)))
    sources=[];cache={};total=0
    for task in native['detect']:
        key=task['source']
        if not isinstance(key,str)or not 1<=len(key.encode('utf-8'))<=256:raise ValueError('invalid crop image ID')
        if key not in cache:cache[key]=_read_bytes(plan['images'][key],64_000_000)
        total+=len(cache[key])
        if total>128_000_000:raise ValueError('detection input bytes exceed budget')
        sources.append(cache[key])
    result,images=prepare_model_detection_bytes(json.dumps(dict(plan=native,page_index=page_index),allow_nan=False),sources)
    snapshot.update(json.loads(result));snapshot['images']=dict(plan['images'])|dict(images)
    if 'page_image'in plan:snapshot['page_image']=plan['page_image']
    return snapshot


def convert_model_layout(regions,*,layout_shape_mode,markdown_ignore_labels=None,formula_enabled=True,formula_level=0,use_custom_ocr=False,filter_overlaps=True):
    """旧PP-DocLayoutV2/V3类别与BatchAnalyze过滤；仅显式rect模式，返回layout_dets。"""
    request=dict(regions=regions,layout_shape_mode=layout_shape_mode,markdown_ignore_labels=markdown_ignore_labels,formula_enabled=formula_enabled,
        formula_level=formula_level,use_custom_ocr=use_custom_ocr,filter_overlaps=filter_overlaps)
    return json.loads(legacy_pipeline_json('convert_layout',json.dumps(request,allow_nan=False)))


def filter_model_layout(layout_dets,*,use_custom_ocr=False):
    """旧filter_overlap_boxes，保留原顺序与未知字段；多边形须等待对应路径。"""
    return json.loads(legacy_pipeline_json('filter_layout',json.dumps(dict(layout_dets=layout_dets,use_custom_ocr=use_custom_ocr),allow_nan=False)))


def predict_model_layout(source,*,layout_model,layout_shape_mode,threshold=0.5,**options):
    """执行实际布局模型后转换旧layout_dets；尚不执行OCR/公式/表格识别。"""
    convert_model_layout([],layout_shape_mode=layout_shape_mode,**options)
    if not isinstance(threshold,(int,float))or isinstance(threshold,bool)or not math.isfinite(threshold)or not 0<=threshold<=1:raise ValueError('invalid layout threshold')
    data=_read_bytes(source,64_000_000)
    return convert_model_layout(layout_model.layout_png(data,threshold=threshold),layout_shape_mode=layout_shape_mode,**options)


def prepare_model_crops(source,model,*,page_index=0,formula_bbox_expand_px=2):
    """按旧规则收集模型区域和RGB裁图，返回model/ocr/tables/formulas及独立images映射。"""
    request=json.dumps(dict(model=model,page_index=page_index,formula_bbox_expand_px=formula_bbox_expand_px),allow_nan=False)
    plan,images=prepare_model_crops_bytes(request,_read_bytes(source,64_000_000),mask_polygons=_HOST_PAGE.get())
    result=json.loads(plan);result['images']=dict(images)
    return result


def prepare_pdf_model_inference(source,*,layout_model,layout_shape_mode,page_index=0,dpi=144.0,max_pixels=16_000_000,pdfium_library=None,password=None,
                                threshold=0.5,formula_bbox_expand_px=2,**options):
    """PDF→RGB→真实布局推理→旧区域任务/裁图；其余模型执行和回填尚待接入。"""
    convert_model_layout([],layout_shape_mode=layout_shape_mode,**options)
    library,data=_prepare_pdf(source,pdfium_library,None)
    facts=read_native_pdf_page(data,page_index=page_index,pdfium_library=library,password=password)
    page=render_pdf_page(data,page_index=page_index,dpi=dpi,max_pixels=max_pixels,pdfium_library=library,password=password,color_mode='RGB')
    model=predict_model_layout(page.png,layout_model=layout_model,layout_shape_mode=layout_shape_mode,threshold=threshold,**options)
    plan=prepare_model_crops(page.png,model,page_index=page_index,formula_bbox_expand_px=formula_bbox_expand_px)
    plan.update(native_page=facts,page_image=page.png,scale=dpi/72.)
    return plan


@_host_call
def parse_pdf_models(source,*,layout_model=None,detector=None,recognizer=None,table_model=None,
                     formula_model=None,orientation=None,table_variant='slanet-plus',
                     classifier=None,secondary_classifier=None,wired_model=None,use_compare_table=False,cls_conf_threshold=.9,
                     native_table_recoverer=None,use_rule_table=True,rule_table_score_threshold=.9,
                     native_text=True,formula_enabled=True,start_page=0,end_page=None,
                     dpi=144.0,pdfium_library=None,password=None,threshold=0.5,
                     max_pages=128,max_pixels=16_000_000,max_total_pixels=128_000_000,
                     merge_tables=True,docvortex_mode='off',native_only=False,
                     supplement_layout_model=None,markdown_ignore_labels=None,table_enabled=True,lang=None,use_det_mode='auto',progress=None,
                     extract_original_image=False,extract_original_image_iou_thresh=.9,table_force_ocr=False,scan_native_ocr_region=False,compat_render=False):
    """PDF模型预览：V3矩形版面→旧OCR/公式/UNET或SLANet→多页Markdown、JSON和图片bytes。

    off 使用布局模型；rule/rule_plus_s 接入完整原生全文，native_only 可不传布局模型。
    按实际回退、补充与OCR/表格/公式任务延迟要求模型，缺失时报错。
    table_variant可选unet、slanet-plus/1m或unet-slanet-plus/1m；双模型用classifier/secondary_classifier及wired_model。
    use_rule_table=True 内置完整原生区域恢复，显式native_table_recoverer可覆盖。
    原生模式保留全文阅读顺序；完整原生输出进入现有OCR/Markdown/资产链。
    不含seal、checkbox或Office；默认无空格补识别器。
    native_text=True使用原生文字优先路径，False强制OCR；不做自动文档分类。
    返回pages/content_list/markdown/assets，其中assets键与Markdown中的images/路径对应。
    无外部写出；失败不返回部分文档，不关闭调用方流。
    """
    for value in (native_text,formula_enabled,merge_tables,use_compare_table,use_rule_table,native_only,table_enabled,extract_original_image,table_force_ocr,scan_native_ocr_region,compat_render):
        if type(value)is not bool:raise ValueError('document flags must be bool')
    for name,value,bound in [('max_pages',max_pages,256),('max_pixels',max_pixels,16_000_000),('max_total_pixels',max_total_pixels,128_000_000)]:
        if type(value)is not int or not 1<=value<=bound:raise ValueError(f'{name} must be 1..{bound}')
    if table_variant not in ('slanet-plus','slanet-1m','unet','unet-slanet-plus','unet-slanet-1m'):raise ValueError('invalid legacy table variant')
    if isinstance(cls_conf_threshold,bool)or not isinstance(cls_conf_threshold,(int,float))or not math.isfinite(cls_conf_threshold)or not 0<=cls_conf_threshold<=1:raise ValueError('invalid classifier confidence threshold')
    if isinstance(rule_table_score_threshold,bool)or not isinstance(rule_table_score_threshold,(int,float))or not math.isfinite(rule_table_score_threshold)or not 0<=rule_table_score_threshold<=1:raise ValueError('invalid native table score threshold')
    if native_table_recoverer is not None and not callable(getattr(native_table_recoverer,'recover_table',None)):raise TypeError('native recoverer must implement recover_table')
    if docvortex_mode not in ('off','rule','rule_plus_s'):raise ValueError('invalid docvortex_mode')
    if native_only and docvortex_mode=='off':raise ValueError('native_only requires rule or rule_plus_s')
    if docvortex_mode=='off'and not callable(getattr(layout_model,'layout_png',None)):raise ValueError('layout model is required')
    if markdown_ignore_labels is not None and (not isinstance(markdown_ignore_labels,(list,tuple,set,frozenset))or any(not isinstance(s,str)for s in markdown_ignore_labels)):raise ValueError('invalid markdown_ignore_labels')
    ignored=None if markdown_ignore_labels is None else list(markdown_ignore_labels)
    if ignored is None and docvortex_mode!='off':ignored=['number','header','header_image','footer','footer_image','aside_text']
    if isinstance(dpi,bool)or not isinstance(dpi,(int,float))or not math.isfinite(dpi)or not 1<=dpi<=600:raise ValueError('dpi must be 1..600')
    if isinstance(threshold,bool)or not isinstance(threshold,(int,float))or not math.isfinite(threshold)or not 0<=threshold<=1:raise ValueError('invalid layout threshold')
    if progress is not None and not callable(progress):raise TypeError('progress must be callable')
    if use_det_mode not in ('auto','txt','ocr'):raise ValueError('invalid use_det_mode')
    def report(stage,page_index=None,detail=None):
        if progress is not None:progress(dict(stage=stage,page_index=page_index,detail=detail))
    library,data=_prepare_pdf(source,pdfium_library,None)
    info=parse_pdf_text(data,pdfium_library=library,password=password,start_page=0,end_page=0,allow_partial=True)
    count=info.document['source_page_count'];first,last=resolve_page_range(start_page,end_page,count)
    if last-first>max_pages:raise ValueError('selected pages exceed budget')
    native_pages=None;layout_error=None;layout_routes=[]
    if docvortex_mode!='off' and first<last:
        report('原生全文分析')
        try:native_pages=analyze_native_pdf(data,pdfium_library=library,password=password,max_pixels=max_pixels)['pages']
        except ImportError:raise
        except Exception as e:
            if native_only:raise
            layout_error=type(e).__name__+': '+str(e)
            report('原生分析失败，回退版面模型',detail=layout_error)
    if native_table_recoverer is None and use_rule_table and native_text:
        native_table_recoverer=NativePdfTableRecoverer()
    class Assets:
        _parent_dir='images'
        def __init__(self):self.data={};self.size=0
        def write(self,name,content):
            if not isinstance(name,str)or not name or Path(name).name!=name or '/'in name or '\\'in name:raise ValueError('invalid asset name')
            key='images/'+name;content=bytes(content)
            if key in self.data:
                if self.data[key]!=content:raise ValueError('conflicting document asset name')
                return
            self.size+=len(content)
            if self.size>128_000_000 or len(self.data)>=4096:raise ValueError('document assets exceed budget')
            self.data[key]=content
    assets=Assets();pages=[];pixels=0;scale=dpi/72.;post_images={};post_bytes=0;native_routes=[]
    prepared=[];recognition_images=[];prepared_bytes=0
    for index in range(first,last):
        report('读取文字与渲染',index)
        facts=read_native_pdf_page(data,page_index=index,pdfium_library=library,password=password)
        if compat_render:
            from rapid_doc_rust._native import render_pdf_model_page_bytes
            width,height,png,scale=render_pdf_model_page_bytes(library,data,index,dpi,max_pixels,password)
            image=RenderedPage(width,height,png)
        else:
            image=render_pdf_page(data,page_index=index,dpi=dpi,max_pixels=max_pixels,pdfium_library=library,password=password,color_mode='RGB')
            scale=dpi/72.
        pixels+=image.width*image.height
        if pixels>max_total_pixels:raise ValueError('document render pixels exceed budget')
        if docvortex_mode=='off' or (native_pages is None and not native_only):
            report('版面模型识别',index)
            if not callable(getattr(layout_model,'layout_png',None)):raise ValueError('native layout fallback requires a layout model: '+str(layout_error))
            model=predict_model_layout(image.png,layout_model=layout_model,layout_shape_mode='rect',threshold=threshold,formula_enabled=formula_enabled,markdown_ignore_labels=ignored)
            if docvortex_mode!='off':layout_routes.append(dict(page_index=index,method='model',reason='native_analysis_error',error=layout_error))
        else:
            from native_pdf_preview import choose_native_layout
            blocks=None if native_pages is None else native_pages[index]
            if scan_native_ocr_region and not native_text and not facts['blocks']:blocks=[dict(type='text',bbox=[0,0,1,1],content='')]
            model,route=choose_native_layout(blocks,image.width,image.height,layout_model=layout_model,supplement_layout_model=supplement_layout_model,source=image.png,mode=docvortex_mode,native_only=native_only,ignored_labels=ignored,native_char_count=len(facts.get('chars')or[]),native_text=native_text,formula_enabled=formula_enabled,threshold=threshold)
            layout_routes.append(dict(page_index=index,**route))
        plan=prepare_model_crops(image.png,model,page_index=index)
        if not table_enabled:plan['tables']=[]
        report('公式识别',index)
        if formula_enabled:plan=execute_model_formulas([plan],recognizer=formula_model,batch_size=1)[0]
        report('正文 OCR 检测与原生文字规划',index)
        plan=prepare_model_ocr(plan,native_page=facts,scale=scale,ocr_enable=not native_text,
                               use_det_mode=use_det_mode if native_text else'ocr',page_index=index)
        plan=_prepare_model_ocr_recognition(plan,detector=detector,recognizer=recognizer,page_index=index,lang=lang)
        recognition_images.extend(plan['images'][t['source']]for t in plan['recognition'])
        prepared_bytes+=sum(len(b)for b in plan['images'].values())+len(image.png)
        if prepared_bytes>128_000_000:raise ValueError('document OCR preparation bytes exceed budget')
        prepared.append((index,facts,image,plan,scale))
    # The original host recognizes body crops together across pages. Per-page
    # batches change padding widths and can change CTC text even on CPU.
    report('正文 OCR 批量识别',detail=f'{len(recognition_images)} 个文字裁图')
    recognized=recognizer.recognize_batch_png(recognition_images,batch_size=6)if recognition_images else[]
    offset=0
    for index,facts,image,plan,scale in prepared:
        n=len(plan['recognition']);values=recognized[offset:offset+n];offset+=n
        plan['model']=apply_model_ocr_recognition(plan['model'],[(r['text'],r['confidence'])for r in values])
        model=plan['model'];dets=model['layout_dets']
        formulas=[d for d in dets if d['category_id']in (8,13,14)]
        checkboxes=[d for d in dets if d['category_id']==200]
        filled=[];page_context=dict(facts,ori_image_list=[])
        guard_images=[]
        if native_table_recoverer is not None and use_rule_table and native_text and facts['blocks']and plan['tables']:
            guard_images=extract_original_pdf_images(data,page_index=index,pdfium_library=library,password=password,render=False)['images']
        for task in plan['tables']:
            report('表格识别',index)
            if native_table_recoverer is not None:
                native=try_native_table_rule(data,dets[task['layout_index']],recoverer=native_table_recoverer,page_index=index,
                    pdfium_library=library,password=password,score_threshold=rule_table_score_threshold,scale=scale,
                    config=dict(use_rule_table=use_rule_table),rotate_label=facts.get('rotate_label'),ocr_enable=not native_text,
                    has_source_pdf=native_text and bool(facts['blocks']),formulas=formulas,checkboxes=checkboxes,images=[dict(bbox=i['bbox'])for i in guard_images])
                native_routes.append(dict(page_index=index,layout_index=task['layout_index'],**native['diagnostic']))
                if native['accepted']:
                    dets[task['layout_index']]=native['table']
                    continue
            hints=prepare_table_hints(page_context,dets[task['layout_index']],useful_list=task['useful_list'],
                formulas=formulas,checkboxes=checkboxes,scale=scale,formula_enabled=formula_enabled)
            page_context=hints['page']
            text=recognize_table_text_stage_png(plan['images'][task['source']],native_page=facts,table=hints['table'],
                useful_list=task['useful_list'],detector=detector,recognizer=recognizer,orientation=orientation,
                masks=hints['formulas'],scale=scale,force_ocr=not native_text or table_force_ocr,ocr_enable=not native_text)
            table_options=dict(structure_model=table_model,detector=detector,recognizer=recognizer,fill_images=hints['fill_images'],formulas=hints['formulas'])
            if table_variant=='unet':result=recognize_legacy_unet_table_png(text['png'],text['ocr_result'],**table_options)
            elif table_variant.startswith('unet-slanet-'):
                table_options.pop('structure_model')
                result=recognize_legacy_table_auto_png(text['png'],text['ocr_result'],classifier=classifier,secondary_classifier=secondary_classifier,
                    wired_model=wired_model,wireless_model=table_model,wireless_variant=table_variant[5:],
                    use_compare_table=use_compare_table,cls_conf_threshold=cls_conf_threshold,**table_options)
            else:result=recognize_legacy_slanet_table_png(text['png'],text['ocr_result'],variant=table_variant,**table_options)
            dets[task['layout_index']]=apply_table_prediction(hints['table'],result['html'],formulas=formulas+checkboxes,
                fill_images=hints['fill_images'],scale=scale)
            filled.extend(dict(uuid=f['uuid'],bbox=f['bbox'],source=plan['images'][f['source']])for f in hints['fill_images'])
        report('页面组装与图片资产',index)
        originals=extract_original_pdf_images(data,page_index=index,pdfium_library=library,password=password)['images']if extract_original_image else[]
        assembled=_assemble_model_page_assets(image.png,model,image_writer=assets,page_width=facts['size'][0],page_height=facts['size'][1],
            page_index=index,scale=scale,formula_enabled=formula_enabled,table_fill_images=filled,native_page=facts if native_text else None,
            original_images=originals,extract_original_image=extract_original_image,extract_original_image_iou_thresh=extract_original_image_iou_thresh)
        page=assembled['page']
        if page is None:page=dict(preproc_blocks=[],page_idx=index,page_size=[int(v)for v in facts['size']],discarded_blocks=[])
        for key,content in assembled['ocr_images'].items():
            if key in post_images:raise ValueError('duplicate document OCR image ID')
            post_bytes+=len(content)
            if post_bytes>128_000_000:raise ValueError('document post OCR bytes exceed budget')
            post_images[key]=content
        pages.append(page)
    report('补识别、段落和跨页表格')
    pages=post_process_model_ocr(pages,recognizer=recognizer,images=post_images)
    pages=split_paragraphs(optimize_formula_numbers(pages))
    if merge_tables:pages=merge_cross_page_tables(pages)
    result=dict(source_page_count=count,page_range=[first,last],pages=pages,
                markdown=render_model_pages(pages,image_path_prefix='images'),
                content_list=render_model_pages(pages,make_mode='content_list',image_path_prefix='images'),assets=assets.data,
                native_table_routes=native_routes)
    if docvortex_mode!='off':result['native_layout_routes']=layout_routes
    return result


def extract_original_pdf_images(source, *, page_index=0, pdfium_library=None, password=None,
                                max_depth=15,max_pixels=32_000_000,render=True,scale_to_original=True):
    """返回 {images,diagnostics}；images可直接传original_images或table_fill_images。"""
    if not isinstance(render,bool)or not isinstance(scale_to_original,bool):raise TypeError('render and scale_to_original must be bool')
    library,data=_prepare_pdf(source,pdfium_library,None)
    metadata,images=extract_original_pdf_images_bytes(library,data,page_index,password,max_depth,max_pixels,render,scale_to_original)
    result=json.loads(metadata);images=dict(images)
    for image in result['images']:image['source']=images[image['uuid']]
    return result


def assemble_pdf_model_page(source,model,*,image_writer,page_index=0,dpi=144.0,max_pixels=16_000_000,
                            pdfium_library=None,password=None,native_text=True,extract_original_image=False,
                            extract_original_image_iou_thresh=0.9,formula_enabled=True):
    """PDF与已有版面/识别结果组装；缺失OCR仍通过ocr_images返回供后置识别。"""
    if not isinstance(native_text,bool)or not isinstance(extract_original_image,bool):raise TypeError('native_text and extract_original_image must be bool')
    library,data=_prepare_pdf(source,pdfium_library,None)
    facts=read_native_pdf_page(data,page_index=page_index,pdfium_library=library,password=password)
    page=render_pdf_page(data,page_index=page_index,dpi=dpi,max_pixels=max_pixels,pdfium_library=library,password=password,color_mode='RGB')
    originals=extract_original_pdf_images(data,page_index=page_index,pdfium_library=library,password=password)if extract_original_image else dict(images=[],diagnostics=[])
    result=_assemble_model_page_assets(page.png,model,image_writer=image_writer,page_width=facts['size'][0],page_height=facts['size'][1],page_index=page_index,
        scale=dpi/72.,formula_enabled=formula_enabled,native_page=facts if native_text else None,original_images=originals['images'],
        extract_original_image=extract_original_image,extract_original_image_iou_thresh=extract_original_image_iou_thresh)
    result['diagnostics']=originals['diagnostics']
    return result


def deduplicate_native_characters(chars):
    """按旧DocVortex规则处理已恢复代理对的普通字符JSON；返回独立副本。"""
    return json.loads(deduplicate_native_characters_json(json.dumps(chars,allow_nan=False)))


def classify_pdf_font_name(face, *, charset=1):
    """按旧五编码、明确字体别名和字符集判断固定 CJK 替代，返回字符集或 None。"""
    if not isinstance(face,(bytes,bytearray,memoryview)):raise TypeError('face must be bytes')
    if (face.nbytes if isinstance(face,memoryview)else len(face))>4096:raise ValueError('font face exceeds 4096 bytes')
    if isinstance(charset,bool)or not isinstance(charset,int):raise TypeError('charset must be int')
    return classify_pdf_font_name_bytes(bytes(face),charset)


def pdfium_font_runtime_info(*, pdfium_library=None):
    """初始化同库固定字体并返回身份/真实替代计数；不暴露句柄。"""
    library,_=_prepare_pdf(b'',pdfium_library,1)
    return json.loads(pdfium_font_runtime_info_json(library))


def read_pdf_native_text_geometry(source, *, page_index=0, pdfium_library=None, password=None, include_geometry=True):
    """读取PDF、恢复代理对并完成旧字符去重；返回字符及几何索引，未过滤页面可见区。"""
    if isinstance(page_index,bool) or not isinstance(page_index,int) or page_index<0:
        raise ValueError('page_index must be a nonnegative integer')
    if not isinstance(include_geometry,bool):raise ValueError('include_geometry must be bool')
    library,data=_prepare_pdf(source,pdfium_library,None)
    return json.loads(read_pdf_native_text_geometry_json(library,data,page_index,password,include_geometry))


def read_pdf_character_primitives(source, *, page_index=0, pdfium_library=None, password=None, include_geometry=True):
    """读取旧DocVortex去重前字符及代理对恢复结果；不做可见区过滤或重复文字删除。"""
    if isinstance(page_index,bool) or not isinstance(page_index,int) or page_index<0:
        raise ValueError('page_index must be a nonnegative integer')
    if not isinstance(include_geometry,bool):raise ValueError('include_geometry must be bool')
    library,data=_prepare_pdf(source,pdfium_library,None)
    return json.loads(read_pdf_character_primitives_json(library,data,page_index,password,include_geometry))


def read_pdf_text_paint(source, indices, *, page_index=0, pdfium_library=None, password=None):
    """读取仅填色文字RGBA；描边、裁剪和无效索引不返回证据。"""
    if isinstance(page_index,bool) or not isinstance(page_index,int) or page_index<0:
        raise ValueError('page_index must be a nonnegative integer')
    indices=list(indices)
    if any(isinstance(i,bool)or not isinstance(i,int)for i in indices):raise ValueError('indices must be integers')
    library,data=_prepare_pdf(source,pdfium_library,None)
    return json.loads(read_pdf_text_paint_json(library,data,indices,page_index,password))


def read_pdf_image_infos(source, *, page_index=0, pdfium_library=None, password=None):
    """自动读取旧PDF图片区域、原始流指纹、背景/白边证据及页面尺寸/旋转。"""
    if isinstance(page_index,bool) or not isinstance(page_index,int) or page_index<0:
        raise ValueError('page_index must be a nonnegative integer')
    library,data=_prepare_pdf(source,pdfium_library,None)
    return json.loads(read_pdf_image_infos_json(library,data,page_index,password))


def read_pdf_link_annotations(source, *, page_index=0, pdfium_library=None, password=None):
    """按旧URI/可见性/QuadPoints规则读取外链注解，返回独立target/bboxes/source_index。"""
    if isinstance(page_index,bool) or not isinstance(page_index,int) or page_index<0:
        raise ValueError('page_index must be a nonnegative integer')
    library,data=_prepare_pdf(source,pdfium_library,None)
    return json.loads(read_pdf_link_annotations_json(library,data,page_index,password))


def read_pdf_form_geometry(source, *, page_index=0, pdfium_library=None, password=None):
    """实际Form调用树/声明、字符路径归属和裁剪；返回稳定编号，不包含PDFium地址。"""
    from rapid_doc_rust._native import read_pdf_form_geometry_json
    if isinstance(page_index,bool) or not isinstance(page_index,int) or page_index<0:
        raise ValueError('page_index must be a nonnegative integer')
    library,data=_prepare_pdf(source,pdfium_library,None)
    return json.loads(read_pdf_form_geometry_json(library,data,page_index,password))


def prepare_native_document_pages(pages, *, classify_text=False, classify_footnotes=False, complete_page=False):
    """原始 source/geometry 全文→分类/容器认领→几何重拆/标签回收/行首分式。

    页面可附 style_lines/link_lines；返回 pages/plan/header_separator_bboxes。
    classify_text=True 继续图题/术语/附录、引用、展示标题及异常中文段组。
    classify_footnotes=True 包含上述阶段，并继续图像/页面脚注和表后字段保护。
    complete_page=True 包含上述阶段并完成整页合并、上/下标、字符压缩及页辅助分类。
    全文语义、后续公式和文字编排仍待，尚非默认 PDF 完整入口。
    """
    if not isinstance(classify_text,bool):raise ValueError('classify_text must be bool')
    if not isinstance(classify_footnotes,bool):raise ValueError('classify_footnotes must be bool')
    if not isinstance(complete_page,bool):raise ValueError('complete_page must be bool')
    return json.loads(legacy_pipeline_json('prepare_native_document_pages',json.dumps(dict(pages=pages,include_post_text_roles=classify_text,include_post_footnotes=classify_footnotes,prepare_page=complete_page),allow_nan=False)))


def prepare_native_document_auxiliary(pages):
    """原页源/几何→完整逐页准备→跨页末表传递→重复矢量装饰→全文辅助文本与正文画像。

    输入与 prepare_native_document_pages 一致，额外返回 body_profile/canonical_body_profile。
    尚未包含全文标题原型、结构标题及最终公式/文字编排，不是默认 PDF 完整入口。
    """
    return json.loads(legacy_pipeline_json('prepare_native_document_auxiliary',json.dumps(dict(pages=pages),allow_nan=False)))


def classify_native_document_auxiliary(pages):
    """已准备页面的全文辅助文本与正文画像；保留输入中的额外字段。"""
    return json.loads(legacy_pipeline_json('classify_native_document_auxiliary',json.dumps(dict(pages=pages),allow_nan=False)))


def prepare_native_document_text(pages):
    """原始页源/几何→完整逐页准备→全文辅助分类/正文画像→标题原型/结构/重复标题。

    完整执行旧 _classify_document_text；返回额外的 title_profile。
    文献、最终公式恢复/文字编排仍待后续，不是默认 PDF 转换入口。
    """
    return json.loads(legacy_pipeline_json('prepare_native_document_text',json.dumps(dict(pages=pages),allow_nan=False)))


def prepare_native_document_references(pages):
    """原始页源/几何→全文文字分类→文献区域/编号条目分组→小字号强调标题。

    保留跨页文献状态、附录/译文屏障及脚注来源组回收。
    尚未包含最终页面的公式、文字及行内编排，不是默认 PDF 转换入口。
    """
    return json.loads(legacy_pipeline_json('prepare_native_document_references',json.dumps(dict(pages=pages),allow_nan=False)))


def classify_native_document_references(pages):
    """已准备页面的完整文献区域、编号分组和小字号强调标题处理。"""
    return json.loads(legacy_pipeline_json('classify_native_document_references',json.dumps(dict(pages=pages),allow_nan=False)))


def group_native_reference_lines(page):
    """对已确认文献区域执行原悬挂条目分组；亦供最终页面编排复用。"""
    return json.loads(legacy_pipeline_json('group_native_reference_lines',json.dumps(dict(page=page),allow_nan=False)))


def recover_native_formula_blocks(lines, page_size, *, table_bboxes=None, drawing_lines=None):
    """完整旧公式认领和替换仲裁；消费原生行，返回公式、未认领行及行状态副本。

    保留编号、分式、正文宿主回退和紧裁图；不是整页 PDF 输出接口。
    """
    request=dict(lines=lines,page_size=page_size,table_bboxes=[] if table_bboxes is None else table_bboxes,
                 drawing_lines=[] if drawing_lines is None else drawing_lines)
    return json.loads(legacy_pipeline_json('recover_native_formula_blocks',json.dumps(request,allow_nan=False)))


def prepare_native_formula_input(lines, page_size, *, table_bboxes=None, drawing_lines=None):
    """原密集视觉行恢复→同基线合并→完整公式恢复，返回公式、未认领行及最终输入行。

    预分类语义分流/目录、canonical 重放及最终页面装配由后续调用方衔接。
    """
    request=dict(lines=lines,page_size=page_size,table_bboxes=[] if table_bboxes is None else table_bboxes,
                 drawing_lines=[] if drawing_lines is None else drawing_lines)
    return json.loads(legacy_pipeline_json('prepare_native_formula_input',json.dumps(request,allow_nan=False)))


def prepare_native_page_formulas(page):
    """已准备页→最终公式检查点：语义/目录分流、公式恢复、canonical 仲裁、墨迹与样式恢复。

    返回 formula_blocks/index_blocks/semantic_lines/remaining_lines 及已清空的 canonical 快照。
    最终文字、行内和块排序尚未执行；不修改输入页面。
    """
    return json.loads(legacy_pipeline_json('prepare_native_page_formulas',json.dumps(dict(page=page),allow_nan=False)))


def prepare_native_document_formulas(pages):
    """原始页源/几何→全文文献准备→最终页面公式检查点；返回额外 formula_pages。

    pages 保留文献阶段状态，formula_pages 按页保存后续检查点；不是默认 PDF 完整输出。
    """
    return json.loads(legacy_pipeline_json('prepare_native_document_formulas',json.dumps(dict(pages=pages),allow_nan=False)))


def prepare_native_final_page_lines(page, *, page_index=0, body_profile=None,
                                    canonical_body_profile=None, title_profile=None):
    """完整现有 Python 最终页语义和行准备：公式→标题收尾→同行聚合→文献/首页分组。

    止于 _build_text_blocks 前，返回待构建文字块的 remaining_lines；不修改输入。
    后续段落块聚合、块排序和默认整篇 PDF 输出尚未完成。
    """
    if isinstance(page_index,bool) or not isinstance(page_index,int) or page_index<0:
        raise ValueError('page_index must be a nonnegative integer')
    request=dict(page=page,page_index=page_index,document_body_profile=body_profile,
                 canonical_body_profile=canonical_body_profile,document_title_profile=title_profile)
    return json.loads(legacy_pipeline_json('prepare_native_final_page_lines',json.dumps(request,allow_nan=False)))


def prepare_native_final_document_lines(pages):
    """原始页源/几何→全文文献→最终标题及行准备；额外返回 text_pages 检查点。

    pages 保存文献阶段状态，text_pages 保存每页最终行状态；不是整篇 PDF 转换完成接口。
    """
    return json.loads(legacy_pipeline_json('prepare_native_final_document_lines',json.dumps(dict(pages=pages),allow_nan=False)))


def build_native_text_blocks(lines, page_size, *, table_bboxes=None, drawing_lines=None,
                             page_footnote_groups=None, page_index=None, visual_bboxes=None):
    """完整现有 Python 文字块组装：段界、悬挂缩进、脚注及全部块级补合并。"""
    if page_index is not None and (isinstance(page_index, bool) or not isinstance(page_index, int) or page_index < 0):
        raise ValueError('page_index must be a nonnegative integer or None')
    request=dict(lines=lines,page_size=page_size,table_bboxes=[] if table_bboxes is None else table_bboxes,
                 drawing_lines=[] if drawing_lines is None else drawing_lines,
                 page_footnote_groups=[] if page_footnote_groups is None else page_footnote_groups,
                 page_index=page_index,visual_bboxes=[] if visual_bboxes is None else visual_bboxes)
    return json.loads(legacy_pipeline_json('build_native_text_blocks',json.dumps(request,allow_nan=False)))


def prepare_native_final_page_blocks(page, *, page_index=0, body_profile=None,
                                     canonical_body_profile=None, title_profile=None, refine_blocks=False,
                                     finalize_output=False):
    """连续最终页语句至完整文字块；refine_blocks=True 继续成员/标题/图注/局部列表。

    finalize_output=True 继续完整视觉注释、阅读排序和归一化，额外返回 model_list。
    """
    if isinstance(page_index,bool) or not isinstance(page_index,int) or page_index<0:
        raise ValueError('page_index must be a nonnegative integer')
    if not isinstance(refine_blocks,bool):raise ValueError('refine_blocks must be a bool')
    if not isinstance(finalize_output,bool):raise ValueError('finalize_output must be a bool')
    request=dict(page=page,page_index=page_index,document_body_profile=body_profile,
                 canonical_body_profile=canonical_body_profile,document_title_profile=title_profile,refine_blocks=refine_blocks,finalize_output=finalize_output)
    return json.loads(legacy_pipeline_json('prepare_native_final_page_blocks',json.dumps(request,allow_nan=False)))


def prepare_native_final_document_blocks(pages, *, refine_blocks=False, finalize_output=False,
                                         materialize_inline=False):
    """原始页源/几何→文献→最终行→完整文字块；返回 block_pages 检查点。"""
    if not isinstance(refine_blocks,bool):raise ValueError('refine_blocks must be a bool')
    if not isinstance(finalize_output,bool):raise ValueError('finalize_output must be a bool')
    if not isinstance(materialize_inline,bool):raise ValueError('materialize_inline must be a bool')
    return json.loads(legacy_pipeline_json('prepare_native_final_document_blocks',json.dumps(dict(pages=pages,refine_blocks=refine_blocks,finalize_output=finalize_output,materialize_inline=materialize_inline),allow_nan=False)))


def prepare_native_spacing_lines(lines, *, space_before=None):
    """原始视觉行→缺失词界/生成中文空格/连字空格证据，保持源字符只读。"""
    return json.loads(legacy_pipeline_json('prepare_native_spacing_lines',json.dumps(dict(lines=lines,space_before=space_before),allow_nan=False)))


def apply_native_spacing_lines(blocks, spacing_lines, page_size):
    """在最终自然语言块投影补/删空格，先于链接、样式及上下标物化。"""
    return json.loads(legacy_pipeline_json('apply_native_spacing_lines',json.dumps(dict(blocks=blocks,spacing_lines=spacing_lines,page_size=page_size),allow_nan=False)))


def materialize_native_document_inline(finalized_pages, prepared_pages, *, page_sizes,
                                       page_style_lines, page_link_lines, page_spacing_lines=(),
                                       script_diagnostics=None):
    """完整原文档行内收尾；严格逐页对齐并保留可选脚本诊断快照。"""
    q=dict(finalized_pages=finalized_pages,prepared_pages=prepared_pages,page_sizes=page_sizes,
           page_style_lines=page_style_lines,page_link_lines=page_link_lines,
           page_spacing_lines=page_spacing_lines,script_diagnostics=script_diagnostics)
    return json.loads(legacy_pipeline_json('materialize_native_document_inline',json.dumps(q,allow_nan=False)))


def finalize_native_document(pages, source, *, pdfium_library=None, password=None,
                             max_pixels=16_000_000):
    """已有原始页源/几何→全文最终块、空格/行内语义→表格媒体；返回逐页块。"""
    if isinstance(max_pixels,bool)or not isinstance(max_pixels,int)or max_pixels<=0:
        raise ValueError('max_pixels must be a positive integer')
    result=prepare_native_final_document_blocks(pages,materialize_inline=True)
    return materialize_native_table_media(result['finalized_pages'],source,
                                         pdfium_library=pdfium_library,password=password,
                                         max_pixels=max_pixels)


def collect_native_pdf_sources(source, *, pdfium_library=None, password=None, max_pixels=16_000_000):
    """PDF 文件→原生可见文字/绘图/Form/签名/图片/样式/链接及跨页图像与噪声裁决。"""
    if type(max_pixels)is not int or max_pixels<=0:raise ValueError('max_pixels must be a positive integer')
    library,data=_prepare_pdf(source,pdfium_library,None)
    return json.loads(native_pdf_document_json(library,data,password,True,max_pixels))


def analyze_native_pdf(source, *, pdfium_library=None, password=None, max_pixels=16_000_000):
    """完整原 PdfModel 文件入口，返回 pages、source_page_count、page_sizes；不执行 OCR。"""
    if type(max_pixels)is not int or max_pixels<=0:raise ValueError('max_pixels must be a positive integer')
    library,data=_prepare_pdf(source,pdfium_library,None)
    return json.loads(native_pdf_document_json(library,data,password,False,max_pixels))


def read_pdf_visible_text_geometry(source, *, page_index=0, pdfium_library=None, password=None):
    """按原生分析规则读取可见字符与裁剪几何；基础全字符接口仍单独可用。"""
    if type(page_index)is not int or page_index<0:raise ValueError('page_index must be a nonnegative integer')
    library,data=_prepare_pdf(source,pdfium_library,None)
    return json.loads(read_pdf_visible_text_geometry_json(library,data,page_index,password))


def plan_native_glyph_weight(lines, *, tight_bboxes=None):
    """给出原同页同字符字形对照分组；candidates 为空时无需渲染。"""
    return json.loads(legacy_pipeline_json('plan_native_glyph_weight',json.dumps(dict(lines=lines,tight_bboxes=tight_bboxes or {}),allow_nan=False)))


def detect_native_glyph_weight(lines, page_png=None, *, tight_bboxes=None, scale=1.):
    """原灰度/最近邻字形墨迹对照→补充粗体样式，保持原字符和行只读。"""
    q=json.dumps(dict(lines=lines,tight_bboxes=tight_bboxes or {},scale=scale),allow_nan=False)
    data=None if page_png is None else _read_bytes(page_png,64_000_000)
    return json.loads(native_glyph_weight_json(q,data))


def normalize_native_pdf_model_text(pages):
    """原 PdfModel 最末可见全角英数/符号清洗；保留公式、代码、URL 和 Span 结构。"""
    return json.loads(legacy_pipeline_json('normalize_native_pdf_model_text',json.dumps(pages,allow_nan=False)))


def classify_native_page_titles(lines, page_size, *, page_index=0, container_bboxes=None,
                                caption_container_bboxes=None, body_profile=None, title_profile=None):
    """完整旧页面标题分类；包括全部栏内/跨栏判断和正文/图注回退。"""
    if isinstance(page_index,bool) or not isinstance(page_index,int) or page_index<0:
        raise ValueError('page_index must be a nonnegative integer')
    request=dict(lines=lines,page_size=page_size,page_index=page_index,container_bboxes=[] if container_bboxes is None else container_bboxes,
                 caption_container_bboxes=[] if caption_container_bboxes is None else caption_container_bboxes,
                 document_body_profile=body_profile,document_title_profile=title_profile,classify_page=True)
    return json.loads(legacy_pipeline_json('classify_native_page_titles',json.dumps(request,allow_nan=False)))['lines']


def infer_native_document_title_profile(pages, body_profile):
    """用完整旧页面标题探测构建全文标题原型；不回写探测副本的分类。"""
    return json.loads(legacy_pipeline_json('infer_native_document_title_profile',json.dumps(dict(pages=pages,body_profile=body_profile),allow_nan=False)))


def classify_native_document_titles(pages, body_profile, canonical_body_profile):
    """已准备页面的完整全文标题原型/结构及未知字重重复标题处理。"""
    return json.loads(legacy_pipeline_json('classify_native_document_titles',json.dumps(dict(pages=pages,body_profile=body_profile,canonical_body_profile=canonical_body_profile),allow_nan=False)))


def finish_native_geometry_page(prepared, plan, *, page_index=0, style_lines=None, link_lines=None, classify_text=False, classify_footnotes=False, complete_page=False, tight_bboxes=None, origins=None):
    """容器阶段后应用 Y 修复、重拆/证据对齐、标签回收和行首分式恢复。"""
    if isinstance(page_index,bool) or not isinstance(page_index,int) or page_index<0:
        raise ValueError('page_index must be a nonnegative integer')
    if not isinstance(classify_text,bool):raise ValueError('classify_text must be bool')
    if not isinstance(classify_footnotes,bool):raise ValueError('classify_footnotes must be bool')
    if not isinstance(complete_page,bool):raise ValueError('complete_page must be bool')
    return json.loads(legacy_pipeline_json('finish_native_geometry_page',json.dumps(dict(prepared=prepared,plan=plan,page_index=page_index,style_lines=[] if style_lines is None else style_lines,link_lines=[] if link_lines is None else link_lines,include_post_text_roles=classify_text,include_post_footnotes=classify_footnotes,prepare_page=complete_page,tight_bboxes={} if tight_bboxes is None else tight_bboxes,origins={} if origins is None else origins),allow_nan=False)))


def read_pdf_vector_geometry(source, *, page_index=0, pdfium_library=None, password=None):
    """按旧DocVortex规则读取视觉页面线、路径与矩形；不提取字符或执行表格恢复。"""
    if isinstance(page_index,bool) or not isinstance(page_index,int) or page_index<0:
        raise ValueError('page_index must be a nonnegative integer')
    library,data=_prepare_pdf(source,pdfium_library,None)
    return json.loads(read_pdf_vector_geometry_json(library,data,page_index,password))


def read_native_pdf_page(source, *, page_index=0, pdfium_library=None, password=None, quote_loosebox=True):
    """按旧 pdftext 规则读取原生字符及 span/行/块；不执行 OCR 或重排阅读顺序。"""
    if not isinstance(quote_loosebox,bool):raise ValueError('quote_loosebox must be bool')
    library,data=_prepare_pdf(source,pdfium_library,None)
    return json.loads(read_native_pdf_page_json(library,data,page_index,password,quote_loosebox))


def table_line_geometry(mask, *, source_width, source_height, morph_close=True, **config):
    """UNET 类别图→原图尺度的横/竖线，完成闭运算、连通域、补连和端点延长。

    mask 来自 table_line_mask_png；尚未执行抗锯齿绘线、旋转修正和闭合格发现。
    """
    return json.loads(unet_line_geometry_json(json.dumps(mask,allow_nan=False),source_width,source_height,morph_close,json.dumps(config,allow_nan=False)))


def connect_table_lines(rows, columns, **config):
    """UNET 线段补连及端点延长；保留 FP32 线段顺序，不执行绘图或旋转修正。"""
    return json.loads(connect_table_lines_json(rows,columns,json.dumps(config,allow_nan=False)))


def recover_table_polygons(polygons, *, row_threshold=10.0, column_threshold=15.0):
    """UNET 已排序格框→原索引对应的闭区间逻辑格；四角为左上/左下/右下/右上。

    不对输入重新排序，不执行线检测或填补缺格；阈值及坐标使用双精度。
    """
    return json.loads(recover_table_polygons_json(polygons, row_threshold, column_threshold))


def _merge_table_classifications(primary, secondary):
    return dict(label=primary['label']if primary['label']==secondary['label']else 'wireless',
                score=min(primary['score'],secondary['score']),members=[primary,secondary])


def select_table_candidate(ocr_texts, *, wired_html, wireless_html):
    """对 Rust 输出器生成的两个候选表格执行旧选优规则，返回选中分支、原因及计数。

    ocr_texts 是未转义字符串；HTML 必须闭合，仅含规范表格标签/跨度及转义文本。
    不修复任意浏览器 HTML，也不屏蔽无效候选或预算错误。
    """
    return json.loads(select_table_candidate_json(list(ocr_texts),wired_html,wireless_html))


def recognize_table_auto_png(source, *, classifier, wired_model, wireless_model, detector, recognizer,
                             secondary_classifier=None, classifier_output_type=None, wireless_variant="slanet-plus",
                             rotated_fix=True, min_confidence=0.0, blank_cell_rec=True,
                             use_compare_table=False, cls_conf_threshold=0.9):
    """裁图分类路由；低于阈值或显式要求时比较两个表格候选，共用一次 OCR。"""
    if not isinstance(cls_conf_threshold,(int,float))or not math.isfinite(cls_conf_threshold)or not 0<=cls_conf_threshold<=1:
        raise ValueError("cls_conf_threshold must be finite and within [0, 1]")
    data=_read_bytes(source,256*1024*1024)
    classification=classifier.classify_table_png(data,output_type=classifier_output_type)
    if secondary_classifier is not None:
        secondary=secondary_classifier.classify_table_png(data,variant="qanything")
        classification=_merge_table_classifications(classification,secondary)
    route=classification['label']
    compare=bool(use_compare_table or classification['score']<cls_conf_threshold)
    required=('wired','wireless')if compare else(route,)
    models=dict(wired=wired_model,wireless=wireless_model)
    if any(models[name]is None for name in required):
        raise ValueError("table routing/comparison requires models: "+", ".join(required))
    lines=ocr_png(data,detector=detector,recognizer=recognizer,min_confidence=min_confidence)
    def predict(name):
        if name=='wired':
            cells=wired_model.table_cells_png(data,rotated_fix=rotated_fix)
            return _complete_wired_table(data,cells,lines,recognizer,blank_cell_rec)
        structure=wireless_model.table_structure_png(data,variant=wireless_variant)
        return dict(structure=structure,ocr_lines=lines,**match_table(structure,lines))
    candidates={route:predict(route)}
    comparison=None
    if compare:
        alternate='wireless'if route=='wired'else'wired'
        candidates[alternate]=predict(alternate)
        comparison=select_table_candidate([line['text']for line in lines],wired_html=candidates['wired']['html'],wireless_html=candidates['wireless']['html'])
        route=comparison['route']
    return dict(classification=classification,route=route,comparison=comparison,**candidates[route])


def recognize_page_tables_png(source, *, layout_model, classifier, wired_model, wireless_model, detector, recognizer,
                               secondary_classifier=None, layout_threshold=0.5, crop_grid=5, max_tables=32,
                               max_pixels=16_000_000, max_total_crop_pixels=32_000_000, classifier_output_type=None,
                               wireless_variant="slanet-plus", rotated_fix=True, min_confidence=0.0, blank_cell_rec=True,
                               use_compare_table=False, cls_conf_threshold=0.9):
    """PP-DocLayoutV3 rect→表格裁图→分类/选优；保留局部结果并提供页面像素几何。

    不执行整页段落/正文解析、表格方向自动旋转或跨页表格合并。任一候选失败时传出错误。
    """
    _validate_page_options(layout_threshold,min_confidence,cls_conf_threshold,wireless_variant,classifier_output_type)
    data=_read_bytes(source,256*1024*1024)
    image=_PageImage(data,max_pixels)
    image.plan_json('[]',crop_grid,max_tables,max_total_crop_pixels)
    layout=layout_model.layout_png(data,threshold=layout_threshold)
    plan=json.loads(image.plan_json(json.dumps(layout,allow_nan=False),crop_grid,max_tables,max_total_crop_pixels))
    tables=[]
    for crop in plan['tables']:
        recognition=recognize_table_auto_png(image.crop_png(crop['bounds']),classifier=classifier,wired_model=wired_model,
            wireless_model=wireless_model,detector=detector,recognizer=recognizer,secondary_classifier=secondary_classifier,
            classifier_output_type=classifier_output_type,wireless_variant=wireless_variant,rotated_fix=rotated_fix,
            min_confidence=min_confidence,blank_cell_rec=blank_cell_rec,use_compare_table=use_compare_table,cls_conf_threshold=cls_conf_threshold)
        geometry=json.loads(image.project_json(json.dumps(recognition,allow_nan=False),crop['bounds']))
        tables.append(dict(layout_index=crop['layout_index'],region=layout[crop['layout_index']],crop=crop,
                           recognition=recognition,page_geometry=geometry))
    return dict(width=image.width,height=image.height,coordinate_space='page_pixels',layout=layout,tables=tables,crop_pixels=plan['crop_pixels'])


def _validate_page_options(layout_threshold,min_confidence,cls_conf_threshold,wireless_variant,classifier_output_type):
    for name,value in [('layout_threshold',layout_threshold),('min_confidence',min_confidence),('cls_conf_threshold',cls_conf_threshold)]:
        if not isinstance(value,(int,float))or not math.isfinite(value)or not 0<=value<=1:
            raise ValueError(name+" must be finite and within [0, 1]")
    if wireless_variant not in ('slanet-plus','slanet-1m')or classifier_output_type not in (None,'probabilities','logits'):
        raise ValueError("invalid table model variant/output type")


def recognize_page_png(source, *, layout_model, detector, recognizer, classifier=None, wired_model=None, wireless_model=None,
                       secondary_classifier=None, layout_threshold=0.5, crop_grid=5, max_regions=256,
                       max_pixels=16_000_000, max_total_crop_pixels=32_000_000, classifier_output_type=None,
                       wireless_variant="slanet-plus", rotated_fix=True, min_confidence=0.0, blank_cell_rec=True,
                       use_compare_table=False, cls_conf_threshold=0.9):
    """单页矩形布局→文字区域 OCR/表格→按模型 order 稳定排序的页面块。

    原始 layout 与局部 OCR 均保留，blocks[*].lines 为页面像素坐标。
    text 按块连接文字及表格 OCR 行；无段落重排、图文去重、旋转矫正或原生 PDF 融合。
    图片/公式等未处理区域仍作为 unsupported 块返回，并附诊断；空布局不回退整页 OCR。
    """
    return _recognize_page_png(source,layout_model=layout_model,detector=detector,recognizer=recognizer,
        classifier=classifier,wired_model=wired_model,wireless_model=wireless_model,secondary_classifier=secondary_classifier,
        layout_threshold=layout_threshold,crop_grid=crop_grid,max_regions=max_regions,max_pixels=max_pixels,
        max_total_crop_pixels=max_total_crop_pixels,classifier_output_type=classifier_output_type,wireless_variant=wireless_variant,
        rotated_fix=rotated_fix,min_confidence=min_confidence,blank_cell_rec=blank_cell_rec,
        use_compare_table=use_compare_table,cls_conf_threshold=cls_conf_threshold)


def _recognize_page_png(source, *, layout_model, detector, recognizer, classifier=None, wired_model=None, wireless_model=None,
                        secondary_classifier=None, layout_threshold=0.5, crop_grid=5, max_regions=256,
                        max_pixels=16_000_000, max_total_crop_pixels=32_000_000, classifier_output_type=None,
                        wireless_variant="slanet-plus", rotated_fix=True, min_confidence=0.0, blank_cell_rec=True,
                        use_compare_table=False, cls_conf_threshold=0.9, text_reader=None, table_resolver=None):
    _validate_page_options(layout_threshold,min_confidence,cls_conf_threshold,wireless_variant,classifier_output_type)
    data=_read_bytes(source,256*1024*1024)
    image=_PageImage(data,max_pixels)
    image.plan_blocks_json('[]',crop_grid,max_regions,max_total_crop_pixels)
    layout=layout_model.layout_png(data,threshold=layout_threshold)
    plan=json.loads(image.plan_blocks_json(json.dumps(layout,allow_nan=False),crop_grid,max_regions,max_total_crop_pixels))
    crops={crop['layout_index']:crop for crop in plan['regions']}
    # 在任意识别开始前检查整批预算和必需的基础会话。
    table_choices=table_resolver(layout,plan['regions'])if table_resolver is not None else{}
    model_tables=[index for index in crops if layout[index]['label']=='table'and table_choices.get(index,{}).get('method')!='native']
    has_text=any(layout[index]['label']!='table'for index in crops)
    if (has_text or model_tables)and(detector is None or (recognizer is None and (text_reader is None or model_tables))):
        raise ValueError("text/table regions require detector and recognizer")
    if model_tables and classifier is None:
        raise ValueError("table regions require classifier")
    blocks=[];diagnostics=[]
    for index,region in sorted(enumerate(layout),key=lambda item:(item[1]['order'],item[0])):
        block=dict(layout_index=index,region=region)
        crop=crops.get(index)
        if crop is None:
            block.update(kind='unsupported',text=None)
            diagnostics.append(dict(code='unsupported_layout_region',layout_index=index,label=region['label']))
        elif region['label']=='table':
            selection=table_choices.get(index)
            if selection is not None and selection['method']=='native':
                recognition=selection['native']
                block.update(kind='table',crop=crop,recognition=recognition,page_geometry=recognition['page_geometry'],
                    lines=[],html=recognition['html'],text=recognition['text'],
                    table_selection=dict(method=selection['method'],reason=selection['reason']))
                blocks.append(block)
                continue
            recognition=recognize_table_auto_png(image.crop_png(crop['bounds']),classifier=classifier,wired_model=wired_model,
                wireless_model=wireless_model,detector=detector,recognizer=recognizer,secondary_classifier=secondary_classifier,
                classifier_output_type=classifier_output_type,wireless_variant=wireless_variant,rotated_fix=rotated_fix,
                min_confidence=min_confidence,blank_cell_rec=blank_cell_rec,use_compare_table=use_compare_table,cls_conf_threshold=cls_conf_threshold)
            geometry=json.loads(image.project_json(json.dumps(recognition,allow_nan=False),crop['bounds']))
            block.update(kind='table',crop=crop,recognition=recognition,page_geometry=geometry,
                         lines=geometry['ocr_lines'],html=recognition['html'])
            block['text']='\n'.join(line['text']for line in block['lines'])
            if selection is not None:
                block['table_selection']=dict(method=selection['method'],reason=selection['reason'])
        else:
            if text_reader is None:
                local=ocr_png(image.crop_png(crop['bounds']),detector=detector,recognizer=recognizer,min_confidence=min_confidence)
                lines=json.loads(image.project_ocr_json(json.dumps(local,allow_nan=False),crop['bounds']))
            else:
                local,lines=text_reader(image.crop_png(crop['bounds']),crop['bounds'],min_confidence)
            block.update(kind='text',crop=crop,local_lines=local,lines=lines,text='\n'.join(line['text']for line in lines))
        blocks.append(block)
    return dict(width=image.width,height=image.height,coordinate_space='page_pixels',layout=layout,blocks=blocks,
                text='\n\n'.join(block['text']for block in blocks if block['text']),
                crop_pixels=plan['crop_pixels'],diagnostics=diagnostics)


def recognize_pdf_page(source, *, layout_model, detector, recognizer, page_index=0, dpi=144.0,
                        max_pixels=16_000_000, pdfium_library=None, password=None, **options):
    """渲染指定 PDF 页并生成有序模型块，坐标为渲染图像素；不执行原生/模型融合。"""
    page=render_pdf_page(source,page_index=page_index,dpi=dpi,max_pixels=max_pixels,pdfium_library=pdfium_library,password=password)
    result=recognize_page_png(page.png,layout_model=layout_model,detector=detector,recognizer=recognizer,max_pixels=max_pixels,**options)
    return dict(page_index=page_index,dpi=dpi,**result)


def _write_model_assets(writer, assets, inline_count=0):
    # 只允许 Rust 生成的平铺 SHA256 名称；先验证所有名称再执行 I/O。
    for name,data in assets:
        digest,sep,extension=name.partition('.')
        if len(digest)!=64 or any(c not in '0123456789abcdef'for c in digest) or not sep or not 1<=len(extension)<=32 or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_'for c in extension):
            raise ValueError('invalid generated asset name')
    if isinstance(writer,(str,os.PathLike)):
        import tempfile
        directory=Path(writer).expanduser().resolve()
        directory.mkdir(parents=True,exist_ok=True)
        for name,data in assets:
            temporary=None
            try:
                with tempfile.NamedTemporaryFile(dir=directory,prefix='.rapiddoc-',delete=False)as stream:
                    temporary=Path(stream.name);stream.write(data)
                os.replace(temporary,directory/name)
            finally:
                if temporary is not None:temporary.unlink(missing_ok=True)
    else:
        written=getattr(writer,'_mineru_written_image_paths',None)
        if inline_count and written is None:
            written=set()
            try:setattr(writer,'_mineru_written_image_paths',written)
            except Exception:written=None
        for index,(name,data) in enumerate(assets):
            once=index<inline_count and written is not None
            if once and name in written:continue
            writer.write(name,data)
            if once:written.add(name)


def replace_inline_base64_img_src(markup, image_writer, page_index=0, image_path_prefix=''):
    """按旧双引号 src 规则保存内联原字节；返回 HTML，不修改其他属性。"""
    if not markup or not image_writer or 'base64,' not in markup:return markup
    html,assets=replace_inline_image_assets(markup,image_path_prefix)
    _write_model_assets(image_writer,assets,len(assets))
    return html


def assemble_model_page_with_assets(source, model, *, image_writer, page_width, page_height, page_index=0,
                                    scale=1.0, formula_enabled=True, original_images=(),
                                    extract_original_image=False, extract_original_image_iou_thresh=0.9,
                                    table_fill_images=()):
    """旧单页组装与图片写出；PNG 8-bit L/LA/RGB/RGBA，writer 为目录或 write(name,bytes) 对象。"""
    return _assemble_model_page_assets(source,model,image_writer=image_writer,page_width=page_width,page_height=page_height,
        page_index=page_index,scale=scale,formula_enabled=formula_enabled,original_images=original_images,
        extract_original_image=extract_original_image,extract_original_image_iou_thresh=extract_original_image_iou_thresh,table_fill_images=table_fill_images)['page']


def assemble_model_page_with_native_text(source,model,*,native_page,**options):
    """旧 ocr_enable=False 字符填充/补识别裁图；返回 {page,ocr_images}，仍需后置 OCR。"""
    if not isinstance(native_page,dict):raise TypeError('native_page must be a dict of normalized legacy PDF facts')
    return _assemble_model_page_assets(source,model,native_page=native_page,**options)


def fill_model_text_spans(source,spans,*,native_page,blocks,discarded_blocks=(),scale=1.0,page_index=0):
    """独立旧 txt_spans_extract；blocks 为旧 prepare_block_bboxes 数组。"""
    request=dict(spans=spans,native_page=native_page,blocks=blocks,discarded_blocks=discarded_blocks,scale=scale,page_index=page_index)
    spans,images=fill_native_text_bytes(json.dumps(request,allow_nan=False),_read_bytes(source,64_000_000))
    return dict(spans=json.loads(spans),ocr_images=dict(images))


def _assemble_model_page_assets(source, model, *, image_writer, page_width, page_height, page_index=0,
                                    scale=1.0, formula_enabled=True, original_images=(),
                                    extract_original_image=False, extract_original_image_iou_thresh=0.9,
                                    table_fill_images=(),native_page=None):
    """旧单页组装与图片写出；PNG 8-bit L/LA/RGB/RGBA，writer 为目录或 write(name,bytes) 对象。"""
    if not isinstance(image_writer,(str,os.PathLike))and not callable(getattr(image_writer,'write',None)):
        raise TypeError('image_writer must be a directory or write(name, bytes) object')
    if not isinstance(extract_original_image,bool):raise TypeError('extract_original_image must be bool')
    originals=[];original_bytes=0
    if extract_original_image:
        if len(original_images)>256:raise ValueError('too many original images')
        for image in original_images:
            data=_read_bytes(image['source'],64_000_000)
            original_bytes+=len(data)
            if original_bytes>128_000_000:raise ValueError('original image bytes exceed budget')
            originals.append((image['bbox'],data))
    if len(table_fill_images)>256:raise ValueError('too many table fill images')
    fills=[];fill_bytes=0
    for image in table_fill_images:
        data=_read_bytes(image['source'],64_000_000)
        fill_bytes+=len(data)
        if fill_bytes>128_000_000:raise ValueError('table fill image bytes exceed budget')
        fills.append((image['uuid'],image['bbox'],data))
    request=dict(model=model,page_width=page_width,page_height=page_height,page_index=page_index,
                 scale=scale,formula_enabled=formula_enabled)
    if native_page is not None:request['native_page']=native_page
    parent=image_writer if isinstance(image_writer,(str,os.PathLike)) else getattr(image_writer,'_parent_dir','images')
    image_dir=os.path.basename(str(parent))
    page,assets,inline_count,ocr_images=assemble_model_page_assets_bytes(json.dumps(request,allow_nan=False),_read_bytes(source,64_000_000),
                                                originals,extract_original_image,extract_original_image_iou_thresh,
                                                image_dir+'/' if image_dir else '',fills)
    page=json.loads(page)
    _write_model_assets(image_writer,assets,inline_count)
    return dict(page=page,ocr_images=dict(ocr_images))


def assemble_model_page(model, *, page_width, page_height, page_index=0, scale=1.0, formula_enabled=True):
    """旧 layout_dets→page_info；OCR 已识别、writer=None，返回副本或空页 None。"""
    request=dict(model=model,page_width=page_width,page_height=page_height,page_index=page_index,
                 scale=scale,formula_enabled=formula_enabled)
    return json.loads(legacy_pipeline_json("assemble_page",json.dumps(request,allow_nan=False)))


def assemble_model_pages(pages, *, optimize_formula_numbers=False, merge_tables=False):
    """单页组装→旧 para_split；公式编号、跨页表格均须显式开启。"""
    if not isinstance(pages,list):raise TypeError("pages must be a list of page assembly requests")
    if not isinstance(optimize_formula_numbers,bool):raise TypeError("optimize_formula_numbers must be bool")
    if not isinstance(merge_tables,bool):raise TypeError("merge_tables must be bool")
    requests=[]
    for i,page in enumerate(pages):
        if not isinstance(page,dict):raise TypeError("page request must be a dict")
        requests.append(dict(page_index=i,scale=1.0,formula_enabled=True)|page)
    operation="assemble_pages_with_formulas"if optimize_formula_numbers else "assemble_pages"
    result=json.loads(legacy_pipeline_json(operation,json.dumps(requests,allow_nan=False)))
    return merge_cross_page_tables(result) if merge_tables else result


def merge_cross_page_tables(pages):
    """按旧规则合并相邻页边界表格，返回副本；严格闭合、非嵌套表格 HTML。"""
    if not isinstance(pages,list):raise TypeError('pages must be a list')
    return json.loads(legacy_pipeline_json('merge_tables',json.dumps(pages,allow_nan=False)))


def apply_model_ocr_results(pages, results):
    """按旧后置 OCR 顺序回填 [(text,score)]，返回副本；不执行推理。"""
    request=dict(pages=pages,results=results)
    return json.loads(legacy_pipeline_json('apply_post_ocr',json.dumps(request,allow_nan=False)))


def post_process_model_ocr(pages, *, recognizer, images, batch_size=6):
    """np_img 使用字符串图片 ID，images 映射到 PNG 路径/bytes/流；先于编号和段落执行。"""
    snapshot=json.dumps(pages,allow_nan=False)
    plan=json.loads(legacy_pipeline_json('plan_post_ocr',snapshot))
    pages=json.loads(snapshot)
    if not plan:return apply_model_ocr_results(pages,[])
    if _HOST_PAGE.get() and not isinstance(recognizer,OnnxSession):
        from rapid_doc_rust._runtime import LazySession
        if isinstance(recognizer,LazySession):recognizer=recognizer.session()
    if not isinstance(recognizer,OnnxSession):raise TypeError('recognizer must be OnnxSession')
    if not isinstance(batch_size,int)or isinstance(batch_size,bool)or not 1<=batch_size<=64:raise ValueError('batch_size must be 1..64')
    refs=[]
    for target in plan:
        image_id=target['image']
        if not isinstance(image_id,str)or not 1<=len(image_id.encode('utf-8'))<=256:raise ValueError('np_img must be a nonempty image ID <=256 bytes')
        refs.append((image_id,images[image_id]))
    sources=[];cache={};total=0
    for image_id,source in refs:
        if image_id not in cache:cache[image_id]=_read_bytes(source,64_000_000)
        data=cache[image_id];total+=len(data)
        if total>128_000_000:raise ValueError('recognition bytes exceed budget')
        sources.append(data)
    results=recognizer.recognize_batch_png(sources,batch_size=batch_size)
    return apply_model_ocr_results(pages,[(r['text'],r['confidence'])for r in results])


def optimize_formula_numbers(pages):
    """旧公式编号关联，返回副本；应先于 split_paragraphs 调用，不重算已有 para_blocks。"""
    if not isinstance(pages,list):raise TypeError("pages must be a page_info_list")
    return json.loads(legacy_pipeline_json("optimize_formulas",json.dumps(pages,allow_nan=False)))


def sort_model_blocks(blocks, *, page_width, page_height, footnote_blocks=None):
    """旧 block_sort 的模型顺序分支，返回副本；缺少 original_order 时明确报 unsupported。"""
    import json
    if not isinstance(blocks,list):
        raise TypeError("blocks must be a list")
    request=dict(blocks=blocks,page_width=page_width,page_height=page_height,
                 footnote_blocks=[]if footnote_blocks is None else footnote_blocks)
    return json.loads(sort_model_blocks_json(json.dumps(request,allow_nan=False)))


def split_paragraphs(pages):
    """按旧 para_split 规则处理 page_info_list，返回副本，保留原 preproc_blocks 和未知字段。

    输入必须已完成旧流程的排序/分组；本方法不执行 OCR、阅读顺序或 Markdown 导出。
    正几何、有效 Unicode、唯一整数页号；资源上限见兼容性记录。
    """
    if not isinstance(pages,list):raise TypeError('pages must be a page_info_list')
    return json.loads(split_paragraphs_json(json.dumps(pages,allow_nan=False)))


def organize_page_text(page, *, experimental=False, split_columns=True, line_order="source", block_order="source", max_block_order_work=2_000_000):
    """保守整理页面块内段落，返回来源行索引；输入页面及原始逐行文字保持不变。

    独立的 x 投影列可按列排列；不跨布局块合并，不补漏检，不解析任意 HTML/公式。
    line_order='geometry' 仅排序可信水平栏，歧义时保留原序并提供诊断。
    block_order='geometry' 可重排清晰留白分隔的块，整页歧义时保留原序。
    这是显式后处理，原识别接口的 text 和 blocks 不会自动变化。
    """
    if experimental is not True:
        raise ValueError('organize_page_text requires experimental=True; paragraph/list/order rules are not Python-parser compatible')
    if not isinstance(page,dict):raise TypeError('page must be a page result dictionary')
    if not isinstance(split_columns,bool):raise ValueError('split_columns must be boolean')
    if not isinstance(line_order,str)or line_order not in ('source','geometry'):
        raise ValueError('line_order must be source or geometry')
    if not isinstance(block_order,str)or block_order not in ('source','geometry'):
        raise ValueError('block_order must be source or geometry')
    if isinstance(max_block_order_work,bool)or not isinstance(max_block_order_work,int)or not 1<=max_block_order_work<=2_000_000:
        raise ValueError('max_block_order_work must be an integer in 1..2000000')
    return json.loads(organize_page_text_json(json.dumps(page,allow_nan=False),split_columns,line_order,block_order,max_block_order_work))


def recognize_pdf_page_tables(source, *, layout_model, classifier, wired_model, wireless_model, detector, recognizer,
                               page_index=0, dpi=144.0, max_pixels=16_000_000, pdfium_library=None, password=None, **options):
    """渲染指定 PDF 页后识别表格；坐标为渲染图像素，页号保留原始索引。"""
    page=render_pdf_page(source,page_index=page_index,dpi=dpi,max_pixels=max_pixels,pdfium_library=pdfium_library,password=password)
    result=recognize_page_tables_png(page.png,layout_model=layout_model,classifier=classifier,wired_model=wired_model,
        wireless_model=wireless_model,detector=detector,recognizer=recognizer,max_pixels=max_pixels,**options)
    return dict(page_index=page_index,dpi=dpi,**result)


def recognize_pdf_page_hybrid(source, *, layout_model, detector=None, recognizer=None, page_index=0, dpi=144.0,
                               experimental=False, max_pixels=16_000_000, max_native_work=10_000_000, prefer_native_tables=False,
                               native_table_margin_pt=6.0, max_native_table_work=10_000_000,
                               pdfium_library=None, password=None, **options):
    """实验性的正文原生/OCR 融合；原生表格优先须另行显式启用。

    native 行 confidence=None，带 PDFium character_indices 和选择原因；OCR 阈值只过滤 OCR 行。
    仍依赖布局/检测，无漏检补偿；旋转页、图片重叠、Unicode/覆盖不足均回到 OCR。
    """
    if experimental is not True:
        raise ValueError('recognize_pdf_page_hybrid requires experimental=True; native/OCR arbitration is not Python-parser compatible')
    if not isinstance(max_native_work,int)or not 0<=max_native_work<=10_000_000:
        raise ValueError('max_native_work must be within [0, 10000000]')
    if not isinstance(prefer_native_tables,bool):
        raise ValueError('prefer_native_tables must be boolean')
    if not isinstance(native_table_margin_pt,(int,float))or not math.isfinite(native_table_margin_pt)or not 0<=native_table_margin_pt<=32:
        raise ValueError('native_table_margin_pt must be within [0, 32]')
    if not isinstance(max_native_table_work,int)or not 0<=max_native_table_work<=10_000_000:
        raise ValueError('max_native_table_work must be within [0, 10000000]')
    if 'text_reader'in options or 'table_resolver'in options:
        raise TypeError('text_reader/table_resolver are internal')
    library,data=_prepare_pdf(source,pdfium_library,None)
    facts=parse_pdf_text(data,pdfium_library=library,start_page=page_index,end_page=page_index+1,password=password,allow_partial=True).pages[0]
    page=render_pdf_page(data,page_index=page_index,dpi=dpi,max_pixels=max_pixels,pdfium_library=library,password=password)
    facts_json=json.dumps(facts,allow_nan=False)
    native=_NativeTextPage(facts_json,page.width,page.height)
    native_tables=_NativeTablePage(facts_json,page.width,page.height)if prefer_native_tables else None
    table_plan=dict(selections=[],work=0)
    def resolve_tables(layout,crops):
        nonlocal table_plan
        proposals=[dict(layout_index=crop['layout_index'],bounds=layout[crop['layout_index']]['coordinate'])for crop in crops
                   if layout[crop['layout_index']]['label']=='table']
        table_plan=json.loads(native_tables.plan_json(json.dumps(proposals,allow_nan=False),native_table_margin_pt,max_native_table_work))
        return {selection['layout_index']:selection for selection in table_plan['selections']}
    remaining=max_native_work;routes=[]
    def read_text(png,bounds,min_confidence):
        nonlocal remaining
        detected=json.loads(detector._native.detect_png_json(png,True))
        matched=json.loads(native.match_json(json.dumps(detected,allow_nan=False),remaining,tuple(bounds[:2])))
        remaining-=matched['work']
        missing=[i for i,item in enumerate(matched['lines'])if item['method']=='ocr']
        if missing and recognizer is None:
            raise ValueError('OCR fallback requires recognizer')
        recognized=recognizer.recognize_regions_png(png,[detected[i]['points']for i in missing])if missing else[]
        by_index=dict(zip(missing,recognized));local=[];lines=[]
        for i,(box,match)in enumerate(zip(detected,matched['lines'])):
            routes.append(dict(bounds=bounds,line_index=i,**match))
            if match['method']=='native':
                line=dict(text=match['text'],confidence=None,character_indices=match['character_indices'])
            else:
                line=by_index[i]
                if line['confidence']<min_confidence:continue
            line=dict(line,method=match['method'],reason=match['reason'],detection_score=box['score'])
            local.append(dict(line,points=box['points']))
            lines.append(dict(line,points=matched['points'][i]))
        return local,lines
    result=_recognize_page_png(page.png,layout_model=layout_model,detector=detector,recognizer=recognizer,
        max_pixels=max_pixels,text_reader=read_text,table_resolver=resolve_tables if native_tables is not None else None,**options)
    return dict(page_index=page_index,dpi=dpi,native_page=facts,native_routes=routes,
                native_work=max_native_work-remaining,native_table_work=table_plan['work'],
                native_table_routes=[{key:selection[key]for key in ('layout_index','method','reason')}for selection in table_plan['selections']],**result)


@dataclass
class PdfOcrOutput:
    """原生/OCR 页面路由预览；pages 的 OCR 坐标为视觉左上 point，document 保留原生事实。"""
    text: str
    document: dict
    pages: list


def parse_pdf_ocr(source, *, detector=None, recognizer=None, pdfium_library=None,
                  start_page=0, end_page=None, password=None, dpi=144.0,
                  max_pixels=16_000_000, max_total_pixels=128_000_000, max_ocr_pages=256):
    """保留原生文字，仅对无原生文字且需 OCR 的页面识别；需要时缺模型会报错。

    尚未补识别部分原生文字页面中的图片块，不做版面/段落推断。
    原始 facts.text 保持原生文字，pages[*].text 才是路由后的最终逐页文字。
    """
    library, data = _prepare_pdf(source, pdfium_library,None)
    output = parse_pdf_ocr_json(library, data, detector._native if detector is not None else None,
        recognizer._native if recognizer is not None else None, start_page, end_page, password,
        dpi, max_pixels, max_total_pixels, max_ocr_pages)
    return PdfOcrOutput(**json.loads(output))


def compatibility_report():
    """返回构建时的来源哈希、证据范围和偏差分类；不把作者合同当成 Python 整文档等价。"""
    return json.loads((Path(__file__).resolve().parents[2]/'packaging/compatibility-audit.json').read_text(encoding='utf-8'))


def capabilities():
    """只报告实际已实现的能力，不将编译成功等同于解析支持。"""
    return {"stage": "native_text_preview", "contracts": True, "formats": ["pdf_native_text"],
            "ocr": False, "layout": False, "tables": False, "images": False, "cuda_verified": False, "table_region_recovery": True, "page_render": True,
            "png_detection": True, "png_ocr": True, "pdf_page_ocr": True,
            "orientation_classification": True,
            "native_ocr_page_routing": True,
            "png_layout_rect": True,
            "png_table_structure": True,
            "table_ocr_matching": True,
            "table_line_mask": True,
            "table_line_geometry": True,
            "table_polygon_topology": True,
            "png_table_cells": True,
            "png_wired_table_ocr": True,
            "paddle_table_classification": True,
            "qanything_table_classification": True,
            "table_classifier_consensus": True,
            "png_table_routing": True,
            "table_quality_selection": True,
            "png_page_tables": True,
            "pdf_page_tables": True,
            "png_page_blocks": True,
            "pdf_page_blocks": True,
            "pdf_native_line_fusion": True,
            "pdf_native_table_selection": True,
            "text_line_joining": True,
            "page_paragraphs": True,
            "legacy_paragraph_rules": True,
            "legacy_model_block_order": True,
            "legacy_model_page_assembly": True,
            "legacy_formula_number_rules": True,
            "legacy_page_asset_writer": True,
            "legacy_cross_page_tables": True,
            "legacy_post_ocr": True,
            "legacy_native_text": True,
            "legacy_native_pdf_page": True,
            "pdf_original_images": True,
            "legacy_model_layout": True,
            "legacy_model_crop_plan": True,
            "legacy_native_ocr_plan": True,
            "legacy_model_ocr_execution": True,
            "legacy_table_text_execution": True,
            "legacy_table_direction": True,
            "legacy_table_native_text": True,
            "legacy_table_hints": True,
            "legacy_table_model_payload": True,
            "legacy_table_full_ocr": True,
            "legacy_slanet_matcher": True,
            "legacy_slanet_raw_inference": True,
            "legacy_table_html_normalization": True,
            "legacy_unet_inference": True,
            "legacy_table_model_routing": True,
            "legacy_native_table_policy": True,
            "native_table_candidate_arbitration": True,
            "native_table_text_preparation": True,
            "native_table_cell_text": True,
            "native_table_candidate_builder": True,
            "native_table_vector_evidence": True,
            "native_table_vector_topology": True,
            "native_table_vector_generation": True,
            "native_table_text_candidates": True,
            "native_table_engine_policy": True,
            "native_table_engine": True,
            "docvortex_pdf_vector_geometry": True,
            "docvortex_pdf_character_primitives": True,
            "docvortex_character_dedup": True,
            "native_script_geometry": True,
            "native_script_token_refinement": True,
            "native_line_typography": True,
            "native_text_lanes": True,
            "native_inline_clusters": True,
            "native_table_materialization": True,
            "native_table_block_preparation": True,
            "native_table_annotations": True,
            "native_table_block_commit": True,
            "native_table_media_png": True,
            "native_visual_lines": True,
            "native_source_lines": True,
            "native_inline_evidence": True,
            "pdf_link_annotations": True,
            "native_inline_output": True,
            "native_script_output": True,
            "native_image_regions": True,
            "pdf_image_infos": True,
            "native_text_noise": True,
            "native_raster_charts": True,
            "native_stacked_graphics": True,
            "native_path_graphics": True,
            "native_drawing_graphics": True,
            "native_isolated_graphics": True,
            "native_graphic_arbitration": True,
            "native_concentric_graphics": True,
            "native_raster_diagrams": True,
            "native_diagram_components": True,
            "native_strong_graphics": True,
            "native_table_page_commit": True,
            "native_code_containers": True,
            "native_index_containers": True,
            "native_form_selection": True,
            "native_form_runtime": True,
            "native_form_document_containers": True,
            "native_document_geometry_plan": True,
            "native_raw_page_marginals": True,
            "native_document_calibrated_pages": True,
            "native_post_container_text_roles": True,
            "native_early_footnotes": True,
            "native_complete_page_preparation": True,
            "native_document_auxiliary_preparation": True,
            "native_document_text_classification": True,
            "native_document_reference_preparation": True,
            "native_formula_block_recovery": True,
            "native_formula_input_preparation": True,
            "native_final_page_formula_preparation": True,
            "native_final_text_line_preparation": True,
            "native_text_block_assembly": True,
            "native_page_block_refinement": True,
            "native_visual_annotation_binding": True,
            "native_complete_grouped_order": True,
            "native_final_page_output": True,
            "native_final_text_spacing": True,
            "native_document_inline_materialization": True,
            "native_pdf_file_collection": True,
            "native_pdf_model_analysis": True,
            "native_pdf_host_route": True,
            "native_pdf_table_region": True,
            "native_visible_text_geometry": True,
            "native_pixel_glyph_weight": True,
            "native_container_table_page": True,
            "native_code_table_page": True,
            "native_closed_rule_pipeline": True,
            "native_structural_table_pipeline": True,
            "native_table_page_policies": True,
            "native_complete_table_candidates": True,
            "native_numeric_candidate_pipeline": True,
            "native_table_candidate_producers": True,
            "native_fixed_cjk_font_runtime": True,
            "native_rule_candidate_expansion": True,
            "native_table_footnotes": True,
            "native_rule_annotation_preparation": True,
            "native_prose_rule_conflicts": True,
            "native_table_candidate_merge": True,
            "native_table_replacements": True,
            "native_closed_grids": True,
            "native_filled_grid_tables": True,
            "native_graphic_captions": True,
            "native_bar_graphics": True,
            "native_bar_groups": True,
            "native_tiled_tables": True,
            "native_bar_geometry": True,
            "pdf_text_paint": True,
            "native_table_cell_script_roles": True,
            "native_table_script_preparation": True,
            "native_script_fraction_protection": True,
            "native_script_line_roles": True,
            "native_table_styled_cell": True,
            "docvortex_pdf_native_text_geometry": True,
            "native_table_aligned_numeric": True,
            "native_table_rule_band": True,
            "native_table_sparse_columns": True,
            "native_table_sparse_hybrid": True,
            "native_table_multiline": True,
            "native_table_banded_numeric": True,
            "native_table_spatial_projection": True,
            "legacy_native_table_recovery": False,
            "model_pdf_preview": True,
            "legacy_table_model_result": True,
            "legacy_table_execution": False,
            "legacy_model_formula_tasks": True,
            "formula_model_inference": True,
            "formula_full_ftfy": True,
            "formula_ftfy_version": "6.3.1",
            "formula_validated_models": ["PP-FormulaNet_plus-M"],
            "formula_preprocessing": True,
            "compatibility": {"python_parser": False, "document_quality_verified": False,
                              "document_performance_verified": False,
                              "experimental_requires_opt_in": ["organize_page_text", "recognize_pdf_page_hybrid"],
                              "native_tables_default": False},
            "facts": {"characters": True, "paths": True, "image_bounds": True, "uri_links": True}}


__all__ = ['__version__', 'capabilities', 'resolve_page_range', 'validate_rect', 'parse_pdf_text', 'PdfTextOutput', 'parse_pdf_ocr', 'PdfOcrOutput', 'RapidDocError', 'OnnxSession', 'ocr_png', 'ocr_pdf_page', 'OcrPageOutput', 'recover_pdf_table', 'RecoveredTable', 'render_pdf_page', 'RenderedPage', 'match_table', 'recognize_table_png', 'table_line_geometry', 'connect_table_lines', 'recover_table_polygons', 'match_wired_table', 'recognize_wired_table_png', 'recognize_table_auto_png', 'select_table_candidate', 'recognize_page_tables_png', 'recognize_pdf_page_tables', 'recognize_page_png', 'recognize_pdf_page', 'recognize_pdf_page_hybrid', 'merge_text_lines', 'organize_page_text', 'compatibility_report', 'split_paragraphs', 'sort_model_blocks', 'assemble_model_page', 'assemble_model_pages', 'render_model_pages', 'optimize_formula_numbers', 'assemble_model_page_with_assets', 'assemble_model_page_with_native_text', 'fill_model_text_spans', 'read_native_pdf_page', 'extract_original_pdf_images', 'assemble_pdf_model_page', 'convert_model_layout', 'filter_model_layout', 'predict_model_layout', 'prepare_model_crops', 'prepare_pdf_model_inference', 'derive_native_ocr_boxes', 'prepare_model_ocr', 'organize_model_detection_boxes', 'prepare_model_recognition', 'apply_model_ocr_recognition', 'execute_model_ocr', 'apply_model_formula_results', 'execute_model_formulas', 'prepare_formula_input', 'FormulaDecoder', 'repair_formula_latex', 'fix_formula_text', 'organize_table_detection_boxes', 'apply_table_ocr_results', 'apply_table_prediction', 'recognize_table_text_png', 'plan_table_direction', 'prepare_oriented_table_text_png', 'extract_table_native_text_png', 'recognize_table_text_stage_png', 'prepare_table_hints', 'prepare_table_model_payload_png', 'recognize_table_full_ocr_png', 'prepare_table_model_input_png', 'match_legacy_slanet_table', 'recognize_legacy_slanet_raw_png', 'normalize_legacy_table_html', 'recognize_legacy_slanet_table_png', 'parse_pdf_models', 'match_legacy_unet_table', 'recognize_legacy_unet_table_png', 'select_legacy_table_candidate', 'recognize_legacy_table_auto_png', 'plan_native_table_rule', 'accept_native_table_rule', 'select_native_table_candidates', 'try_native_table_rule', 'prepare_native_table_glyphs', 'build_native_table_text', 'build_native_table_cell_text', 'inspect_native_table_candidate', 'build_native_table_candidate', 'render_native_table_candidate', 'prepare_native_table_rule_evidence', 'canonicalize_native_table_tracks', 'build_native_table_vector_candidate', 'generate_native_table_vector_candidates', 'inspect_native_table_vector_tracks', 'generate_native_table_text_candidates', 'inspect_native_table_engine_policy', 'regroup_native_table_numeric_text', 'build_native_table_aligned_numeric_candidate', 'generate_native_table_rule_band_candidates', 'inspect_native_table_sparse_columns', 'generate_native_table_sparse_hybrid_candidates', 'generate_native_table_multiline_candidates', 'project_native_table_spatial_text', 'build_native_table_banded_numeric_candidate', 'evaluate_native_table_primitives', 'recover_native_table_primitives', 'diagnose_native_table_primitives', 'merge_cross_page_tables', 'apply_model_ocr_results', 'post_process_model_ocr', 'replace_inline_base64_img_src', 'read_pdf_vector_geometry', 'classify_pdf_font_name', 'pdfium_font_runtime_info', 'read_pdf_character_primitives', 'deduplicate_native_characters', 'read_pdf_native_text_geometry', 'classify_native_script_roles', 'native_numeric_script_indices', 'render_native_table_styled_cell', 'refine_native_script_tokens', 'classify_native_script_runs', 'classify_native_script_line', 'native_fraction_member_sources', 'measure_native_line_typography', 'prepare_native_table_script_cells', 'classify_native_table_script_line', 'infer_native_text_lanes', 'merge_native_inline_clusters', 'classify_native_table_script_cell', 'NativeTableRecoveryError', 'render_native_table_html_with_scripts', 'recover_native_table_result', 'prepare_native_table_block', 'prepare_native_table_annotations', 'restore_native_front_matter_panel', 'materialize_native_table_blocks', 'build_native_visual_lines', 'prepare_native_source_lines', 'prepare_native_image_regions', 'detect_native_inline_evidence', 'read_pdf_link_annotations', 'read_pdf_image_infos', 'prepare_native_text_noise', 'apply_native_text_noise', 'read_pdf_text_paint', 'filter_pdf_native_text_noise', 'group_native_raster_chart_descriptions', 'detect_native_stacked_graphics', 'detect_native_path_graphics', 'detect_native_drawing_graphics', 'detect_native_isolated_graphics', 'arbitrate_native_graphic_candidates', 'inspect_native_graphic_grids', 'detect_native_concentric_graphics', 'detect_native_raster_diagrams', 'detect_native_diagram_components', 'detect_native_strong_graphics', 'prepare_native_table_page', 'build_native_code_blocks', 'prepare_native_code_table_page', 'extract_native_index_blocks', 'select_native_form_regions', 'prepare_native_container_table_page', 'prepare_native_bar_geometry', 'apply_native_inline_evidence', 'prepare_native_script_lines', 'trace_native_inline_evidence', 'materialize_native_table_media_page', 'materialize_native_table_media', 'detect_native_tiled_tables', 'prepare_native_bar_groups', 'protect_native_graphic_captions', 'detect_native_bar_graphics', 'detect_native_filled_grid_tables', 'inspect_native_closed_grids', 'prefer_complete_native_table_replacements', 'merge_native_table_candidates', 'exclude_native_prose_spanning_rule_drafts', 'prepare_native_rule_table_annotations', 'collect_native_table_footnotes', 'expand_native_rule_table_candidate', 'inspect_native_rule_table_geometry', 'expand_native_connected_table_grids', 'detect_native_closed_rule_tables', 'detect_native_physical_table_candidates', 'detect_native_rule_table_candidates', 'detect_native_supplemental_tables', 'detect_native_structural_table_candidates', 'detect_native_table_candidates', 'apply_native_table_page_policies', 'trailing_native_table_grid', 'read_pdf_form_geometry', 'finish_native_geometry_page', 'prepare_native_document_pages', 'prepare_native_document_auxiliary', 'classify_native_document_auxiliary', 'prepare_native_document_text', 'classify_native_page_titles', 'infer_native_document_title_profile', 'classify_native_document_titles', 'prepare_native_document_references', 'classify_native_document_references', 'group_native_reference_lines', 'recover_native_formula_blocks', 'prepare_native_formula_input', 'prepare_native_page_formulas', 'prepare_native_document_formulas', 'prepare_native_final_page_lines', 'prepare_native_final_document_lines', 'build_native_text_blocks', 'prepare_native_final_page_blocks', 'prepare_native_final_document_blocks', 'prepare_native_spacing_lines', 'apply_native_spacing_lines', 'materialize_native_document_inline', 'finalize_native_document', 'collect_native_pdf_sources', 'analyze_native_pdf', 'read_pdf_visible_text_geometry', 'plan_native_glyph_weight', 'detect_native_glyph_weight', 'normalize_native_pdf_model_text', 'convert_native_pdf_layout', 'native_pdf_layout_fallback_reason', 'merge_native_pdf_supplements', 'native_pdf_table_text_angle', 'NativePdfTableRecoverer', 'RapidDoc', 'RapidDocOutput']


