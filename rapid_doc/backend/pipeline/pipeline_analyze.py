import os
import time
from typing import List, Tuple
from PIL import Image
from loguru import logger

from .model_init import MineruPipelineModel
from ...utils.config_reader import get_device
from ...utils.enum_class import ImageType
from ...utils.hash_utils import make_hashable
from ...utils.pdf_classify import classify
from ...utils.pdf_image_tools import load_images_from_pdf, get_ori_image
from ...utils.model_utils import get_vram, clean_memory
from ...utils.pdf_text_tool import get_page

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 让mps可以fallback
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'  # 禁止albumentations检查更新

class ModelSingleton:
    _instance = None
    _models = {}

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(
        self,
        lang=None,
        formula_enable=None,
        table_enable=None,
        layout_config=None,
        ocr_config=None,
        formula_config=None,
        table_config=None,
    ):
        key = (lang, formula_enable, table_enable, make_hashable(layout_config), make_hashable(ocr_config), make_hashable(formula_config), make_hashable(table_config))
        if key not in self._models:
            self._models[key] = custom_model_init(
                lang=lang,
                formula_enable=formula_enable,
                table_enable=table_enable,
                layout_config=layout_config,
                ocr_config=ocr_config,
                formula_config=formula_config,
                table_config=table_config,
            )
        return self._models[key]


def custom_model_init(
    lang=None,
    formula_enable=True,
    table_enable=True,
    layout_config=None,
    ocr_config=None,
    formula_config=None,
    table_config=None,
):
    model_init_start = time.time()
    # 从配置文件读取model-dir和device
    device = get_device()

    final_formula_config = {"enable": formula_enable}
    if formula_config is not None:
        final_formula_config.update(formula_config)  # 合并传入的配置
    final_table_config = {"enable": table_enable}
    if table_config is not None:
        final_table_config.update(table_config)  # 合并传入的配置

    model_input = {
        'device': device,
        'layout_config': layout_config,
        'ocr_config': ocr_config,
        'table_config': final_table_config,
        'formula_config': final_formula_config,
        'lang': lang,
    }

    custom_model = MineruPipelineModel(**model_input)

    model_init_cost = time.time() - model_init_start
    logger.info(f'model init cost: {model_init_cost}')

    return custom_model


def doc_analyze(
        pdf_bytes_list,
        lang_list: list[str] = None,
        parse_method: str = 'auto',
        formula_enable=True,
        table_enable=True,
        layout_config=None,
        ocr_config=None,
        formula_config=None,
        table_config=None,
        checkbox_config=None,
):
    """
    适当调大MIN_BATCH_INFERENCE_SIZE可以提高性能，更大的 MIN_BATCH_INFERENCE_SIZE会消耗更多内存，
    可通过环境变量MINERU_MIN_BATCH_INFERENCE_SIZE设置，默认值为384。
    """
    if lang_list is None:
        lang_list = ["ch"] * len(pdf_bytes_list)
    min_batch_inference_size = int(os.environ.get('MINERU_MIN_BATCH_INFERENCE_SIZE', 384))

    # 收集所有页面信息
    all_pages_info = []  # 存储(dataset_index, page_index, img, ocr, lang, width, height)

    all_image_lists = []
    all_pdf_docs = []
    ocr_enabled_list = []
    for pdf_idx, pdf_bytes in enumerate(pdf_bytes_list):
        # 确定OCR设置
        _ocr_enable = False
        if parse_method == 'auto':
            if classify(pdf_bytes) == 'ocr':
                _ocr_enable = True
        elif parse_method == 'ocr':
            _ocr_enable = True

        ocr_enabled_list.append(_ocr_enable)
        _lang = lang_list[pdf_idx]

        # 收集每个数据集中的页面
        images_list, pdf_doc_list = load_images_from_pdf(pdf_bytes, image_type=ImageType.PIL)
        all_image_lists.append(images_list)

        all_pdf_dict = []
        for pdf_doc in pdf_doc_list:
            # 获取pdf的文字和图片的字典对象
            page_dict = get_page(pdf_doc)
            if page_dict['blocks']:
                page_dict['ori_image_list'] = get_ori_image(pdf_doc) # 从 PDF 中提取所有原始图片
            else:
                page_dict['ori_image_list'] = [] # 提取不到文字视为扫描版，不需要提取图片
            pdf_doc.close()
            all_pdf_dict.append(page_dict)
        all_pdf_docs.append(all_pdf_dict)
        for page_idx in range(len(images_list)):
            img_dict = images_list[page_idx]
            all_pages_info.append((
                pdf_idx, page_idx,
                img_dict['img_pil'], img_dict['scale'], _ocr_enable, _lang,
            ))

    # 准备批处理
    images_with_extra_info = [(info[2], info[3], info[4], info[5], all_pdf_docs[info[0]][info[1]]) for info in all_pages_info]
    batch_size = min_batch_inference_size
    batch_images = [
        images_with_extra_info[i:i + batch_size]
        for i in range(0, len(images_with_extra_info), batch_size)
    ]

    # 执行批处理
    results = []
    processed_images_count = 0
    for index, batch_image in enumerate(batch_images):
        processed_images_count += len(batch_image)
        logger.info(
            f'Batch {index + 1}/{len(batch_images)}: '
            f'{processed_images_count} pages/{len(images_with_extra_info)} pages'
        )
        batch_results = batch_image_analyze(batch_image, formula_enable, table_enable, layout_config, ocr_config, formula_config, table_config, checkbox_config)
        results.extend(batch_results)

    # 构建返回结果
    infer_results = []

    for _ in range(len(pdf_bytes_list)):
        infer_results.append([])

    for i, page_info in enumerate(all_pages_info):
        pdf_idx, page_idx, pil_img, _, _, _ = page_info
        result = results[i]

        page_info_dict = {'page_no': page_idx, 'width': pil_img.width, 'height': pil_img.height}
        page_dict = {'layout_dets': result, 'page_info': page_info_dict}

        infer_results[pdf_idx].append(page_dict)

    return infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list


def batch_image_analyze(
        images_with_extra_info: List[Tuple[Image.Image, float, bool, str, dict]],
        formula_enable=True,
        table_enable=True,
        layout_config=None,
        ocr_config=None,
        formula_config=None,
        table_config=None,
        checkbox_config=None,):

    from .batch_analyze import BatchAnalyze

    model_manager = ModelSingleton()

    batch_ratio = 1
    device = get_device()

    if str(device).startswith('npu'):
        try:
            import torch_npu
            if torch_npu.npu.is_available():
                torch_npu.npu.set_compile_mode(jit_compile=False)
        except Exception as e:
            raise RuntimeError(
                "NPU is selected as device, but torch_npu is not available. "
                "Please ensure that the torch_npu package is installed correctly."
            ) from e

    if str(device).startswith('npu') or str(device).startswith('cuda'):
        vram = get_vram(device)
        if vram is not None:
            gpu_memory = int(os.getenv('MINERU_VIRTUAL_VRAM_SIZE', round(vram)))
            if gpu_memory >= 16:
                batch_ratio = 16
            elif gpu_memory >= 12:
                batch_ratio = 8
            elif gpu_memory >= 8:
                batch_ratio = 4
            elif gpu_memory >= 6:
                batch_ratio = 2
            else:
                batch_ratio = 1
            logger.info(f'gpu_memory: {gpu_memory} GB, batch_ratio: {batch_ratio}')
        else:
            # Default batch_ratio when VRAM can't be determined
            batch_ratio = 1
            logger.info(f'Could not determine GPU memory, using default batch_ratio: {batch_ratio}')

    enable_ocr_det_batch = True
    batch_model = BatchAnalyze(model_manager, batch_ratio, formula_enable, table_enable, enable_ocr_det_batch, layout_config, ocr_config, formula_config, table_config, checkbox_config)
    results = batch_model(images_with_extra_info)

    clean_memory(get_device())

    return results