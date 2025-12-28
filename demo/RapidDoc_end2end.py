import os
from pathlib import Path
# # 使用默认 GPU（cuda:0）
os.environ['MINERU_DEVICE_MODE'] = "cuda"
# # 或指定 GPU 编号，例如使用第二块 GPU（cuda:1）
# os.environ['MINERU_DEVICE_MODE'] = "cuda:1"
# # 模型文件存储目录
os.environ['RAPID_MODELS_DIR'] = r'/root/hzkitty/models' #模型文件存储目录，如果不设置会默认下载到rapid_doc项目里面
# os.environ['RAPID_MODELS_DIR'] = r'D:\CodeProjects\doc\RapidAI\models' #模型文件存储目录，如果不设置会默认下载到rapid_doc项目里面

from loguru import logger
import time
from rapid_doc.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, read_fn
from rapid_doc.data.data_reader_writer import FileBasedDataWriter
from rapid_doc.utils.enum_class import MakeMode
from rapid_doc.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from rapid_doc.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json

def do_parse(
    output_dir,
    pdf_file_names: list[str],
    pdf_bytes_list: list[bytes],
    parse_method="auto",
    formula_enable=True,
    table_enable=True,
    f_dump_md=True,
    f_make_md_mode=MakeMode.MM_MD,
    start_page_id=0,
    end_page_id=None,
):
    layout_config = {
    }

    ocr_config = {
    }

    formula_config = {
    }

    table_config = {
    }

    checkbox_config = {
    }

    image_config = {
    }
    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
        pdf_bytes_list[idx] = new_pdf_bytes

    infer_results, all_image_lists, all_page_dicts, lang_list, ocr_enabled_list = pipeline_doc_analyze(pdf_bytes_list, parse_method=parse_method, formula_enable=formula_enable,table_enable=table_enable
                                                                                                     ,layout_config=layout_config, ocr_config=ocr_config, formula_config=formula_config, table_config=table_config, checkbox_config=checkbox_config)

    def prepare_env(output_dir, pdf_file_name, parse_method):
        local_md_dir = output_dir
        local_image_dir = os.path.join(str(local_md_dir), "images")
        os.makedirs(local_image_dir, exist_ok=True)
        os.makedirs(local_md_dir, exist_ok=True)
        return local_image_dir, local_md_dir

    for idx, model_list in enumerate(infer_results):
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
        # image_writer = None # 不保存图片

        images_list = all_image_lists[idx]
        pdf_dict = all_page_dicts[idx]
        _lang = lang_list[idx]
        _ocr_enable = ocr_enabled_list[idx]
        middle_json = pipeline_result_to_middle_json(model_list, images_list, pdf_dict, image_writer, _lang, _ocr_enable, formula_enable, ocr_config=ocr_config, image_config=image_config)

        pdf_info = middle_json["pdf_info"]

        if f_dump_md:
            image_dir = str(os.path.basename(local_image_dir))
            md_content_str = pipeline_union_make(pdf_info, f_make_md_mode, image_dir)
            md_writer.write_string(
                f"{pdf_file_name}.md",
                md_content_str,
            )
        logger.info(f"local output dir is {local_md_dir}/{pdf_file_name}")


def parse_doc(
        path_list: list[Path],
        output_dir,
        method="auto",
        start_page_id=0,
        end_page_id=None
):
    try:
        file_name_list = []
        pdf_bytes_list = []
        for path in path_list:
            file_name = str(Path(path).stem)
            pdf_bytes = read_fn(path)
            file_name_list.append(file_name)
            pdf_bytes_list.append(pdf_bytes)
        do_parse(
            output_dir=output_dir,
            pdf_file_names=file_name_list,
            pdf_bytes_list=pdf_bytes_list,
            parse_method=method,
            start_page_id=start_page_id,
            end_page_id=end_page_id
        )
    except Exception as e:
        logger.exception(e)


if __name__ == '__main__':
    files_dir = r"/root/hzkitty/OmniDocBenchFiles/images"
    output_dir = r"/root/hzkitty/OmniDocBenchFiles/layout_plus_l-ocr_mobile-image"

    # files_dir = r"D:\Download\OmniDocBench\images"
    # output_dir = r"D:\Download\OmniDocBench\layout_plus_l-ocr_mobile-image"

    suffixes = [".pdf", ".png", ".jpg", ".jpeg"]
    batch_size = 100

    doc_path_list = []
    start_time = time.time()

    # 遍历 imgs 目录
    for doc_path in Path(files_dir).glob('*'):
        if doc_path.suffix.lower() in suffixes:
            doc_path_list.append(doc_path)

    # 按批次运行 parse_doc
    for i in range(0, len(doc_path_list), batch_size):
        batch = doc_path_list[i:i + batch_size]
        print(f"处理第 {i // batch_size + 1} 批，共 {len(batch)} 个文件")
        parse_doc(batch, output_dir)

    print(f"总运行时间: {time.time() - start_time} 秒")