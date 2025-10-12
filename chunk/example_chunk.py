import os
from pathlib import Path

from chunk.text_splitters import MarkdownTextSplitter
from chunk.get_bbox_page_fast import get_bbox_for_chunk, get_blocks_from_middle
from rapid_doc.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, prepare_env, read_fn
from rapid_doc.data.data_reader_writer import FileBasedDataWriter
from rapid_doc.utils.enum_class import MakeMode
from rapid_doc.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from rapid_doc.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json


def do_parse(path_list, output_dir):
    pdf_file_names = []
    pdf_bytes_list = []
    for path in path_list:
        file_name = str(Path(path).stem)
        pdf_bytes = read_fn(path)
        pdf_file_names.append(file_name)
        pdf_bytes_list.append(pdf_bytes)

    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes)
        pdf_bytes_list[idx] = new_pdf_bytes
    infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = pipeline_doc_analyze(pdf_bytes_list)

    result_list = []
    for idx, model_list in enumerate(infer_results):

        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, "auto")
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

        images_list = all_image_lists[idx]
        pdf_doc = all_pdf_docs[idx]
        _lang = lang_list[idx]
        _ocr_enable = ocr_enabled_list[idx]
        middle_json = pipeline_result_to_middle_json(model_list, images_list, pdf_doc, image_writer, _lang, _ocr_enable)

        pdf_info = middle_json["pdf_info"]
        image_dir = str(os.path.basename(local_image_dir))
        md_content_str = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)

        content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)

        result_list.append((middle_json, md_content_str, content_list, image_dir))
    return result_list

if __name__ == '__main__':
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(__dir__, "output")

    doc_path_list = [
        "D:\\file\\text-pdf\\示例1-论文模板.pdf",
    ]
    result_list = do_parse(doc_path_list, output_dir)

    for middle_json_content, markdown_document, content_list, _ in result_list:
        # 分块
        text_splitter = MarkdownTextSplitter(
            chunk_token_num=100, min_chunk_tokens=50
        )
        chunk_list = text_splitter.split_text(markdown_document)
        # 定位分块原始位置
        block_list = get_blocks_from_middle(middle_json_content)
        matched_global_indices = set()
        for i, chunk in enumerate(chunk_list):
            position_int_temp = get_bbox_for_chunk(chunk.strip(), block_list, matched_global_indices)
            print(position_int_temp)
