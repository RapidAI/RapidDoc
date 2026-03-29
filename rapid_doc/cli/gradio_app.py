# Copyright (c) Opendatalab. All rights reserved.

import base64
import html
import os
import re
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

import click
import gradio as gr
from dotenv import load_dotenv
from gradio_pdf import PDF
from loguru import logger

from rapid_doc.cli.common import prepare_env, read_fn, aio_do_parse, pdf_suffixes, image_suffixes, office_suffixes, \
    old_office_suffixes
from rapid_doc.utils.hash_utils import str_sha256
from rapid_doc.utils.office_converter import convert_legacy_office_to_modern
from rapid_doc.version import __version__

OFFICE_VIEWER_HEAD = """
<link rel="stylesheet" href="https://unpkg.com/jit-viewer@1.1.0/dist/iife/jit-viewer.min.css">
<script src="https://unpkg.com/jit-viewer@1.1.0/dist/iife/jit-viewer.min.js"></script>
<style>
  .office-preview-shell {
    display: flex;
    flex-direction: column;
    min-height: 800px;
    background: #fff;
    border: 1px solid #d9d9e3;
    border-radius: 12px;
    overflow: hidden;
  }
  .office-preview-host {
    min-height: 800px;
  }
  .office-preview-empty,
  .office-preview-error {
    padding: 24px 16px;
    color: #6b7280;
  }
  .office-preview-error {
    color: #b91c1c;
  }
  #viewer {
    position: relative;
    height: 800px;
  }

  .jv-viewer__branding[data-v-12fe006b] {
    position: absolute !important;
    left: 0;
    right: 0;
    bottom: 0 !important;
    top: auto !important;
    z-index: 10;
  }
</style>
<script>
(() => {
  function initOfficePreview(root) {
    if (!root) return;

    const viewerHost = root.querySelector("[data-office-viewer-host]");
    const errorHost = root.querySelector("[data-office-preview-error]");
    const fileUrl = root.dataset.fileUrl;
    const fileName = root.dataset.fileName || undefined;

    if (!viewerHost) return;

    if (!fileUrl) {
      viewerHost.innerHTML = "";
      if (errorHost) {
        errorHost.textContent = "";
      }
      root.dataset.initializedFileUrl = "";
      return;
    }

    if (root.dataset.initializedFileUrl === fileUrl) {
      return;
    }

    root.dataset.initializedFileUrl = fileUrl;
    viewerHost.innerHTML = "";
    if (errorHost) {
      errorHost.textContent = "";
    }

    if (!window.JitViewer || typeof window.JitViewer.createViewer !== "function") {
      if (errorHost) {
        errorHost.textContent = "jit-viewer loaded failed.";
      }
      return;
    }

    try {
      const { createViewer } = window.JitViewer;
      const viewer = createViewer({
        file: fileUrl,
        filename: fileName,
        toolbar: false,
        theme: "light",
        width: "100%",
        height: "800px",
        onError: (err) => {
          console.error("Office preview failed:", err);
          if (errorHost) {
            errorHost.textContent = "Office preview failed. See browser console for details.";
          }
        }
      });
      if (typeof viewer.mount === "function") {
        try {
          viewer.mount(viewerHost);
        } catch (mountErr) {
          if (!viewerHost.id) {
            viewerHost.id = `office-viewer-${Math.random().toString(36).slice(2, 10)}`;
          }
          viewer.mount();
        }
      }
    } catch (err) {
      console.error("Office preview failed:", err);
      if (errorHost) {
        errorHost.textContent = "Office preview failed. See browser console for details.";
      }
    }
  }

  function scanOfficePreviews() {
    document
      .querySelectorAll("[data-office-preview-root]")
      .forEach((root) => initOfficePreview(root));
  }

  window.__rapidDocInitOfficePreview = scanOfficePreviews;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scanOfficePreviews);
  } else {
    setTimeout(scanOfficePreviews, 0);
  }

  const observer = new MutationObserver(() => {
    window.requestAnimationFrame(scanOfficePreviews);
  });
  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["data-file-url", "data-file-name"],
  });
})();
</script>
"""


async def parse_doc(doc_path, output_dir, end_page_id, is_ocr, formula_enable, table_enable, language, backend, url):
    os.makedirs(output_dir, exist_ok=True)

    try:
        file_name = f'{safe_stem(Path(doc_path).stem)}_{time.strftime("%y%m%d_%H%M%S")}'
        pdf_data = read_fn(doc_path)
        # 检测 office 文件类型
        file_suffix = Path(doc_path).suffix.lower().lstrip('.')
        if file_suffix in office_suffixes:
            # office 文件使用固定的 env_name，parse_method 设为默认值（aio_do_parse 内部会处理）
            env_name = "office"
            parse_method = "auto"
        else:
            parse_method = 'ocr' if is_ocr else 'auto'
            env_name = parse_method

        local_image_dir, local_md_dir = prepare_env(output_dir, file_name, env_name)
        await aio_do_parse(
            output_dir=output_dir,
            pdf_file_names=[file_name],
            pdf_bytes_list=[pdf_data],
            p_lang_list=[language],
            parse_method=parse_method,
            end_page_id=end_page_id,
            formula_enable=formula_enable,
            table_enable=table_enable,
            backend=backend,
            server_url=url,
        )
        return local_md_dir, file_name
    except Exception as e:
        logger.exception(e)
        return None


def compress_directory_to_zip(directory_path, output_zip_path):
    """压缩指定目录到一个 ZIP 文件。

    :param directory_path: 要压缩的目录路径
    :param output_zip_path: 输出的 ZIP 文件路径
    """
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

            # 遍历目录中的所有文件和子目录
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    # 构建完整的文件路径
                    file_path = os.path.join(root, file)
                    # 计算相对路径
                    arcname = os.path.relpath(file_path, directory_path)
                    # 添加文件到 ZIP 文件
                    zipf.write(file_path, arcname)
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def replace_image_with_base64(markdown_text, image_dir_path):
    # 匹配Markdown中的图片标签
    pattern = r'\!\[(?:[^\]]*)\]\(([^)]+)\)'

    # 替换图片链接
    def replace(match):
        relative_path = match.group(1)
        # 只处理以.jpg .png结尾的图片
        if relative_path.endswith('.jpg'):
            full_path = os.path.join(image_dir_path, relative_path)
            base64_image = image_to_base64(full_path)
            return f'![{relative_path}](data:image/jpeg;base64,{base64_image})'
        elif relative_path.endswith('.png'):
            full_path = os.path.join(image_dir_path, relative_path)
            base64_image = image_to_base64(full_path)
            return f'![{relative_path}](data:image/png;base64,{base64_image})'
        else:
            # 其他格式的图片保持原样
            return match.group(0)
    # 应用替换
    return re.sub(pattern, replace, markdown_text)


async def to_markdown(file_path, end_pages=10, is_ocr=False, formula_enable=True, table_enable=True, language="ch", backend="pipeline", url=None):
    if file_path is None:
        return None, None, None, *build_preview_updates(None)

    file_suffix = Path(file_path).suffix[1:].lower()
    if file_suffix in old_office_suffixes:
        file_path = convert_legacy_office_to_modern(file_path)
    # office 文件不需要转换为 PDF，直接使用原始文件路径
    file_suffix = Path(file_path).suffix.lower().lstrip('.')
    is_office = file_suffix in office_suffixes
    if is_office:
        parse_file_path = file_path
    else:
        parse_file_path = to_pdf(file_path)
    # 获取识别的md文件以及压缩包文件路径
    local_md_dir, file_name = await parse_doc(parse_file_path, '../../docker/output', end_pages - 1, is_ocr, formula_enable, table_enable, language, backend, url)
    archive_zip_path = os.path.join('../../docker/output', str_sha256(local_md_dir) + '.zip')
    zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
    if zip_archive_success == 0:
        logger.info('Compression successful')
    else:
        logger.error('Compression failed')
    md_path = os.path.join(local_md_dir, file_name + '.md')
    with open(md_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()
    md_content = replace_image_with_base64(txt_content, local_md_dir)
    # office 文件没有 layout PDF，返回 None；其他格式返回转换后的 PDF 路径
    if is_office:
        preview_path = resolve_preview_office_path(local_md_dir, file_name)
    else:
        preview_path = resolve_preview_pdf_path(local_md_dir, file_name)

    if not os.path.exists(archive_zip_path):
        archive_zip_path = None

    pdf_preview_update, office_preview_update = build_preview_updates(preview_path)
    return md_content, txt_content, archive_zip_path, pdf_preview_update, office_preview_update


def resolve_preview_pdf_path(local_md_dir, file_name):
    layout_pdf_path = os.path.join(local_md_dir, file_name + '_layout.pdf')
    if os.path.exists(layout_pdf_path):
        return layout_pdf_path

    origin_pdf_path = os.path.join(local_md_dir, file_name + '_origin.pdf')
    if os.path.exists(origin_pdf_path):
        logger.warning(
            f"Layout preview PDF not found for {file_name}, "
            f"falling back to origin PDF: {origin_pdf_path}"
        )
        return origin_pdf_path

    logger.warning(f"No preview PDF found for {file_name} under {local_md_dir}")
    return None


def resolve_preview_office_path(local_md_dir, file_name):
    for office_suffix in office_suffixes:
        origin_office_path = os.path.join(local_md_dir, file_name + f'_origin.{office_suffix}')
        if os.path.exists(origin_office_path):
            return origin_office_path

    logger.warning(f"No preview office file found for {file_name} under {local_md_dir}")
    return None


def build_gradio_file_url(file_path):
    return f"/gradio_api/file={quote(os.path.abspath(file_path), safe='')}"


def render_office_preview_html(file_path=None):
    if not file_path:
        return (
            '<div class="office-preview-shell" data-office-preview-root>'
            '<div class="office-preview-empty">Upload a DOCX, PPTX, or XLSX file to preview it here.</div>'
            '</div>'
        )

    abs_path = os.path.abspath(file_path)
    file_url = html.escape(build_gradio_file_url(abs_path), quote=True)
    file_name = html.escape(os.path.basename(abs_path), quote=True)
    return f"""
<div class="office-preview-shell" data-office-preview-root data-file-url="{file_url}" data-file-name="{file_name}">
  <div class="office-preview-host" data-office-viewer-host></div>
  <div class="office-preview-error" data-office-preview-error></div>
</div>
"""


def build_preview_updates(file_path):
    if file_path is None:
        return (
            gr.update(value=None, visible=True),
            gr.update(value=render_office_preview_html(), visible=False),
        )

    file_suffix = Path(file_path).suffix.lower().lstrip('.')
    if file_suffix in office_suffixes:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=render_office_preview_html(file_path), visible=True),
        )

    if file_suffix in pdf_suffixes:
        preview_pdf_path = file_path
    else:
        # preview_pdf_path = to_pdf(file_path)
        preview_pdf_path = None


    return (
        gr.update(value=preview_pdf_path, visible=True),
        gr.update(value=render_office_preview_html(), visible=False),
    )

latex_delimiters_type_a = [
    {'left': '$$', 'right': '$$', 'display': True},
    {'left': '$', 'right': '$', 'display': False},
]
latex_delimiters_type_b = [
    {'left': '\\(', 'right': '\\)', 'display': False},
    {'left': '\\[', 'right': '\\]', 'display': True},
]
latex_delimiters_type_all = latex_delimiters_type_a + latex_delimiters_type_b

header_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'header.html')

with open(header_path, 'r') as header_file:
    header = header_file.read()

all_lang = ['ch']

def safe_stem(file_path):
    stem = Path(file_path).stem
    # 只保留字母、数字、下划线和点，其他字符替换为下划线
    return re.sub(r'[^\w.]', '_', stem)


def to_pdf(file_path):

    if file_path is None:
        return None

    pdf_bytes = read_fn(file_path)

    # unique_filename = f'{uuid.uuid4()}.pdf'
    unique_filename = f'{safe_stem(file_path)}.pdf'

    # 构建完整的文件路径
    tmp_file_path = os.path.join(os.path.dirname(file_path), unique_filename)

    # 将字节数据写入文件
    with open(tmp_file_path, 'wb') as tmp_pdf_file:
        tmp_pdf_file.write(pdf_bytes)

    return tmp_file_path


def update_file_preview(file_path):
    return build_preview_updates(file_path)


# 更新界面函数
def update_interface(backend_choice):
    if backend_choice in ["pipeline"]:
        return gr.update(visible=False), gr.update(visible=True)
    else:
        pass

@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.option(
    '--enable-example',
    'example_enable',
    type=bool,
    help="Enable example files for input."
         "The example files to be input need to be placed in the `example` folder within the directory where the command is currently executed.",
    default=True,
)
@click.option(
    '--enable-vllm-engine',
    'vllm_engine_enable',
    type=bool,
    help="Enable vLLM engine backend for faster processing.",
    default=False,
)
@click.option(
    '--enable-api',
    'api_enable',
    type=bool,
    help="Enable gradio API for serving the application.",
    default=True,
)
@click.option(
    '--max-convert-pages',
    'max_convert_pages',
    type=int,
    help="Set the maximum number of pages to convert from PDF to Markdown.",
    default=1000,
)
@click.option(
    '--server-name',
    'server_name',
    type=str,
    help="Set the server name for the Gradio app.",
    default=None,
)
@click.option(
    '--server-port',
    'server_port',
    type=int,
    help="Set the server port for the Gradio app.",
    default=None,
)
@click.option(
    '--latex-delimiters-type',
    'latex_delimiters_type',
    type=click.Choice(['a', 'b', 'all']),
    help="Set the type of LaTeX delimiters to use in Markdown rendering:"
         "'a' for type '$', 'b' for type '()[]', 'all' for both types.",
    default='all',
)

def main(ctx,
        example_enable, vllm_engine_enable, api_enable, max_convert_pages,
        server_name, server_port, latex_delimiters_type, **kwargs
):

    if latex_delimiters_type == 'a':
        latex_delimiters = latex_delimiters_type_a
    elif latex_delimiters_type == 'b':
        latex_delimiters = latex_delimiters_type_b
    elif latex_delimiters_type == 'all':
        latex_delimiters = latex_delimiters_type_all
    else:
        raise ValueError(f"Invalid latex delimiters type: {latex_delimiters_type}.")

    # suffixes = [f".{suffix}" for suffix in pdf_suffixes + image_suffixes]
    suffixes = [f".{suffix}" for suffix in pdf_suffixes + image_suffixes + office_suffixes + old_office_suffixes]
    with gr.Blocks(head=OFFICE_VIEWER_HEAD) as demo:
        gr.HTML(header)
        with gr.Row():
            with gr.Column(variant='panel', scale=5):
                with gr.Row():
                    input_file = gr.File(label='Please upload a PDF, image, or Office file', file_types=suffixes)
                with gr.Row():
                    max_pages = gr.Slider(1, max_convert_pages, int(max_convert_pages/2), step=1, label=f'Max convert pages  v{__version__}')
                with gr.Row():
                    drop_list = ["pipeline"]
                    preferred_option = "pipeline"
                    backend = gr.Dropdown(drop_list, label="Backend", value=preferred_option)
                with gr.Row(visible=False) as client_options:
                    url = gr.Textbox(label='Server URL', value='http://localhost:30000', placeholder='http://localhost:30000')
                with gr.Row(equal_height=True):
                    with gr.Column():
                        gr.Markdown("**Recognition Options:**")
                        formula_enable = gr.Checkbox(label='Enable formula recognition', value=True)
                        table_enable = gr.Checkbox(label='Enable table recognition', value=True)
                    with gr.Column(visible=False) as ocr_options:
                        language = gr.Dropdown(all_lang, label='Language', value='ch')
                        is_ocr = gr.Checkbox(label='Force enable OCR', value=False)
                with gr.Row():
                    change_bu = gr.Button('Convert')
                    clear_bu = gr.ClearButton(value='Clear')
                pdf_show = PDF(label='PDF preview', interactive=False, visible=True, height=800)
                office_show = gr.HTML(
                    label='Office preview',
                    value=render_office_preview_html(),
                    visible=False,
                    container=True,
                    min_height=800,
                )
                if example_enable:
                    example_root = os.path.join(os.getcwd(), 'examples')
                    if os.path.exists(example_root):
                        with gr.Accordion('Examples:'):
                            gr.Examples(
                                examples=[os.path.join(example_root, _) for _ in os.listdir(example_root) if
                                          _.endswith(tuple(suffixes))],
                                inputs=input_file
                            )

            with gr.Column(variant='panel', scale=5):
                output_file = gr.File(label='convert result', interactive=False)
                with gr.Tabs():
                    with gr.Tab('Markdown rendering'):
                        md = gr.Markdown(label='Markdown rendering', height=1100, show_copy_button=True,
                                         latex_delimiters=latex_delimiters,
                                         line_breaks=True)
                    with gr.Tab('Markdown text'):
                        md_text = gr.TextArea(lines=45, show_copy_button=True)

        # 添加事件处理
        backend.change(
            fn=update_interface,
            inputs=[backend],
            outputs=[client_options, ocr_options],
            api_name=False
        )
        # 添加demo.load事件，在页面加载时触发一次界面更新
        demo.load(
            fn=update_interface,
            inputs=[backend],
            outputs=[client_options, ocr_options],
            api_name=False
        )
        clear_bu.add([input_file, md, pdf_show, office_show, md_text, output_file, is_ocr])

        if api_enable:
            api_name = None
        else:
            api_name = False

        input_file.change(
            fn=update_file_preview,
            inputs=input_file,
            outputs=[pdf_show, office_show],
            api_name=api_name
        )
        change_bu.click(
            fn=to_markdown,
            inputs=[input_file, max_pages, is_ocr, formula_enable, table_enable, language, backend, url],
            outputs=[md, md_text, output_file, pdf_show, office_show],
            api_name=api_name
        )

    output_root = os.path.abspath("../../docker/output")

    demo.launch(server_name=server_name, server_port=server_port, show_api=api_enable, allowed_paths=[output_root])


if __name__ == '__main__':
    load_dotenv()
    main()