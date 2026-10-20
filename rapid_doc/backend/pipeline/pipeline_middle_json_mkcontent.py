import re
from loguru import logger

from rapid_doc.utils.config_reader import get_latex_delimiter_config
from rapid_doc.backend.pipeline.para_split import ListLineTag
from rapid_doc.utils.enum_class import BlockType, ContentType, MakeMode
from rapid_doc.utils.guess_suffix_or_lang import guess_language_by_text
from rapid_doc.utils.text_boundary import resolve_text_line_boundary, remove_invalid_surrogates


def make_blocks_to_markdown(paras_of_layout,
                                      mode,
                                      img_buket_path='',
                                      ):
    page_markdown = []
    for para_block in paras_of_layout:
        para_text = ''
        para_type = para_block['type']
        if para_type in [BlockType.TEXT, BlockType.LIST, BlockType.INDEX]:
            para_text = merge_para_with_text(para_block)
        elif para_type == BlockType.TITLE:
            title_level = get_title_level(para_block)
            para_text = f'{"#" * title_level} {merge_para_with_text(para_block)}'
            para_text = f"{para_text}".replace("-\n", "").replace(
                "\n", " "
            )
        elif para_type == BlockType.INTERLINE_EQUATION:
            if len(para_block['lines']) == 0 or len(para_block['lines'][0]['spans']) == 0:
                continue
            if para_block['lines'][0]['spans'][0].get('content', ''):
                para_text = merge_para_with_text(para_block)
            else:
                para_text += f"![]({img_buket_path}/{para_block['lines'][0]['spans'][0]['image_path']})"
        elif para_type == BlockType.IMAGE:
            if mode == MakeMode.NLP_MD:
                continue
            elif mode == MakeMode.MM_MD:
                # 检测是否存在图片脚注
                has_image_footnote = any(block['type'] == BlockType.IMAGE_FOOTNOTE for block in para_block['blocks'])
                # 如果存在图片脚注，则将图片脚注拼接到图片正文后面
                if has_image_footnote:
                    for block in para_block['blocks']:  # 1st.拼image_caption
                        if block['type'] == BlockType.IMAGE_CAPTION:
                            para_text += merge_para_with_text(block) + '  \n'
                    for block in para_block['blocks']:  # 2nd.拼image_body
                        if block['type'] == BlockType.IMAGE_BODY:
                            for line in block['lines']:
                                for span in line['spans']:
                                    if span['type'] == ContentType.IMAGE:
                                        if span.get('image_path', ''):
                                            para_text += f"![]({img_buket_path}/{span['image_path']})"
                    for block in para_block['blocks']:  # 3rd.拼image_footnote
                        if block['type'] == BlockType.IMAGE_FOOTNOTE:
                            para_text += '  \n' + merge_para_with_text(block)
                else:
                    for block in para_block['blocks']:  # 1st.拼image_body
                        if block['type'] == BlockType.IMAGE_BODY:
                            for line in block['lines']:
                                for span in line['spans']:
                                    if span['type'] == ContentType.IMAGE:
                                        if span.get('image_path', ''):
                                            para_text += f"![]({img_buket_path}/{span['image_path']})"
                                        if 'seal' == span.get("original_label"):
                                            content = _get_seal_text(span)
                                            if content:
                                                para_text += f"  \n{content}"
                    for block in para_block['blocks']:  # 2nd.拼image_caption
                        if block['type'] == BlockType.IMAGE_CAPTION:
                            para_text += '  \n' + merge_para_with_text(block)
        elif para_type == BlockType.TABLE:
            if mode == MakeMode.NLP_MD:
                continue
            elif mode == MakeMode.MM_MD:
                for block in para_block['blocks']:  # 1st.拼table_caption
                    if block['type'] == BlockType.TABLE_CAPTION:
                        para_text += merge_para_with_text(block) + '  \n'
                for block in para_block['blocks']:  # 2nd.拼table_body
                    if block['type'] == BlockType.TABLE_BODY:
                        for line in block['lines']:
                            for span in line['spans']:
                                if span['type'] == ContentType.TABLE:
                                    # if processed by table model
                                    if span.get('html', ''):
                                        para_text += f"\n{span['html']}\n"
                                    elif span.get('image_path', ''):
                                        para_text += f"![]({img_buket_path}/{span['image_path']})"
                for block in para_block['blocks']:  # 3rd.拼table_footnote
                    if block['type'] == BlockType.TABLE_FOOTNOTE:
                        para_text += '\n' + merge_para_with_text(block) + '  '

        if para_text.strip() == '':
            continue
        else:
            # page_markdown.append(para_text.strip() + '  ')
            page_markdown.append(para_text.strip())

    return page_markdown


def full_to_half(text: str) -> str:
    """Convert full-width characters to half-width characters using code point manipulation.

    Args:
        text: String containing full-width characters

    Returns:
        String with full-width characters converted to half-width
    """
    result = []
    for char in text:
        code = ord(char)
        # Full-width letters and numbers (FF21-FF3A for A-Z, FF41-FF5A for a-z, FF10-FF19 for 0-9)
        if (0xFF21 <= code <= 0xFF3A) or (0xFF41 <= code <= 0xFF5A) or (0xFF10 <= code <= 0xFF19):
            result.append(chr(code - 0xFEE0))  # Shift to ASCII range
        else:
            result.append(char)
    return ''.join(result)

latex_delimiters_config = get_latex_delimiter_config()

default_delimiters = {
    'display': {'left': '$$', 'right': '$$'},
    'inline': {'left': '$', 'right': '$'}
}

delimiters = latex_delimiters_config if latex_delimiters_config else default_delimiters

display_left_delimiter = delimiters['display']['left']
display_right_delimiter = delimiters['display']['right']
inline_left_delimiter = delimiters['inline']['left']
inline_right_delimiter = delimiters['inline']['right']

def _inline_separator(previous, current):
    left, right = str(previous.get('content', '')), str(current.get('content', ''))
    if previous['type'] != ContentType.TEXT or current['type'] != ContentType.TEXT:
        return ' '
    if left.endswith((' ', '\t')) or right.startswith((' ', '\t')):
        return ' '
    first, second = previous.get('bbox'), current.get('bbox')
    if first and second:
        height = min(first[3] - first[1], second[3] - second[1])
        if height > 0:
            return ' ' if second[0] - first[2] > 0.15 * height else ''
    # No geometry: use visible boundary characters without rewriting a span.
    return resolve_text_line_boundary(left, next_content=right)[1]


def merge_para_with_text(para_block):
    if _is_algorithm_block(para_block):
        return _merge_algorithm_block(para_block)
    rendered, raw = '', ''
    previous_tail = None
    for line in para_block['lines']:
        line_rendered, line_raw = '', ''
        previous = None
        first = None
        for span in line['spans']:
            kind = span['type']
            value = remove_invalid_surrogates(str(span.get('content', '')))
            if kind == ContentType.TEXT:
                value = full_to_half(value)
                content = escape_special_markdown_char(value).strip()
            elif kind == ContentType.INLINE_EQUATION:
                content = f"{inline_left_delimiter}{value}{inline_right_delimiter}" if value else ''
            elif kind == ContentType.INTERLINE_EQUATION:
                content = f"\n{display_left_delimiter}\n{value}\n{display_right_delimiter}\n" if value else ''
            elif kind == ContentType.CHECKBOX:
                content = value.strip()
            else:
                continue
            if not content:
                continue
            normalized = {**span, 'content': value}
            separator = _inline_separator(previous, normalized) if previous else ''
            line_rendered += separator + content
            line_raw += separator + value.strip()
            first = first or normalized
            previous = normalized
        if not line_rendered:
            continue
        if rendered:
            if line.get(ListLineTag.IS_LIST_START_LINE, False):
                separator = '  \n'
                raw += '\n'
            elif previous_tail['type'] == ContentType.TEXT and first['type'] == ContentType.TEXT:
                adjusted, separator = resolve_text_line_boundary(raw, next_content=line_raw)
                if len(adjusted) < len(raw) and rendered.endswith('-'):
                    rendered = rendered[:-1]
                elif len(adjusted) < len(raw) and rendered[-1:] in {'\u00ad', '\u2010', '\u2011', '\u2043'}:
                    rendered = rendered[:-1]
                raw = adjusted
            else:
                separator = ' '
            rendered += separator
            raw += separator
        rendered += line_rendered
        raw += line_raw
        previous_tail = previous
    return rendered.strip()


def _merge_algorithm_block(para_block):
    code_lines = []
    for line in para_block.get('lines', []):
        line_text = ''
        for span in line.get('spans', []):
            span_type = span.get('type')
            if span_type == ContentType.TEXT:
                line_text += full_to_half(str(span.get('content', '')))
            elif span_type == ContentType.INLINE_EQUATION:
                line_text += str(span.get('content', ''))
            elif span_type == ContentType.INTERLINE_EQUATION:
                line_text += str(span.get('content', ''))
            elif span_type == ContentType.CHECKBOX:
                line_text += str(span.get('content', ''))
        code_lines.append(line_text.rstrip())

    code_text = '\n'.join(code_lines).strip('\n')
    if not code_text:
        return ''

    guess_lang = guess_language_by_text(code_text)
    return f"```{guess_lang}\n{code_text}\n```"


def _is_algorithm_block(para_block):
    if para_block.get('original_label') == BlockType.ALGORITHM:
        return True

    for line in para_block.get('lines', []):
        for span in line.get('spans', []):
            if span.get('original_label') == BlockType.ALGORITHM:
                return True
    return False


def make_blocks_to_content_list(para_block, img_buket_path, page_idx, page_size):
    para_type = para_block['type']
    para_content = {}
    if para_type in [BlockType.TEXT, BlockType.LIST, BlockType.INDEX]:
        para_content = {
            'type': ContentType.TEXT,
            'text': merge_para_with_text(para_block),
        }
    elif para_type == BlockType.DISCARDED:
        para_content = {
            'type': para_type,
            'text': merge_para_with_text(para_block),
        }
    elif para_type == BlockType.TITLE:
        para_content = {
            'type': ContentType.TEXT,
            'text': merge_para_with_text(para_block),
        }
        title_level = get_title_level(para_block)
        if title_level != 0:
            para_content['text_level'] = title_level
    elif para_type == BlockType.INTERLINE_EQUATION:
        if len(para_block['lines']) == 0 or len(para_block['lines'][0]['spans']) == 0:
            return None
        para_content = {
            'type': ContentType.EQUATION,
            'img_path': f"{img_buket_path}/{para_block['lines'][0]['spans'][0].get('image_path', '')}",
        }
        if para_block['lines'][0]['spans'][0].get('content', ''):
            para_content['text'] = merge_para_with_text(para_block)
            para_content['text_format'] = 'latex'
    elif para_type == BlockType.IMAGE:
        para_content = {'type': ContentType.IMAGE, 'img_path': '', BlockType.IMAGE_CAPTION: [], BlockType.IMAGE_FOOTNOTE: []}
        for block in para_block['blocks']:
            if block['type'] == BlockType.IMAGE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if 'seal' == span.get("original_label"):
                            para_content['text'] = _get_seal_text(span)
                        if span['type'] == ContentType.IMAGE:
                            if span.get('image_path', ''):
                                para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"
            if block['type'] == BlockType.IMAGE_CAPTION:
                para_content[BlockType.IMAGE_CAPTION].append(merge_para_with_text(block))
            if block['type'] == BlockType.IMAGE_FOOTNOTE:
                para_content[BlockType.IMAGE_FOOTNOTE].append(merge_para_with_text(block))
    elif para_type == BlockType.TABLE:
        para_content = {'type': ContentType.TABLE, 'img_path': '', BlockType.TABLE_CAPTION: [], BlockType.TABLE_FOOTNOTE: []}
        for block in para_block['blocks']:
            if block['type'] == BlockType.TABLE_BODY:
                for line in block['lines']:
                    for span in line['spans']:
                        if span['type'] == ContentType.TABLE:
                            if span.get('html', ''):
                                para_content[BlockType.TABLE_BODY] = f"{span['html']}"

                            if span.get('image_path', ''):
                                para_content['img_path'] = f"{img_buket_path}/{span['image_path']}"

            if block['type'] == BlockType.TABLE_CAPTION:
                para_content[BlockType.TABLE_CAPTION].append(merge_para_with_text(block))
            if block['type'] == BlockType.TABLE_FOOTNOTE:
                para_content[BlockType.TABLE_FOOTNOTE].append(merge_para_with_text(block))

    page_width, page_height = page_size
    para_bbox = para_block.get('bbox')
    if para_bbox:
        x0, y0, x1, y1 = para_bbox
        para_content['bbox'] = [
            int(x0 * 1000 / page_width),
            int(y0 * 1000 / page_height),
            int(x1 * 1000 / page_width),
            int(y1 * 1000 / page_height),
        ]

    para_content['page_idx'] = page_idx

    return para_content


def union_make(pdf_info_dict: list,
               make_mode: str,
               img_buket_path: str = '',
               ):
    output_content = []
    for page_info in pdf_info_dict:
        paras_of_layout = page_info.get('para_blocks')
        paras_of_discarded = page_info.get('discarded_blocks')
        page_idx = page_info.get('page_idx')
        page_size = page_info.get('page_size')
        if not paras_of_layout:
            continue
        if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
            if not paras_of_layout:
                continue
            page_markdown = make_blocks_to_markdown(paras_of_layout, make_mode, img_buket_path)
            output_content.extend(page_markdown)
        elif make_mode == MakeMode.CONTENT_LIST:
            para_blocks = (paras_of_layout or []) + (paras_of_discarded or [])
            if not para_blocks:
                continue
            for para_block in para_blocks:
                para_content = make_blocks_to_content_list(para_block, img_buket_path, page_idx, page_size)
                if para_content:
                    output_content.append(para_content)

    if make_mode in [MakeMode.MM_MD, MakeMode.NLP_MD]:
        return '\n\n'.join(output_content)
    elif make_mode == MakeMode.CONTENT_LIST:
        return output_content
    else:
        logger.error(f"Unsupported make mode: {make_mode}")
        return None


def get_title_level(block):
    title_level = block.get('level', 1)
    if title_level > 4:
        title_level = 4
    elif title_level < 1:
        title_level = 0
    return title_level

def _get_seal_text(seal_span):
    content = seal_span.get('content', '')
    if isinstance(content, list):
        return ' '.join(str(item) for item in content if str(item).strip())
    if isinstance(content, str):
        return content.strip()
    return ''


def escape_special_markdown_char(content):
    """
    转义正文里对markdown语法有特殊意义的字符
    """
    # special_chars = ["*", "`", "~", "$"]
    special_chars = ["*", "`", "~"] # vl模型识别文本的时候，公式用$包裹的
    for char in special_chars:
        content = content.replace(char, "\\" + char)

    return content
