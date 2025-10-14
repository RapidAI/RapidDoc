import json
import time
from rapidfuzz import fuzz
from tqdm import tqdm

from chunk.text_splitters import MarkdownTextSplitter

def get_blocks_from_middle(middle_json_content):
    try:
        data = middle_json_content
        block_list = []

        for page_idx, page in enumerate(data['pdf_info']):
            for block in page['preproc_blocks']:
                bbox = block.get('bbox')
                if not bbox:
                    continue

                # 提取文本内容
                text_content = ''
                if 'lines' in block:
                    for line in block['lines']:
                        if 'spans' in line:
                            for span in line['spans']:
                                if 'content' in span:
                                    text_content += span['content']

                block_data = {
                    'bbox': bbox,
                    'content': text_content.strip(),
                    'page_number': page_idx,
                }
                block_list.append(block_data)

        return block_list
    except Exception as e:
        print(f"[ERROR] 获取块列表失败: {e}")
        return []

def get_bbox_for_chunk(chunk_content, block_list=None, matched_global_indices=None):
    """
    根据 chunk 内容，返回构成该 chunk 的连续 block 的 bbox 列表。
    采用 rapidfuzz 找出最相似的 block（相似度最高），
    然后从该锚点向前后扩展，寻找同样存在于 chunk 中的连续 block。
    支持外部传入 block_list，避免重复解析。
    支持Pipeline模式的数据结构。
    匹配到的块会通过 matched_global_indices 记录，避免后续 chunk 重复匹配。
    """
    try:
        if matched_global_indices is None:
            matched_global_indices = set()
        if not block_list:
            # print(f"[WARNING] 无法获取块列表，跳过位置信息获取")
            return None

        chunk_content_clean = chunk_content.strip()
        if not chunk_content_clean:
            return None

        # 用 rapidfuzz 找最相似的 block
        best_idx = -1
        best_ratio = 0.0
        for i, block in enumerate(block_list):
            if i in matched_global_indices:
                continue
            block_text = block.get('content', '').strip()
            if not block_text:
                continue
            ratio = fuzz.ratio(chunk_content_clean, block_text) / 100.0

            if ratio > best_ratio:
                best_ratio = ratio
                best_idx = i
                if best_ratio > 0.95:  # 提前停止（相似度极高）
                    break
        if best_idx == -1 or best_ratio < 0.1:  # 阈值可调整
            # print(f"[WARNING] 未找到足够相似的块 (最高相似度: {best_ratio:.3f})")
            return None

        # 从锚点扩展
        matched_indices = [best_idx]
        # 向前扩展
        for i in range(best_idx - 1, -1, -1):
            if i in matched_global_indices:
                continue
            block_text = block_list[i].get('content', '').strip()
            if block_text and block_text in chunk_content_clean:
                matched_indices.insert(0, i)
            else:
                break
        # 向后扩展
        for i in range(best_idx + 1, len(block_list)):
            if i in matched_global_indices:
                continue
            block_text = block_list[i].get('content', '').strip()
            if block_text and block_text in chunk_content_clean:
                matched_indices.append(i)
            else:
                break
        # 提取位置信息
        positions = []
        for idx in matched_indices:
            block = block_list[idx]
            bbox = block.get('bbox')
            page_number = block.get('page_number')
            if bbox and page_number is not None:
                position = [page_number, bbox[0], bbox[2], bbox[1], bbox[3]]
                positions.append(position)
        # 记录已匹配 block 索引
        matched_global_indices.update(matched_indices)
        if positions:
            # print(f"[INFO] 为chunk找到{len(positions)}个位置（最高相似度: {best_ratio:.3f}），并已记录 matched_global_indices")
            return positions
        else:
            print(f"[WARNING] 未能提取到有效的位置信息")
            return None
    except Exception as e:
        print(f"[ERROR] 获取chunk位置失败: {e}")
        return None

if __name__ == '__main__':

    # 字符文本分割器（markdown + 递归）
    markdown_path = r"D:\CodeProjects\doc\RapidAI\RapidDoc\output888\1 - 副本 (3)\auto\1 - 副本 (3).md"

    with open(markdown_path, 'r', encoding='utf-8') as f:
        markdown_document = f.read()
    start_time0 = time.time()
    smart_text_splitter = MarkdownTextSplitter(
        chunk_token_num=512, min_chunk_tokens=50
    )
    print(f"分块时间: {time.time() - start_time0}秒")
    chunk_list = smart_text_splitter.split_text(markdown_document)

    mineru_middle_path = r"D:\CodeProjects\doc\RapidAI\RapidDoc\output888\1 - 副本 (3)\auto\1 - 副本 (3)_middle.json"
    with open(mineru_middle_path, 'r', encoding='utf-8') as f:
        middle_json_content = json.load(f)
        print(position_int_temp)
    start_time = time.time()
    block_list = get_blocks_from_middle(middle_json_content)
    matched_global_indices = set()
    for chunk in tqdm(chunk_list, desc="Chunk-position Predict"):
        position_int_temp = get_bbox_for_chunk(chunk.strip(), block_list, matched_global_indices)
    print(f"总运行时间: {time.time() - start_time}秒")
