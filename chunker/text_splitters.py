#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import os
import re
import tiktoken
import tempfile
from markdown import markdown as md_to_html
from markdown_it import MarkdownIt
from markdown_it.tree import SyntaxTreeNode


# 设置tiktoken缓存目录，优先使用环境变量，否则使用默认路径
tiktoken_cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR", tempfile.gettempdir())
# 确保缓存目录存在
os.makedirs(tiktoken_cache_dir, exist_ok=True)
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
# encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
encoder = tiktoken.get_encoding("cl100k_base")
HTML_TABLE_PATTERN = re.compile(r"(?is)<table\b.*?</table>")
HTML_TABLE_ROW_PATTERN = re.compile(r"(?is)<tr\b.*?</tr>\s*")
HTML_TABLE_CELL_PATTERN = re.compile(r"(?is)<t[hd]\b.*?</t[hd]>\s*")

def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    try:
        return len(encoder.encode(string))
    except Exception:
        return 0

class MarkdownTextSplitter:

    def __init__(self, chunk_token_num=512, min_chunk_tokens=50, max_table_tokens=8000, char_max_length=60000, max_tokens=None) -> None:
        """Create a new TextSplitter."""
        self.chunk_token_num = chunk_token_num
        self.min_chunk_tokens = min_chunk_tokens
        self.max_table_tokens = max_table_tokens
        self.char_max_length = char_max_length
        self.max_tokens = chunk_token_num * 2 if max_tokens is None else max_tokens

    def split_text(self, txt) -> list[str]:
        """
        基于 markdown-it-py AST 的智能分块方法，解决 RAG Markdown 文件分块问题：
        1. 基于语义切分（使用 AST）
        2. 维护表格完整性，即使超出了最大 tokens
        3. 考虑 markdown 父子分块关系
        """
        if not txt or not txt.strip():
            return []

        # 初始化 markdown-it 解析器
        md = MarkdownIt("commonmark", {"breaks": True, "html": True})
        md.enable(['table'])


        # 解析为 AST
        tokens = md.parse(txt)
        tree = SyntaxTreeNode(tokens)

        # 基于 AST 进行智能分块
        chunks = []
        current_chunk = []
        current_tokens = 0
        context_stack = []  # 维护标题层级栈

        for block in self._merge_html_table_blocks(tree.children):
            if isinstance(block, str):
                chunk_data = block
                should_break = True
            else:
                chunk_data, should_break = self._process_ast_node(
                    block, context_stack, self.chunk_token_num, self.min_chunk_tokens
                )

            if should_break and current_chunk and current_tokens >= self.min_chunk_tokens:
                # 完成当前块
                chunk_content = self._finalize_ast_chunk(current_chunk, context_stack)
                if chunk_content.strip():
                    chunks.extend(self._force_split_if_oversize(chunk_content))
                current_chunk = []
                current_tokens = 0

            if chunk_data:
                # 兼容：html_block 返回的是多个 segment
                if isinstance(chunk_data, list):
                    for seg in chunk_data:
                        seg_tokens = num_tokens_from_string(seg)
                        # 仍然遵循 chunk_token_num 限制
                        if (current_tokens + seg_tokens > self.chunk_token_num and
                                current_chunk and current_tokens >= self.min_chunk_tokens):

                            chunk_content = self._finalize_ast_chunk(current_chunk, context_stack)
                            if chunk_content.strip():
                                chunks.extend(self._force_split_if_oversize(chunk_content))
                            current_chunk = []
                            current_tokens = 0
                        current_chunk.append(seg)
                        current_tokens += seg_tokens
                    continue
                chunk_tokens = num_tokens_from_string(chunk_data)

                # 检查是否需要分块
                if (current_tokens + chunk_tokens > self.chunk_token_num and
                        current_chunk and current_tokens >= self.min_chunk_tokens):

                    chunk_content = self._finalize_ast_chunk(current_chunk, context_stack)
                    if chunk_content.strip():
                        chunks.extend(self._force_split_if_oversize(chunk_content))
                    current_chunk = []
                    current_tokens = 0

                current_chunk.append(chunk_data)
                current_tokens += chunk_tokens

        # 处理最后的块
        if current_chunk:
            chunk_content = self._finalize_ast_chunk(current_chunk, context_stack)
            if chunk_content.strip():
                chunks.extend(self._force_split_if_oversize(chunk_content))

        return [chunk for chunk in chunks if chunk.strip()]

    def _merge_html_table_blocks(self, nodes):
        """
        markdown-it 在 table 内部出现换行/空行时，可能把一个 <table> 拆成多个 html_block。
        这里先按 <table>...</table> 平衡关系合并，避免主循环在碎片之间切块。
        """
        merged_blocks = []
        table_parts = []
        table_depth = 0

        for node in nodes:
            if table_parts:
                table_parts.append(self._extract_raw_node_content(node))
                table_depth += self._count_html_table_balance(
                    table_parts[-1]
                )
                if table_depth <= 0:
                    merged_blocks.append("".join(table_parts))
                    table_parts = []
                    table_depth = 0
                continue

            raw_content = self._extract_raw_node_content(node)
            table_balance = self._count_html_table_balance(raw_content)

            if table_balance > 0:
                table_parts = [raw_content]
                table_depth = table_balance
                if table_depth <= 0:
                    merged_blocks.append("".join(table_parts))
                    table_parts = []
                    table_depth = 0
            else:
                merged_blocks.append(node)

        if table_parts:
            merged_blocks.append("".join(table_parts))

        return merged_blocks

    def _extract_raw_node_content(self, node):
        """优先取 markdown-it 原始节点内容，拿不到时退回文本提取。"""
        if hasattr(node, "content") and node.content:
            return node.content
        return self._extract_text_from_node(node)

    def _count_html_table_balance(self, text: str) -> int:
        """统计 HTML table 起止标签平衡值。"""
        if not text:
            return 0
        open_count = len(re.findall(r"(?is)<table\b", text))
        close_count = len(re.findall(r"(?is)</table\s*>", text))
        return open_count - close_count

    def _process_ast_node(self, node, context_stack, chunk_token_num, min_chunk_tokens):
        """
        处理 AST 节点，返回 (内容, 是否应该分块)
        """
        node_type = node.type
        should_break = False
        content = ""

        if node_type == "heading":
            # 标题处理
            level = int(node.tag[1])  # h1 -> 1, h2 -> 2, etc.
            title_text = self._extract_text_from_node(node)

            # 更新上下文栈
            self._update_context_stack(context_stack, level, title_text)

            content = node.markup + " " + title_text
            should_break = True  # 标题通常作为分块边界

        elif node_type == "table":
            # 表格处理 - 保持完整性
            content = self._render_table_from_ast(node)
            table_tokens = num_tokens_from_string(content)

            # 表格过大时也要保持完整性
            if table_tokens > chunk_token_num:
                should_break = True

        elif node_type == "code_block":
            # 代码块处理
            content = f"```{node.info or ''}\n{node.content}```"

        elif node_type == "blockquote":
            # 引用块处理
            content = self._render_blockquote_from_ast(node)

        elif node_type == "list":
            # 列表处理
            content = self._render_list_from_ast(node)

        elif node_type == "paragraph":
            # 段落处理
            content = self._extract_text_from_node(node)

        elif node_type == "hr":
            # 分隔符
            content = "---"
            should_break = True

        elif node_type == "html_block":
            # 判断是否是 HTML 表格
            if node.content.strip().startswith("<table"):
                # 若表格过大，则进行拆分
                table_segments = self._split_html_table_if_needed(node.content)
                return table_segments, True  # 强制断块

        else:
            # 其他类型节点
            content = self._extract_text_from_node(node)

        return content, should_break


    def _update_context_stack(self, context_stack, level, title):
        """更新标题上下文栈"""
        # 移除比当前级别更深的标题
        while context_stack and context_stack[-1]['level'] >= level:
            context_stack.pop()

        # 添加当前标题
        context_stack.append({'level': level, 'title': title})


    def _extract_text_from_node(self, node):
        """从 AST 节点提取文本内容"""
        if hasattr(node, 'content') and node.content:
            return node.content

        text_parts = []
        if hasattr(node, 'children') and node.children:
            for child in node.children:
                if child.type == "text":
                    text_parts.append(child.content)
                elif child.type == "code_inline":
                    text_parts.append(f"`{child.content}`")
                elif child.type == "strong":
                    text_parts.append(f"**{self._extract_text_from_node(child)}**")
                elif child.type == "em":
                    text_parts.append(f"*{self._extract_text_from_node(child)}*")
                elif child.type == "link":
                    link_text = self._extract_text_from_node(child)
                    text_parts.append(f"[{link_text}]({child.attrGet('href') or ''})")
                else:
                    text_parts.append(self._extract_text_from_node(child))

        return "".join(text_parts)


    def _render_table_from_ast(self, table_node):
        """从 AST 渲染表格为 HTML"""
        try:
            # 构建表格的 markdown 表示
            table_md = []

            for child in table_node.children:
                if child.type == "thead":
                    # 表头处理
                    for row in child.children:
                        if row.type == "tr":
                            cells = []
                            for cell in row.children:
                                if cell.type in ["th", "td"]:
                                    cells.append(self._extract_text_from_node(cell))
                            table_md.append("| " + " | ".join(cells) + " |")

                    # 添加分隔符
                    if table_md:
                        separator = "| " + " | ".join(["---"] * len(cells)) + " |"
                        table_md.append(separator)

                elif child.type == "tbody":
                    # 表体处理
                    for row in child.children:
                        if row.type == "tr":
                            cells = []
                            for cell in row.children:
                                if cell.type in ["th", "td"]:
                                    cells.append(self._extract_text_from_node(cell))
                            table_md.append("| " + " | ".join(cells) + " |")

            # 转换为 HTML
            table_markdown = "\n".join(table_md)
            return md_to_html(table_markdown, extensions=['markdown.extensions.tables'])

        except Exception as e:
            print(f"Table rendering error: {e}")
            return self._extract_text_from_node(table_node)


    def _render_list_from_ast(self, list_node):
        """从 AST 渲染列表"""
        list_items = []
        list_type = list_node.attrGet('type') or 'bullet'

        for i, item in enumerate(list_node.children):
            if item.type == "list_item":
                item_content = self._extract_text_from_node(item)
                if list_type == 'ordered':
                    list_items.append(f"{i + 1}. {item_content}")
                else:
                    list_items.append(f"- {item_content}")

        return "\n".join(list_items)


    def _render_blockquote_from_ast(self, blockquote_node):
        """从 AST 渲染引用块"""
        content = self._extract_text_from_node(blockquote_node)
        lines = content.split('\n')
        return '\n'.join(f"> {line}" for line in lines)


    def _finalize_ast_chunk(self, chunk_parts, context_stack):
        """完成基于 AST 的 chunk 格式化"""
        chunk_content = "\n\n".join(chunk_parts).strip()

        # 可以根据需要添加上下文信息
        # 例如，如果chunk没有标题，可以考虑添加父级标题作为上下文

        return chunk_content

    def _split_html_table_if_needed(self, html_table: str):
        """
        如果 HTML 表格超过 max_table_tokens，则按行拆分成多个较小表格，
        每个表格保持 HTML 合法性。
        返回：分段后的 HTML 表格列表
        """
        total_tokens = num_tokens_from_string(html_table)

        if total_tokens <= self.max_table_tokens:
            return [html_table]  # 不需要拆分

        # --- 解析行 <tr> ---
        import re
        rows = re.findall(r"<tr.*?>.*?</tr>", html_table, flags=re.S)

        if not rows:
            return [html_table]  # 如果没解析到行，不拆

        # 提取表头（如果存在）
        header = ""
        body_rows = rows

        # 检测第一行是否为表头（简单判断 th 是否出现）
        if "<th" in rows[0]:
            header = rows[0]
            body_rows = rows[1:]

        table_segments = []
        current_rows = []
        current_tokens = 0

        for row in body_rows:
            row_tokens = num_tokens_from_string(row)

            # 新表格的 token 超出限制 → 保存前一个，开启新表格
            if current_rows and current_tokens + row_tokens > self.max_table_tokens:
                html_seg = self._build_html_table(header, current_rows)
                table_segments.append(html_seg)
                current_rows = []
                current_tokens = 0

            current_rows.append(row)
            current_tokens += row_tokens

        # 最后一个块
        if current_rows:
            html_seg = self._build_html_table(header, current_rows)
            table_segments.append(html_seg)

        return table_segments

    def _build_html_table(self, header, body_rows):
        table = ["<table>"]
        if header:
            table.append(header)
        table.extend(body_rows)
        table.append("</table>")
        return "".join(table)

    def _force_split_if_oversize(self, text: str):
        """
        如果 chunk 超过最大 token 或字符限制，进行硬拆分
        """
        token_limit = self.chunk_token_num * 2
        if (num_tokens_from_string(text) <= token_limit
                and len(text) <= 60000):
            return [text]

        if HTML_TABLE_PATTERN.search(text):
            return self._force_split_with_html_tables(text, token_limit)

        return self._split_plain_text_by_lines(text, token_limit)

    def _force_split_with_html_tables(self, text: str, token_limit: int):
        """优先保持 HTML table / tr 完整性，再进行硬拆分。"""
        segments = []
        current_segment = ""
        current_tokens = 0

        for block in self._split_blocks_preserving_tables(text):
            if HTML_TABLE_PATTERN.fullmatch(block.strip()):
                block_parts = self._split_html_table_block(block, token_limit)
            else:
                block_parts = self._split_plain_text_by_lines(block, token_limit)

            for part in block_parts:
                if not part or not part.strip():
                    continue

                part_tokens = num_tokens_from_string(part)
                if current_segment and current_tokens + part_tokens > token_limit:
                    segments.append(current_segment.strip())
                    current_segment = ""
                    current_tokens = 0

                current_segment += part
                current_tokens += part_tokens

        if current_segment.strip():
            segments.append(current_segment.strip())

        return segments

    def _split_blocks_preserving_tables(self, text: str):
        """把文本拆成 table/non-table 交替块，避免先打散 table。"""
        blocks = []
        cursor = 0
        for match in HTML_TABLE_PATTERN.finditer(text):
            start, end = match.span()
            if start > cursor:
                blocks.append(text[cursor:start])
            blocks.append(text[start:end])
            cursor = end

        if cursor < len(text):
            blocks.append(text[cursor:])

        return blocks if blocks else [text]

    def _split_html_table_block(self, table_html: str, token_limit: int):
        """按完整 <tr> 聚合拆表；只有单行超限时才继续拆行内内容。"""
        if num_tokens_from_string(table_html) <= token_limit and len(table_html) <= 60000:
            return [table_html]

        row_matches = list(HTML_TABLE_ROW_PATTERN.finditer(table_html))
        if not row_matches:
            return self._split_plain_text_by_lines(table_html, token_limit)

        prefix = table_html[:row_matches[0].start()]
        suffix = table_html[row_matches[-1].end():]

        segments = []
        current_rows = []

        def build_table(rows):
            return f"{prefix}{''.join(rows)}{suffix}"

        for row_match in row_matches:
            row_html = row_match.group(0)
            row_as_table = build_table([row_html])

            if num_tokens_from_string(row_as_table) > token_limit or len(row_as_table) > 60000:
                if current_rows:
                    segments.append(build_table(current_rows))
                    current_rows = []
                segments.extend(self._split_oversize_table_row(prefix, row_html, suffix, token_limit))
                continue

            candidate_rows = current_rows + [row_html]
            candidate_table = build_table(candidate_rows)
            if (current_rows
                    and (num_tokens_from_string(candidate_table) > token_limit
                         or len(candidate_table) > 60000)):
                segments.append(build_table(current_rows))
                current_rows = [row_html]
            else:
                current_rows = candidate_rows

        if current_rows:
            segments.append(build_table(current_rows))

        return segments

    def _split_oversize_table_row(self, table_prefix: str, row_html: str, table_suffix: str, token_limit: int):
        """单个 <tr> 超限时，优先按单元格拆；再不行才回退到普通换行拆分。"""
        cell_matches = list(HTML_TABLE_CELL_PATTERN.finditer(row_html))
        if not cell_matches:
            return self._split_plain_text_by_lines(
                f"{table_prefix}{row_html}{table_suffix}", token_limit
            )

        row_prefix = row_html[:cell_matches[0].start()]
        row_suffix = row_html[cell_matches[-1].end():]
        segments = []
        current_cells = []

        def build_row(cells):
            return f"{table_prefix}{row_prefix}{''.join(cells)}{row_suffix}{table_suffix}"

        for cell_match in cell_matches:
            cell_html = cell_match.group(0)
            cell_as_row = build_row([cell_html])

            if num_tokens_from_string(cell_as_row) > token_limit or len(cell_as_row) > 60000:
                if current_cells:
                    segments.append(build_row(current_cells))
                    current_cells = []
                segments.extend(self._split_plain_text_by_lines(cell_as_row, token_limit))
                continue

            candidate_cells = current_cells + [cell_html]
            candidate_row = build_row(candidate_cells)
            if (current_cells
                    and (num_tokens_from_string(candidate_row) > token_limit
                         or len(candidate_row) > 60000)):
                segments.append(build_row(current_cells))
                current_cells = [cell_html]
            else:
                current_cells = candidate_cells

        if current_cells:
            segments.append(build_row(current_cells))

        return segments

    def _split_plain_text_by_lines(self, text: str, token_limit: int):
        """普通文本兜底拆分，按换行尽量保持块完整。"""
        if (num_tokens_from_string(text) <= token_limit
                and len(text) <= 60000):
            return [text]

        segments = []
        current = []
        current_tokens = 0

        # 以换行作为软切点
        for line in text.splitlines(keepends=True):
            line_tokens = num_tokens_from_string(line)

            if (current_tokens + line_tokens > token_limit
                    and current):
                segments.append("".join(current))
                current = []
                current_tokens = 0

            current.append(line)
            current_tokens += line_tokens

        if current:
            segments.append("".join(current))

        return segments


if __name__ == '__main__':
    markdown_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mineru.md')

    with open(markdown_path, 'r', encoding='utf-8') as f:
        txt = f.read()

    text_splitter = MarkdownTextSplitter(
        chunk_token_num=256, min_chunk_tokens=10
    )
    chunks = text_splitter.split_text(txt)

    print(chunks)