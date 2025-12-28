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

        for node in tree.children:
            chunk_data, should_break = self._process_ast_node(
                node, context_stack, self.chunk_token_num, self.min_chunk_tokens
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
        if (num_tokens_from_string(text) <= self.max_tokens
                and len(text) <= self.char_max_length):
            return [text]

        segments = []
        current = []
        current_tokens = 0

        # 以换行作为软切点
        for line in text.split("\n"):
            line_tokens = num_tokens_from_string(line)

            if (current_tokens + line_tokens > self.max_tokens
                    and current):
                segments.append("\n".join(current))
                current = []
                current_tokens = 0

            current.append(line)
            current_tokens += line_tokens

        if current:
            segments.append("\n".join(current))

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