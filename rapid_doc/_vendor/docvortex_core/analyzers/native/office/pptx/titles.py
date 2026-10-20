"""PPTX 幻灯片标题判定，复用当前转换器的单文档状态。"""

from collections import Counter
from typing import Optional
from .....schema import BlockType

from .context import (
    _EFFECTIVE_FONT_SIZE_KEY,
    _EFFECTIVE_ALL_BOLD_KEY,
    _PPTX_TITLE_CANDIDATE_KEY,
    _PPTX_TITLE_ROLE_KEY,
    _PPTX_TITLE_ROLE_CENTER,
    _PPTX_TITLE_ROLE_SUBTITLE,
)


class _PptxTitles:
    """集中维护幻灯片标题判定，不改变文档生命周期和公开入口。"""

    @staticmethod
    def _most_common_size(font_sizes: list[float]) -> Optional[float]:
        """按原有幻灯片标题判定规则执行 _most_common_size，保持输入顺序与降级行为。"""
        if not font_sizes:
            return None

        counts = Counter(font_sizes)
        return min(
            counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]

    def _promote_slide_text_blocks_to_titles(self, slide_blocks: list[dict]) -> None:
        """按原有幻灯片标题判定规则执行 _promote_slide_text_blocks_to_titles，保持输入顺序与降级行为。"""
        body_font_size_pt = self._most_common_size(
            [
                block[_EFFECTIVE_FONT_SIZE_KEY]
                for block in slide_blocks
                if (
                    block.get("type") == BlockType.TEXT
                    and block.get(_PPTX_TITLE_CANDIDATE_KEY) is not True
                    and block.get(_EFFECTIVE_FONT_SIZE_KEY) is not None
                    and not block.get(_EFFECTIVE_ALL_BOLD_KEY, False)
                )
            ]
        )

        self._promote_level2_text_blocks(slide_blocks, body_font_size_pt)
        self._promote_level3_text_blocks(slide_blocks, body_font_size_pt)

    def _promote_level2_text_blocks(
        self,
        slide_blocks: list[dict],
        body_font_size_pt: Optional[float],
    ) -> None:
        """按原有幻灯片标题判定规则执行 _promote_level2_text_blocks，保持输入顺序与降级行为。"""
        bold_text_blocks = [
            block
            for block in slide_blocks
            if (
                block.get("type") == BlockType.TEXT
                and block.get(_PPTX_TITLE_CANDIDATE_KEY) is not True
                and block.get(_EFFECTIVE_ALL_BOLD_KEY, False)
                and block.get(_EFFECTIVE_FONT_SIZE_KEY) is not None
            )
        ]
        if not bold_text_blocks:
            return

        bold_font_sizes = sorted(
            {block[_EFFECTIVE_FONT_SIZE_KEY] for block in bold_text_blocks},
            reverse=True,
        )
        level2_font_size_pt = bold_font_sizes[0]
        level2_candidates = [block for block in bold_text_blocks if block[_EFFECTIVE_FONT_SIZE_KEY] == level2_font_size_pt]

        if len(level2_candidates) != 1:
            return

        if body_font_size_pt is not None and level2_font_size_pt < body_font_size_pt + 4:
            return

        if len(bold_font_sizes) > 1 and level2_font_size_pt < bold_font_sizes[1] + 2:
            return

        level2_candidates[0][_PPTX_TITLE_CANDIDATE_KEY] = True
        level2_candidates[0]["level"] = 2

    def _promote_level3_text_blocks(
        self,
        slide_blocks: list[dict],
        body_font_size_pt: Optional[float],
    ) -> None:
        """按原有幻灯片标题判定规则执行 _promote_level3_text_blocks，保持输入顺序与降级行为。"""
        if body_font_size_pt is None:
            return

        level2_font_sizes = sorted(
            {
                block[_EFFECTIVE_FONT_SIZE_KEY]
                for block in slide_blocks
                if (
                    block.get(_PPTX_TITLE_CANDIDATE_KEY) is True
                    and block.get("level") == 2
                    and block.get(_EFFECTIVE_FONT_SIZE_KEY) is not None
                )
            },
            reverse=True,
        )
        if not level2_font_sizes:
            return

        level2_font_size_pt = level2_font_sizes[0]
        level3_font_sizes = sorted(
            {
                block[_EFFECTIVE_FONT_SIZE_KEY]
                for block in slide_blocks
                if (
                    block.get("type") == BlockType.TEXT
                    and block.get(_PPTX_TITLE_CANDIDATE_KEY) is not True
                    and block.get(_EFFECTIVE_ALL_BOLD_KEY, False)
                    and block.get(_EFFECTIVE_FONT_SIZE_KEY) is not None
                    and block[_EFFECTIVE_FONT_SIZE_KEY] < level2_font_size_pt
                )
            },
            reverse=True,
        )
        if not level3_font_sizes:
            return

        level3_font_size_pt = level3_font_sizes[0]
        if level3_font_size_pt < body_font_size_pt + 2:
            return
        if level2_font_size_pt < level3_font_size_pt + 2:
            return

        for block in slide_blocks:
            if (
                block.get("type") == BlockType.TEXT
                and block.get(_PPTX_TITLE_CANDIDATE_KEY) is not True
                and block.get(_EFFECTIVE_ALL_BOLD_KEY, False)
                and block.get(_EFFECTIVE_FONT_SIZE_KEY) == level3_font_size_pt
            ):
                block[_PPTX_TITLE_CANDIDATE_KEY] = True
                block["level"] = 3

    @staticmethod
    def _finalize_slide_title_types(
        slide_blocks: list[dict],
        *,
        is_first_visible_slide: bool,
    ) -> None:
        """将 PPTX 标题候选统一拆分为文档标题、段落标题或普通文本。"""
        for block in slide_blocks:
            is_title_candidate = block.pop(_PPTX_TITLE_CANDIDATE_KEY, False) is True
            title_role = block.pop(_PPTX_TITLE_ROLE_KEY, None)
            if not is_title_candidate:
                continue
            if title_role == _PPTX_TITLE_ROLE_SUBTITLE:
                block["type"] = BlockType.TEXT
                block.pop("level", None)
                block.pop("is_numbered_style", None)
                continue

            if title_role == _PPTX_TITLE_ROLE_CENTER and is_first_visible_slide:
                block["type"] = BlockType.DOC_TITLE
                block["level"] = 1
                block.pop("is_numbered_style", None)
            else:
                block["type"] = BlockType.PARAGRAPH_TITLE

    @staticmethod
    def _cleanup_slide_text_block_metadata(slide_blocks: list[dict]) -> None:
        """按原有幻灯片标题判定规则执行 _cleanup_slide_text_block_metadata，保持输入顺序与降级行为。"""
        for block in slide_blocks:
            block.pop(_EFFECTIVE_FONT_SIZE_KEY, None)
            block.pop(_EFFECTIVE_ALL_BOLD_KEY, None)
            block.pop(_PPTX_TITLE_CANDIDATE_KEY, None)
            block.pop(_PPTX_TITLE_ROLE_KEY, None)
