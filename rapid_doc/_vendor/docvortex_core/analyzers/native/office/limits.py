"""Flash Office 二进制、嵌入对象与 RTF 解析共享的固定安全限制。"""

from typing import Final

MAX_ENTRY_BYTES: Final = 128 * 1024 * 1024
MAX_TOTAL_BYTES: Final = 512 * 1024 * 1024
MAX_ASSET_TOTAL_BYTES: Final = 128 * 1024 * 1024
# 公式探测单独计费，避免普通图片耗尽嵌入素材预算；初始值与素材预算保持一致。
MAX_EQUATION_CANDIDATE_TOTAL_BYTES: Final = 128 * 1024 * 1024
MAX_GRID_SLOTS: Final = 4_000_000
MAX_RECORD_DEPTH: Final = 64
MAX_RECORDS: Final = 16_000_000
MAX_PICTURE_RECORDS: Final = 100_000
MAX_USER_EDIT_CHAIN: Final = 100
