"""原生 ModelJson 视觉素材处理使用的共享常量。"""

from ...schema import BlockType

IMAGE_BLOCK_CONTAINMENT_THRESHOLD = 0.8
MODEL_JSON_VISUAL_BLOCK_TYPES = {BlockType.IMAGE, BlockType.CHART, BlockType.TABLE, BlockType.EQUATION}
