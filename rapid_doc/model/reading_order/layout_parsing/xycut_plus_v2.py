# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from typing import List, Dict, Any
import numpy as np

from rapid_doc.model.layout.rapid_layout_self import RapidLayoutInput, ModelType, RapidLayout
from rapid_doc.model.reading_order.layout_parsing.layout_objects import LayoutBlock, LayoutRegion
from rapid_doc.model.reading_order.layout_parsing.setting import BLOCK_LABEL_MAP
from rapid_doc.model.reading_order.layout_parsing.utils import update_region_box, caculate_bbox_area, \
    remove_overlap_blocks
from rapid_doc.model.reading_order.layout_parsing.xycut_enhanced import xycut_enhanced
from rapid_doc.model.reading_order.utils import visualize_reading_order


def sort_layout_parsing_blocks(
        layout_parsing_page: LayoutRegion
) -> List[LayoutBlock]:
    layout_parsing_regions = xycut_enhanced(layout_parsing_page)
    parsing_res_list = []
    for region in layout_parsing_regions:
        layout_parsing_blocks = xycut_enhanced(region)
        parsing_res_list.extend(layout_parsing_blocks)

    return parsing_res_list


def standardized_data(
    layout_det_res,
    region_det_res=None,
) -> list:
    """
    根据布局检测结果获取布局解析结果。

    参数：
        layout_det_res: 包含布局检测结果的对象，包括检测到的布局框及其标签。结构预期如下：
            - "boxes": 字典列表，每个字典包含 "coordinate"（框坐标）和 "block_label"（内容类型）。

    返回：
        list: 表示布局解析结果的字典列表。
    """
    if not region_det_res:
        region_det_res = {'boxes': []}
    region_to_block_map = {}
    block_to_ocr_map = {}

    base_region_bbox = [65535, 65535, 0, 0]
    layout_det_res = remove_overlap_blocks(
        layout_det_res,
        threshold=0.5,
        smaller=True,
    )

    # 遍历布局块
    for box_idx, box_info in enumerate(layout_det_res["boxes"]):
        box = box_info["coordinate"]
        base_region_bbox = update_region_box(box, base_region_bbox)

    block_bboxes = [box["coordinate"] for box in layout_det_res["boxes"]]
    region_det_res["boxes"] = sorted(
        region_det_res["boxes"],
        key=lambda item: caculate_bbox_area(item["coordinate"]),
    )
    if len(region_det_res["boxes"]) == 0:
        region_det_res["boxes"] = [
            {
                "coordinate": base_region_bbox,
                "label": "SupplementaryRegion",
                "score": 1,
            }
        ]
        region_to_block_map[0] = range(len(block_bboxes))

    region_block_ocr_idx_map = dict(
        region_to_block_map=region_to_block_map,
        block_to_ocr_map=block_to_ocr_map,
    )

    return region_block_ocr_idx_map, region_det_res, layout_det_res


def get_layout_parsing_objects(
        region_block_ocr_idx_map: dict,
        region_det_res,
        layout_det_res,
) -> list:
    """
    从 OCR 结果和布局检测结果中提取结构化信息。

    参数：
        layout_det_res (DetResult): 包含布局检测结果的对象，包括检测到的布局框及其标签。结构预期如下：
            - "boxes": 字典列表，每个字典包含 "coordinate"（框坐标）和 "block_label"（内容类型）。

    返回：
        list: 结构化块列表，每项为字典，包含：
            - "block_label": 内容的标签（如 'table'、'chart'、'image'）。
            - 以标签为键的值，包含表格 HTML 或图像数据及文本。
            - "block_bbox": 布局框的坐标。
    """
    layout_parsing_blocks: List[LayoutBlock] = []

    for box_idx, box_info in enumerate(layout_det_res["boxes"]):
        label = box_info["label"]
        block_bbox = box_info["coordinate"]
        block = LayoutBlock(label=label, bbox=block_bbox)

        if label == "table":
            block.content = ""
        elif label == "seal":
            block.content = ""
        elif label == "chart":
            block.content = ""
        else:
            block.content = ""

        if (
            label
            in ["seal", "table", "formula", "chart"]
            + BLOCK_LABEL_MAP["image_labels"]
        ):
            block.image = {"path": "", "img": None}

        layout_parsing_blocks.append(block)

    page_region_bbox = [65535, 65535, 0, 0]
    layout_parsing_regions: List[LayoutRegion] = []
    for region_idx, region_info in enumerate(region_det_res["boxes"]):
        region_bbox = np.array(region_info["coordinate"]).astype("int")
        region_blocks = [
            layout_parsing_blocks[idx]
            for idx in region_block_ocr_idx_map["region_to_block_map"][region_idx]
        ]
        if region_blocks:
            page_region_bbox = update_region_box(region_bbox, page_region_bbox)
            region = LayoutRegion(bbox=region_bbox, blocks=region_blocks)
            layout_parsing_regions.append(region)

    layout_parsing_page = LayoutRegion(
        bbox=np.array(page_region_bbox).astype("int"), blocks=layout_parsing_regions
    )

    return layout_parsing_page


def xycut_plus_sort_v2(layout_det_res):
    layout_det_res = {'boxes': layout_det_res}
    # Standardize data
    region_block_ocr_idx_map, region_det_res, layout_det_res = (
        standardized_data(
            layout_det_res=layout_det_res,
        )
    )
    # Format layout parsing block
    layout_parsing_page = get_layout_parsing_objects(
        region_block_ocr_idx_map=region_block_ocr_idx_map,
        region_det_res=region_det_res,
        layout_det_res=layout_det_res,
    )
    return sort_layout_parsing_blocks(layout_parsing_page)

if __name__ == '__main__':

    import pickle

    cfg = RapidLayoutInput(model_type=ModelType.PP_DOCLAYOUT_L, conf_thresh=0.4)
    model = RapidLayout(cfg=cfg)
    all_results = model(img_contents=[r'13a3fa7b-80b6-4b51-9978-e1ec84988ef1.png'])
    data_list = [list(map(float, box)) for box in all_results[0].boxes]
    result = all_results[0]

    layout_det_res = [
        {
            "coordinate": box,
            "label": result.class_names[idx],
            "score": result.scores[idx],
        }
        for idx, box in enumerate(result.boxes)
    ]

    layout_det_res = {'boxes': layout_det_res}

    with open(r"C:\ocr\models\ppmodel\layout\PP-DocLayout_plus-L\layout_det_res_v2.pkl", "rb") as f:
        layout_det_res = pickle.load(f)

    region_det_res = {'boxes': []}
    # Standardize data
    region_block_ocr_idx_map, region_det_res, layout_det_res = (
        standardized_data(
            region_det_res=region_det_res,
            layout_det_res=layout_det_res,
        )
    )

    # Format layout parsing block
    layout_parsing_page = get_layout_parsing_objects(
        region_block_ocr_idx_map=region_block_ocr_idx_map,
        region_det_res=region_det_res,
        layout_det_res=layout_det_res,
    )

    parsing_res_list = sort_layout_parsing_blocks(layout_parsing_page)
    sorted_data: List[Dict[str, Any]] = []
    for order, parsing_res in enumerate(parsing_res_list):
        sorted_data.append({
                "bbox": parsing_res.bbox,
                "reading_order": order,
                "label": "text"
            })

    print("=== 阅读顺序排序结果 ===")
    for item in sorted_data:
        print(f"顺序: {item['reading_order']}, 标签: {item['label']}, 边界框: {item['bbox']}")

    visualize_reading_order(sorted_data)