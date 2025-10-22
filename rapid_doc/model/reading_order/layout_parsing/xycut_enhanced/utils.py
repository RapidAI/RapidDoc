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

from typing import List, Tuple

import numpy as np

from ..layout_objects import LayoutBlock, LayoutRegion
from ..setting import BLOCK_LABEL_MAP, XYCUT_SETTINGS
from ..utils import (
    calculate_overlap_ratio,
    calculate_projection_overlap_ratio,
    get_seg_flag,
)


def get_nearest_edge_distance(
    bbox1: List[int],
    bbox2: List[int],
    weight: List[float] = [1.0, 1.0, 1.0, 1.0],
) -> Tuple[float]:
    """
    计算两个边界框（bounding boxes）之间的最近边缘距离，并考虑方向权重。

    参数：
        bbox1 (list): 输入对象的边界框坐标 [x1, y1, x2, y2]。
        bbox2 (list): 用于匹配的目标对象边界框坐标 [x1', y1', x2', y2']。
        weight (list, 可选): 边缘距离的方向权重 [左, 右, 上, 下]，默认值为 [1, 1, 1, 1]。

    返回：
        float: 计算得到的两个边界框之间的最小边缘距离。
    """
    x1, y1, x2, y2 = bbox1
    x1_prime, y1_prime, x2_prime, y2_prime = bbox2
    min_x_distance, min_y_distance = 0, 0
    horizontal_iou = calculate_projection_overlap_ratio(bbox1, bbox2, "horizontal")
    vertical_iou = calculate_projection_overlap_ratio(bbox1, bbox2, "vertical")
    if horizontal_iou > 0 and vertical_iou > 0:
        return 0.0
    if horizontal_iou == 0:
        min_x_distance = min(abs(x1 - x2_prime), abs(x2 - x1_prime)) * (
            weight[0] if x2 < x1_prime else weight[1]
        )
    if vertical_iou == 0:
        min_y_distance = min(abs(y1 - y2_prime), abs(y2 - y1_prime)) * (
            weight[2] if y2 < y1_prime else weight[3]
        )

    return min_x_distance + min_y_distance


def projection_by_bboxes(boxes: np.ndarray, axis: int) -> np.ndarray:
    """
    根据指定的轴，从边界框（bounding boxes）生成一维投影直方图。

    参数：
        boxes: 形状为 (N, 4) 的边界框数组，每个边界框由 [x_min, y_min, x_max, y_max] 定义。
        axis: 投影轴方向；0 表示水平方向（x 轴），1 表示垂直方向（y 轴）。

    返回：
        一维 numpy 数组，表示根据边界框区间生成的投影直方图。
    """
    assert axis in [0, 1]

    if np.min(boxes[:, axis::2]) < 0:
        max_length = abs(np.min(boxes[:, axis::2]))
    else:
        max_length = np.max(boxes[:, axis::2])

    projection = np.zeros(max_length, dtype=int)

    # Increment projection histogram over the interval defined by each bounding box
    for start, end in boxes[:, axis::2]:
        start = abs(start)
        end = abs(end)
        projection[start:end] += 1

    return projection


def split_projection_profile(arr_values: np.ndarray, min_value: float, min_gap: float):
    """
    根据指定的阈值将投影轮廓（projection profile）分割成若干段。

    参数：
        arr_values: 一维数组，表示投影轮廓数据。
        min_value: 最小值阈值，用于判断一个轮廓段是否具有意义。
        min_gap: 最小间隔宽度，用于判断各段之间的分隔距离。

    返回：
        一个包含各段起始索引和结束索引的元组，表示符合条件的分段范围。
    """
    # Identify indices where the projection exceeds the minimum value
    significant_indices = np.where(arr_values > min_value)[0]
    if not len(significant_indices):
        return

    # Calculate gaps between significant indices
    index_diffs = significant_indices[1:] - significant_indices[:-1]
    gap_indices = np.where(index_diffs > min_gap)[0]

    # Determine start and end indices of segments
    segment_starts = np.insert(
        significant_indices[gap_indices + 1],
        0,
        significant_indices[0],
    )
    segment_ends = np.append(
        significant_indices[gap_indices],
        significant_indices[-1] + 1,
    )

    return segment_starts, segment_ends


def recursive_yx_cut(
    boxes: np.ndarray, indices: List[int], res: List[int], min_gap: int = 1
):
    """
    递归地对边界框（bounding boxes）进行投影和分段，先沿 Y 轴投影，再沿 X 轴分段。

    参数：
        boxes: 形状为 (N, 4) 的边界框数组。
        indices: 表示边界框在原始列表中位置的索引列表。
        res: 用于存储最终分段后的边界框索引的列表。
        min_gap (int): 在 X 轴上判断段之间分隔的最小间隔宽度，默认值为 1。

    返回：
        None: 该函数会直接修改 `res` 列表。
    """
    assert len(boxes) == len(
        indices
    ), "The length of boxes and indices must be the same."

    # Sort by y_min for Y-axis projection
    y_sorted_indices = boxes[:, 1].argsort()
    y_sorted_boxes = boxes[y_sorted_indices]
    y_sorted_indices = np.array(indices)[y_sorted_indices]

    # Perform Y-axis projection
    y_projection = projection_by_bboxes(boxes=y_sorted_boxes, axis=1)
    y_intervals = split_projection_profile(y_projection, 0, 1)

    if not y_intervals:
        return

    # Process each segment defined by Y-axis projection
    for y_start, y_end in zip(*y_intervals):
        # Select boxes within the current y interval
        y_interval_indices = (y_start <= y_sorted_boxes[:, 1]) & (
            y_sorted_boxes[:, 1] < y_end
        )
        y_boxes_chunk = y_sorted_boxes[y_interval_indices]
        y_indices_chunk = y_sorted_indices[y_interval_indices]

        # Sort by x_min for X-axis projection
        x_sorted_indices = y_boxes_chunk[:, 0].argsort()
        x_sorted_boxes_chunk = y_boxes_chunk[x_sorted_indices]
        x_sorted_indices_chunk = y_indices_chunk[x_sorted_indices]

        # Perform X-axis projection
        x_projection = projection_by_bboxes(boxes=x_sorted_boxes_chunk, axis=0)
        x_intervals = split_projection_profile(x_projection, 0, min_gap)

        if not x_intervals:
            continue

        # If X-axis cannot be further segmented, add current indices to results
        if len(x_intervals[0]) == 1:
            res.extend(x_sorted_indices_chunk)
            continue

        if np.min(x_sorted_boxes_chunk[:, 0]) < 0:
            x_intervals = np.flip(x_intervals, axis=1)
        # Recursively process each segment defined by X-axis projection
        for x_start, x_end in zip(*x_intervals):
            x_interval_indices = (x_start <= abs(x_sorted_boxes_chunk[:, 0])) & (
                abs(x_sorted_boxes_chunk[:, 0]) < x_end
            )
            recursive_yx_cut(
                x_sorted_boxes_chunk[x_interval_indices],
                x_sorted_indices_chunk[x_interval_indices],
                res,
            )


def recursive_xy_cut(
    boxes: np.ndarray, indices: List[int], res: List[int], min_gap: int = 1
):
    """
    递归地先沿 X 轴投影，再沿 Y 轴投影，以分段边界框（bounding boxes）。

    参数：
        boxes: 形状为 (N, 4) 的边界框数组，每个框表示为 [x_min, y_min, x_max, y_max]。
        indices: 表示边界框在原始数据中位置的索引列表。
        res: 用于存储符合条件的边界框索引的列表。
        min_gap (int): 在 X 轴上判断段之间分隔的最小间隔宽度，默认值为 1。

    返回：
        None: 该函数会直接修改 `res` 列表。
    """
    # Ensure boxes and indices have the same length
    assert len(boxes) == len(
        indices
    ), "The length of boxes and indices must be the same."

    # Sort by x_min to prepare for X-axis projection
    x_sorted_indices = boxes[:, 0].argsort()
    x_sorted_boxes = boxes[x_sorted_indices]
    x_sorted_indices = np.array(indices)[x_sorted_indices]

    # Perform X-axis projection
    x_projection = projection_by_bboxes(boxes=x_sorted_boxes, axis=0)
    x_intervals = split_projection_profile(x_projection, 0, 1)

    if not x_intervals:
        return

    if np.min(x_sorted_boxes[:, 0]) < 0:
        x_intervals = np.flip(x_intervals, axis=1)
    # Process each segment defined by X-axis projection
    for x_start, x_end in zip(*x_intervals):
        # Select boxes within the current x interval
        x_interval_indices = (x_start <= abs(x_sorted_boxes[:, 0])) & (
            abs(x_sorted_boxes[:, 0]) < x_end
        )
        x_boxes_chunk = x_sorted_boxes[x_interval_indices]
        x_indices_chunk = x_sorted_indices[x_interval_indices]

        # Sort selected boxes by y_min to prepare for Y-axis projection
        y_sorted_indices = x_boxes_chunk[:, 1].argsort()
        y_sorted_boxes_chunk = x_boxes_chunk[y_sorted_indices]
        y_sorted_indices_chunk = x_indices_chunk[y_sorted_indices]

        # Perform Y-axis projection
        y_projection = projection_by_bboxes(boxes=y_sorted_boxes_chunk, axis=1)
        y_intervals = split_projection_profile(y_projection, 0, min_gap)

        if not y_intervals:
            continue

        # If Y-axis cannot be further segmented, add current indices to results
        if len(y_intervals[0]) == 1:
            res.extend(y_sorted_indices_chunk)
            continue

        # Recursively process each segment defined by Y-axis projection
        for y_start, y_end in zip(*y_intervals):
            y_interval_indices = (y_start <= y_sorted_boxes_chunk[:, 1]) & (
                y_sorted_boxes_chunk[:, 1] < y_end
            )
            recursive_xy_cut(
                y_sorted_boxes_chunk[y_interval_indices],
                y_sorted_indices_chunk[y_interval_indices],
                res,
            )


def reference_insert(
    block: LayoutBlock,
    sorted_blocks: List[LayoutBlock],
    **kwargs,
):
    """
    根据块与最近已排序块之间的距离，将参考块（reference block）插入到已排序块列表中。

    参数：
        block: 待插入的块。
        sorted_blocks: 已排序的块列表，新块将插入到此列表中。
        config: 包含布局解析相关参数的配置字典。
        median_width: 文档的中位宽度，默认值为 0.0。

    返回：
        sorted_blocks: 插入参考块后更新的已排序块列表。
    """
    min_distance = float("inf")
    nearest_sorted_block_index = 0
    for sorted_block_idx, sorted_block in enumerate(sorted_blocks):
        if sorted_block.bbox[3] <= block.bbox[1]:
            distance = -(sorted_block.bbox[2] * 10 + sorted_block.bbox[3])
        if distance < min_distance:
            min_distance = distance
            nearest_sorted_block_index = sorted_block_idx

    sorted_blocks.insert(nearest_sorted_block_index + 1, block)
    return sorted_blocks


def manhattan_insert(
    block: LayoutBlock,
    sorted_blocks: List[LayoutBlock],
    **kwargs,
):
    """
    根据块与最近已排序块之间的曼哈顿距离，将块插入到已排序块列表中。

    参数：
        block: 待插入的块。
        sorted_blocks: 已排序的块列表，新块将插入到此列表中。
        config: 包含布局解析相关参数的配置字典。
        median_width: 文档的中位宽度，默认值为 0.0。

    返回：
        sorted_blocks: 插入块后更新的已排序块列表。
    """
    min_distance = float("inf")
    nearest_sorted_block_index = 0
    for sorted_block_idx, sorted_block in enumerate(sorted_blocks):
        distance = _manhattan_distance(block.bbox, sorted_block.bbox)
        if distance < min_distance:
            min_distance = distance
            nearest_sorted_block_index = sorted_block_idx

    sorted_blocks.insert(nearest_sorted_block_index + 1, block)
    return sorted_blocks


def euclidean_insert(
    block: LayoutRegion,
    sorted_blocks: List[LayoutRegion],
    **kwargs,
):
    """
    根据块与最近已排序块之间的欧几里得距离，将块插入到已排序块列表中。

    参数：
        block: 待插入的块。
        sorted_blocks: 已排序的块列表，新块将插入到此列表中。
        config: 包含布局解析相关参数的配置字典。
        median_width: 文档的中位宽度，默认值为 0.0。

    返回：
        sorted_blocks: 插入块后更新的已排序块列表。
    """
    nearest_sorted_block_index = len(sorted_blocks)
    block_euclidean_distance = block.euclidean_distance
    for sorted_block_idx, sorted_block in enumerate(sorted_blocks):
        distance = sorted_block.euclidean_distance
        if distance > block_euclidean_distance:
            nearest_sorted_block_index = sorted_block_idx
            break
    sorted_blocks.insert(nearest_sorted_block_index, block)
    return sorted_blocks


def weighted_distance_insert(
    block: LayoutBlock,
    sorted_blocks: List[LayoutBlock],
    region: LayoutRegion,
):
    """
    根据块与最近已排序块之间的加权距离，将块插入到已排序块列表中。

    参数：
        block: 待插入的块。
        sorted_blocks: 已排序的块列表，新块将插入到此列表中。
        config: 包含布局解析相关参数的配置字典。
        median_width: 文档的中位宽度，默认值为 0.0。

    返回：
        sorted_blocks: 插入块后更新的已排序块列表。
    """

    tolerance_len = XYCUT_SETTINGS["edge_distance_compare_tolerance_len"]
    x1, y1, x2, y2 = block.bbox
    min_weighted_distance, min_edge_distance, min_up_edge_distance = (
        float("inf"),
        float("inf"),
        float("inf"),
    )
    nearest_sorted_block_index = 0
    for sorted_block_idx, sorted_block in enumerate(sorted_blocks):

        x1_prime, y1_prime, x2_prime, y2_prime = sorted_block.bbox

        # Calculate edge distance
        weight = _get_weights(block.order_label, block.direction)
        edge_distance = get_nearest_edge_distance(block.bbox, sorted_block.bbox, weight)

        if block.label in BLOCK_LABEL_MAP["doc_title_labels"]:
            disperse = max(1, region.text_line_width)
            tolerance_len = max(tolerance_len, disperse)
        if block.label == "abstract":
            tolerance_len *= 2
            edge_distance = max(0.1, edge_distance) * 10

        # Calculate up edge distances
        up_edge_distance = y1_prime if region.direction == "horizontal" else -x2_prime
        left_edge_distance = x1_prime if region.direction == "horizontal" else y1_prime
        is_below_sorted_block = (
            y2_prime < y1 if region.direction == "horizontal" else x1_prime > x2
        )

        if (
            block.label not in BLOCK_LABEL_MAP["unordered_labels"]
            or block.label in BLOCK_LABEL_MAP["doc_title_labels"]
            or block.label in BLOCK_LABEL_MAP["paragraph_title_labels"]
            or block.label in BLOCK_LABEL_MAP["vision_labels"]
        ) and is_below_sorted_block:
            up_edge_distance = -up_edge_distance
            left_edge_distance = -left_edge_distance

        if abs(min_up_edge_distance - up_edge_distance) <= tolerance_len:
            up_edge_distance = min_up_edge_distance

        # Calculate weighted distance
        weighted_distance = (
            +edge_distance
            * XYCUT_SETTINGS["distance_weight_map"].get("edge_weight", 10**4)
            + up_edge_distance
            * XYCUT_SETTINGS["distance_weight_map"].get("up_edge_weight", 1)
            + left_edge_distance
            * XYCUT_SETTINGS["distance_weight_map"].get("left_edge_weight", 0.0001)
        )

        min_edge_distance = min(edge_distance, min_edge_distance)
        min_up_edge_distance = min(up_edge_distance, min_up_edge_distance)

        if weighted_distance < min_weighted_distance:
            nearest_sorted_block_index = sorted_block_idx
            min_weighted_distance = weighted_distance
            if abs(y1 // 2 - y1_prime // 2) > 0:
                sorted_distance = y1_prime
                block_distance = y1
            else:
                if region.direction == "horizontal":
                    if abs(x1 // 2 - x2 // 2) > 0:
                        sorted_distance = x1_prime
                        block_distance = x1
                    else:
                        # distance with (0,0)
                        sorted_block_center_x, sorted_block_center_y = (
                            sorted_block.get_centroid()
                        )
                        block_center_x, block_center_y = block.get_centroid()
                        sorted_distance = (
                            sorted_block_center_x**2 + sorted_block_center_y**2
                        )
                        block_distance = block_center_x**2 + block_center_y**2
                else:
                    if abs(x1 - x2) > 0:
                        sorted_distance = -x2_prime
                        block_distance = -x2
                    else:
                        # distance with (max,0)
                        sorted_block_center_x, sorted_block_center_y = (
                            sorted_block.get_centroid()
                        )
                        block_center_x, block_center_y = block.get_centroid()
                        sorted_distance = (
                            sorted_block_center_x**2 + sorted_block_center_y**2
                        )
                        block_distance = block_center_x**2 + block_center_y**2
            if block_distance > sorted_distance:
                nearest_sorted_block_index = sorted_block_idx + 1
                if (
                    sorted_block_idx < len(sorted_blocks) - 1
                    and block.label
                    in BLOCK_LABEL_MAP["vision_labels"]
                    + BLOCK_LABEL_MAP["vision_title_labels"]
                ):
                    seg_start_flag, _ = get_seg_flag(
                        sorted_blocks[sorted_block_idx + 1],
                        sorted_blocks[sorted_block_idx],
                    )
                    if not seg_start_flag:
                        nearest_sorted_block_index += 1
            else:
                if (
                    sorted_block_idx > 0
                    and block.label
                    in BLOCK_LABEL_MAP["vision_labels"]
                    + BLOCK_LABEL_MAP["vision_title_labels"]
                ):
                    seg_start_flag, _ = get_seg_flag(
                        sorted_blocks[sorted_block_idx],
                        sorted_blocks[sorted_block_idx - 1],
                    )
                    if not seg_start_flag:
                        nearest_sorted_block_index = sorted_block_idx - 1

    sorted_blocks.insert(nearest_sorted_block_index, block)
    return sorted_blocks


def insert_child_blocks(
    block: LayoutBlock,
    block_idx: int,
    sorted_blocks: List[LayoutBlock],
) -> List[LayoutBlock]:
    """
    将某个块（block）的子块插入到已排序块列表中。

    参数：
        block: 需要插入其子块的父块。
        block_idx: 父块在已排序块列表中的索引位置。
        sorted_blocks: 已排序块列表，子块将插入到该列表中。

    返回：
        sorted_blocks: 插入子块后更新的已排序块列表。
    """
    if block.child_blocks:
        sub_blocks = block.get_child_blocks()
        sub_blocks.append(block)
        sub_blocks = sort_child_blocks(sub_blocks, sub_blocks[0].direction)
        sorted_blocks[block_idx] = sub_blocks[0]
        for block in sub_blocks[1:]:
            block_idx += 1
            sorted_blocks.insert(block_idx, block)
    return sorted_blocks


def sort_child_blocks(
    blocks: List[LayoutRegion], direction="horizontal"
) -> List[LayoutBlock]:
    """
    根据子块的边界框坐标对其进行排序。

    参数：
        blocks: 表示子块的 LayoutBlock 对象列表。
        direction: 子块的排序方向（'horizontal' 或 'vertical'），默认值为 'horizontal'。

    返回：
        sorted_blocks: 排序后的 LayoutBlock 对象列表。
    """
    if blocks[0].label != "region":
        if direction == "horizontal":
            blocks.sort(
                key=lambda x: (
                    x.bbox[1],
                    x.bbox[0],
                    x.get_centroid()[0] ** 2 + x.get_centroid()[1] ** 2,
                ),  # distance with (0,0)
            )
        else:
            blocks.sort(
                key=lambda x: (
                    -x.bbox[2],
                    x.bbox[1],
                    -x.get_centroid()[0] ** 2 + x.get_centroid()[1] ** 2,
                ),  # distance with (max,0)
            )
    else:
        blocks.sort(key=lambda x: x.euclidean_distance)
    return blocks


def _get_weights(label, direction="horizontal"):
    """Define weights based on the label and direction."""
    if label == "doc_title":
        return (
            [1, 0.1, 0.1, 1] if direction == "horizontal" else [0.2, 0.1, 1, 1]
        )  # left-down ,  right-left
    elif label in [
        "paragraph_title",
        "table_title",
        "abstract",
        "image",
        "seal",
        "chart",
        "figure",
    ]:
        return [1, 1, 0.1, 1]  # down
    else:
        return [1, 1, 1, 0.1]  # up


def _manhattan_distance(
    point1: Tuple[float, float],
    point2: Tuple[float, float],
    weight_x: float = 1.0,
    weight_y: float = 1.0,
) -> float:
    """
    计算两点之间的加权曼哈顿距离。

    参数：
        point1 (Tuple[float, float]): 第一个点，表示为 (x, y)。
        point2 (Tuple[float, float]): 第二个点，表示为 (x, y)。
        weight_x (float): x 轴距离的权重，默认值为 1.0。
        weight_y (float): y 轴距离的权重，默认值为 1.0。

    返回：
        float: 两点之间的加权曼哈顿距离。
    """
    return weight_x * abs(point1[0] - point2[0]) + weight_y * abs(point1[1] - point2[1])


def sort_normal_blocks(
    blocks, text_line_height, text_line_width, region_direction
) -> List[LayoutBlock]:
    """
    根据块在页面中的位置对块进行排序。

    参数：
        blocks (List[LayoutBlock]): 待排序的块列表。
        text_line_height (int): 每行文本的高度。
        text_line_width (int): 每行文本的宽度。
        region_direction (str): 区域方向，取值为 "horizontal"（水平）或 "vertical"（垂直）。

    返回：
        List[LayoutBlock]: 排序后的块列表。
    """
    if region_direction == "horizontal":
        blocks.sort(
            key=lambda x: (
                x.bbox[1] // text_line_height,
                x.bbox[0] // text_line_width,
                x.get_centroid()[0] ** 2 + x.get_centroid()[1] ** 2,
            ),
        )
    else:
        blocks.sort(
            key=lambda x: (
                -x.bbox[2] // text_line_width,
                x.bbox[1] // text_line_height,
                -x.get_centroid()[0] ** 2 + x.get_centroid()[1] ** 2,
            ),
        )
    return blocks


def get_cut_blocks(blocks, cut_direction, cut_coordinates, mask_labels=[]):
    """
    根据指定的切分方向和坐标对块进行切分。

    参数：
        blocks (list): 待切分的块列表。
        cut_direction (str): 切分方向，取值为 "horizontal"（水平）或 "vertical"（垂直）。
        cut_coordinates (list): 切分坐标列表。

    返回：
        list: 包含切分后块及其对应平均宽度的元组列表。
    """
    cuted_list = []
    # filter out mask blocks,including header, footer, unordered and child_blocks

    # 0: horizontal, 1: vertical
    cut_aixis = 0 if cut_direction == "horizontal" else 1
    blocks.sort(key=lambda x: x.bbox[cut_aixis + 2])
    cut_coordinates.append(float("inf"))

    cut_coordinates = list(set(cut_coordinates))
    cut_coordinates.sort()

    cut_idx = 0
    for cut_coordinate in cut_coordinates:
        group_blocks = []
        block_idx = cut_idx
        while block_idx < len(blocks):
            block = blocks[block_idx]
            if block.bbox[cut_aixis + 2] > cut_coordinate:
                break
            elif block.order_label not in mask_labels:
                group_blocks.append(block)
            block_idx += 1
        cut_idx = block_idx
        if group_blocks:
            cuted_list.append(group_blocks)

    return cuted_list


def get_blocks_by_direction_interval(
    blocks: List[LayoutBlock],
    start_index: int,
    end_index: int,
    direction: str = "horizontal",
) -> List[LayoutBlock]:
    """
    Get blocks within a specified direction interval.

    Args:
        blocks (List[LayoutBlock]): A list of blocks.
        start_index (int): The starting index of the direction.
        end_index (int): The ending index of the direction.
        direction (str, optional): The direction to consider. Defaults to "horizontal".

    Returns:
        List[LayoutBlock]: A list of blocks within the specified direction interval.
    """
    interval_blocks = []
    aixis = 0 if direction == "horizontal" else 1
    blocks.sort(key=lambda x: x.bbox[aixis + 2])

    for block in blocks:
        if block.bbox[aixis] >= start_index and block.bbox[aixis + 2] <= end_index:
            interval_blocks.append(block)

    return interval_blocks


def get_nearest_blocks(
    block: LayoutBlock,
    ref_blocks: List[LayoutBlock],
    overlap_threshold,
    direction="horizontal",
) -> List:
    """
    Get the adjacent blocks with the same direction as the current block.
    Args:
        block (LayoutBlock): The current block.
        blocks (List[LayoutBlock]): A list of all blocks.
        ref_block_idxes (List[int]): A list of indices of reference blocks.
        iou_threshold (float): The IOU threshold to determine if two blocks are considered adjacent.
    Returns:
        Int: The index of the previous block with same direction.
        Int: The index of the following block with same direction.
    """
    prev_blocks: List[LayoutBlock] = []
    post_blocks: List[LayoutBlock] = []
    sort_index = 1 if direction == "horizontal" else 0
    for ref_block in ref_blocks:
        if ref_block.index == block.index:
            continue
        overlap_ratio = calculate_projection_overlap_ratio(
            block.bbox, ref_block.bbox, direction, mode="small"
        )
        if overlap_ratio > overlap_threshold:
            if ref_block.bbox[sort_index] <= block.bbox[sort_index]:
                prev_blocks.append(ref_block)
            else:
                post_blocks.append(ref_block)

    if prev_blocks:
        prev_blocks.sort(key=lambda x: x.bbox[sort_index], reverse=True)
    if post_blocks:
        post_blocks.sort(key=lambda x: x.bbox[sort_index])

    return prev_blocks, post_blocks


def update_doc_title_child_blocks(
    block: LayoutBlock,
    region: LayoutRegion,
) -> None:
    """
    Update the child blocks of a document title block.

    The child blocks need to meet the following conditions:
        1. They must be adjacent
        2. They must have the same direction as the parent block.
        3. Their short side length should be less than 80% of the parent's short side length.
        4. Their long side length should be less than 150% of the parent's long side length.
        5. The child block must be text block.
        6. The nearest edge distance should be less than 2 times of the text line height.

    Args:
        blocks (List[LayoutBlock]): overall blocks.
        block (LayoutBlock): document title block.
        prev_idx (int): previous block index, None if not exist.
        post_idx (int): post block index, None if not exist.
        config (dict): configurations.

    Returns:
        None

    """
    ref_blocks = [region.block_map[idx] for idx in region.normal_text_block_idxes]
    overlap_threshold = XYCUT_SETTINGS["child_block_overlap_ratio_threshold"]
    prev_blocks, post_blocks = get_nearest_blocks(
        block, ref_blocks, overlap_threshold, block.direction
    )
    prev_block = None
    post_block = None

    if prev_blocks:
        prev_block = prev_blocks[0]
    if post_blocks:
        post_block = post_blocks[0]

    for ref_block in [prev_block, post_block]:
        if ref_block is None:
            continue
        with_seem_direction = ref_block.direction == block.direction

        short_side_length_condition = (
            ref_block.short_side_length < block.short_side_length * 0.8
        )

        long_side_length_condition = (
            ref_block.long_side_length < block.long_side_length
            or ref_block.long_side_length > 1.5 * block.long_side_length
        )

        nearest_edge_distance = get_nearest_edge_distance(block.bbox, ref_block.bbox)

        if (
            with_seem_direction
            and ref_block.label in BLOCK_LABEL_MAP["text_labels"]
            and short_side_length_condition
            and long_side_length_condition
            and ref_block.num_of_lines < 3
            and nearest_edge_distance < ref_block.text_line_height * 2
        ):
            ref_block.order_label = "doc_title_text"
            block.append_child_block(ref_block)
            region.normal_text_block_idxes.remove(ref_block.index)

    for ref_block in ref_blocks:
        if ref_block.order_label == "doc_title_text":
            continue
        with_seem_direction = ref_block.direction == block.direction

        overlap_ratio = calculate_overlap_ratio(
            block.bbox, ref_block.bbox, mode="small"
        )

        if overlap_ratio > 0.9 and with_seem_direction:
            ref_block.order_label = "doc_title_text"
            block.append_child_block(ref_block)
            region.normal_text_block_idxes.remove(ref_block.index)


def update_paragraph_title_child_blocks(
    block: LayoutBlock,
    region: LayoutRegion,
) -> None:
    """
    Update the child blocks of a paragraph title block.

    The child blocks need to meet the following conditions:
        1. They must be adjacent
        2. They must have the same direction as the parent block.
        3. The child block must be paragraph title block.

    Args:
        blocks (List[LayoutBlock]): overall blocks.
        block (LayoutBlock): document title block.
        prev_idx (int): previous block index, None if not exist.
        post_idx (int): post block index, None if not exist.
        config (dict): configurations.

    Returns:
        None

    """
    if block.order_label == "sub_paragraph_title":
        return
    ref_blocks = [
        region.block_map[idx]
        for idx in region.paragraph_title_block_idxes + region.normal_text_block_idxes
    ]
    overlap_threshold = XYCUT_SETTINGS["child_block_overlap_ratio_threshold"]
    prev_blocks, post_blocks = get_nearest_blocks(
        block, ref_blocks, overlap_threshold, block.direction
    )
    for ref_blocks in [prev_blocks, post_blocks]:
        for ref_block in ref_blocks:
            if ref_block.label not in BLOCK_LABEL_MAP["paragraph_title_labels"]:
                break
            min_text_line_height = min(
                block.text_line_height, ref_block.text_line_height
            )
            nearest_edge_distance = get_nearest_edge_distance(
                block.bbox, ref_block.bbox
            )
            with_seem_direction = ref_block.direction == block.direction
            with_seem_start = (
                abs(ref_block.start_coordinate - block.start_coordinate)
                < min_text_line_height * 2
            )
            if (
                with_seem_direction
                and with_seem_start
                and nearest_edge_distance <= min_text_line_height * 1.5
            ):
                ref_block.order_label = "sub_paragraph_title"
                block.append_child_block(ref_block)
                region.paragraph_title_block_idxes.remove(ref_block.index)


def update_vision_child_blocks(
    block: LayoutBlock,
    region: LayoutRegion,
) -> None:
    """
    Update the child blocks of a paragraph title block.

    The child blocks need to meet the following conditions:
    - For Both:
        1. They must be adjacent
        2. The child block must be vision_title or text block.
    - For vision_title:
        1. The distance between the child block and the parent block should be less than 1/2 of the parent's height.
    - For text block:
        1. The distance between the child block and the parent block should be less than 15.
        2. The child short_side_length should be less than the parent's short side length.
        3. The child long_side_length should be less than 50% of the parent's long side length.
        4. The difference between their centers is very small.

    Args:
        blocks (List[LayoutBlock]): overall blocks.
        block (LayoutBlock): document title block.
        ref_block_idxes (List[int]): A list of indices of reference blocks.
        prev_idx (int): previous block index, None if not exist.
        post_idx (int): post block index, None if not exist.
        config (dict): configurations.

    Returns:
        None

    """
    ref_blocks = [
        region.block_map[idx]
        for idx in region.normal_text_block_idxes + region.vision_title_block_idxes
    ]
    overlap_threshold = XYCUT_SETTINGS["child_block_overlap_ratio_threshold"]
    has_vision_footnote = False
    has_vision_title = False
    for direction in [block.direction, block.secondary_direction]:
        prev_blocks, post_blocks = get_nearest_blocks(
            block, ref_blocks, overlap_threshold, direction
        )
        for ref_block in prev_blocks:
            if (
                ref_block.label
                not in BLOCK_LABEL_MAP["text_labels"]
                + BLOCK_LABEL_MAP["vision_title_labels"]
            ):
                break
            nearest_edge_distance = get_nearest_edge_distance(
                block.bbox, ref_block.bbox
            )
            block_center = block.get_centroid()
            ref_block_center = ref_block.get_centroid()
            if (
                ref_block.label in BLOCK_LABEL_MAP["vision_title_labels"]
                and nearest_edge_distance <= ref_block.text_line_height * 2
            ):
                has_vision_title = True
                ref_block.order_label = "vision_title"
                block.append_child_block(ref_block)
                region.vision_title_block_idxes.remove(ref_block.index)
            if ref_block.label in BLOCK_LABEL_MAP["text_labels"]:
                if (
                    not has_vision_footnote
                    and ref_block.direction == block.direction
                    and ref_block.long_side_length < block.long_side_length
                    and nearest_edge_distance <= ref_block.text_line_height * 2
                ):
                    if (
                        (
                            ref_block.short_side_length < block.short_side_length
                            and ref_block.long_side_length
                            < 0.5 * block.long_side_length
                            and abs(block_center[0] - ref_block_center[0]) < 10
                        )
                        or (
                            block.bbox[0] - ref_block.bbox[0] < 10
                            and ref_block.num_of_lines == 1
                        )
                        or (
                            block.bbox[2] - ref_block.bbox[2] < 10
                            and ref_block.num_of_lines == 1
                        )
                    ):
                        has_vision_footnote = True
                        ref_block.order_label = "vision_footnote"
                        block.append_child_block(ref_block)
                        region.normal_text_block_idxes.remove(ref_block.index)
                break
        for ref_block in post_blocks:
            if (
                has_vision_footnote
                and ref_block.label in BLOCK_LABEL_MAP["text_labels"]
            ):
                break
            nearest_edge_distance = get_nearest_edge_distance(
                block.bbox, ref_block.bbox
            )
            block_center = block.get_centroid()
            ref_block_center = ref_block.get_centroid()
            if (
                ref_block.label in BLOCK_LABEL_MAP["vision_title_labels"]
                and nearest_edge_distance <= ref_block.text_line_height * 2
            ):
                has_vision_title = True
                ref_block.order_label = "vision_title"
                block.append_child_block(ref_block)
                region.vision_title_block_idxes.remove(ref_block.index)
            if ref_block.label in BLOCK_LABEL_MAP["text_labels"]:
                if (
                    not has_vision_footnote
                    and ref_block.direction == block.direction
                    and ref_block.long_side_length < block.long_side_length
                    and nearest_edge_distance <= ref_block.text_line_height * 2
                ):
                    if (
                        (
                            ref_block.short_side_length < block.short_side_length
                            and ref_block.long_side_length
                            < 0.5 * block.long_side_length
                            and abs(block_center[0] - ref_block_center[0]) < 10
                        )
                        or (
                            block.bbox[0] - ref_block.bbox[0] < 10
                            and ref_block.num_of_lines == 1
                        )
                        or (
                            block.bbox[2] - ref_block.bbox[2] < 10
                            and ref_block.num_of_lines == 1
                        )
                    ):
                        has_vision_footnote = True
                        ref_block.label = "vision_footnote"
                        ref_block.order_label = "vision_footnote"
                        block.append_child_block(ref_block)
                        region.normal_text_block_idxes.remove(ref_block.index)
                break
        if has_vision_title:
            break

    for ref_block in ref_blocks:
        if ref_block.index not in region.normal_text_block_idxes:
            continue

        overlap_ratio = calculate_overlap_ratio(
            block.bbox, ref_block.bbox, mode="small"
        )

        if overlap_ratio > 0.9:
            ref_block.label = "vision_footnote"
            ref_block.order_label = "vision_footnote"
            block.append_child_block(ref_block)
            region.normal_text_block_idxes.remove(ref_block.index)


def update_region_child_blocks(
    block: LayoutBlock,
    region: LayoutRegion,
) -> None:
    """Update child blocks of a region.

    Args:
        block (LayoutBlock): document title block.
        region (LayoutRegion): layout region.

    Returns:
        None
    """
    for ref_block in region.block_map.values():
        if block.index != ref_block.index:
            bbox_iou = calculate_overlap_ratio(block.bbox, ref_block.bbox)
            if (
                bbox_iou > 0
                and block.area > ref_block.area
                and ref_block.order_label != "sub_region"
            ):
                ref_block.order_label = "sub_region"
                block.append_child_block(ref_block)
                region.normal_text_block_idxes.remove(ref_block.index)


def calculate_discontinuous_projection(
    boxes, direction="horizontal", return_num=False
) -> List:
    """
    Calculate the discontinuous projection of boxes along the specified direction.

    Args:
        boxes (ndarray): Array of bounding boxes represented by [[x_min, y_min, x_max, y_max]].
        direction (str): direction along which to perform the projection ('horizontal' or 'vertical').

    Returns:
        list: List of tuples representing the merged intervals.
    """
    boxes = np.array(boxes)
    if direction == "horizontal":
        intervals = boxes[:, [0, 2]]
    elif direction == "vertical":
        intervals = boxes[:, [1, 3]]
    else:
        raise ValueError("direction must be 'horizontal' or 'vertical'")

    intervals = intervals[np.argsort(intervals[:, 0])]

    merged_intervals = []
    num = 1
    current_start, current_end = intervals[0]
    num_list = []

    for start, end in intervals[1:]:
        if start <= current_end:
            num += 1
            current_end = max(current_end, end)
        else:
            num_list.append(num)
            merged_intervals.append((current_start, current_end))
            num = 1
            current_start, current_end = start, end

    num_list.append(num)
    merged_intervals.append((current_start, current_end))
    if return_num:
        return merged_intervals, num_list
    return merged_intervals


def shrink_overlapping_boxes(
    boxes, direction="horizontal", min_threshold=0, max_threshold=0.1
) -> List:
    """
    Shrink overlapping boxes along the specified direction.

    Args:
        boxes (ndarray): Array of bounding boxes represented by [[x_min, y_min, x_max, y_max]].
        direction (str): direction along which to perform the shrinking ('horizontal' or 'vertical').
        min_threshold (float): Minimum threshold for shrinking. Default is 0.
        max_threshold (float): Maximum threshold for shrinking. Default is 0.2.

    Returns:
        list: List of tuples representing the merged intervals.
    """
    current_block = boxes[0]
    for block in boxes[1:]:
        x1, y1, x2, y2 = current_block.bbox
        x1_prime, y1_prime, x2_prime, y2_prime = block.bbox
        cut_iou = calculate_projection_overlap_ratio(
            current_block.bbox, block.bbox, direction=direction
        )
        match_iou = calculate_projection_overlap_ratio(
            current_block.bbox,
            block.bbox,
            direction="horizontal" if direction == "vertical" else "vertical",
        )
        if direction == "vertical":
            if (
                (match_iou > 0 and cut_iou > min_threshold and cut_iou < max_threshold)
                or y2 == y1_prime
                or abs(y2 - y1_prime) <= 3
            ):
                overlap_y_min = max(y1, y1_prime)
                overlap_y_max = min(y2, y2_prime)
                split_y = int((overlap_y_min + overlap_y_max) / 2)
                overlap_y_min = split_y - 1
                overlap_y_max = split_y + 1
                if y1 < y1_prime:
                    current_block.bbox = [x1, y1, x2, overlap_y_min]
                    block.bbox = [x1_prime, overlap_y_max, x2_prime, y2_prime]
                else:
                    current_block.bbox = [x1, overlap_y_min, x2, y2]
                    block.bbox = [x1_prime, y1_prime, x2_prime, overlap_y_max]
        else:
            if (
                (match_iou > 0 and cut_iou > min_threshold and cut_iou < max_threshold)
                or x2 == x1_prime
                or abs(x2 - x1_prime) <= 3
            ):
                overlap_x_min = max(x1, x1_prime)
                overlap_x_max = min(x2, x2_prime)
                split_x = int((overlap_x_min + overlap_x_max) / 2)
                overlap_x_min = split_x - 1
                overlap_x_max = split_x + 1
                if x1 < x1_prime:
                    current_block.bbox = [x1, y1, overlap_x_min, y2]
                    block.bbox = [overlap_x_max, y1_prime, x2_prime, y2_prime]
                else:
                    current_block.bbox = [overlap_x_min, y1, x2, y2]
                    block.bbox = [x1_prime, y1_prime, overlap_x_max, y2_prime]
        current_block = block
    return boxes


def find_local_minima_flat_regions(arr) -> List:
    """
    Find all local minima regions in a flat array.

    Args:
        arr (list): The input array.

    Returns:
        list: A list of tuples containing the indices of the local minima regions.
    """
    n = len(arr)
    if n == 0:
        return []

    flat_minima_regions = []
    start = 0

    for i in range(1, n):
        if arr[i] != arr[i - 1]:
            if (start == 0 or arr[start - 1] > arr[start]) and (
                i == n or arr[i] > arr[start]
            ):
                flat_minima_regions.append((start, i - 1))
            start = i

    return flat_minima_regions[1:] if len(flat_minima_regions) > 1 else None
