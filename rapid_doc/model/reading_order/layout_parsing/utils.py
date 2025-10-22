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

import re
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union


def calculate_projection_overlap_ratio(
    bbox1: List[float],
    bbox2: List[float],
    direction: str = "horizontal",
    mode="union",
) -> float:
    """
    Calculate the IoU of lines between two bounding boxes.

    Args:
        bbox1 (List[float]): First bounding box [x_min, y_min, x_max, y_max].
        bbox2 (List[float]): Second bounding box [x_min, y_min, x_max, y_max].
        direction (str): direction of the projection, "horizontal" or "vertical".

    Returns:
        float: Line overlap ratio. Returns 0 if there is no overlap.
    """
    start_index, end_index = 1, 3
    if direction == "horizontal":
        start_index, end_index = 0, 2

    intersection_start = max(bbox1[start_index], bbox2[start_index])
    intersection_end = min(bbox1[end_index], bbox2[end_index])
    overlap = intersection_end - intersection_start
    if overlap <= 0:
        return 0

    if mode == "union":
        ref_width = max(bbox1[end_index], bbox2[end_index]) - min(
            bbox1[start_index], bbox2[start_index]
        )
    elif mode == "small":
        ref_width = min(
            bbox1[end_index] - bbox1[start_index], bbox2[end_index] - bbox2[start_index]
        )
    elif mode == "large":
        ref_width = max(
            bbox1[end_index] - bbox1[start_index], bbox2[end_index] - bbox2[start_index]
        )
    else:
        raise ValueError(
            f"Invalid mode {mode}, must be one of ['union', 'small', 'large']."
        )

    return overlap / ref_width if ref_width > 0 else 0.0


def calculate_overlap_ratio(
    bbox1: Union[list, tuple], bbox2: Union[list, tuple], mode="union"
) -> float:
    """
    Calculate the overlap ratio between two bounding boxes.

    Args:
        bbox1 (list or tuple): The first bounding box, format [x_min, y_min, x_max, y_max]
        bbox2 (list or tuple): The second bounding box, format [x_min, y_min, x_max, y_max]
        mode (str): The mode of calculation, either 'union', 'small', or 'large'.

    Returns:
        float: The overlap ratio value between the two bounding boxes
    """
    x_min_inter = max(bbox1[0], bbox2[0])
    y_min_inter = max(bbox1[1], bbox2[1])
    x_max_inter = min(bbox1[2], bbox2[2])
    y_max_inter = min(bbox1[3], bbox2[3])

    inter_width = max(0, x_max_inter - x_min_inter)
    inter_height = max(0, y_max_inter - y_min_inter)

    inter_area = float(inter_width) * float(inter_height)

    bbox1_area = caculate_bbox_area(bbox1)
    bbox2_area = caculate_bbox_area(bbox2)

    if mode == "union":
        ref_area = bbox1_area + bbox2_area - inter_area
    elif mode == "small":
        ref_area = min(bbox1_area, bbox2_area)
    elif mode == "large":
        ref_area = max(bbox1_area, bbox2_area)
    else:
        raise ValueError(
            f"Invalid mode {mode}, must be one of ['union', 'small', 'large']."
        )

    if ref_area == 0:
        return 0.0

    return inter_area / ref_area

def is_english_letter(char):
    """check if the char is english letter"""
    return bool(re.match(r"^[A-Za-z]$", char))


def is_numeric(char):
    """check if the char is numeric"""
    return bool(re.match(r"^[\d]+$", char))


def is_non_breaking_punctuation(char):
    """
    check if the char is non-breaking punctuation

    Args:
        char (str): character to check

    Returns:
        bool: True if the char is non-breaking punctuation
    """
    non_breaking_punctuations = {
        ",",
        "，",
        "、",
        ";",
        "；",
        ":",
        "：",
        "-",
        "'",
        '"',
        "“",
    }

    return char in non_breaking_punctuations


def caculate_bbox_area(bbox):
    """Calculate bounding box area"""
    x1, y1, x2, y2 = map(float, bbox)
    area = abs((x2 - x1) * (y2 - y1))
    return area


def caculate_euclidean_dist(point1, point2):
    """Calculate euclidean distance between two points"""
    x1, y1 = point1
    x2, y2 = point2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_seg_flag(block, prev_block):
    """Get segment start flag and end flag based on previous block

    Args:
        block (Block): Current block
        prev_block (Block): Previous block

    Returns:
        seg_start_flag (bool): Segment start flag
        seg_end_flag (bool): Segment end flag
    """

    seg_start_flag = True
    seg_end_flag = True

    context_left_coordinate = block.start_coordinate
    context_right_coordinate = block.end_coordinate
    seg_start_coordinate = block.seg_start_coordinate
    seg_end_coordinate = block.seg_end_coordinate

    if prev_block is not None:
        num_of_prev_lines = prev_block.num_of_lines
        pre_block_seg_end_coordinate = prev_block.seg_end_coordinate
        prev_end_space_small = (
            abs(prev_block.end_coordinate - pre_block_seg_end_coordinate) < 10
        )
        prev_lines_more_than_one = num_of_prev_lines > 1

        overlap_blocks = (
            context_left_coordinate < prev_block.end_coordinate
            and context_right_coordinate > prev_block.start_coordinate
        )

        # update context_left_coordinate and context_right_coordinate
        if overlap_blocks:
            context_left_coordinate = min(
                prev_block.start_coordinate, context_left_coordinate
            )
            context_right_coordinate = max(
                prev_block.end_coordinate, context_right_coordinate
            )
            prev_end_space_small = (
                abs(context_right_coordinate - pre_block_seg_end_coordinate) < 10
            )
            edge_distance = 0
        else:
            edge_distance = abs(block.start_coordinate - prev_block.end_coordinate)

        current_start_space_small = seg_start_coordinate - context_left_coordinate < 10

        if (
            prev_end_space_small
            and current_start_space_small
            and prev_lines_more_than_one
            and edge_distance < max(prev_block.width, block.width)
        ):
            seg_start_flag = False
    else:
        if seg_start_coordinate - context_left_coordinate < 10:
            seg_start_flag = False

    if context_right_coordinate - seg_end_coordinate < 10:
        seg_end_flag = False

    return seg_start_flag, seg_end_flag

def update_region_box(bbox, region_box):
    """Update region box with bbox"""
    if region_box is None:
        return bbox

    x1, y1, x2, y2 = bbox
    x1_region, y1_region, x2_region, y2_region = region_box

    x1_region = int(min(x1, x1_region))
    y1_region = int(min(y1, y1_region))
    x2_region = int(max(x2, x2_region))
    y2_region = int(max(y2, y2_region))

    region_box = [x1_region, y1_region, x2_region, y2_region]

    return region_box

def _get_minbox_if_overlap_by_ratio(
    bbox1: Union[List[int], Tuple[int, int, int, int]],
    bbox2: Union[List[int], Tuple[int, int, int, int]],
    ratio: float,
    smaller: bool = True,
) -> Optional[Union[List[int], Tuple[int, int, int, int]]]:
    """
    Determine if the overlap area between two bounding boxes exceeds a given ratio
    and return the smaller (or larger) bounding box based on the `smaller` flag.

    Args:
        bbox1 (Union[List[int], Tuple[int, int, int, int]]): Coordinates of the first bounding box [x_min, y_min, x_max, y_max].
        bbox2 (Union[List[int], Tuple[int, int, int, int]]): Coordinates of the second bounding box [x_min, y_min, x_max, y_max].
        ratio (float): The overlap ratio threshold.
        smaller (bool): If True, return the smaller bounding box; otherwise, return the larger one.

    Returns:
        Optional[Union[List[int], Tuple[int, int, int, int]]]:
            The selected bounding box or None if the overlap ratio is not exceeded.
    """
    # Calculate the areas of both bounding boxes
    area1 = caculate_bbox_area(bbox1)
    area2 = caculate_bbox_area(bbox2)
    # Calculate the overlap ratio using a helper function
    overlap_ratio = calculate_overlap_ratio(bbox1, bbox2, mode="small")
    # Check if the overlap ratio exceeds the threshold
    if overlap_ratio > ratio:
        if (area1 <= area2 and smaller) or (area1 >= area2 and not smaller):
            return 1
        else:
            return 2
    return None

def remove_overlap_blocks(
    blocks: List[Dict[str, List[int]]], threshold: float = 0.65, smaller: bool = True
) -> Tuple[List[Dict[str, List[int]]], List[Dict[str, List[int]]]]:
    """
    Remove overlapping blocks based on a specified overlap ratio threshold.

    Args:
        blocks (List[Dict[str, List[int]]]): List of block dictionaries, each containing a 'block_bbox' key.
        threshold (float): Ratio threshold to determine significant overlap.
        smaller (bool): If True, the smaller block in overlap is removed.

    Returns:
        Tuple[List[Dict[str, List[int]]], List[Dict[str, List[int]]]]:
            A tuple containing the updated list of blocks and a list of dropped blocks.
    """
    dropped_indexes = set()
    blocks = deepcopy(blocks)
    overlap_image_blocks = []
    # Iterate over each pair of blocks to find overlaps
    for i, block1 in enumerate(blocks["boxes"]):
        for j in range(i + 1, len(blocks["boxes"])):
            block2 = blocks["boxes"][j]
            # Skip blocks that are already marked for removal
            if i in dropped_indexes or j in dropped_indexes:
                continue
            # Check for overlap and determine which block to remove
            overlap_box_index = _get_minbox_if_overlap_by_ratio(
                block1["coordinate"],
                block2["coordinate"],
                threshold,
                smaller=smaller,
            )
            if overlap_box_index is not None:
                is_block1_image = block1["label"] == "image"
                is_block2_image = block2["label"] == "image"

                if is_block1_image != is_block2_image:
                    # 如果只有一个块在视觉标签中，删除在视觉标签中的那个块
                    drop_index = i if is_block1_image else j
                    overlap_image_blocks.append(blocks["boxes"][drop_index])
                else:
                    # 如果两个块都在或都不在视觉标签中，根据 overlap_box_index 决定删除哪个块
                    drop_index = i if overlap_box_index == 1 else j

                dropped_indexes.add(drop_index)

    # Remove marked blocks from the original list
    for index in sorted(dropped_indexes, reverse=True):
        del blocks["boxes"][index]

    return blocks
