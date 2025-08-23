from copy import deepcopy
from typing import Any, List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def caculate_bbox_area(bbox):
    """Calculate bounding box area"""
    x1, y1, x2, y2 = map(float, bbox)
    area = abs((x2 - x1) * (y2 - y1))
    return area

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

def _get_minbox_if_overlap_by_ratio(
    bbox1: Union[List[int], Tuple[int, int, int, int]],
    bbox2: Union[List[int], Tuple[int, int, int, int]],
    ratio: float,
    smaller: bool = True,
) -> Optional[int]:
    """
    Determine if the overlap area between two bounding boxes exceeds a given ratio
    and return which one to drop (1 for bbox1, 2 for bbox2).

    Args:
        bbox1: Coordinates of the first bounding box [x_min, y_min, x_max, y_max].
        bbox2: Coordinates of the second bounding box [x_min, y_min, x_max, y_max].
        ratio: The overlap ratio threshold.
        smaller: If True, drop the smaller bounding box; otherwise, drop the larger one.

    Returns:
        Optional[int]: 1 if bbox1 should be dropped, 2 if bbox2 should be dropped, None otherwise.
    """
    area1 = caculate_bbox_area(bbox1)
    area2 = caculate_bbox_area(bbox2)
    overlap_ratio = calculate_overlap_ratio(bbox1, bbox2, mode="small")

    if overlap_ratio > ratio:
        if (area1 <= area2 and smaller) or (area1 >= area2 and not smaller):
            return 1
        else:
            return 2
    return None


def remove_overlap_blocks(
    bboxes: List[List[int]], threshold: float = 0.65, smaller: bool = True
) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Remove overlapping bounding boxes based on a specified overlap ratio threshold.

    Args:
        bboxes: List of bounding boxes, each in format [x_min, y_min, x_max, y_max].
        threshold: Ratio threshold to determine significant overlap.
        smaller: If True, the smaller block in overlap is removed.

    Returns:
        Tuple[List[List[int]], List[List[int]]]:
            A tuple containing the updated list of bounding boxes and a list of dropped boxes.
    """
    dropped_indexes = set()
    bboxes = deepcopy(bboxes)
    dropped_boxes = []

    for i, bbox1 in enumerate(bboxes):
        for j in range(i + 1, len(bboxes)):
            bbox2 = bboxes[j]
            if i in dropped_indexes or j in dropped_indexes:
                continue

            drop_flag = _get_minbox_if_overlap_by_ratio(
                bbox1, bbox2, threshold, smaller=smaller
            )

            if drop_flag is not None:
                drop_index = i if drop_flag == 1 else j
                dropped_indexes.add(drop_index)

    for index in sorted(dropped_indexes, reverse=True):
        dropped_boxes.append(bboxes[index])
        del bboxes[index]

    return bboxes, dropped_boxes


def visualize_reading_order(blocks: List[Dict[str, Any]], image_path: str = "layout_res_out.png", figsize=(12, 16)):
    """
    可视化阅读顺序

    参数:
        blocks: 带有阅读顺序的区块列表
        image_path: 输出图像路径
        figsize: 图像大小
    """
    if not blocks:
        print("No blocks to visualize")
        return

    # 计算页面大小
    max_x = max([block["bbox"][2] for block in blocks])
    max_y = max([block["bbox"][3] for block in blocks])

    # 创建图像
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, max_x)
    ax.set_ylim(max_y, 0)  # 反转y轴，使得原点在左上角

    # 绘制每个区块
    for block in blocks:
        x1, y1, x2, y2 = block["bbox"]
        width = x2 - x1
        height = y2 - y1

        # 获取区块颜色
        color = "white"

        # 绘制矩形
        rect = patches.Rectangle((x1, y1), width, height, linewidth=1,
                                 edgecolor="black", facecolor=color, alpha=0.5)
        ax.add_patch(rect)

        # 添加标签和阅读顺序
        ax.text(x1 + width / 2, y1 + height / 2,
                f"{block['reading_order']}\n{block['label']}",
                ha="center", va="center", fontsize=8)

    # 保存图像
    plt.title("Document Reading Order Visualization")
    plt.savefig(image_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved to {image_path}")