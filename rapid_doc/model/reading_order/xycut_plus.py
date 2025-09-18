from rapid_doc.model.layout.rapid_layout_self import ModelType, RapidLayout, RapidLayoutInput
import numpy as np
from typing import Any, List, Dict

from rapid_doc.model.reading_order.utils import remove_overlap_blocks, visualize_reading_order


def projection_by_bboxes(boxes: np.ndarray, axis: int) -> np.ndarray:
    """
    Generate a 1D projection histogram from bounding boxes along a specified axis.

    Args:
        boxes: A (N, 4) array of bounding boxes defined by [x_min, y_min, x_max, y_max].
        axis: Axis for projection; 0 for horizontal (x-axis), 1 for vertical (y-axis).

    Returns:
        A 1D numpy array representing the projection histogram based on bounding box intervals.
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
    Split the projection profile into segments based on specified thresholds.

    Args:
        arr_values: 1D array representing the projection profile.
        min_value: Minimum value threshold to consider a profile segment significant.
        min_gap: Minimum gap width to consider a separation between segments.

    Returns:
        A tuple of start and end indices for each segment that meets the criteria.
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
    Recursively project and segment bounding boxes, starting with Y-axis and followed by X-axis.

    Args:
        boxes: A (N, 4) array representing bounding boxes.
        indices: List of indices indicating the original position of boxes.
        res: List to store indices of the final segmented bounding boxes.
        min_gap (int): Minimum gap width to consider a separation between segments on the X-axis. Defaults to 1.

    Returns:
        None: This function modifies the `res` list in place.
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
    Recursively performs X-axis projection followed by Y-axis projection to segment bounding boxes.

    Args:
        boxes: A (N, 4) array representing bounding boxes with [x_min, y_min, x_max, y_max].
        indices: A list of indices representing the position of boxes in the original data.
        res: A list to store indices of bounding boxes that meet the criteria.
        min_gap (int): Minimum gap width to consider a separation between segments on the X-axis. Defaults to 1.

    Returns:
        None: This function modifies the `res` list in place.
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

def get_bbox_direction(width, height, direction_ratio: float = 1.0) -> str:
    """
    Determine if a bounding box is horizontal or vertical.

    Args:
        direction_ratio (float): Ratio for determining direction. Default is 1.0.

    Returns:
        str: "horizontal" or "vertical".
    """
    return (
        "horizontal" if width * direction_ratio >= height else "vertical"
    )


def calculate_text_line_direction(
    bboxes: List[List[int]], direction_ratio: float = 1.5
) -> str:
    """
    Calculate the direction of the text based on the bounding boxes.

    Args:
        bboxes (list): A list of bounding boxes.
        direction_ratio (float): Ratio for determining direction. Default is 1.5.

    Returns:
        str: "horizontal" or "vertical".
    """

    horizontal_box_num = 0
    for bbox in bboxes:
        if len(bbox) != 4:
            raise ValueError(
                "Invalid bounding box format. Expected a list of length 4."
            )
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        horizontal_box_num += 1 if width * direction_ratio >= height else 0

    return "horizontal" if horizontal_box_num >= len(bboxes) * 0.5 else "vertical"


def sort_by_xycut(
    block_bboxes: List,
    direction: str = "vertical",
    min_gap: int = 1,
) -> List[int]:
    """
    Sort bounding boxes using recursive XY cut method based on the specified direction.

    Args:
        block_bboxes (Union[np.ndarray, List[List[int]]]): An array or list of bounding boxes,
                                                           where each box is represented as
                                                           [x_min, y_min, x_max, y_max].
        direction (int): direction for the initial cut. Use 1 for Y-axis first and 0 for X-axis first.
                         Defaults to 0.
        min_gap (int): Minimum gap width to consider a separation between segments. Defaults to 1.

    Returns:
        List[int]: A list of indices representing the order of sorted bounding boxes.
    """
    block_bboxes = np.asarray(block_bboxes).astype(int)
    res = []
    if direction == "vertical":
        recursive_yx_cut(
            block_bboxes,
            np.arange(len(block_bboxes)).tolist(),
            res,
            min_gap,
        )
    else:
        recursive_xy_cut(
            block_bboxes,
            np.arange(len(block_bboxes)).tolist(),
            res,
            min_gap,
        )
    return res


def xycut_plus_sort(bboxes: List, direction: str = None) -> List[int]:
    """
    基于xycut++算法对版面识别结果进行阅读顺序排序，并返回排序索引列表

    参数:
        bboxes: 版面识别结果bbox列表
        direction: 排序方向，"horizontal"表示从左到右从上到下，
                   "vertical"表示从上到下从左到右

    返回:
        List[int]: 排序后的索引列表，每个元素是原bboxes的索引，类型为int
    """
    if not bboxes:
        return []

    if not direction:
        direction = calculate_text_line_direction(bboxes)
        # print("xycut_pro_sort direction", direction)

    # 使用xycut算法获取排序索引
    sorted_indices = sort_by_xycut(bboxes, direction=direction)

    # 转成Python int列表返回
    return [int(idx) for idx in sorted_indices]


# =========================以下为测试方法=========================

# 测试xycut++算法
if __name__ == "__main__":
    cfg = RapidLayoutInput(model_type=ModelType.PP_DOCLAYOUT_L, conf_thresh=0.5,
                           model_dir_or_path="C:\ocr\models\ppmodel\layout\PP-DocLayout-L\pp_doclayout_l.onnx")
    layout_engine = RapidLayout(cfg=cfg)

    img_path = r"C:\ocr\453897009-46be64ca-3adb-471c-a43d-0226fee10f73.png"
    results = layout_engine([img_path])
    results[0].vis(r"C:\ocr\453897009-46be64ca-3adb-471c-a43d-0226fee10f73_vis.png")
    data_list = [list(map(float, box)) for box in results[0].boxes]

    # data_list, _ = remove_overlap_blocks(
    #     data_list,
    #     threshold=0.5,
    #     smaller=True,
    # )

    # 对data_list进行阅读顺序排序
    res = []
    sorted_indices = xycut_plus_sort(data_list)

    sorted_data: List[Dict[str, Any]] = []
    for order, idx in enumerate(sorted_indices):
        sorted_data.append({
            "bbox": data_list[idx],
            "reading_order": order,
            "label": "text"
        })


    print("=== 阅读顺序排序结果 ===")
    for item in sorted_data:
        print(f"顺序: {item['reading_order']}, 标签: {item['label']}, 边界框: {item['bbox']}")

    visualize_reading_order(sorted_data)

    print("\n=== 按标签分组的阅读顺序 ===")
    label_groups = {}
    for item in sorted_data:
        label = item['label']
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(item['reading_order'])

    for label, orders in label_groups.items():
        print(f"{label}: {orders}")
