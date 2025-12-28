from typing import List, Tuple, Union

import numpy as np


class PPPostProcess:
    def __init__(self, labels, conf_thres: Union[float, dict] =0.4, iou_thres=0.5, layout_nms=True):
        self.labels = labels
        self.strides = [8, 16, 32, 64]
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.layout_nms = layout_nms
        self.nms_top_k = 1000
        self.keep_top_k = 100

    def __call__(
        self,
        boxes,
        img_size,
    ):
        """Apply post-processing to the detection boxes.

        Args:
            boxes (ndarray): The input detection boxes with scores.
            img_size (tuple): The original image size.

        Returns:
            Boxes: The post-processed detection boxes.
        """
        # === 1️⃣ 按置信度阈值过滤 ===
        if isinstance(self.conf_thres, float):
            # 单一阈值
            expect_boxes = (boxes[:, 1] >= self.conf_thres) & (boxes[:, 0] > -1)
            boxes = boxes[expect_boxes, :]
        elif isinstance(self.conf_thres, dict):
            # 每个类别使用不同阈值
            category_filtered_boxes = []
            for cat_id in np.unique(boxes[:, 0]):
                category_boxes = boxes[boxes[:, 0] == cat_id]
                category_threshold = self.conf_thres.get(int(cat_id), 0.5)
                selected_indices = (category_boxes[:, 1] >= category_threshold) & (
                        category_boxes[:, 0] > -1
                )
                category_filtered_boxes.append(category_boxes[selected_indices])
            boxes = (
                np.vstack(category_filtered_boxes)
                if category_filtered_boxes
                else np.array([])
            )
        # === 2️⃣ NMS (非极大值抑制) ===
        if self.layout_nms:
            selected_indices = nms(boxes[:, :6], iou_same=0.6, iou_diff=0.98)
            boxes = np.array(boxes[selected_indices])
        # === 3️⃣ 过滤大面积 image 框 ===
        filter_large_image = True
        # boxes.shape[1] == 6 is object detection, 8 is ordered object detection
        if filter_large_image and len(boxes) > 1 and boxes.shape[1] in [6, 8]:
            if img_size[0] > img_size[1]:
                area_thres = 0.82
            else:
                area_thres = 0.93
            image_index = self.labels.index("image") if "image" in self.labels else None
            img_area = img_size[0] * img_size[1]
            filtered_boxes = []
            for box in boxes:
                (
                    label_index,
                    score,
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                ) = box[:6]
                if label_index == image_index:
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(img_size[0], xmax)
                    ymax = min(img_size[1], ymax)
                    box_area = (xmax - xmin) * (ymax - ymin)
                    if box_area <= area_thres * img_area:
                        filtered_boxes.append(box)
                else:
                    filtered_boxes.append(box)
            if len(filtered_boxes) == 0:
                filtered_boxes = boxes
            boxes = np.array(filtered_boxes)
        # === 4️⃣ 格式化输出 ===
        if boxes.size == 0:
            return []

        if boxes.shape[1] == 8:
            # Sort boxes by their order
            sorted_idx = np.lexsort((-boxes[:, 7], boxes[:, 6]))
            sorted_boxes = boxes[sorted_idx]
            boxes = sorted_boxes[:, :6]

        if boxes.shape[1] == 6:
            """For Normal Object Detection"""
            boxes = restructured_boxes(boxes, self.labels, img_size)
        elif boxes.shape[1] == 10:
            """Adapt For Rotated Object Detection"""
            boxes = restructured_rotated_boxes(boxes, self.labels, img_size)
        else:
            """Unexpected Input Box Shape"""
            raise ValueError(
                f"The shape of boxes should be 6 or 10, instead of {boxes.shape[1]}"
            )
        return boxes

def nms(boxes, iou_same=0.6, iou_diff=0.95):
    """Perform Non-Maximum Suppression (NMS) with different IoU thresholds for same and different classes."""
    # Extract class scores
    scores = boxes[:, 1]

    # Sort indices by scores in descending order
    indices = np.argsort(scores)[::-1]
    selected_boxes = []

    while len(indices) > 0:
        current = indices[0]
        current_box = boxes[current]
        current_class = current_box[0]
        current_box[1]
        current_coords = current_box[2:]

        selected_boxes.append(current)
        indices = indices[1:]

        filtered_indices = []
        for i in indices:
            box = boxes[i]
            box_class = box[0]
            box_coords = box[2:]
            iou_value = iou(current_coords, box_coords)
            threshold = iou_same if current_class == box_class else iou_diff

            # If the IoU is below the threshold, keep the box
            if iou_value < threshold:
                filtered_indices.append(i)
        indices = filtered_indices
    return selected_boxes


def iou(box1, box2):
    """Compute the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # Compute the intersection coordinates
    x1_i = max(x1, x1_p)
    y1_i = max(y1, y1_p)
    x2_i = min(x2, x2_p)
    y2_i = min(y2, y2_p)

    # Compute the area of intersection
    inter_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)

    # Compute the area of both bounding boxes
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)

    # Compute the IoU
    iou_value = inter_area / float(box1_area + box2_area - inter_area)

    return iou_value

def restructured_boxes(
    boxes, labels: List[str], img_size: Tuple[int, int]
):
    """
    Restructure the given bounding boxes and labels based on the image size.

    Args:
        boxes (ndarray): A 2D array of bounding boxes with each box represented as [cls_id, score, xmin, ymin, xmax, ymax].
        labels (List[str]): A list of class labels corresponding to the class ids.
        img_size (Tuple[int, int]): A tuple representing the width and height of the image.

    Returns:
        Boxes: A list of dictionaries, each containing 'cls_id', 'label', 'score', and 'coordinate' keys.
    """
    box_list = []
    w, h = img_size

    for box in boxes:
        xmin, ymin, xmax, ymax = box[2:]
        xmin = float(max(0, xmin))
        ymin = float(max(0, ymin))
        xmax = float(min(w, xmax))
        ymax = float(min(h, ymax))
        if xmax <= xmin or ymax <= ymin:
            continue
        box_list.append(
            {
                "cls_id": int(box[0]),
                "label": labels[int(box[0])],
                "score": float(box[1]),
                "coordinate": [xmin, ymin, xmax, ymax],
            }
        )

    return box_list


def restructured_rotated_boxes(
    boxes, labels: List[str], img_size: Tuple[int, int]
):
    """
    Restructure the given rotated bounding boxes and labels based on the image size.

    Args:
        boxes (ndarray): A 2D array of rotated bounding boxes with each box represented as [cls_id, score, x1, y1, x2, y2, x3, y3, x4, y4].
        labels (List[str]): A list of class labels corresponding to the class ids.
        img_size (Tuple[int, int]): A tuple representing the width and height of the image.

    Returns:
        Boxes: A list of dictionaries, each containing 'cls_id', 'label', 'score', and 'coordinate' keys.
    """
    box_list = []
    w, h = img_size

    assert boxes.shape[1] == 10, "The shape of rotated boxes should be [N, 10]"
    for box in boxes:
        x1, y1, x2, y2, x3, y3, x4, y4 = box[2:]
        x1 = min(max(0, x1), w)
        y1 = min(max(0, y1), h)
        x2 = min(max(0, x2), w)
        y2 = min(max(0, y2), h)
        x3 = min(max(0, x3), w)
        y3 = min(max(0, y3), h)
        x4 = min(max(0, x4), w)
        y4 = min(max(0, y4), h)
        box_list.append(
            {
                "cls_id": int(box[0]),
                "label": labels[int(box[0])],
                "score": float(box[1]),
                "coordinate": [x1, y1, x2, y2, x3, y3, x4, y4],
            }
        )

    return box_list
