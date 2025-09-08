import re
from typing import List
from sklearn.cluster import KMeans
import numpy as np

from .table_recognition_post_processing_v2 import get_table_recognition_res

class TableMatchPipeline:

    def __call__(self, pred_structures, cell_bboxes, dt_boxes, rec_res, cell_results, cells_trans_to_html=False):

        table_cells_result, table_cells_score = cell_results
        table_cells_result, table_cells_score = self.cells_det_results_nms(
            table_cells_result, table_cells_score
        )

        if cells_trans_to_html == True:
            table_structure_result = self.trans_cells_det_results_to_html(
                table_cells_result
            )
        else:
            table_structure_result = pred_structures
            table_cells_result = self.cells_det_results_reprocessing(
                table_cells_result,
                table_cells_score,
                dt_boxes,
                len(cell_bboxes),
            )

        # pred_html = get_table_recognition_res(pred_structures, cell_bboxes, dt_boxes, rec_res)
        ocr_texts_res = [text for text, score in rec_res]
        pred_html, cell_bboxes = get_table_recognition_res(
            table_structure_result,
            table_cells_result,
            dt_boxes,
            ocr_texts_res,
        )
        return pred_html, np.array(cell_bboxes)


    def split_string_by_keywords(self, html_string):
        """
        Split HTML string by keywords.

        Args:
            html_string (str): The HTML string.
        Returns:
            split_html (list): The list of html keywords.
        """

        keywords = [
            "<thead>",
            "</thead>",
            "<tbody>",
            "</tbody>",
            "<tr>",
            "</tr>",
            "<td>",
            "<td",
            ">",
            "</td>",
            'colspan="2"',
            'colspan="3"',
            'colspan="4"',
            'colspan="5"',
            'colspan="6"',
            'colspan="7"',
            'colspan="8"',
            'colspan="9"',
            'colspan="10"',
            'colspan="11"',
            'colspan="12"',
            'colspan="13"',
            'colspan="14"',
            'colspan="15"',
            'colspan="16"',
            'colspan="17"',
            'colspan="18"',
            'colspan="19"',
            'colspan="20"',
            'rowspan="2"',
            'rowspan="3"',
            'rowspan="4"',
            'rowspan="5"',
            'rowspan="6"',
            'rowspan="7"',
            'rowspan="8"',
            'rowspan="9"',
            'rowspan="10"',
            'rowspan="11"',
            'rowspan="12"',
            'rowspan="13"',
            'rowspan="14"',
            'rowspan="15"',
            'rowspan="16"',
            'rowspan="17"',
            'rowspan="18"',
            'rowspan="19"',
            'rowspan="20"',
        ]
        regex_pattern = "|".join(re.escape(keyword) for keyword in keywords)
        split_result = re.split(f"({regex_pattern})", html_string)
        split_html = [part for part in split_result if part]
        return split_html

    def cluster_positions(self, positions, tolerance):
        if not positions:
            return []
        positions = sorted(set(positions))
        clustered = []
        current_cluster = [positions[0]]
        for pos in positions[1:]:
            if abs(pos - current_cluster[-1]) <= tolerance:
                current_cluster.append(pos)
            else:
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [pos]
        clustered.append(sum(current_cluster) / len(current_cluster))
        return clustered


    def trans_cells_det_results_to_html(self, cells_det_results):
        """
        Trans table cells bboxes to HTML.

        Args:
            cells_det_results (list): The table cells detection results.
        Returns:
            html (list): The list of html keywords.
        """

        tolerance = 5
        x_coords = [x for cell in cells_det_results for x in (cell[0], cell[2])]
        y_coords = [y for cell in cells_det_results for y in (cell[1], cell[3])]
        x_positions = self.cluster_positions(x_coords, tolerance)
        y_positions = self.cluster_positions(y_coords, tolerance)
        x_position_to_index = {x: i for i, x in enumerate(x_positions)}
        y_position_to_index = {y: i for i, y in enumerate(y_positions)}
        num_rows = len(y_positions) - 1
        num_cols = len(x_positions) - 1
        grid = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        cells_info = []
        cell_index = 0
        cell_map = {}
        for index, cell in enumerate(cells_det_results):
            x1, y1, x2, y2 = cell
            x1_idx = min(
                range(len(x_positions)), key=lambda i: abs(x_positions[i] - x1)
            )
            x2_idx = min(
                range(len(x_positions)), key=lambda i: abs(x_positions[i] - x2)
            )
            y1_idx = min(
                range(len(y_positions)), key=lambda i: abs(y_positions[i] - y1)
            )
            y2_idx = min(
                range(len(y_positions)), key=lambda i: abs(y_positions[i] - y2)
            )
            col_start = min(x1_idx, x2_idx)
            col_end = max(x1_idx, x2_idx)
            row_start = min(y1_idx, y2_idx)
            row_end = max(y1_idx, y2_idx)
            rowspan = row_end - row_start
            colspan = col_end - col_start
            if rowspan == 0:
                rowspan = 1
            if colspan == 0:
                colspan = 1
            cells_info.append(
                {
                    "row_start": row_start,
                    "col_start": col_start,
                    "rowspan": rowspan,
                    "colspan": colspan,
                    "content": "",
                }
            )
            for r in range(row_start, row_start + rowspan):
                for c in range(col_start, col_start + colspan):
                    key = (r, c)
                    if key in cell_map:
                        continue
                    else:
                        cell_map[key] = index
        html = "<table><tbody>"
        for r in range(num_rows):
            html += "<tr>"
            c = 0
            while c < num_cols:
                key = (r, c)
                if key in cell_map:
                    cell_index = cell_map[key]
                    cell_info = cells_info[cell_index]
                    if cell_info["row_start"] == r and cell_info["col_start"] == c:
                        rowspan = cell_info["rowspan"]
                        colspan = cell_info["colspan"]
                        rowspan_attr = f' rowspan="{rowspan}"' if rowspan > 1 else ""
                        colspan_attr = f' colspan="{colspan}"' if colspan > 1 else ""
                        content = cell_info["content"]
                        html += f"<td{rowspan_attr}{colspan_attr}>{content}</td>"
                    c += cell_info["colspan"]
                else:
                    html += "<td></td>"
                    c += 1
            html += "</tr>"
        html += "</tbody></table>"
        html = self.split_string_by_keywords(html)
        return html


    def cells_det_results_nms(
        self, cells_det_results, cells_det_scores, cells_det_threshold=0.3
    ):
        """
        Apply Non-Maximum Suppression (NMS) on detection results to remove redundant overlapping bounding boxes.

        Args:
            cells_det_results (list): List of bounding boxes, each box is in format [x1, y1, x2, y2].
            cells_det_scores (list): List of confidence scores corresponding to the bounding boxes.
            cells_det_threshold (float): IoU threshold for suppression. Boxes with IoU greater than this threshold
                                        will be suppressed. Default is 0.5.

        Returns:
        Tuple[list, list]: A tuple containing the list of bounding boxes and confidence scores after NMS,
                            while maintaining one-to-one correspondence.
        """
        # Convert lists to numpy arrays for efficient computation
        boxes = np.array(cells_det_results)
        scores = np.array(cells_det_scores)
        # Initialize list for picked indices
        picked_indices = []
        # Get coordinates of bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        # Compute the area of the bounding boxes
        areas = (x2 - x1) * (y2 - y1)
        # Sort the bounding boxes by the confidence scores in descending order
        order = scores.argsort()[::-1]
        # Process the boxes
        while order.size > 0:
            # Index of the current highest score box
            i = order[0]
            picked_indices.append(i)
            # Compute IoU between the highest score box and the rest
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            # Compute the width and height of the overlapping area
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            # Compute the ratio of overlap (IoU)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            # Indices of boxes with IoU less than threshold
            inds = np.where(ovr <= cells_det_threshold)[0]
            # Update order, only keep boxes with IoU less than threshold
            order = order[
                inds + 1
            ]  # inds shifted by 1 because order[0] is the current box
        # Select the boxes and scores based on picked indices
        final_boxes = boxes[picked_indices].tolist()
        final_scores = scores[picked_indices].tolist()
        return final_boxes, final_scores



    def cells_det_results_reprocessing(
            self, cells_det_results, cells_det_scores, ocr_det_results, html_pred_boxes_nums
    ):
        """
        Process and filter cells_det_results based on ocr_det_results and html_pred_boxes_nums.

        Args:
            cells_det_results (List[List[float]]): List of detected cell rectangles [[x1, y1, x2, y2], ...].
            cells_det_scores (List[float]): List of confidence scores for each rectangle in cells_det_results.
            ocr_det_results (List[List[float]]): List of OCR detected rectangles [[x1, y1, x2, y2], ...].
            html_pred_boxes_nums (int): The desired number of rectangles in the final output.

        Returns:
            List[List[float]]: The processed list of rectangles.
        """

        # Function to compute IoU between two rectangles
        def compute_iou(box1, box2):
            """
            Compute the Intersection over Union (IoU) between two rectangles.

            Args:
                box1 (array-like): [x1, y1, x2, y2] of the first rectangle.
                box2 (array-like): [x1, y1, x2, y2] of the second rectangle.

            Returns:
                float: The IoU between the two rectangles.
            """
            # Determine the coordinates of the intersection rectangle
            x_left = max(box1[0], box2[0])
            y_top = max(box1[1], box2[1])
            x_right = min(box1[2], box2[2])
            y_bottom = min(box1[3], box2[3])
            if x_right <= x_left or y_bottom <= y_top:
                return 0.0
            # Calculate the area of intersection rectangle
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            # Calculate the area of both rectangles
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            (box2[2] - box2[0]) * (box2[3] - box2[1])
            # Calculate the IoU
            iou = intersection_area / float(box1_area)
            return iou

        # Function to combine rectangles into N rectangles
        def combine_rectangles(rectangles, N):
            """
            Combine rectangles into N rectangles based on geometric proximity.

            Args:
                rectangles (list of list of int): A list of rectangles, each represented by [x1, y1, x2, y2].
                N (int): The desired number of combined rectangles.

            Returns:
                list of list of int: A list of N combined rectangles.
            """
            # Number of input rectangles
            num_rects = len(rectangles)
            # If N is greater than or equal to the number of rectangles, return the original rectangles
            if N >= num_rects:
                return rectangles
            # Compute the center points of the rectangles
            centers = np.array(
                [
                    [
                        (rect[0] + rect[2]) / 2,  # Center x-coordinate
                        (rect[1] + rect[3]) / 2,  # Center y-coordinate
                    ]
                    for rect in rectangles
                ]
            )
            # Perform KMeans clustering on the center points to group them into N clusters
            kmeans = KMeans(n_clusters=N, random_state=0, n_init="auto")
            labels = kmeans.fit_predict(centers)
            # Initialize a list to store the combined rectangles
            combined_rectangles = []
            # For each cluster, compute the minimal bounding rectangle that covers all rectangles in the cluster
            for i in range(N):
                # Get the indices of rectangles that belong to cluster i
                indices = np.where(labels == i)[0]
                if len(indices) == 0:
                    # If no rectangles in this cluster, skip it
                    continue
                # Extract the rectangles in cluster i
                cluster_rects = np.array([rectangles[idx] for idx in indices])
                # Compute the minimal x1, y1 (top-left corner) and maximal x2, y2 (bottom-right corner)
                x1_min = np.min(cluster_rects[:, 0])
                y1_min = np.min(cluster_rects[:, 1])
                x2_max = np.max(cluster_rects[:, 2])
                y2_max = np.max(cluster_rects[:, 3])
                # Append the combined rectangle to the list
                combined_rectangles.append([x1_min, y1_min, x2_max, y2_max])
            return combined_rectangles

        # Ensure that the inputs are numpy arrays for efficient computation
        cells_det_results = np.array(cells_det_results)
        cells_det_scores = np.array(cells_det_scores)
        ocr_det_results = np.array(ocr_det_results)
        more_cells_flag = False
        if len(cells_det_results) == html_pred_boxes_nums:
            return cells_det_results
        # Step 1: If cells_det_results has more rectangles than html_pred_boxes_nums
        elif len(cells_det_results) > html_pred_boxes_nums:
            more_cells_flag = True
            # Select the indices of the top html_pred_boxes_nums scores
            top_indices = np.argsort(-cells_det_scores)[:html_pred_boxes_nums]
            # Adjust the corresponding rectangles
            cells_det_results = cells_det_results[top_indices].tolist()
        # Threshold for IoU
        iou_threshold = 0.6
        # List to store ocr_miss_boxes
        ocr_miss_boxes = []
        # For each rectangle in ocr_det_results
        for ocr_rect in ocr_det_results:
            merge_ocr_box_iou = []
            # Flag to indicate if ocr_rect has IoU >= threshold with any cell_rect
            has_large_iou = False
            # For each rectangle in cells_det_results
            for cell_rect in cells_det_results:
                # Compute IoU
                iou = compute_iou(ocr_rect, cell_rect)
                if iou > 0:
                    merge_ocr_box_iou.append(iou)
                if (iou >= iou_threshold) or (sum(merge_ocr_box_iou) >= iou_threshold):
                    has_large_iou = True
                    break
            if not has_large_iou:
                ocr_miss_boxes.append(ocr_rect)
        # If no ocr_miss_boxes, return cells_det_results
        if len(ocr_miss_boxes) == 0:
            final_results = (
                cells_det_results
                if more_cells_flag == True
                else cells_det_results.tolist()
            )
        else:
            if more_cells_flag == True:
                final_results = combine_rectangles(
                    cells_det_results + ocr_miss_boxes, html_pred_boxes_nums
                )
            else:
                # Need to combine ocr_miss_boxes into N rectangles
                N = html_pred_boxes_nums - len(cells_det_results)
                # Combine ocr_miss_boxes into N rectangles
                ocr_supp_boxes = combine_rectangles(ocr_miss_boxes, N)
                # Combine cells_det_results and ocr_supp_boxes
                final_results = np.concatenate(
                    (cells_det_results, ocr_supp_boxes), axis=0
                ).tolist()
        if len(final_results) <= 0.6 * html_pred_boxes_nums:
            final_results = combine_rectangles(ocr_det_results, html_pred_boxes_nums)
        return final_results