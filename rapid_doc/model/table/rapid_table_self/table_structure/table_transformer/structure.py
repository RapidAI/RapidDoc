"""Selected Microsoft TATR postprocessing; see THIRD_PARTY_NOTICES.md."""
from collections import defaultdict
from statistics import median

from .geometry import Rect
from . import postprocess

def iob(bbox1, bbox2):
    """
    Compute the intersection area over box area, for bbox1.
    """
    intersection = Rect(bbox1).intersect(bbox2)
    
    bbox1_area = Rect(bbox1).get_area()
    if bbox1_area > 0:
        return intersection.get_area() / bbox1_area
    
    return 0


def align_headers(headers, rows):
    """
    Adjust the header boundary to be the convex hull of the rows it intersects
    at least 50% of the height of.

    For now, we are not supporting tables with multiple headers, so we need to
    eliminate anything besides the top-most header.
    """
    
    aligned_headers = []

    for row in rows:
        row['column header'] = False

    header_row_nums = []
    for header in headers:
        for row_num, row in enumerate(rows):
            row_height = row['bbox'][3] - row['bbox'][1]
            min_row_overlap = max(row['bbox'][1], header['bbox'][1])
            max_row_overlap = min(row['bbox'][3], header['bbox'][3])
            overlap_height = max_row_overlap - min_row_overlap
            if overlap_height / row_height >= 0.5:
                header_row_nums.append(row_num)

    if len(header_row_nums) == 0:
        return aligned_headers

    header_rect = Rect()
    if header_row_nums[0] > 0:
        header_row_nums = list(range(header_row_nums[0]+1)) + header_row_nums

    last_row_num = -1
    for row_num in header_row_nums:
        if row_num == last_row_num + 1:
            row = rows[row_num]
            row['column header'] = True
            header_rect = header_rect.include_rect(row['bbox'])
            last_row_num = row_num
        else:
            # Break as soon as a non-header row is encountered.
            # This ignores any subsequent rows in the table labeled as a header.
            # Having more than 1 header is not supported currently.
            break

    header = {'bbox': list(header_rect)}
    aligned_headers.append(header)

    return aligned_headers


def refine_table_structure(table_structure, class_thresholds):
    """
    Apply operations to the detected table structure objects such as
    thresholding, NMS, and alignment.
    """
    rows = table_structure["rows"]
    columns = table_structure['columns']

    # Process the headers
    column_headers = table_structure['column headers']
    column_headers = postprocess.apply_threshold(column_headers, class_thresholds["table column header"])
    column_headers = postprocess.nms(column_headers)
    column_headers = align_headers(column_headers, rows)

    # Process spanning cells
    spanning_cells = [elem for elem in table_structure['spanning cells'] if not elem['projected row header']]
    projected_row_headers = [elem for elem in table_structure['spanning cells'] if elem['projected row header']]
    spanning_cells = postprocess.apply_threshold(spanning_cells, class_thresholds["table spanning cell"])
    projected_row_headers = postprocess.apply_threshold(projected_row_headers,
                                                        class_thresholds["table projected row header"])
    spanning_cells += projected_row_headers
    # Align before NMS for spanning cells because alignment brings them into agreement
    # with rows and columns first; if spanning cells still overlap after this operation,
    # the threshold for NMS can basically be lowered to just above 0
    spanning_cells = postprocess.align_supercells(spanning_cells, rows, columns)
    spanning_cells = postprocess.nms_supercells(spanning_cells)

    postprocess.header_supercell_tree(spanning_cells)

    table_structure['columns'] = columns
    table_structure['rows'] = rows
    table_structure['spanning cells'] = spanning_cells
    table_structure['column headers'] = column_headers

    return table_structure


def objects_to_structures(objects, tokens, class_thresholds):
    """
    Process the bounding boxes produced by the table structure recognition model into
    a *consistent* set of table structures (rows, columns, spanning cells, headers).
    This entails resolving conflicts/overlaps, and ensuring the boxes meet certain alignment
    conditions (for example: rows should all have the same width, etc.).
    """

    tables = [obj for obj in objects if obj['label'] == 'table']
    table_structures = []

    for table in tables:
        table_objects = [obj for obj in objects if iob(obj['bbox'], table['bbox']) >= 0.5]
        table_tokens = [token for token in tokens if iob(token['bbox'], table['bbox']) >= 0.5]
        
        structure = {}

        columns = [obj for obj in table_objects if obj['label'] == 'table column']
        rows = [obj for obj in table_objects if obj['label'] == 'table row']
        column_headers = [obj for obj in table_objects if obj['label'] == 'table column header']
        spanning_cells = [obj for obj in table_objects if obj['label'] == 'table spanning cell']
        for obj in spanning_cells:
            obj['projected row header'] = False
        projected_row_headers = [obj for obj in table_objects if obj['label'] == 'table projected row header']
        for obj in projected_row_headers:
            obj['projected row header'] = True
        spanning_cells += projected_row_headers
        for obj in rows:
            obj['column header'] = False
            for header_obj in column_headers:
                if iob(obj['bbox'], header_obj['bbox']) >= 0.5:
                    obj['column header'] = True

        # Refine table structures
        rows = postprocess.refine_rows(rows, table_tokens, class_thresholds['table row'])
        columns = postprocess.refine_columns(columns, table_tokens, class_thresholds['table column'])

        # Shrink table bbox to just the total height of the rows
        # and the total width of the columns
        row_rect = Rect()
        for obj in rows:
            row_rect.include_rect(obj['bbox'])
        column_rect = Rect() 
        for obj in columns:
            column_rect.include_rect(obj['bbox'])
        table['row_column_bbox'] = [column_rect[0], row_rect[1], column_rect[2], row_rect[3]]
        table['bbox'] = table['row_column_bbox']

        # Process the rows and columns into a complete segmented table
        columns = postprocess.align_columns(columns, table['row_column_bbox'])
        rows = postprocess.align_rows(rows, table['row_column_bbox'])

        structure['rows'] = rows
        structure['columns'] = columns
        structure['column headers'] = column_headers
        structure['spanning cells'] = spanning_cells

        if rows and columns:
            structure = refine_table_structure(structure, class_thresholds)

        table_structures.append(structure)

    return table_structures


def restore_missing_numeric_rows(table_structure, tokens, is_numeric):
    """Restore a missing data row when OCR supplies a complete aligned band."""
    rows, columns = table_structure['rows'], table_structure['columns']
    if len(rows) < 3 or len(columns) < 4:
        return
    heights = [row['bbox'][3] - row['bbox'][1] for row in rows]
    normal_height = median(heights)
    gaps = [(i, left['bbox'][3], right['bbox'][1])
            for i, (left, right) in enumerate(zip(rows, rows[1:]))
            if not left.get('column header') and not right.get('column header')
            and .65 * normal_height <= right['bbox'][1] - left['bbox'][3] <= 1.4 * normal_height]
    if not gaps:
        return
    spans = table_structure['spanning cells']
    insertions = {}
    for i, top, bottom in gaps:
        if any(min(span['row_numbers']) <= i < max(span['row_numbers']) for span in spans):
            continue
        band = [token for token in tokens
                if top <= (token['bbox'][1] + token['bbox'][3]) / 2 <= bottom
                and token['bbox'][1] >= top - .12 * normal_height
                and token['bbox'][3] <= bottom + .12 * normal_height]
        centers = [(token['bbox'][1] + token['bbox'][3]) / 2 for token in band]
        if not centers or max(centers) - min(centers) > .45 * normal_height:
            continue
        slots, _, _ = postprocess.slot_into_containers(columns, band)
        if not any(not is_numeric(band[j]['text']) for j in slots[0]):
            continue
        if sum(len(ids) == 1 and is_numeric(band[ids[0]]['text']) for ids in slots[1:]) < 3:
            continue
        left, _, right, _ = rows[i]['bbox']
        insertions[i] = {'bbox': [left, top, right, bottom], 'column header': False}

    if insertions:
        rebuilt, row_indices = [], []
        for i, row in enumerate(rows):
            row_indices.append(len(rebuilt))
            rebuilt.append(row)
            if i in insertions:
                rebuilt.append(insertions[i])
        table_structure['rows'] = rebuilt
        for span in spans:
            span['row_numbers'] = [row_indices[index] for index in span['row_numbers']]


def restore_missing_numeric_columns(table_structure, tokens, is_numeric):
    """Return the logical indices of columns supported by repeated OCR."""
    rows, columns = table_structure['rows'], table_structure['columns']
    if len(rows) < 3 or len(columns) < 3:
        return []
    normal_width = median(col['bbox'][2] - col['bbox'][0] for col in columns)
    if normal_width <= 0:
        return []
    insertions = {}
    for i, (left, right) in enumerate(zip(columns, columns[1:])):
        x0, x1 = left['bbox'][2], right['bbox'][0]
        if x1 - x0 < .5 * normal_width:
            continue
        # 仅从网格内部的空档恢复列。多个数据行都必须有完整、对齐的独立数值，
        # 同一行出现多个候选值时无法确定这是一列还是多列，保留原网格。
        band = [token for token in tokens if token.get('confidence', 0) >= .85
                and x0 <= token['bbox'][0] < token['bbox'][2] <= x1
                and is_numeric(token['text'].strip())]
        if len(band) < 3:
            continue
        slots, _, _ = postprocess.slot_into_containers(rows, band, overlap_threshold=.75)
        data_slots = [ids for row, ids in zip(rows, slots) if not row.get('column header') and ids]
        # 稀疏的孤立数值不足以推断整列。要求空档在可用数据行中反复出现。
        if len(data_slots) < max(3, .25 * len(rows)) or any(len(ids) != 1 for ids in data_slots):
            continue
        boxes = [band[ids[0]]['bbox'] for ids in data_slots]
        median_width = median(box[2] - box[0] for box in boxes)
        if min(max(box[k] for box in boxes) - min(box[k] for box in boxes) for k in (0, 2)) > .45 * median_width:
            continue
        _, y0, _, y1 = left['bbox']
        insertions[i] = {'bbox': [x0, y0, x1, y1]}

    if insertions:
        rebuilt, indices = [], []
        added = []
        for i, column in enumerate(columns):
            indices.append(len(rebuilt))
            rebuilt.append(column)
            if i in insertions:
                added.append(len(rebuilt))
                rebuilt.append(insertions[i])
        table_structure['columns'] = rebuilt
        for span in table_structure['spanning cells']:
            start, end = min(span['column_numbers']), max(span['column_numbers'])
            span['column_numbers'] = list(range(indices[start], indices[end] + 1))
        return added
    return []


def structure_to_cells(table_structure, tokens, *, compute_confidence=True):
    """
    Assuming the row, column, spanning cell, and header bounding boxes have
    been refined into a set of consistent table structures, process these
    table structures into table cells. This is a universal representation
    format for the table, which can later be exported to Pandas or CSV formats.
    Classify the cells as header/access cells or data cells
    based on if they intersect with the header bounding box.
    """
    columns = table_structure['columns']
    rows = table_structure['rows']
    spanning_cells = table_structure['spanning cells']
    cells = []
    covered = set()
    merged = []
    # NMS can shrink the indices of a span while leaving its original bbox.
    # Assign every slot exactly once using the refined logical coordinates.
    for span in spanning_cells:
        row_nums = sorted(span['row_numbers'])
        col_nums = sorted(span['column_numbers'])
        if not row_nums or not col_nums:
            continue
        if (row_nums[-1] - row_nums[0] + 1 != len(row_nums)
                or col_nums[-1] - col_nums[0] + 1 != len(col_nums)):
            continue
        slots = {(r, c) for r in row_nums for c in col_nums}
        if not covered.isdisjoint(slots):
            continue
        covered.update(slots)
        merged.append({
            'bbox': [
                columns[col_nums[0]]['bbox'][0], rows[row_nums[0]]['bbox'][1],
                columns[col_nums[-1]]['bbox'][2], rows[row_nums[-1]]['bbox'][3],
            ],
            'column_nums': col_nums, 'row_nums': row_nums,
            'column header': all(rows[r].get('column header', False) for r in row_nums),
            'projected row header': span['projected row header'],
        })

    for column_num, column in enumerate(columns):
        for row_num, row in enumerate(rows):
            if (row_num, column_num) in covered:
                continue
            cell_rect = Rect(list(row['bbox'])).intersect(Rect(list(column['bbox'])))
            cells.append({
                'bbox': list(cell_rect), 'column_nums': [column_num], 'row_nums': [row_num],
                'column header': row.get('column header', False), 'projected row header': False,
            })
    cells.extend(merged)

    # Compute a confidence score based on how well the page tokens
    # slot into the cells reported by the model
    confidence_score = 0
    if compute_confidence:
        _, _, scores = postprocess.slot_into_containers(cells, tokens)
        if scores:
            confidence_score = (sum(scores) / len(scores) + min(scores)) / 2

    # Dilate rows and columns before final extraction
    #dilated_columns = fill_column_gaps(columns, table_bbox)
    dilated_columns = columns
    #dilated_rows = fill_row_gaps(rows, table_bbox)
    dilated_rows = rows
    for cell in cells:
        column_rect = Rect()
        for column_num in cell['column_nums']:
            column_rect.include_rect(list(dilated_columns[column_num]['bbox']))
        row_rect = Rect()
        for row_num in cell['row_nums']:
            row_rect.include_rect(list(dilated_rows[row_num]['bbox']))
        cell_rect = column_rect.intersect(row_rect)
        cell['bbox'] = list(cell_rect)

    span_nums_by_cell, _, _ = postprocess.slot_into_containers(cells, tokens, overlap_threshold=0.001,
                                                               unique_assignment=True, forced_assignment=False)

    for cell, cell_span_nums in zip(cells, span_nums_by_cell):
        cell_spans = [tokens[num] for num in cell_span_nums]
        # TODO: Refine how text is extracted; should be character-based, not span-based;
        # but need to associate 
        cell['cell text'] = postprocess.extract_text_from_spans(cell_spans, remove_integer_superscripts=False)
        cell['spans'] = cell_spans
        
    # Adjust the row, column, and cell bounding boxes to reflect the extracted text
    num_rows = len(rows)
    rows = postprocess.sort_objects_top_to_bottom(rows)
    num_columns = len(columns)
    columns = postprocess.sort_objects_left_to_right(columns)
    min_y_values_by_row = defaultdict(list)
    max_y_values_by_row = defaultdict(list)
    min_x_values_by_column = defaultdict(list)
    max_x_values_by_column = defaultdict(list)
    for cell in cells:
        min_row = min(cell["row_nums"])
        max_row = max(cell["row_nums"])
        min_column = min(cell["column_nums"])
        max_column = max(cell["column_nums"])
        for span in cell['spans']:
            min_x_values_by_column[min_column].append(span['bbox'][0])
            min_y_values_by_row[min_row].append(span['bbox'][1])
            max_x_values_by_column[max_column].append(span['bbox'][2])
            max_y_values_by_row[max_row].append(span['bbox'][3])
    for row_num, row in enumerate(rows):
        if len(min_x_values_by_column[0]) > 0:
            row['bbox'][0] = min(min_x_values_by_column[0])
        if len(min_y_values_by_row[row_num]) > 0:
            row['bbox'][1] = min(min_y_values_by_row[row_num])
        if len(max_x_values_by_column[num_columns-1]) > 0:
            row['bbox'][2] = max(max_x_values_by_column[num_columns-1])
        if len(max_y_values_by_row[row_num]) > 0:
            row['bbox'][3] = max(max_y_values_by_row[row_num])
    for column_num, column in enumerate(columns):
        if len(min_x_values_by_column[column_num]) > 0:
            column['bbox'][0] = min(min_x_values_by_column[column_num])
        if len(min_y_values_by_row[0]) > 0:
            column['bbox'][1] = min(min_y_values_by_row[0])
        if len(max_x_values_by_column[column_num]) > 0:
            column['bbox'][2] = max(max_x_values_by_column[column_num])
        if len(max_y_values_by_row[num_rows-1]) > 0:
            column['bbox'][3] = max(max_y_values_by_row[num_rows-1])
    for cell in cells:
        row_rect = Rect()
        column_rect = Rect()
        for row_num in cell['row_nums']:
            row_rect.include_rect(list(rows[row_num]['bbox']))
        for column_num in cell['column_nums']:
            column_rect.include_rect(list(columns[column_num]['bbox']))
        cell_rect = row_rect.intersect(column_rect)
        if cell_rect.get_area() > 0:
            cell['bbox'] = list(cell_rect)
            pass

    return cells, confidence_score
