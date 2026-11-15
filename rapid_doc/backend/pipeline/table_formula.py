"""Orient table-owned formula crops before recognition; retain page coordinates."""

from ...utils.boxbase import get_rotate_image
from ...utils.ocr_utils import get_adjusted_mfdetrec_res


def rotate_table_hint_boxes(hints, source_shape, label):
    if not hints or label not in ('90', '270'):
        return hints
    height, width = source_shape[:2]
    result = []
    for hint in hints:
        x0, y0, x1, y1 = hint['bbox']
        bbox = ([y0, width - x1, y1, width - x0] if label == '90'
                else [height - y1, x0, height - y0, x1])
        result.append({**hint, 'bbox': bbox})
    return result


def prepare_table_formula_crops(formulas, tables, pages, scales, manager, table_config, ocr_config):
    from .analyze_utils import _get_table_ocr_model, _probe_table_direction

    owned = {}
    for formula in formulas:
        box = formula['formula_res']['bbox']
        x0, y0, x1, y1 = box
        area = max(0, x1 - x0) * max(0, y1 - y0)
        if not area:
            continue
        candidates = []
        for index, table in enumerate(tables):
            if table['page_idx'] != formula['page_idx']:
                continue
            poly = table['table_res']['poly']
            tx0, ty0, tx1, ty1 = poly[0], poly[1], poly[4], poly[5]
            intersection = max(0, min(x1, tx1) - max(x0, tx0)) * max(0, min(y1, ty1) - max(y0, ty0))
            if (tx0 <= (x0 + x1) / 2 <= tx1 and ty0 <= (y0 + y1) / 2 <= ty1
                    and intersection >= area * .5):
                # A formula belongs to one table only, preferring coverage and
                # then the smaller region for nested/overlapping detections.
                candidates.append((-intersection, (tx1 - tx0) * (ty1 - ty0), index))
        if candidates:
            owned.setdefault(min(candidates)[2], []).append(formula)

    for index, entries in owned.items():
        table = tables[index]
        masks = get_adjusted_mfdetrec_res(
            table['single_page_mfdetrec_res'] + table['checkbox_res'],
            table['useful_list'], return_text=True,
        )
        ocr_model = _get_table_ocr_model(manager, table['lang'], ocr_config)
        probe = _probe_table_direction(
            table, pages[table['page_idx']], scales[table['page_idx']],
            manager, ocr_model, table_config, masks,
        )
        table['_formula_direction_probe'] = probe
        label = probe[1]
        if label in ('90', '270'):
            for formula in entries:
                formula['formula_img'] = get_rotate_image(formula['formula_img'], label)
