"""复用不变路径的基线事实；文本成员与最终图体仍由各阶段独立判断。"""

import math

from ...._compute_backend import get_native


def compound_baseline_flags(source, em):
    """普通有限矩形按对象身份和实际字号缓存，保留引用以避免身份被复用。"""
    if type(em) not in (int, float) or not math.isfinite(em) or abs(em) > 1e100:
        return {}
    output, pending, keys = {}, [], []
    cache = source.compound_baseline_cache
    for path in source.path_infos:
        boxes = path.rectangle_bboxes
        if not path.fill_visible or len(boxes) <= 1:
            continue
        key = (id(boxes), em)
        cached = cache.get(key)
        if cached is not None and cached[0] is boxes:
            output[id(boxes)] = cached[1]
            continue
        if type(boxes) is not tuple or any(
            type(box) is not tuple
            or len(box) != 4
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or abs(v) > (1e100 if type(v) is float else 2**50)
                for v in box
            )
            for box in boxes
        ):
            continue
        pending.append(boxes)
        keys.append(key)
    native = get_native()
    if native is not None:
        values = native.compound_baselines(pending, em) if pending else []
    else:
        values = [
            tuple(
                any(
                    sum(min(abs(box[e] - baseline) for e in (edge, edge + 2)) <= 0.35 * em for box in boxes)
                    >= 0.75 * len(boxes)
                    for seed in boxes
                    for baseline in (seed[edge], seed[edge + 2])
                )
                for edge in (0, 1)
            )
            for boxes in pending
        ]
    for key, boxes, flags in zip(keys, pending, values):
        cache[key] = (boxes, flags)
        output[id(boxes)] = flags
    return output
