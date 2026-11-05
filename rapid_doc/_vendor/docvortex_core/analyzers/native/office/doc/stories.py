"""MS-DOC auxiliary story indices; damaged ranges are skipped, never guessed."""
from ..legacy.binary import bounded_slice
from .records import parse_plc
from loguru import logger


def textbox_ranges(table, text_pair, shape_pair, base, count, budget):
    text = bounded_slice(table, text_pair.fc, text_pair.lcb)
    shapes = bounded_slice(table, shape_pair.fc, shape_pair.lcb)
    if text is None or shapes is None:
        if text_pair.lcb:
            logger.warning('DOC_TEXTBOX_INDEX: invalid textbox/shape range')
        return []
    cps, entries = parse_plc(text, item_size=22, budget=budget)
    if count and not entries:
        logger.warning('DOC_TEXTBOX_INDEX: malformed textbox PLC')
    anchors, descriptors = parse_plc(shapes, item_size=26, budget=budget)
    by_shape = {}
    for cp, entry in zip(anchors, descriptors):
        by_shape.setdefault(int.from_bytes(entry[:4], 'little'), cp)
    result = []
    for i, entry in enumerate(entries[:-1]):
        if int.from_bytes(entry[8:10], 'little'):
            continue
        lid = int.from_bytes(entry[14:18], 'little')
        if lid not in by_shape or not 0 <= cps[i] < cps[i+1] <= count or cps[i+1] - cps[i] <= 1:
            logger.warning('DOC_TEXTBOX_INDEX: skipped unbound or invalid textbox {}', i)
            continue
        result.append((by_shape[lid], base + cps[i], base + cps[i+1]))
    return result


def comment_ranges(table, ref_pair, text_pair, base, count, main_count, budget):
    refs = bounded_slice(table, ref_pair.fc, ref_pair.lcb)
    text = bounded_slice(table, text_pair.fc, text_pair.lcb)
    if refs is None or text is None:
        return {}, []
    cps, entries = parse_plc(refs, item_size=30, budget=budget)
    texts, _ = parse_plc(text, item_size=0, budget=budget)
    references, result = {}, []
    for i, _ in enumerate(entries):
        if i + 1 >= len(texts):
            break
        if not 0 <= cps[i] < main_count or not 0 <= texts[i] < texts[i+1] <= count:
            continue
        if cps[i] in references:
            continue
        label = f'comment {i+1}'
        references[cps[i]] = label
        result.append((cps[i], base + texts[i], base + texts[i+1], label))
    return references, result
