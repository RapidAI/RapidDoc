"""Raster/OCR adapter for native table recovery. Images are RGB; OCR is crop-local.

Pipeline text is HTML-escaped. Words remain atomic; no synthetic char positions.
"""
from __future__ import annotations

from dataclasses import replace
from html import unescape
import json
from pathlib import Path
import statistics
import math
from uuid import uuid4

import cv2
import numpy as np

from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.contracts import NativeTableInput, NativeTableRule
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.engine import recover_native_pdf_table
from rapid_doc._vendor.docvortex_core.analyzers.native.pdf._table_recovery.text import build_native_table_text, _tokenize_row


def extract_raster_rules(image, words=()):
    """Extract straight strokes, joining only 1-3px gaps along their own axis."""
    height, width = image.shape[:2]
    if min(height, width) < 16:
        return ()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    block = min(41, (min(height, width)//2)*2-1)
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block, 12)
    rules = []
    for orientation, size in (("horizontal", width), ("vertical", height)):
        horizontal = orientation == "horizontal"
        gap = max(1,min(3,round(size/500)))
        minimum = max(12, round(size*.025))
        def kernel(length):
            return np.ones((1,length) if horizontal else (length,1), np.uint8)
        connected = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, kernel(gap+1))
        mask = cv2.morphologyEx(connected, cv2.MORPH_OPEN, kernel(minimum))
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for x,y,w,h,area in stats[1:count]:
            length, thickness = (w,h) if horizontal else (h,w)
            if length < minimum or length < 5*thickness or thickness > max(5,min(width,height)*.025):
                continue
            # A slightly sloped thin rule has a sparse enclosing rectangle.
            if area/(w*h) < .15:
                continue
            # Native contracts require positive-area stroke bounds, not two
            # coincident coordinates along the thin axis.
            bbox = (float(x),float(y),float(x+w),float(y+h))
            # Letter stems, underscores and fraction bars are not table lines.
            # Discard a stroke when >=80% lies inside a single OCR word box.
            if any(max(0,min(bbox[2],b[2])-max(bbox[0],b[0])) *
                   max(0,min(bbox[3],b[3])-max(bbox[1],b[1])) >= .8*w*h
                   for b in (word['bbox'] for word in words)):
                continue
            rules.append(NativeTableRule(bbox=bbox, width=float(thickness), orientation=orientation))
    return tuple(rules)


def _deskew(image, chars):
    """Estimate small skew from long strokes, transforming pixels and words together."""
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    height,width = gray.shape
    edges = cv2.Canny(gray,50,150)
    lines = cv2.HoughLinesP(edges,1,np.pi/1800,40,minLineLength=max(30,min(width,height)*.25),maxLineGap=8)
    estimates = []
    # OpenCV 4 returns N x 1 x 4, while OpenCV 5 returns N x 4.
    for x0,y0,x1,y1 in (() if lines is None else np.asarray(lines).reshape(-1,4)):
        angle = math.degrees(math.atan2(float(y1-y0),float(x1-x0)))
        angle = (angle+45)%90-45
        if abs(angle) <= 5:
            estimates.append((angle,math.hypot(float(x1-x0),float(y1-y0))))
    if len(estimates) < 3:
        return image,chars,0.
    estimates.sort()
    midpoint = sum(length for _,length in estimates)/2
    total = 0.
    for angle,length in estimates:
        total += length
        if total >= midpoint:
            break
    if abs(angle) < .1:
        return image,chars,0.
    matrix = cv2.getRotationMatrix2D((width/2,height/2),angle,1.)
    new_width = math.ceil(height*abs(matrix[0,1])+width*abs(matrix[0,0]))
    new_height = math.ceil(height*abs(matrix[0,0])+width*abs(matrix[0,1]))
    matrix[:,2] += [(new_width-width)/2,(new_height-height)/2]
    transformed = []
    for char in chars:
        x0,y0,x1,y1 = char['bbox']
        corners = np.array([[x0,y0,1],[x1,y0,1],[x1,y1,1],[x0,y1,1]])@matrix.T
        transformed.append(dict(char,bbox=(*corners.min(axis=0),*corners.max(axis=0))))
    corrected = cv2.warpAffine(image,matrix,(new_width,new_height),flags=cv2.INTER_CUBIC,borderValue=(255,255,255))
    return corrected,transformed,float(angle)


def prepare_ocr_table(image, ocr_result, *, diagnostics=None):
    """Convert words without splitting them or silently discarding uncertain text."""
    if image is None or image.ndim != 3 or image.shape[2] != 3 or not image.size:
        raise ValueError("docvortex_ocr requires a nonempty RGB image")
    if ocr_result is None or len(ocr_result) == 0:
        return None
    if len(ocr_result) != 3 or len({len(items) for items in ocr_result}) != 1:
        raise ValueError("OCR boxes/texts/scores length mismatch")
    height,width = image.shape[:2]
    chars, widths = [], []
    for box, text, score in zip(*ocr_result):
        points = np.asarray(box,np.float64).reshape(-1,2)
        if not len(points) or not np.isfinite(points).all() or not np.isfinite(float(score)):
            raise ValueError("OCR contains invalid coordinates or confidence")
        text = unescape(str(text)).strip()
        if not text:
            continue
        if float(score) < .5:
            return None
        x0,y0 = points.min(axis=0)
        x1,y1 = points.max(axis=0)
        if x0 < -1 or y0 < -1 or x1 > width+1 or y1 > height+1 or x1 <= x0 or y1 <= y0:
            raise ValueError("OCR boxes must use crop-local image coordinates")
        x0,y0,x1,y1 = max(0.,x0),max(0.,y0),min(float(width),x1),min(float(height),y1)
        chars.append({"char":text,"bbox":(x0,y0,x1,y1),"char_idx":len(chars)})
        widths.append((x1-x0)/max(1,len(text)))
    if not chars:
        return None
    image,chars,angle = _deskew(image,chars)
    height,width = image.shape[:2]
    rules = extract_raster_rules(image,chars)
    table_input = NativeTableInput(table_bbox=(0.,0.,float(width),float(height)),
                                  page_size=(float(width),float(height)),angle=0,
                                  chars=tuple(chars),drawing_lines=rules)
    text = build_native_table_text(table_input)
    if text is None:
        return None
    if len(text.glyphs)!=len(chars) or {glyph.source_index for glyph in text.glyphs} != {char['char_idx'] for char in chars}:
        if diagnostics is not None:
            diagnostics.update(reason='ocr_word_filtered_during_preparation',accepted=False)
        return None
    # Estimate character SCALE only. Full word rectangles stay unchanged.
    median_width = max(.1,statistics.median(widths))
    glyphs = {g.glyph_id:g for g in text.glyphs}
    rows = tuple(replace(row,tokens=_tokenize_row([glyphs[i] for i in row.glyph_ids],
                                                  median_width,text.median_glyph_height)) for row in text.rows)
    return table_input,replace(text,median_glyph_width=median_width,rows=rows),angle


def _has_physical_outer_frame(result, rules):
    """Reject inferred outer edges: raster rules need observed frame coverage.

    The native PDF engine may infer an outer track from line endpoints. That
    remains unchanged for PDF input; this optional raster route is stricter.
    """
    if not result.cells:
        return False
    x0=min(c.bbox[0] for c in result.cells)
    y0=min(c.bbox[1] for c in result.cells)
    x1=max(c.bbox[2] for c in result.cells)
    y1=max(c.bbox[3] for c in result.cells)
    for orientation,coord,start,end in (
        ('horizontal',y0,x0,x1),('horizontal',y1,x0,x1),
        ('vertical',x0,y0,y1),('vertical',x1,y0,y1),
    ):
        horizontal=orientation=='horizontal'
        intervals=[]
        allowance=3.
        for rule in rules:
            if rule.orientation!=orientation:
                continue
            box=rule.bbox
            center=(box[1]+box[3])/2 if horizontal else (box[0]+box[2])/2
            tolerance=max(2.,rule.width*1.5)
            if abs(center-coord)>tolerance:
                continue
            a,b=(box[0],box[2]) if horizontal else (box[1],box[3])
            a,b=max(start,a),min(end,b)
            if b>a:
                intervals.append((a,b))
                allowance=max(allowance,rule.width*2)
        if not intervals:
            return False
        cursor=start
        for a,b in sorted(intervals):
            if a-cursor>allowance:
                return False
            cursor=max(cursor,b)
        if end-cursor>allowance:
            return False
    return True


def recover_ocr_table(image, ocr_result, *, diagnostics=None):
    """Return a validated candidate or None so the host can fall back to its model."""
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(reason="missing_or_low_confidence_ocr",accepted=False)
    prepared = prepare_ocr_table(image,ocr_result,diagnostics=diagnostics)
    if prepared is None:
        return None
    table_input,text,angle = prepared
    result = recover_native_pdf_table(table_input,prepared_text=text)
    if diagnostics is not None:
        diagnostics.update(reason="recovered" if result else "ambiguous_or_unsupported_structure",
                           lines=[{"bbox":r.bbox,"width":r.width,"orientation":r.orientation}
                                  for r in table_input.drawing_lines],
                           words=list(table_input.chars),accepted=result is not None,
                           deskew_angle=angle,coordinate_space='deskewed_crop')
    if result is not None:
        if diagnostics is not None:
            diagnostics.update(source=result.source,confidence=result.confidence,rows=result.rows,cols=result.cols,
                               cells=[{"bbox":c.bbox,"row":c.row,"col":c.col,"rowspan":c.rowspan,
                                       "colspan":c.colspan,"text":c.content} for c in result.cells])
        if result.source!='vector_grid' or not _has_physical_outer_frame(result,table_input.drawing_lines):
            if diagnostics is not None:
                diagnostics.update(reason='physical_closed_grid_required',accepted=False)
            return None
        # Geometric confidence is not OCR accuracy. Every atomic word needs one
        # owner covering >=80% of its box; cross-cell words must trigger fallback.
        for glyph in text.glyphs:
            b = glyph.bbox
            area = (b[2]-b[0])*(b[3]-b[1])
            matches = []
            for cell in result.cells:
                c = cell.bbox
                overlap = max(0,min(b[2],c[2])-max(b[0],c[0]))*max(0,min(b[3],c[3])-max(b[1],c[1]))
                if overlap/area >= .8:
                    matches.append(cell)
            if len(matches) != 1:
                if diagnostics is not None:
                    diagnostics.update(reason="word_crosses_cell_boundary",accepted=False)
                return None
    return result


def save_ocr_diagnostics(directory,image,report):
    """Save per-table evidence with unique names for concurrent workers."""
    directory = Path(directory)/uuid4().hex
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    drawn = _deskew(image,[])[0] if report.get('deskew_angle') else image.copy()
    for rule in report.get('lines',[]):
        x0,y0,x1,y1 = map(round,rule['bbox'])
        cv2.line(drawn,(x0,y0),(x1,y1),(0,128,255),1)
    for word in report.get('words',[]):
        x0,y0,x1,y1 = map(round,word['bbox'])
        cv2.rectangle(drawn,(x0,y0),(x1,y1),(0,160,0),1)
    from PIL import Image
    Image.fromarray(drawn).save(directory/'evidence.png')
    return directory
