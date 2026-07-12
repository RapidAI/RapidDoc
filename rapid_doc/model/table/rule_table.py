"""Conservative PDF-native ruled-table reconstruction.

The geometry gates mirror liteparse's ruled-grid path: axis-aligned rules are
clustered, connected components are formed, text is assigned by centroid, and
sparse/conflicting grids are rejected instead of guessed.
"""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Iterable
import re


AXIS_TOL = 1.0
GRID_TOL = 2.0
CROSS_TOL = 3.0
COL_TOL = 6.0


@dataclass
class RuleTableResult:
    html: str
    score: float
    rows: int
    columns: int
    text_coverage: float


def _looks_like_header(row: list[str]) -> bool:
    nonempty=[x for x in row if x.strip()]
    if not nonempty:
        return False
    alpha=sum(any(ch.isalpha() for ch in x) for x in nonempty)
    value=re.compile(r'^\s*[-+]?\d[\d.,/%\-]*\s*$')
    values=sum(bool(value.match(x)) for x in nonempty)
    return alpha/len(nonempty) >= .5 and values/len(nonempty) < .5


def _cell_text(items) -> str:
    # pdftext exposes synthetic word-separator spaces as zero-width, tiny
    # boxes on a shifted baseline. They are layout hints, not visible cell
    # content; retaining them creates artifacts such as ``6 36,903``.
    items=[item for item in items if not (str(item[2]).isspace() and len(item)>4 and item[4]-item[1] <= .1)]
    items.sort()
    logical_lines=[]
    for item in items:
        if not logical_lines or abs(item[0]-logical_lines[-1][0]) > max(item[3],logical_lines[-1][2])*.5:
            logical_lines.append([item[0],[item],item[3]])
        else:
            logical_lines[-1][1].append(item)
            logical_lines[-1][2]=max(logical_lines[-1][2],item[3])
    texts=[]
    for _,line,_ in logical_lines:
        line.sort(key=lambda x:x[1])
        texts.append(" ".join("".join(x[2] for x in line).split()))
    return " ".join(x for x in texts if x)


def _cluster(values: Iterable[float], tolerance: float) -> list[float]:
    groups: list[list[float]] = []
    for value in sorted(values):
        if groups and abs(value - sum(groups[-1]) / len(groups[-1])) <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(group) / len(group) for group in groups]


def _overlap(a0, a1, b0, b1, tol=0.0):
    return min(a1, b1) + tol >= max(a0, b0)


def _bbox_intersection_ratio(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1-x0) * max(0.0, y1-y0)
    area = max(1.0, (a[2]-a[0]) * (a[3]-a[1]))
    return inter / area


def _segments(lines, table_bbox):
    hs, vs = [], []
    margin = 3.0
    scope = [table_bbox[0]-margin, table_bbox[1]-margin, table_bbox[2]+margin, table_bbox[3]+margin]
    for line in lines or []:
        x1, y1, x2, y2 = (float(line[k]) for k in ("x1", "y1", "x2", "y2"))
        if not _overlap(min(x1,x2), max(x1,x2), scope[0], scope[2]) or not _overlap(min(y1,y2), max(y1,y2), scope[1], scope[3]):
            continue
        if abs(y1-y2) <= AXIS_TOL and abs(x1-x2) > 1:
            hs.append([min(x1,x2), max(x1,x2), (y1+y2)/2])
        elif abs(x1-x2) <= AXIS_TOL and abs(y1-y2) > 1:
            vs.append([min(y1,y2), max(y1,y2), (x1+x2)/2])
    return hs, vs


def _merge_segments(segs, coord_index):
    out = []
    for seg in sorted(segs, key=lambda s: s[coord_index]):
        if out and abs(out[-1][coord_index]-seg[coord_index]) <= GRID_TOL:
            out[-1][0] = min(out[-1][0], seg[0])
            out[-1][1] = max(out[-1][1], seg[1])
            out[-1][coord_index] = (out[-1][coord_index]+seg[coord_index])/2
        else:
            out.append(seg[:])
    return out


def _best_component(hs, vs, table_bbox):
    # Union crossing H/V segments, matching liteparse's component isolation.
    n = len(hs)+len(vs)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a != b: parent[a]=b
    for i,h in enumerate(hs):
        for j,v in enumerate(vs):
            if h[0]-CROSS_TOL <= v[2] <= h[1]+CROSS_TOL and v[0]-CROSS_TOL <= h[2] <= v[1]+CROSS_TOL:
                union(i, len(hs)+j)
    groups = {}
    for i in range(n): groups.setdefault(find(i), [[],[]])[0 if i < len(hs) else 1].append(i if i < len(hs) else i-len(hs))
    candidates = []
    for hi,vi in groups.values():
        if len(hi) < 3 or len(vi) < 2: continue
        box=[min(vs[i][2] for i in vi), min(hs[i][2] for i in hi), max(vs[i][2] for i in vi), max(hs[i][2] for i in hi)]
        candidates.append((_bbox_intersection_ratio(table_bbox, box), hi, vi))
    return max(candidates, default=None, key=lambda x:x[0])


def _char_text(char):
    # Keep parity with the normal span preprocessing path.  Some embedded PDF
    # fonts expose a discretionary/line-break hyphen as U+0002.
    return str(char.get("char", char.get("text", ""))).replace("\u0002", "-")


def build_rule_table(page_dict: dict, table_bbox: list[float]) -> RuleTableResult | None:
    """Build HTML for one layout table bbox, or return ``None`` when unsure."""
    raw_hs, raw_vs = _segments(page_dict.get("vector_lines"), table_bbox)
    hs, vs = _merge_segments(raw_hs, 2), _merge_segments(raw_vs, 2)
    component = _best_component(hs, vs, table_bbox)
    if not component or component[0] < 0.85:
        return None
    _, hi, vi = component
    xs = _cluster((vs[i][2] for i in vi), COL_TOL)
    ys = _cluster((hs[i][2] for i in hi), GRID_TOL)
    if len(xs) < 3 or len(ys) < 3:
        return None
    rows, cols = len(ys)-1, len(xs)-1
    if rows*cols > 10000:
        return None
    grid = [[[] for _ in range(cols)] for _ in range(rows)]
    native = []
    all_table_chars = []
    for ch in page_dict.get("chars", []):
        # Native PDF watermarks are often emitted as rotated text spanning many
        # cells.  They are not cell content and must not participate in either
        # grid assignment or the rule-table confidence score.
        if abs(float(ch.get("rotation", 0) or 0)) > 0.05:
            continue
        text = _char_text(ch)
        if not text or text in ("\r", "\n"):
            continue
        b = ch.get("bbox")
        if b is None: continue
        b = getattr(b, "bbox", b)
        if len(b) < 4: continue
        cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
        if table_bbox[0]-1 <= cx <= table_bbox[2]+1 and table_bbox[1]-1 <= cy <= table_bbox[3]+1:
            all_table_chars.append(text)
        if not (xs[0]-1 <= cx <= xs[-1]+1 and ys[0]-1 <= cy <= ys[-1]+1): continue
        c = next((i for i in range(cols) if xs[i]-1 <= cx <= xs[i+1]+1), None)
        r = next((i for i in range(rows) if ys[i]-1 <= cy <= ys[i+1]+1), None)
        if r is not None and c is not None:
            grid[r][c].append(((b[1]+b[3])/2, b[0], text, max(1.0, b[3]-b[1]), b[2])); native.append(text)
    if not native:
        return None
    texts=[]
    for row in grid:
        out=[]
        for cell in row:
            out.append(_cell_text(cell))
        texts.append(out)
    filled=sum(bool(x) for row in texts for x in row)
    empty_frac=1-filled/(rows*cols)
    col0_fill=sum(bool(texts[r][0]) for r in range(rows))/rows
    if empty_frac > 0.30 and not (empty_frac <= 0.75 and col0_fill >= 0.70 and max((len(texts[r][0]) for r in range(rows)), default=0) <= 60):
        return None
    # Border completeness is the strongest confidence signal. A boundary may
    # be split into many segments, so coverage is tested per logical edge.
    def h_edge(y,x0,x1): return any(abs(h[2]-y)<=GRID_TOL and h[0]<=x0+CROSS_TOL and h[1]>=x1-CROSS_TOL for h in hs)
    def v_edge(x,y0,y1): return any(abs(v[2]-x)<=COL_TOL and v[0]<=y0+CROSS_TOL and v[1]>=y1-CROSS_TOL for v in vs)
    edge_checks=[]
    for y in ys: edge_checks.append(h_edge(y,xs[0],xs[-1]))
    for x in xs: edge_checks.append(v_edge(x,ys[0],ys[-1]))
    edge_score=sum(edge_checks)/len(edge_checks)
    coverage_chars=sum(len(t) for row in texts for t in row)
    native_chars=sum(1 for t in all_table_chars if not t.isspace())
    text_coverage=min(1.0, coverage_chars/max(1,native_chars))
    density_score=1.0-min(1.0, empty_frac/0.75)
    score=0.55*edge_score+0.30*text_coverage+0.15*density_score
    if edge_score < 0.70 or text_coverage < 0.95:
        return None
    # Missing internal dividers encode merged cells. Build only rectangular
    # spans; irregular/L-shaped components are left as individual cells rather
    # than emitting invalid HTML.
    def covers_interval(segments, start, end):
        """Return whether collinear PDF fragments jointly cover an edge.

        PDF generators commonly split one visible rule at every drawing
        object/cell.  Testing individual fragments therefore invents merged
        cells even though the rendered boundary is continuous.
        """
        intervals = sorted((seg[0], seg[1]) for seg in segments if seg[1] >= start - CROSS_TOL and seg[0] <= end + CROSS_TOL)
        cursor = start
        for seg_start, seg_end in intervals:
            if seg_start > cursor + CROSS_TOL:
                break
            cursor = max(cursor, seg_end)
            if cursor >= end - CROSS_TOL:
                return True
        return False

    def has_v(boundary, r):
        return covers_interval(
            (v for v in raw_vs if abs(v[2] - boundary) <= COL_TOL),
            ys[r], ys[r + 1],
        )
    def has_h(boundary, c0, c1):
        return covers_interval(
            (h for h in raw_hs if abs(h[2] - boundary) <= GRID_TOL),
            xs[c0], xs[c1],
        )
    occupied=set(); rendered=[[] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if (r,c) in occupied: continue
            c1=c+1
            while c1 < cols and not has_v(xs[c1], r): c1 += 1
            r1=r+1
            while r1 < rows and not has_h(ys[r1], c, c1): r1 += 1
            members=[(rr,cc) for rr in range(r,r1) for cc in range(c,c1)]
            if any(x in occupied for x in members):
                r1,c1=r+1,c+1; members=[(r,c)]
            occupied.update(members)
            parts=[]
            for rr,cc in members:
                if texts[rr][cc] and texts[rr][cc] not in parts: parts.append(texts[rr][cc])
            attrs=""
            if r1-r > 1: attrs += f' rowspan="{r1-r}"'
            if c1-c > 1: attrs += f' colspan="{c1-c}"'
            tag="th" if r == 0 and _looks_like_header(texts[0]) else "td"
            rendered[r].append(f"<{tag}{attrs}>{escape(' '.join(parts))}</{tag}>")
    body=["<tr>"+"".join(cells)+"</tr>" for cells in rendered]
    return RuleTableResult("<table>"+"".join(body)+"</table>", round(score,4), rows, cols, text_coverage)


def build_track_table(page_dict: dict, table_bbox: list[float]) -> RuleTableResult | None:
    """Reconstruct a semi-ruled table from vertical tracks and text baselines.

    This is the PDF pattern used by dense meteorological/statistical tables:
    stable column rules exist, but data rows have no horizontal borders.
    """
    _, raw_vs=_segments(page_dict.get('vector_lines'),table_bbox)
    merged_vs=_merge_segments(raw_vs,2)
    tall=[v for v in merged_vs if (v[1]-v[0]) >= (table_bbox[3]-table_bbox[1])*.65]
    xs=_cluster((v[2] for v in tall),COL_TOL)
    if len(xs) < 4:
        return None
    cols=len(xs)-1
    chars=[]
    all_count=0
    outside_track_ys=[]
    for ch in page_dict.get('chars',[]):
        if abs(float(ch.get('rotation', 0) or 0)) > 0.05:
            continue
        text=_char_text(ch); b=ch.get('bbox')
        if not text or b is None or text in ('\r','\n'): continue
        b=getattr(b,'bbox',b); cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
        if text.isspace() and b[2]-b[0] <= .1: continue
        if not (table_bbox[0]-1<=cx<=table_bbox[2]+1 and table_bbox[1]-1<=cy<=table_bbox[3]+1): continue
        if not text.isspace(): all_count += 1
        if xs[0]-1<=cx<=xs[-1]+1:
            chars.append((cy,b[0],text,max(1.0,b[3]-b[1]),cx,b[2]))
        elif not text.isspace():
            outside_track_ys.append(cy)
    if not chars: return None
    chars.sort()
    baselines=[]
    for item in chars:
        if not baselines or abs(item[0]-baselines[-1][0]) > max(item[3],baselines[-1][2])*.55:
            baselines.append([item[0],[item],item[3]])
        else:
            baselines[-1][1].append(item); baselines[-1][2]=max(baselines[-1][2],item[3])
    outside_lines=_cluster(outside_track_ys,2.0)
    # Repeated text on either open side means the tracks cover only an
    # interior subset of the table (for example omitted No./Difference cols).
    if len(outside_lines)>=3 and len(outside_lines)>=len(baselines)*.2:
        return None
    rows=[]
    for _,items,_ in baselines:
        buckets=[[] for _ in range(cols)]
        items.sort(key=lambda x:x[1])
        tokens=[]
        for item in items:
            if tokens and item[1]-tokens[-1][-1][5] <= max(item[3],tokens[-1][-1][3])*.8:
                tokens[-1].append(item)
            else:
                tokens.append([item])
        for token in tokens:
            cy=sum(x[0] for x in token)/len(token); x0=token[0][1]; x1=token[-1][5]
            text=''.join(x[2] for x in token); height=max(x[3] for x in token); cx=(x0+x1)/2
            col=next((i for i in range(cols) if xs[i]-1<=cx<=xs[i+1]+1),None)
            if col is not None: buckets[col].append((cy,x0,text,height))
        row=[_cell_text(cell) for cell in buckets]
        if any(row): rows.append(row)
    if len(rows)<5: return None
    # Sparse centered labels often cross narrow numeric tracks (station name,
    # basin name). Rejoin adjacent alphabetic fragments and place the label at
    # the run centroid, matching liteparse's span-centroid assignment.
    for row in rows:
        if sum(bool(x) for x in row)/cols >= .5: continue
        c=0
        while c<cols:
            if not row[c] or not any(ch.isalpha() for ch in row[c]): c+=1; continue
            end=c+1
            while end<cols and row[end] and any(ch.isalpha() for ch in row[end]): end+=1
            if end-c>1:
                text=''.join(row[c:end]); row[c:end]=['']*(end-c); row[(c+end-1)//2]=text
            c=end
    # Multi-baseline headers: month/group labels sit above a dense max/min
    # layer. Replicate each group label across its following blank track and
    # flatten all layers through the first dense header row.
    header_end=next((i for i,row in enumerate(rows[:4]) if _looks_like_header(row) and sum(bool(x) for x in row)/cols>=.8),None)
    if header_end is not None and header_end>=1:
        layers=[r[:] for r in rows[:header_end+1]]
        for layer in layers[:-1]:
            original=layer[:]
            if sum(bool(x) for x in original) < 2: continue
            for c in range(cols-1):
                if original[c] and not original[c+1] and any(ch.isalpha() for ch in original[c]): layer[c+1]=original[c]
        merged=[]
        for c in range(cols):
            parts=[]
            for layer in layers:
                if layer[c] and (not parts or parts[-1]!=layer[c]): parts.append(layer[c])
            merged.append(' '.join(parts))
        rows=[merged]+rows[header_end+1:]
    # Aggregate rows in paired max/min layouts place one value across each
    # two-column group. PDF centroids sit on the shared boundary; liteparse's
    # track anchoring resolves these to the second (min) track.
    paired_header=rows and all(('max' in rows[0][c].lower() and 'min' in rows[0][c+1].lower()) for c in range(1,cols-1,2))
    paired_body=any(sum(bool(x) for x in row[1:])/max(1,cols-1)>=.8 for row in rows)
    if paired_header or paired_body:
        for row in rows[1:]:
            populated=[c for c in range(1,cols) if row[c]]
            label_alpha=bool(row[0] and any(ch.isalpha() for ch in row[0]))
            if label_alpha and len(populated)>=3 and all(c%2==1 for c in populated):
                for c in reversed(populated):
                    if c+1<cols and not row[c+1]: row[c+1],row[c]=row[c],''
    dense=sum(sum(bool(x) for x in row)/cols>=.7 for row in rows)/len(rows)
    covered=sum(len(x) for row in rows for x in row)
    coverage=min(1.0,covered/max(1,all_count))
    if dense<.6 or coverage<.95: return None
    header=_looks_like_header(rows[0])
    html_rows=[]
    for r,row in enumerate(rows):
        tag='th' if r==0 and header else 'td'
        html_rows.append('<tr>'+''.join(f'<{tag}>{escape(x)}</{tag}>' for x in row)+'</tr>')
    score=.55*coverage+.30*dense+.15*min(1.0,len(xs)/4)
    return RuleTableResult('<table>'+''.join(html_rows)+'</table>',round(score,4),len(rows),cols,coverage)
