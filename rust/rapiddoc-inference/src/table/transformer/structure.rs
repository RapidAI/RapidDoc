//! Port of the Python TATR adapter's selected Microsoft structure.py/postprocess.py.
//! Stable ordering, shared-token suppression and supercell shrinking are intentional.
use serde::{Deserialize, Serialize};
type B = [f64; 4];
pub fn is_numeric(text: &str) -> bool {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| {
        let n = r"[+−-]?\d+(?:,\d{3})*(?:\.\d+)?%?";
        regex::Regex::new(&format!(r"\A(?:\({n}\)|{n})\z")).unwrap()
    })
    .is_match(text)
}
fn area(b: B) -> f64 {
    (b[2] - b[0]).max(0.) * (b[3] - b[1]).max(0.)
}
fn intersection(a: B, b: B) -> B {
    if area(b) == 0. {
        return b;
    }
    if area(a) == 0. {
        return a;
    }
    [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
    .map(|v| v as f32 as f64)
}
fn union(a: B, b: B) -> B {
    if area(b) == 0. {
        return a;
    }
    if area(a) == 0. {
        return b;
    }
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
    .map(|v| v as f32 as f64)
}
fn hull(values: impl Iterator<Item = B>) -> B {
    values.reduce(union).unwrap_or([0.; 4])
}
fn iob(a: B, b: B) -> f64 {
    let n = area(a);
    if n > 0. {
        area(intersection(a, b)) / n
    } else {
        0.
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Token {
    pub bbox: B,
    pub text: String,
    #[serde(default)]
    pub block_num: usize,
    #[serde(default)]
    pub line_num: usize,
    #[serde(default)]
    pub span_num: usize,
    #[serde(default)]
    pub confidence: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Object {
    pub label: usize,
    pub score: f64,
    pub bbox: B,
    #[serde(default)]
    pub header: bool,
    #[serde(default)]
    pub rows: Vec<usize>,
    #[serde(default)]
    pub columns: Vec<usize>,
}
impl Object {
    pub fn new(label: usize, score: f64, bbox: B) -> Self {
        Self {
            label,
            score,
            bbox,
            header: false,
            rows: vec![],
            columns: vec![],
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cell {
    pub bbox: B,
    pub row_nums: Vec<usize>,
    pub column_nums: Vec<usize>,
    pub column_header: bool,
    pub text: String,
    #[serde(skip)]
    spans: Vec<usize>,
}
impl Cell {
    pub fn token_indices(&self) -> &[usize] {
        &self.spans
    }
}
fn score_sort(v: &mut [Object]) {
    v.sort_by(|a, b| b.score.total_cmp(&a.score));
}
fn nms(mut v: Vec<Object>, threshold: f64) -> Vec<Object> {
    score_sort(&mut v);
    let mut keep: Vec<Object> = vec![];
    for o in v {
        if keep.iter().all(|k| iob(o.bbox, k.bbox) < threshold) {
            keep.push(o);
        }
    }
    keep
}
fn slots(boxes: &[B], tokens: &[Token], threshold: f64) -> Vec<Vec<usize>> {
    let mut assigned = vec![vec![]; boxes.len()];
    for (i, t) in tokens.iter().enumerate() {
        let best = boxes
            .iter()
            .enumerate()
            .map(|(j, &b)| (j, iob(t.bbox, b)))
            .max_by(|a, b| a.1.total_cmp(&b.1).then(b.0.cmp(&a.0)));
        if let Some((j, score)) = best {
            if score >= threshold {
                assigned[j].push(i);
            }
        }
    }
    assigned
}
fn text(tokens: &[Token], ids: &[usize]) -> String {
    let mut ids = ids.to_vec();
    ids.sort_by_key(|&i| (tokens[i].block_num, tokens[i].line_num, tokens[i].span_num));
    let mut lines: Vec<String> = vec![];
    let mut pieces: Vec<&str> = vec![];
    let mut last = None;
    for i in ids {
        let t = &tokens[i];
        let line = (t.block_num, t.line_num);
        if last.is_some_and(|l| l != line) {
            lines.push(pieces.join(" ").trim().to_owned());
            pieces.clear();
        }
        pieces.push(&t.text);
        last = Some(line);
    }
    if !pieces.is_empty() {
        lines.push(pieces.join(" "));
    }
    lines.join(" ").trim().to_owned()
}
fn refine(mut v: Vec<Object>, tokens: &[Token], row: bool) -> Vec<Object> {
    if tokens.is_empty() {
        v = nms(v, if row { 0.5 } else { 0.25 });
    } else {
        score_sort(&mut v);
        let assigned = slots(&v.iter().map(|o| o.bbox).collect::<Vec<_>>(), tokens, 0.5);
        let mut suppressed = vec![false; v.len()];
        // Original nms_by_containment deliberately keeps the first object even
        // when its assignment is empty, then removes objects without content.
        for j in 1..v.len() {
            if assigned[j].is_empty() {
                suppressed[j] = true;
            }
            for i in 0..j {
                if !suppressed[i] && assigned[j].iter().any(|t| assigned[i].contains(t)) {
                    suppressed[j] = true;
                }
            }
        }
        v = v
            .into_iter()
            .enumerate()
            .filter_map(|(i, o)| {
                let ids = tokens
                    .iter()
                    .enumerate()
                    .filter_map(|(j, t)| (iob(t.bbox, o.bbox) >= 0.5).then_some(j))
                    .collect::<Vec<_>>();
                (!suppressed[i] && !text(tokens, &ids).trim().is_empty()).then_some(o)
            })
            .collect();
    }
    let axis = if row { 1 } else { 0 };
    v.sort_by(|a, b| {
        (a.bbox[axis] + a.bbox[axis + 2]).total_cmp(&(b.bbox[axis] + b.bbox[axis + 2]))
    });
    v
}
fn shrink(a: &Object, b: &mut Object) {
    while b.rows.iter().any(|r| a.rows.contains(r))
        && b.columns.iter().any(|c| a.columns.contains(c))
    {
        let (v, other) = if b.rows.len() < b.columns.len() {
            (&mut b.columns, &a.columns)
        } else {
            (&mut b.rows, &a.rows)
        };
        if other.contains(v.last().unwrap()) {
            v.pop();
        } else if other.contains(&v[0]) {
            v.remove(0);
        } else {
            v.clear();
        }
    }
}

pub fn decode_objects(objects: &[Object], tokens: &[Token]) -> Vec<Cell> {
    let Some(table) = objects
        .iter()
        .filter(|o| o.label == 0 && o.score >= 0.5)
        .max_by(|a, b| {
            area(a.bbox)
                .total_cmp(&area(b.bbox))
                .then(std::cmp::Ordering::Greater)
        })
    else {
        return vec![];
    };
    let selected = objects
        .iter()
        .filter(|o| o.label != 0 && iob(o.bbox, table.bbox) >= 0.5)
        .cloned()
        .collect::<Vec<_>>();
    let table_tokens = tokens
        .iter()
        .filter(|t| iob(t.bbox, table.bbox) >= 0.5)
        .cloned()
        .collect::<Vec<_>>();
    let mut rows = refine(
        selected.iter().filter(|o| o.label == 2).cloned().collect(),
        &table_tokens,
        true,
    );
    let mut cols = refine(
        selected.iter().filter(|o| o.label == 1).cloned().collect(),
        &table_tokens,
        false,
    );
    if rows.is_empty() || cols.is_empty() {
        return vec![];
    }
    let rb = hull(rows.iter().map(|o| o.bbox));
    let cb = hull(cols.iter().map(|o| o.bbox));
    for r in &mut rows {
        r.header = selected
            .iter()
            .any(|h| h.label == 3 && iob(r.bbox, h.bbox) >= 0.5);
        r.bbox[0] = cb[0];
        r.bbox[2] = cb[2];
    }
    for c in &mut cols {
        c.bbox[1] = rb[1];
        c.bbox[3] = rb[3];
    }
    let mut supers = selected
        .iter()
        .filter(|o| matches!(o.label, 4 | 5))
        .cloned()
        .collect::<Vec<_>>();
    if cols.len() > 1 {
        let headers = nms(
            selected
                .iter()
                .filter(|o| o.label == 3 && o.score >= 0.5)
                .cloned()
                .collect(),
            0.05,
        );
        let mut nums = vec![];
        for h in headers {
            for (i, r) in rows.iter().enumerate() {
                if (r.bbox[3].min(h.bbox[3]) - r.bbox[1].max(h.bbox[1])) / (r.bbox[3] - r.bbox[1])
                    >= 0.5
                {
                    nums.push(i);
                }
            }
        }
        if let Some(&first) = nums.first() {
            if first > 0 {
                let mut prefix = (0..=first).collect::<Vec<_>>();
                prefix.extend(nums);
                nums = prefix;
            }
        }
        for r in &mut rows {
            r.header = false;
        }
        let mut last = None;
        for i in nums {
            if i == last.map_or(0, |n| n + 1) {
                rows[i].header = true;
                last = Some(i);
            } else {
                break;
            }
        }
        // Match current Python adapter ordering: ordinary spanning cells then
        // projected headers. Model objects never carry the optional span flag.
        supers.sort_by_key(|o| o.label == 4);
        supers.retain(|o| o.score >= 0.5);
        supers = supers
            .into_iter()
            .filter_map(|mut o| {
                o.rows = rows
                    .iter()
                    .enumerate()
                    .filter_map(|(i, r)| {
                        ((r.bbox[3].min(o.bbox[3]) - r.bbox[1].max(o.bbox[1]))
                            / (r.bbox[3] - r.bbox[1])
                            >= 0.5)
                            .then_some(i)
                    })
                    .collect();
                o.columns = cols
                    .iter()
                    .enumerate()
                    .filter_map(|(i, c)| {
                        ((c.bbox[2].min(o.bbox[2]) - c.bbox[0].max(o.bbox[0]))
                            / (c.bbox[2] - c.bbox[0])
                            >= 0.5)
                            .then_some(i)
                    })
                    .collect();
                if o.rows.is_empty() || o.columns.is_empty() {
                    return None;
                }
                o.bbox = intersection(
                    hull(o.rows.iter().map(|&i| rows[i].bbox)),
                    hull(o.columns.iter().map(|&i| cols[i].bbox)),
                );
                (o.rows.len() > 1 || o.columns.len() > 1).then_some(o)
            })
            .collect();
        score_sort(&mut supers);
        let mut suppressed = vec![false; supers.len()];
        for j in 1..supers.len() {
            let (left, right) = supers.split_at_mut(j);
            for a in left {
                shrink(a, &mut right[0]);
            }
            suppressed[j] = right[0].rows.is_empty()
                || right[0].columns.is_empty()
                || (right[0].rows.len() < 2 && right[0].columns.len() < 2);
        }
        supers = supers
            .into_iter()
            .enumerate()
            .filter_map(|(i, o)| (!suppressed[i]).then_some(o))
            .collect();
    }
    // 合并格经过 NMS 后，框必须按最终逻辑范围重算，避免旧框吞掉邻格。
    for span in &mut supers {
        span.bbox = intersection(
            hull(span.rows.iter().map(|&i| rows[i].bbox)),
            hull(span.columns.iter().map(|&i| cols[i].bbox)),
        );
    }
    restore_rows(&mut rows, &cols, &mut supers, tokens);
    let mut repaired_cols = cols.clone();
    let mut repaired_supers = supers.clone();
    let added = restore_columns(&rows, &mut repaired_cols, &mut repaired_supers, tokens);
    let cells = structure_cells(&rows, &cols, &supers, tokens);
    if added.is_empty() {
        return cells;
    }
    let repaired = structure_cells(&rows, &repaired_cols, &repaired_supers, tokens);
    let original: std::collections::BTreeSet<_> =
        cells.iter().flat_map(|c| c.spans.iter().copied()).collect();
    let retained: std::collections::BTreeSet<_> = repaired
        .iter()
        .flat_map(|c| c.spans.iter().copied())
        .collect();
    // 新增列必须恢复多行独立数值，且不丢掉原候选已有的 OCR 实例。
    if original.is_subset(&retained)
        && added.iter().all(|col| {
            repaired
                .iter()
                .filter(|c| {
                    c.column_nums == [*col]
                        && c.row_nums.len() == 1
                        && c.spans
                            .iter()
                            .any(|i| !original.contains(i) && is_numeric(tokens[*i].text.trim()))
                })
                .count() as f64
                >= 3f64.max(0.25 * rows.len() as f64)
        })
    {
        repaired
    } else {
        cells
    }
}

fn structure_cells(
    source_rows: &[Object],
    source_cols: &[Object],
    supers: &[Object],
    tokens: &[Token],
) -> Vec<Cell> {
    let mut rows = source_rows.to_vec();
    let mut cols = source_cols.to_vec();
    let mut covered = std::collections::BTreeSet::new();
    let mut merged = Vec::new();
    for span in supers {
        let rn = &span.rows;
        let cn = &span.columns;
        if rn.is_empty()
            || cn.is_empty()
            || rn.last().unwrap() - rn[0] + 1 != rn.len()
            || cn.last().unwrap() - cn[0] + 1 != cn.len()
        {
            continue;
        }
        let slots: Vec<_> = rn
            .iter()
            .flat_map(|r| cn.iter().map(move |c| (*r, *c)))
            .collect();
        if slots.iter().any(|p| covered.contains(p)) {
            continue;
        }
        covered.extend(slots);
        merged.push(Cell {
            bbox: [
                cols[cn[0]].bbox[0],
                rows[rn[0]].bbox[1],
                cols[*cn.last().unwrap()].bbox[2],
                rows[*rn.last().unwrap()].bbox[3],
            ],
            row_nums: rn.clone(),
            column_nums: cn.clone(),
            column_header: rn.iter().all(|&r| rows[r].header),
            text: String::new(),
            spans: vec![],
        });
    }
    let mut cells = Vec::new();
    for (c, col) in cols.iter().enumerate() {
        for (r, row) in rows.iter().enumerate() {
            if covered.contains(&(r, c)) {
                continue;
            }
            cells.push(Cell {
                bbox: intersection(row.bbox, col.bbox),
                row_nums: vec![r],
                column_nums: vec![c],
                column_header: row.header,
                text: String::new(),
                spans: vec![],
            });
        }
    }
    cells.extend(merged);
    for cell in &mut cells {
        cell.bbox = intersection(
            hull(cell.row_nums.iter().map(|&i| rows[i].bbox)),
            hull(cell.column_nums.iter().map(|&i| cols[i].bbox)),
        );
    }
    let assigned = slots(
        &cells.iter().map(|c| c.bbox).collect::<Vec<_>>(),
        tokens,
        0.001,
    );
    for (cell, ids) in cells.iter_mut().zip(assigned) {
        cell.text = text(tokens, &ids);
        cell.spans = ids;
    }
    let (nr, nc) = (rows.len(), cols.len());
    let mut minx = vec![vec![]; nc];
    let mut maxx = vec![vec![]; nc];
    let mut miny = vec![vec![]; nr];
    let mut maxy = vec![vec![]; nr];
    for c in &cells {
        for &i in &c.spans {
            let b = tokens[i].bbox;
            minx[c.column_nums[0]].push(b[0]);
            maxx[*c.column_nums.last().unwrap()].push(b[2]);
            miny[c.row_nums[0]].push(b[1]);
            maxy[*c.row_nums.last().unwrap()].push(b[3]);
        }
    }
    fn lo(v: &[f64], d: f64) -> f64 {
        v.iter().copied().reduce(f64::min).unwrap_or(d)
    }
    fn hi(v: &[f64], d: f64) -> f64 {
        v.iter().copied().reduce(f64::max).unwrap_or(d)
    }
    for (i, r) in rows.iter_mut().enumerate() {
        r.bbox = [
            lo(&minx[0], r.bbox[0]),
            lo(&miny[i], r.bbox[1]),
            hi(&maxx[nc - 1], r.bbox[2]),
            hi(&maxy[i], r.bbox[3]),
        ];
    }
    for (i, c) in cols.iter_mut().enumerate() {
        c.bbox = [
            lo(&minx[i], c.bbox[0]),
            lo(&miny[0], c.bbox[1]),
            hi(&maxx[i], c.bbox[2]),
            hi(&maxy[nr - 1], c.bbox[3]),
        ];
    }
    for cell in &mut cells {
        let b = intersection(
            hull(cell.row_nums.iter().map(|&i| rows[i].bbox)),
            hull(cell.column_nums.iter().map(|&i| cols[i].bbox)),
        );
        if area(b) > 0. {
            cell.bbox = b;
        }
    }
    cells
}
fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut values: Vec<_> = values.collect();
    values.sort_by(f64::total_cmp);
    if values.is_empty() {
        return 0.;
    }
    (values[(values.len() - 1) / 2] + values[values.len() / 2]) / 2.
}
fn restore_rows(rows: &mut Vec<Object>, cols: &[Object], supers: &mut [Object], tokens: &[Token]) {
    if rows.len() < 3 || cols.len() < 4 {
        return;
    }
    let height = median(rows.iter().map(|r| r.bbox[3] - r.bbox[1]));
    let mut insertions = std::collections::BTreeMap::new();
    for (i, pair) in rows.windows(2).enumerate() {
        let (top, bottom) = (pair[0].bbox[3], pair[1].bbox[1]);
        let gap = bottom - top;
        if pair[0].header
            || pair[1].header
            || gap < 0.65 * height
            || gap > 1.4 * height
            || supers
                .iter()
                .any(|s| s.rows[0] <= i && i < *s.rows.last().unwrap())
        {
            continue;
        }
        let band: Vec<_> = tokens
            .iter()
            .filter(|t| {
                let b = t.bbox;
                let c = (b[1] + b[3]) / 2.;
                c >= top
                    && c <= bottom
                    && b[1] >= top - 0.12 * height
                    && b[3] <= bottom + 0.12 * height
            })
            .cloned()
            .collect();
        if band.is_empty() {
            continue;
        }
        let centers = band.iter().map(|t| (t.bbox[1] + t.bbox[3]) / 2.);
        let (lo, hi) = centers.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(v), hi.max(v))
        });
        if hi - lo > 0.45 * height {
            continue;
        }
        let assigned = slots(&cols.iter().map(|c| c.bbox).collect::<Vec<_>>(), &band, 0.5);
        if !assigned[0]
            .iter()
            .any(|&j| !is_numeric(band[j].text.trim()))
        {
            continue;
        }
        if assigned[1..]
            .iter()
            .filter(|ids| ids.len() == 1 && is_numeric(band[ids[0]].text.trim()))
            .count()
            < 3
        {
            continue;
        }
        insertions.insert(
            i,
            Object::new(2, 1., [pair[0].bbox[0], top, pair[0].bbox[2], bottom]),
        );
    }
    if insertions.is_empty() {
        return;
    }
    let mut rebuilt = Vec::new();
    let mut indices = Vec::new();
    for (i, row) in rows.iter().enumerate() {
        indices.push(rebuilt.len());
        rebuilt.push(row.clone());
        if let Some(row) = insertions.remove(&i) {
            rebuilt.push(row);
        }
    }
    for span in supers {
        span.rows = span.rows.iter().map(|&i| indices[i]).collect();
    }
    *rows = rebuilt;
}
fn restore_columns(
    rows: &[Object],
    cols: &mut Vec<Object>,
    supers: &mut [Object],
    tokens: &[Token],
) -> Vec<usize> {
    if rows.len() < 3 || cols.len() < 3 {
        return vec![];
    }
    let width = median(cols.iter().map(|c| c.bbox[2] - c.bbox[0]));
    if width <= 0. {
        return vec![];
    }
    let mut insertions = std::collections::BTreeMap::new();
    for (i, pair) in cols.windows(2).enumerate() {
        let (lo, hi) = (pair[0].bbox[2], pair[1].bbox[0]);
        if hi - lo < 0.5 * width {
            continue;
        }
        let band: Vec<_> = tokens
            .iter()
            .filter(|t| {
                t.confidence >= 0.85
                    && is_numeric(t.text.trim())
                    && lo <= t.bbox[0]
                    && t.bbox[0] < t.bbox[2]
                    && t.bbox[2] <= hi
            })
            .cloned()
            .collect();
        if band.len() < 3 {
            continue;
        }
        let assigned = slots(
            &rows.iter().map(|r| r.bbox).collect::<Vec<_>>(),
            &band,
            0.75,
        );
        let data: Vec<_> = rows
            .iter()
            .zip(assigned)
            .filter_map(|(r, ids)| (!r.header && !ids.is_empty()).then_some(ids))
            .collect();
        if (data.len() as f64) < 3f64.max(0.25 * rows.len() as f64)
            || data.iter().any(|ids| ids.len() != 1)
        {
            continue;
        }
        let boxes: Vec<_> = data.iter().map(|ids| band[ids[0]].bbox).collect();
        let spread = |axis: usize| {
            boxes
                .iter()
                .map(|b| b[axis])
                .fold(f64::NEG_INFINITY, f64::max)
                - boxes.iter().map(|b| b[axis]).fold(f64::INFINITY, f64::min)
        };
        let token_width = median(boxes.iter().map(|b| b[2] - b[0]));
        if spread(0).min(spread(2)) > 0.45 * token_width {
            continue;
        }
        insertions.insert(
            i,
            Object::new(1, 1., [lo, pair[0].bbox[1], hi, pair[0].bbox[3]]),
        );
    }
    let mut rebuilt = Vec::new();
    let mut indices = Vec::new();
    let mut added = Vec::new();
    for (i, col) in cols.iter().enumerate() {
        indices.push(rebuilt.len());
        rebuilt.push(col.clone());
        if let Some(col) = insertions.remove(&i) {
            added.push(rebuilt.len());
            rebuilt.push(col);
        }
    }
    for span in supers {
        span.columns =
            (indices[span.columns[0]]..=indices[*span.columns.last().unwrap()]).collect();
    }
    *cols = rebuilt;
    added
}

#[cfg(test)]
mod repair_tests {
    use super::*;
    fn token(bbox: [f64; 4], text: &str, confidence: f64) -> Token {
        Token {
            bbox,
            text: text.into(),
            confidence,
            block_num: 0,
            line_num: 0,
            span_num: 0,
        }
    }
    #[test]
    fn internal_row_needs_label_and_three_values() {
        let rows = vec![
            Object::new(2, 1., [0., 0., 400., 20.]),
            Object::new(2, 1., [0., 40., 400., 60.]),
            Object::new(2, 1., [0., 60., 400., 80.]),
        ];
        let cols: Vec<_> = (0..4)
            .map(|c| Object::new(1, 1., [c as f64 * 100., 0., (c + 1) as f64 * 100., 80.]))
            .collect();
        let tokens: Vec<_> = (0..4)
            .map(|c| {
                token(
                    [c as f64 * 100. + 5., 24., c as f64 * 100. + 70., 36.],
                    if c == 0 { "label" } else { "123" },
                    0.99,
                )
            })
            .collect();
        let mut partial = rows.clone();
        restore_rows(&mut partial, &cols, &mut [], &tokens[..3]);
        assert_eq!(partial.len(), 3);
        let mut complete = rows;
        restore_rows(&mut complete, &cols, &mut [], &tokens);
        assert_eq!(complete.len(), 4);
        assert_eq!(complete[1].bbox, [0., 20., 400., 40.]);
    }
    #[test]
    fn internal_column_needs_repeated_confident_values() {
        let rows: Vec<_> = (0..4)
            .map(|r| Object::new(2, 1., [0., r as f64 * 20., 400., (r + 1) as f64 * 20.]))
            .collect();
        let cols = vec![
            Object::new(1, 1., [0., 0., 100., 80.]),
            Object::new(1, 1., [200., 0., 300., 80.]),
            Object::new(1, 1., [300., 0., 400., 80.]),
        ];
        let mut tokens: Vec<_> = (0..4)
            .map(|r| {
                token(
                    [125., r as f64 * 20. + 3., 175., r as f64 * 20. + 17.],
                    "123",
                    0.8,
                )
            })
            .collect();
        let mut low = cols.clone();
        assert!(restore_columns(&rows, &mut low, &mut [], &tokens).is_empty());
        for t in &mut tokens {
            t.confidence = 0.99;
        }
        let mut high = cols;
        assert_eq!(restore_columns(&rows, &mut high, &mut [], &tokens), [1]);
        assert_eq!(high.len(), 4);
    }
}
