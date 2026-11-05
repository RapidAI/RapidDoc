//! Port of the Python TATR adapter's selected Microsoft structure.py/postprocess.py.
//! Stable ordering, shared-token suppression and supercell shrinking are intentional.
use serde::{Deserialize, Serialize};
type B = [f64; 4];
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
    let mut cells = vec![];
    let mut subcells = vec![];
    for (c, col) in cols.iter().enumerate() {
        for (r, row) in rows.iter().enumerate() {
            let b = intersection(row.bbox, col.bbox);
            let cell = Cell {
                bbox: b,
                row_nums: vec![r],
                column_nums: vec![c],
                column_header: row.header,
                text: String::new(),
                spans: vec![],
            };
            if supers.iter().any(|s| iob(b, s.bbox) > 0.5) {
                subcells.push(cell);
            } else {
                cells.push(cell);
            }
        }
    }
    for s in &supers {
        let selected = subcells
            .iter()
            .filter(|c| iob(c.bbox, s.bbox) > 0.5)
            .collect::<Vec<_>>();
        if selected.is_empty() {
            continue;
        }
        let mut rn = selected
            .iter()
            .flat_map(|c| c.row_nums.iter().copied())
            .collect::<Vec<_>>();
        rn.sort();
        rn.dedup();
        let mut cn = selected
            .iter()
            .flat_map(|c| c.column_nums.iter().copied())
            .collect::<Vec<_>>();
        cn.sort();
        cn.dedup();
        cells.push(Cell {
            bbox: hull(selected.iter().map(|c| c.bbox)),
            row_nums: rn,
            column_nums: cn,
            column_header: selected.iter().all(|c| c.column_header),
            text: String::new(),
            spans: vec![],
        });
    }
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
