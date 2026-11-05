//! 旧candidate.py：给定网格规格的校验、落格、文字/来源、拒绝诊断、质量分及普通HTML。
use super::table_native_text::{self, Glyph, Text};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::NativeGridIndex;
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Deserialize)]
struct Spec {
    row: i64,
    col: i64,
    rowspan: i64,
    colspan: i64,
    bbox: [f64; 4],
}
struct Grid {
    rows: usize,
    cols: usize,
    specs: Vec<Spec>,
    valid: bool,
    index: Option<NativeGridIndex>,
}
fn flag(v: &Value, key: &str) -> Result<bool, Error> {
    v.get(key)
        .map_or(Ok(false), |x| x.as_bool().ok_or_else(invalid))
}
fn integer(v: &Value) -> Result<i64, Error> {
    let n = v.as_i64().ok_or_else(invalid)?;
    if n.unsigned_abs() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    Ok(n)
}
fn number(v: &Value) -> Result<f64, Error> {
    v.as_f64().filter(|n| n.is_finite()).ok_or_else(invalid)
}
pub(super) fn sum(values: &[f64]) -> f64 {
    // CPython3.12+有限float sum的补偿求和；轨道值保持原spec顺序。
    let (mut hi, mut lo) = (0f64, 0f64);
    for &x in values {
        let t = hi + x;
        lo += if hi.abs() >= x.abs() {
            (hi - t) + x
        } else {
            (x - t) + hi
        };
        hi = t;
    }
    hi + lo
}
fn grid(v: &Value, work: &mut Work) -> Result<Grid, Error> {
    let (rows, cols) = (integer(&v["rows"])?, integer(&v["cols"])?);
    let specs = Vec::<Spec>::deserialize(&v["specs"]).map_err(|_| invalid())?;
    if specs.len() > rapiddoc_core::resources::limit(2000)
        || specs.iter().any(|s| {
            [s.row, s.col, s.rowspan, s.colspan]
                .iter()
                .any(|n| n.unsigned_abs() > rapiddoc_core::resources::limit(1000))
                || s.bbox.iter().any(|n| !n.is_finite() || n.abs() > 1e9)
        })
    {
        return Err(limit());
    }
    let mut result = Grid {
        rows: rows.max(0) as usize,
        cols: cols.max(0) as usize,
        specs,
        valid: false,
        index: None,
    };
    if rows < (if flag(v, "allow_single_row")? { 1 } else { 2 })
        || cols
            < (if flag(v, "allow_single_column")? {
                1
            } else {
                2
            })
        || (rows == 1 && cols == 1)
        || result.specs.is_empty()
    {
        return Ok(result);
    }
    let count = result.rows * result.cols;
    if count > rapiddoc_core::resources::limit(10000) {
        return Err(limit());
    }
    work.charge(count)?;
    let mut owners = vec![vec![usize::MAX; result.cols]; result.rows];
    for (i, s) in result.specs.iter().enumerate() {
        if s.row < 0
            || s.col < 0
            || s.rowspan < 1
            || s.colspan < 1
            || s.row + s.rowspan > rows
            || s.col + s.colspan > cols
            || s.bbox[2] <= s.bbox[0]
            || s.bbox[3] <= s.bbox[1]
        {
            return Ok(result);
        }
        work.charge((s.rowspan * s.colspan) as usize)?;
        for row in owners
            .iter_mut()
            .skip(s.row as usize)
            .take(s.rowspan as usize)
        {
            for owner in row.iter_mut().skip(s.col as usize).take(s.colspan as usize) {
                if *owner != usize::MAX {
                    return Ok(result);
                }
                *owner = i;
            }
        }
    }
    if owners.iter().flatten().any(|x| *x == usize::MAX) {
        return Ok(result);
    }
    result.valid = true;
    if flag(v, "use_grid_index")? {
        let mut xs = vec![Vec::new(); result.cols + 1];
        let mut ys = vec![Vec::new(); result.rows + 1];
        for s in &result.specs {
            xs[s.col as usize].push(s.bbox[0]);
            xs[(s.col + s.colspan) as usize].push(s.bbox[2]);
            ys[s.row as usize].push(s.bbox[1]);
            ys[(s.row + s.rowspan) as usize].push(s.bbox[3]);
        }
        if xs.iter().chain(&ys).all(|a| !a.is_empty()) {
            let x: Vec<_> = xs.iter().map(|a| sum(a) / a.len() as f64).collect();
            let y: Vec<_> = ys.iter().map(|a| sum(a) / a.len() as f64).collect();
            if x.windows(2).all(|p| p[1] > p[0]) && y.windows(2).all(|p| p[1] > p[0]) {
                result.index = Some((x, y, owners));
            }
        }
    }
    Ok(result)
}
pub(super) fn text(v: &Value) -> Result<Text, Error> {
    let t = Text::deserialize(v).map_err(|_| invalid())?;
    if t.glyphs.len() > crate::legacy_page::budget::records(3000)
        || t.rows.len() > crate::legacy_page::budget::records(3000)
        || !t.median_glyph_height.is_finite()
        || t.median_glyph_height <= 0.
        || !t.median_glyph_width.is_finite()
    {
        return Err(limit());
    }
    if t.glyphs.iter().any(|g| {
        g.glyph_id > rapiddoc_core::resources::limit(1_000_000)
            || g.source_index.unsigned_abs() > rapiddoc_core::resources::limit(1_000_000_000_000)
            || g.visual_row.unsigned_abs() > rapiddoc_core::resources::limit(10000)
            || g.bbox.iter().any(|x| !x.is_finite() || x.abs() > 1e9)
    }) {
        return Err(limit());
    }
    Ok(t)
}
fn assign(g: &Grid, t: &Text, work: &mut Work) -> Result<Vec<(Option<usize>, bool)>, Error> {
    work.charge(t.glyphs.len() * g.specs.len() * 3)?;
    let kernel = if crate::legacy_page::budget::current().host {
        rapiddoc_native::table_assign_cells_page
    } else {
        rapiddoc_native::table_assign_cells
    };
    kernel(
        t.glyphs.iter().map(|g| g.bbox).collect(),
        g.specs.iter().map(|s| s.bbox).collect(),
        g.index.clone(),
    )
}
pub fn inspect(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let mut work = Work(0);
    let g = grid(v, &mut work)?;
    let assignments = if g.valid {
        Some(assign(&g, &text(&v["text"])?, &mut work)?)
    } else {
        None
    };
    let index = g
        .index
        .as_ref()
        .map(|(x, y, owners)| json!({"x_tracks":x,"y_tracks":y,"owners":owners}));
    Ok(json!({"valid":g.valid,"index":index,"assignments":assignments}))
}
fn rejected(mut diagnostics: Map<String, Value>, gate: &str) -> Value {
    diagnostics.insert("candidate_rejection_gate".into(), json!(gate));
    json!({"candidate":null,"diagnostics":diagnostics,"html":null})
}
pub fn build(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let mut work = Work(0);
    let g = grid(v, &mut work)?;
    let mut diagnostics = match v.get("diagnostics") {
        None => Map::new(),
        Some(x) => x.as_object().cloned().ok_or_else(invalid)?,
    };
    if !g.valid {
        return Ok(rejected(diagnostics, "grid_specs"));
    }
    let t = text(&v["text"])?;
    let matched = assign(&g, &t, &mut work)?;
    let mut assignments = BTreeMap::new();
    let mut groups = vec![Vec::<Glyph>::new(); g.specs.len()];
    let mut ambiguous = 0usize;
    for (glyph, (owner, amb)) in t.glyphs.iter().zip(matched) {
        if let Some(i) = owner {
            assignments.insert(glyph.glyph_id, i);
            groups[i].push(glyph.clone());
            ambiguous += usize::from(amb);
        }
    }
    let count = t.glyphs.len();
    let capture = if count == 0 {
        0.
    } else {
        assignments.len() as f64 / count as f64
    };
    let ratio = if count == 0 {
        0.
    } else {
        ambiguous as f64 / count as f64
    };
    let mut split = 0usize;
    for row in &t.rows {
        for token in &row.tokens {
            work.charge(token.glyph_ids.len())?;
            let owners: BTreeSet<_> = token
                .glyph_ids
                .iter()
                .filter_map(|i| assignments.get(i))
                .collect();
            split += usize::from(owners.len() > 1);
        }
    }
    diagnostics.insert("token_split_count".into(), json!(split));
    diagnostics.insert("ambiguous_glyph_ratio".into(), json!(ratio));
    if capture < 0.98 || ratio > 0.02 {
        return Ok(rejected(diagnostics, "text_assignment"));
    }
    if flag(v, "require_atomic_tokens")? && split > 0 {
        return Ok(rejected(diagnostics, "token_split"));
    }
    if groups
        .iter()
        .any(|glyphs| table_native_text::cell_writing_angle(glyphs).is_none())
    {
        return Ok(rejected(diagnostics, "cell_writing_direction"));
    }
    let mut cells = Vec::new();
    let mut content_size = 0;
    for (i, (s, glyphs)) in g.specs.iter().zip(groups.iter_mut()).enumerate() {
        work.charge(glyphs.len() * glyphs.len() * 2)?;
        let parts = table_native_text::cell_parts(
            &json!({"glyphs":glyphs,"median_height":t.median_glyph_height}),
        )?;
        let content = parts["text"].as_str().ok_or_else(invalid)?;
        content_size += content.len();
        if content_size > rapiddoc_core::resources::limit(1_000_000) {
            return Err(limit());
        }
        glyphs.sort_by(|a, b| {
            a.visual_row
                .cmp(&b.visual_row)
                .then(a.bbox[0].partial_cmp(&b.bbox[0]).unwrap())
                .then(a.glyph_id.cmp(&b.glyph_id))
        });
        cells.push(json!({"row":s.row,"col":s.col,"rowspan":s.rowspan,"colspan":s.colspan,"bbox":v["specs"][i]["bbox"],"content":content,"source_char_indices":if glyphs.iter().any(|g|g.writing_angle.is_some_and(|a|a!=0)){parts["parts"].as_array().unwrap().iter().filter_map(|p|p[1].as_i64()).collect::<Vec<_>>()}else{glyphs.iter().map(|g|g.source_index).collect::<Vec<_>>()}}));
    }
    let by_id: BTreeMap<_, _> = t.glyphs.iter().map(|g| (g.glyph_id, g)).collect();
    let (mut comparable, mut ordered) = (0usize, 0usize);
    for row in &t.rows {
        let mut glyphs: Vec<_> = row
            .glyph_ids
            .iter()
            .filter(|i| assignments.contains_key(i))
            .map(|i| by_id[i])
            .collect();
        glyphs.sort_by(|a, b| {
            a.bbox[0]
                .partial_cmp(&b.bbox[0])
                .unwrap()
                .then(a.glyph_id.cmp(&b.glyph_id))
        });
        let mut previous = None;
        for glyph in glyphs {
            let s = &g.specs[assignments[&glyph.glyph_id]];
            let position = (s.row, s.col);
            if let Some(p) = previous {
                comparable += 1;
                ordered += usize::from(position >= p);
            }
            previous = Some(position);
        }
    }
    let consistency = if comparable == 0 {
        1.
    } else {
        ordered as f64 / comparable as f64
    };
    let structure = number(&v["structure_support"])?.clamp(0., 1.);
    let row = number(&v["row_stability"])?.clamp(0., 1.);
    let col = number(&v["column_stability"])?.clamp(0., 1.);
    let source = v["source"]
        .as_str()
        .filter(|s| {
            matches!(
                *s,
                "vector_grid"
                    | "sparse_hybrid"
                    | "sparse_grid"
                    | "key_value"
                    | "text_grid"
                    | "sparse_multiline"
            )
        })
        .ok_or_else(invalid)?;
    let issues = v.get("issues").cloned().unwrap_or(json!([]));
    if !issues.is_array() {
        return Err(invalid());
    }
    let candidate = json!({"source":source,"rows":g.rows,"cols":g.cols,"cells":cells,"score":capture.min(structure).min(row).min(col).min(consistency),"text_capture":capture,"structure_support":structure,"row_stability":row,"column_stability":col,"order_consistency":consistency,"issues":issues});
    diagnostics.insert("candidate_rejection_gate".into(), Value::Null);
    let html = render(&candidate)?;
    Ok(json!({"candidate":candidate,"diagnostics":diagnostics,"html":html}))
}
pub fn render(v: &Value) -> Result<String, Error> {
    validate_tree(v)?;
    render_with(v, |c| {
        Ok(c["content"]
            .as_str()
            .unwrap()
            .replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;"))
    })
}
pub(super) fn render_with(
    v: &Value,
    mut cell_html: impl FnMut(&Value) -> Result<String, Error>,
) -> Result<String, Error> {
    let rows = integer(&v["rows"])?;
    let cells = v["cells"]
        .as_array()
        .filter(|a| a.len() <= rapiddoc_core::resources::limit(2000))
        .ok_or_else(invalid)?;
    let mut grouped = BTreeMap::<i64, Vec<&Value>>::new();
    let mut bytes = 0;
    for c in cells {
        integer(&c["col"])?;
        integer(&c["rowspan"])?;
        integer(&c["colspan"])?;
        let content = c["content"].as_str().ok_or_else(invalid)?;
        bytes += content.len();
        if bytes > rapiddoc_core::resources::limit(1_000_000) {
            return Err(limit());
        }
        grouped.entry(integer(&c["row"])?).or_default().push(c);
    }
    let mut html = String::from("<table><tbody>");
    for row in 0..rows {
        html.push_str("<tr>");
        if let Some(cells) = grouped.get_mut(&row) {
            cells.sort_by_key(|c| c["col"].as_i64().unwrap());
            for c in cells {
                html.push_str("<td");
                for field in ["rowspan", "colspan"] {
                    let n = c[field].as_i64().unwrap();
                    if n > 1 {
                        html.push_str(&format!(" {field}=\"{n}\""));
                    }
                }
                html.push('>');
                let fragment = cell_html(c)?;
                if html.len().saturating_add(fragment.len())
                    > rapiddoc_core::resources::limit(4_000_000)
                {
                    return Err(limit());
                }
                html.push_str(&fragment);
                html.push_str("</td>");
            }
        }
        html.push_str("</tr>");
    }
    html.push_str("</tbody></table>");
    if html.len() > rapiddoc_core::resources::limit(4_000_000) {
        return Err(limit());
    }
    Ok(html)
}
