//! 旧vector.py轨道硬门和line/rect假设编排；输入为已有PDF原语，不读取PDF。
use super::{
    canonicalize_with_work, coverage, local, normalized, number, prepare_with_work, topology, Rule,
    Track,
};
use crate::legacy_model::table_native_text::Text;
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
mod digit;
#[derive(Clone, Deserialize, Serialize, PartialEq)]
struct Drawing {
    bbox: [f64; 4],
    width: f64,
    orientation: String,
}
type Cache = BTreeMap<(bool, bool, u64), std::sync::Arc<Value>>;
fn angle(input: &Value) -> Result<i64, Error> {
    let n = input.get("angle").map_or(Ok(0.), number)?.trunc() as i64;
    let n = n.rem_euclid(360);
    Ok(if [0, 90, 180, 270].contains(&n) { n } else { 0 })
}
fn bbox(input: &Value) -> Result<[f64; 4], Error> {
    serde::Deserialize::deserialize(&input["table_bbox"]).map_err(|_| invalid())
}
fn drawings(input: &Value) -> Result<Vec<Drawing>, Error> {
    let a: Vec<Drawing> =
        serde_json::from_value(input.get("drawing_lines").cloned().unwrap_or(json!([])))
            .map_err(|_| invalid())?;
    if a.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    if a.iter().any(|r| {
        r.bbox
            .iter()
            .chain([&r.width])
            .any(|x| !x.is_finite() || x.abs() > 1e9)
            || !["horizontal", "vertical"].contains(&r.orientation.as_str())
    }) {
        return Err(invalid());
    }
    Ok(a)
}
fn cov(
    rules: &[Rule],
    orientation: &str,
    aliases: &[f64],
    start: f64,
    end: f64,
    snap: f64,
    work: &mut Work,
) -> Result<f64, Error> {
    work.charge(rules.len() * aliases.len())?;
    coverage(
        rules
            .iter()
            .filter(|r| {
                r.orientation == orientation
                    && aliases.iter().any(|a| (r.coordinate - a).abs() <= snap)
            })
            .map(|r| (r.start, r.end))
            .collect(),
        start,
        end,
    )
}
fn reject(d: &mut Map<String, Value>, gate: &str) -> Option<Value> {
    d.insert("first_rejection_gate".into(), json!(gate));
    None
}
fn repeated(
    rules: &[Rule],
    x: &[f64],
    y: &[f64],
    snap: f64,
    work: &mut Work,
) -> Result<bool, Error> {
    if x.len() < 3 || y.len() < 3 {
        return Ok(false);
    }
    let (mut total, mut present, mut vertical, mut horizontal) = (0usize, 0usize, false, false);
    for (i, c) in x.iter().enumerate() {
        let mut count = 0;
        for p in y.windows(2) {
            let exists = cov(rules, "vertical", &[*c], p[0], p[1], snap, work)? >= 0.8;
            total += 1;
            present += usize::from(exists);
            count += usize::from(exists)
        }
        if i > 0 && i + 1 < x.len() && count >= 2 {
            vertical = true
        }
    }
    for (i, c) in y.iter().enumerate() {
        let mut count = 0;
        for p in x.windows(2) {
            let exists = cov(rules, "horizontal", &[*c], p[0], p[1], snap, work)? >= 0.8;
            total += 1;
            present += usize::from(exists);
            count += usize::from(exists)
        }
        if i > 0 && i + 1 < y.len() && count >= 2 {
            horizontal = true
        }
    }
    Ok(present as f64 / total as f64 >= 0.8 && vertical && horizontal)
}
fn dense(
    text: &Text,
    x: &[f64],
    y: &[f64],
    rules: &[Rule],
    cx: &[Track],
    snap: f64,
    work: &mut Work,
) -> Result<Vec<Value>, Error> {
    let cols = x.len() - 1;
    let minimum = 2usize.max((0.60 * cols as f64).ceil() as usize);
    let mut bands = vec![Vec::<(i64, Vec<usize>)>::new(); y.len() - 1];
    let by_index: BTreeMap<_, _> = text.rows.iter().map(|r| (r.row_index, r)).collect();
    for row in &text.rows {
        work.charge(y.len())?;
        let cy = (row.bbox[1] + row.bbox[3]) / 2.;
        let Some(b) = y.windows(2).position(|p| p[0] <= cy && cy <= p[1]) else {
            continue;
        };
        let mut columns = BTreeSet::new();
        for token in &row.tokens {
            work.charge(x.len())?;
            let c = (token.bbox[0] + token.bbox[2]) / 2.;
            if let Some(i) = x.windows(2).position(|p| p[0] <= c && c <= p[1]) {
                columns.insert(i);
            }
        }
        bands[b].push((row.row_index, columns.into_iter().collect()));
    }
    let mut blank_header = cols == 3
        && bands.len() >= 4
        && bands[0].len() == 2
        && bands[0].iter().all(|e| e.1 == [1, 2])
        && bands.iter().skip(1).filter(|b| !b.is_empty()).count() >= 3
        && bands.iter().skip(1).all(|b| b.iter().all(|e| e.1 == [0]));
    if blank_header {
        for (i, _) in &bands[0] {
            for token in &by_index[i].tokens {
                work.charge(token.text.len())?;
                if token.text.chars().any(|c| {
                    rapiddoc_native::python_decimal(c).is_some()
                        || digit::EXTRA.binary_search(&(c as u32)).is_ok()
                }) {
                    blank_header = false;
                    break;
                }
            }
        }
    }
    let mut out = Vec::new();
    for (band, entries) in bands.iter().enumerate() {
        if band == 0 && blank_header {
            continue;
        }
        let maximum = entries
            .iter()
            .filter(|e| e.1.len() >= minimum)
            .map(|e| e.1.len())
            .max()
            .unwrap_or(0);
        let dense: Vec<_> = entries
            .iter()
            .filter(|e| e.1.len() >= minimum && e.1.len() == maximum)
            .collect();
        if dense.len() < 2 || dense.iter().skip(1).any(|e| e.1 != dense[0].1) {
            continue;
        }
        let original = entries
            .iter()
            .filter(|e| !e.1.is_empty())
            .all(|e| e.1 == dense[0].1);
        if !original {
            let mut separator = false;
            for track in &cx[1..cx.len() - 1] {
                if cov(
                    rules,
                    "vertical",
                    &track.aliases,
                    y[band],
                    y[band + 1],
                    snap,
                    work,
                )? >= 0.8
                {
                    separator = true;
                    break;
                }
            }
            if separator {
                continue;
            }
        }
        for p in dense.windows(2) {
            out.push(
                json!({"physical_row":band,"visual_rows":[p[0].0,p[1].0],"occupied_cols":p[0].1}),
            )
        }
    }
    Ok(out)
}
struct Builder<'a> {
    input: &'a Value,
    text: &'a Text,
    drawing: bool,
    rect: bool,
    supported: bool,
}
impl Builder<'_> {
    fn prep(
        &self,
        halo: f64,
        snap: f64,
        join: f64,
        cache: &mut Cache,
        work: &mut Work,
    ) -> Result<std::sync::Arc<Value>, Error> {
        let key = (self.drawing, self.rect, halo.to_bits());
        if let Some(p) = cache.get(&key) {
            return Ok(p.clone());
        }
        let p = prepare_with_work(
            &json!({"table_input":self.input,"snap_tolerance":snap,"join_gap":join,"evidence_halo":halo,"include_drawing":self.drawing,"include_rectangles":self.rect}),
            work,
        )?;
        let p = std::sync::Arc::new(p);
        cache.insert(key, p.clone());
        Ok(p)
    }
    fn tracks(
        &self,
        d: &mut Map<String, Value>,
        cache: &mut Cache,
        work: &mut Work,
    ) -> Result<Option<Value>, Error> {
        let text = self.text;
        let h = text.median_glyph_height;
        let snap = (0.08 * h).clamp(0.5, 2.5);
        let join = (0.40 * h).clamp(2., 8.);
        let configured = if self.drawing && !self.rect {
            (0.25 * h).clamp(1., 3.)
        } else {
            0.
        };
        let Some(b) = normalized(&self.input["table_bbox"])? else {
            return Ok(reject(d, "table_geometry"));
        };
        let a = angle(self.input)?;
        let (w, height) = if a == 90 || a == 270 {
            (b[3] - b[1], b[2] - b[0])
        } else {
            (b[2] - b[0], b[3] - b[1])
        };
        let exact = self.prep(0., snap, join, cache, work)?;
        let fragments = exact["fragments"].as_array().ok_or_else(invalid)?;
        // Python MAX_PRIMITIVES_PER_TABLE is an algorithm candidate gate,
        // not a document resource budget. It applies in production too.
        if fragments.is_empty() || fragments.len() > 5000 {
            return Ok(reject(d, "raw_fragments"));
        }
        let ex: Vec<f64> =
            serde::Deserialize::deserialize(&exact["inferred"][0]).map_err(|_| invalid())?;
        let ey: Vec<f64> =
            serde::Deserialize::deserialize(&exact["inferred"][1]).map_err(|_| invalid())?;
        let hint = self.drawing
            && !self.rect
            && ey.len() >= 3
            && ex.len() >= 2
            && w > 0.
            && ex
                .iter()
                .all(|t| t.abs().min((w - t).abs()) <= (3. * configured).max(2.));
        let halo = if configured > 0.
            && (hint
                || ey.len() == 2
                || ey.len() == 3
                    && ey
                        .windows(2)
                        .map(|p| p[1] - p[0])
                        .fold(f64::INFINITY, f64::min)
                        <= 0.75 * h)
        {
            configured
        } else {
            0.
        };
        let p = if halo > 0. {
            self.prep(halo, snap, join, cache, work)?
        } else {
            exact
        };
        let rules: Vec<Rule> =
            serde::Deserialize::deserialize(&p["rules"]).map_err(|_| invalid())?;
        let inferred = &p[if self.supported {
            "supported"
        } else {
            "inferred"
        }];
        let x: Vec<f64> = serde::Deserialize::deserialize(&inferred[0]).map_err(|_| invalid())?;
        let y: Vec<f64> = serde::Deserialize::deserialize(&inferred[1]).map_err(|_| invalid())?;
        d.insert(
            "raw_fragment_count".into(),
            json!(p["fragments"].as_array().ok_or_else(invalid)?.len()),
        );
        d.insert(
            "track_hypothesis".into(),
            json!(if self.supported { "supported" } else { "raw" }),
        );
        d.insert("evidence_halo".into(), json!(halo));
        d.insert("removed_horizontal_tracks".into(), inferred[2].clone());
        d.insert("inferred_tracks".into(), json!({"x":x.len(),"y":y.len()}));
        if self.rect && !self.drawing && !repeated(&rules, &x, &y, snap, work)? {
            return Ok(reject(d, "rect_lattice"));
        }
        let mut widths: Vec<_> = drawings(self.input)?
            .iter()
            .filter(|d| d.width > 0.)
            .map(|d| d.width)
            .collect();
        widths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if widths.is_empty() {
            0.
        } else {
            let m = widths.len() / 2;
            if widths.len() % 2 == 1 {
                widths[m]
            } else {
                (widths[m - 1] + widths[m]) / 2.
            }
        };
        let threshold = (0.75 * h).max(3. * median);
        let gx: Vec<_> = text
            .glyphs
            .iter()
            .map(|g| (g.bbox[0] + g.bbox[2]) / 2.)
            .collect();
        let gy: Vec<_> = text
            .glyphs
            .iter()
            .map(|g| (g.bbox[1] + g.bbox[3]) / 2.)
            .collect();
        let xc = canonicalize_with_work(
            &json!({"positions":x,"glyph_centers":gx,"threshold":threshold,"extent":w,"evidence_halo":halo,"minimum_track_count":3,"rules":rules,"orientation":"vertical","separator_extent":height,"snap_tolerance":snap}),
            work,
        )?;
        let yc = canonicalize_with_work(
            &json!({"positions":y,"glyph_centers":gy,"threshold":threshold,"extent":height,"evidence_halo":halo,"minimum_track_count":2,"rules":rules,"orientation":"horizontal","separator_extent":w,"snap_tolerance":snap,"preserve_double_boundary":true,"collapse_narrow_bands":ey.len()<=3,"collapse_leading_edge":true}),
            work,
        )?;
        if xc["alias_conflict"] == true
            || yc["alias_conflict"] == true
            || xc["unique"] != true
            || yc["unique"] != true
        {
            return Ok(reject(d, "canonical_alias"));
        }
        let cx: Vec<Track> =
            serde::Deserialize::deserialize(&xc["tracks"]).map_err(|_| invalid())?;
        let cy: Vec<Track> =
            serde::Deserialize::deserialize(&yc["tracks"]).map_err(|_| invalid())?;
        let x: Vec<_> = cx.iter().map(|t| t.coordinate).collect();
        let y: Vec<_> = cy.iter().map(|t| t.coordinate).collect();
        d.insert("canonical_tracks".into(), xc["tracks"].clone());
        d.insert("canonical_y_tracks".into(), yc["tracks"].clone());
        d.insert(
            "outer_track_collapses".into(),
            json!({"x":xc["outer_collapse_count"],"y":yc["outer_collapse_count"]}),
        );
        work.charge(x.len() * gx.len() + y.len() * gy.len())?;
        if x.windows(2)
            .any(|p| p[1] - p[0] <= threshold && !gx.iter().any(|g| p[0] < *g && *g < p[1]))
        {
            return Ok(reject(d, "remaining_narrow_track"));
        }
        for pair in cy.windows(2) {
            if pair[1].coordinate - pair[0].coordinate <= threshold
                && !gy
                    .iter()
                    .any(|g| pair[0].coordinate < *g && *g < pair[1].coordinate)
            {
                let (Some(left), Some(right)) = (x.first(), x.last()) else {
                    return Err(invalid());
                };
                if cov(
                    &rules,
                    "horizontal",
                    &pair[0].aliases,
                    *left,
                    *right,
                    snap,
                    work,
                )? >= 0.8
                    && cov(
                        &rules,
                        "horizontal",
                        &pair[1].aliases,
                        *left,
                        *right,
                        snap,
                        work,
                    )? >= 0.8
                {
                    return Ok(reject(d, "remaining_narrow_track"));
                }
            }
        }
        let line = self.drawing && !self.rect;
        let single_row = line && y.len() == 2 && x.len() >= 3;
        let single_col = line && x.len() == 2 && y.len() >= 3;
        if x.len() < (if single_col { 2 } else { 3 })
            || y.len() < (if single_row { 2 } else { 3 })
            || x.len() > 200
            || y.len() > 200
        {
            return Ok(reject(d, "track_count"));
        }
        let (rows, cols) = (y.len() - 1, x.len() - 1);
        d.insert("grid".into(), json!({"rows":rows,"cols":cols}));
        if rows * cols > rapiddoc_core::resources::limit(10000) {
            return Ok(reject(d, "atomic_cell_limit"));
        }
        let pairs = if line {
            dense(text, &x, &y, &rules, &cx, snap, work)?
        } else {
            Vec::new()
        };
        d.insert("physical_row_dense_baseline_pairs".into(), json!(pairs));
        if !pairs.is_empty() {
            return Ok(reject(d, "physical_row_undercount"));
        }
        if line {
            let tolerance = 0.5f64.max(0.08 * text.median_glyph_height);
            let mut uncovered = Vec::new();
            for row in &text.rows {
                let center = (row.bbox[1] + row.bbox[3]) / 2.;
                if y[0] - tolerance <= center && center <= y[y.len() - 1] + tolerance {
                    continue;
                }
                let mut occupied = std::collections::BTreeSet::new();
                for token in &row.tokens {
                    let center = (token.bbox[0] + token.bbox[2]) / 2.;
                    for (col, pair) in x.windows(2).enumerate() {
                        if pair[0] <= center && center <= pair[1] {
                            occupied.insert(col);
                        }
                    }
                }
                if occupied.len() >= 2 {
                    uncovered.push(row.row_index);
                }
            }
            if !uncovered.is_empty() {
                d.insert("uncovered_dense_outer_rows".into(), json!(uncovered));
                return Ok(reject(d, "physical_outer_row_undercount"));
            }
        }
        Ok(Some(
            json!({"snap_tolerance":snap,"local_width":w,"rules":rules,"canonical_x_tracks":cx,"canonical_y_tracks":cy,"x_tracks":x,"y_tracks":y,"narrow_empty_threshold":threshold,"is_line_grid":line,"is_single_row_shape":single_row,"is_single_column_shape":single_col,"rows":rows,"cols":cols}),
        ))
    }
}
fn without_decorations(input: &Value, text: &Text, work: &mut Work) -> Result<Value, Error> {
    let b = bbox(input)?;
    let a = angle(input)?;
    let (w, h) = if a == 90 || a == 270 {
        (b[3] - b[1], b[2] - b[0])
    } else {
        (b[2] - b[0], b[3] - b[1])
    };
    let tolerance = (0.08 * text.median_glyph_height).max(0.5);
    let ds = drawings(input)?;
    let vertical: Vec<_> = ds
        .iter()
        .enumerate()
        .filter_map(|(i, r)| {
            local(r.bbox, b, a, 0.)
                .filter(|b| b[3] - b[1] > b[2] - b[0])
                .map(|b| (i, b))
        })
        .collect();
    let mut removed = Vec::new();
    for (i, b) in &vertical {
        let c = (b[0] + b[2]) / 2.;
        let distance = c.min(w - c);
        if !(tolerance < distance && distance < (0.05 * w).min(2. * text.median_glyph_height)) {
            continue;
        }
        work.charge(vertical.len())?;
        let intervals: Vec<_> = vertical
            .iter()
            .filter(|(_, b)| ((b[0] + b[2]) / 2. - c).abs() <= tolerance)
            .map(|(_, b)| (b[1], b[3]))
            .collect();
        if coverage(intervals.clone(), 0., h)? >= 0.30 {
            continue;
        }
        work.charge(text.glyphs.len() * (intervals.len() + 1))?;
        let outside = text.glyphs.iter().any(|g| {
            if c < w / 2. {
                (g.bbox[0] + g.bbox[2]) / 2. < c
            } else {
                (g.bbox[0] + g.bbox[2]) / 2. > c
            }
        });
        let cuts = text.glyphs.iter().any(|g| {
            g.bbox[0] + tolerance < c
                && c < g.bbox[2] - tolerance
                && intervals
                    .iter()
                    .all(|(s, e)| g.bbox[3] < s - tolerance || g.bbox[1] > e + tolerance)
        });
        if !outside || cuts {
            removed.push(ds[*i].clone())
        }
    }
    let mut out = input.clone();
    if !removed.is_empty() {
        work.charge(ds.len() * removed.len())?;
        out["drawing_lines"] = json!(ds
            .into_iter()
            .filter(|d| !removed.contains(d))
            .collect::<Vec<_>>())
    }
    Ok(out)
}
fn preflight(v: &Value) -> Result<Text, Error> {
    validate_tree(v)?;
    let input = &v["table_input"];
    let b = bbox(input)?;
    for n in b {
        if !n.is_finite() || n.abs() > 1e9 {
            return Err(invalid());
        }
    }
    let lines = drawings(input)?;
    let rects = input
        .get("rectangles")
        .map_or(Some(0), |a| a.as_array().map(Vec::len))
        .ok_or_else(invalid)?;
    if lines.len() + rects > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    topology::validate_text(&v["text"])
}
pub fn inspect_tracks(v: &Value) -> Result<Value, Error> {
    let text = preflight(v)?;
    let mut d = Map::new();
    let mut cache = Cache::new();
    let mut work = Work(0);
    let b = Builder {
        input: &v["table_input"],
        text: &text,
        drawing: super::flag(v, "include_drawing", true)?,
        rect: super::flag(v, "include_rectangles", false)?,
        supported: super::flag(v, "prune_unsupported_horizontal", false)?,
    };
    let tracks = b.tracks(&mut d, &mut cache, &mut work)?;
    Ok(json!({"tracks":tracks,"diagnostics":d}))
}
fn hypothesis(
    b: Builder<'_>,
    text: &Value,
    d: &mut Map<String, Value>,
    cache: &mut Cache,
    work: &mut Work,
) -> Result<Value, Error> {
    let label = if b.drawing { "line_grid" } else { "rect_grid" };
    d.insert("evidence".into(), json!(label));
    let Some(tracks) = b.tracks(d, cache, work)? else {
        return Ok(Value::Null);
    };
    let result = topology::build_with_work(
        &json!({"tracks":tracks,"text":text,"evidence_label":label,"diagnostics":d}),
        work,
    )?;
    *d = result["diagnostics"]
        .as_object()
        .cloned()
        .ok_or_else(invalid)?;
    Ok(result["candidate"].clone())
}
pub fn generate(v: &Value) -> Result<Value, Error> {
    generate_with_work(v, &mut Work(0))
}
pub(in crate::processing) fn generate_with_work(
    v: &Value,
    work: &mut Work,
) -> Result<Value, Error> {
    let text = preflight(v)?;
    let mut cache = Cache::new();
    let input = &v["table_input"];
    let line_input = without_decorations(input, &text, work)?;
    let mut raw = Map::new();
    let mut line = hypothesis(
        Builder {
            input: &line_input,
            text: &text,
            drawing: true,
            rect: false,
            supported: false,
        },
        &v["text"],
        &mut raw,
        &mut cache,
        work,
    )?;
    let mut hypotheses = vec![Value::Object(raw.clone())];
    let mut selected = raw;
    let ambiguous = !line.is_null()
        && !line["issues"]
            .as_array()
            .ok_or_else(invalid)?
            .iter()
            .any(|i| i == "ambiguous_separator_ratio=0.0000");
    if line.is_null() || number(&line["score"])? < 0.95 || ambiguous {
        let mut supported = Map::new();
        let next = hypothesis(
            Builder {
                input: &line_input,
                text: &text,
                drawing: true,
                rect: false,
                supported: true,
            },
            &v["text"],
            &mut supported,
            &mut cache,
            work,
        )?;
        if supported
            .get("removed_horizontal_tracks")
            .and_then(Value::as_array)
            .is_some_and(|a| !a.is_empty())
        {
            hypotheses.push(json!(supported));
            selected = supported.clone()
        }
        if !next.is_null() && (line.is_null() || number(&next["score"])? >= 0.95) {
            line = next;
            selected = supported
        }
    }
    selected.insert("track_hypotheses".into(), json!(hypotheses));
    let mut diagnostics = vec![Value::Object(selected)];
    let mut candidates = Vec::new();
    if !line.is_null() {
        candidates.push(line)
    }
    let mut rect = Map::new();
    let candidate = hypothesis(
        Builder {
            input,
            text: &text,
            drawing: false,
            rect: true,
            supported: false,
        },
        &v["text"],
        &mut rect,
        &mut cache,
        work,
    )?;
    diagnostics.push(json!(rect));
    if !candidate.is_null() {
        candidates.push(candidate)
    }
    let html = candidates
        .iter()
        .map(crate::legacy_model::table_native_candidate::render)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({"candidates":candidates,"diagnostics":diagnostics,"html":html}))
}
