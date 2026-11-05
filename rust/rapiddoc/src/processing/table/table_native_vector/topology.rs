//! 旧_build_vector_topology与_materialize_vector_candidate；输入是已验证的规范轨道。
use super::{coverage, number, Rule, Track};
use crate::legacy_model::{table_native_candidate, table_native_text::Text};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::python_space;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;

#[derive(Deserialize)]
struct Tracks {
    snap_tolerance: f64,
    local_width: f64,
    rules: Vec<Rule>,
    canonical_x_tracks: Vec<Track>,
    canonical_y_tracks: Vec<Track>,
    x_tracks: Vec<f64>,
    y_tracks: Vec<f64>,
    narrow_empty_threshold: f64,
    is_line_grid: bool,
    is_single_row_shape: bool,
    is_single_column_shape: bool,
    rows: usize,
    cols: usize,
}
#[derive(Serialize)]
struct PhysicalRow {
    row: usize,
    top_coverage: f64,
    bottom_coverage: f64,
    left_coverage: f64,
    right_coverage: f64,
    height_ratio: f64,
    glyph_crossing: bool,
}
impl PhysicalRow {
    fn reliability(&self) -> f64 {
        if self.glyph_crossing {
            0.
        } else {
            self.top_coverage
                .min(self.bottom_coverage)
                .min(self.left_coverage)
                .min(self.right_coverage)
                .min(self.height_ratio)
        }
    }
    fn diagnostic(&self) -> Value {
        json!({"row":self.row,"reliability":self.reliability(),"verified":self.reliability()>=0.8,"top":self.top_coverage,"bottom":self.bottom_coverage,"left":self.left_coverage,"right":self.right_coverage,"height_ratio":self.height_ratio,"glyph_crossing":self.glyph_crossing})
    }
}
#[derive(Serialize)]
struct SingleRow {
    top_coverage: f64,
    bottom_coverage: f64,
    vertical_coverages: Vec<f64>,
    height_ratio: f64,
    glyph_crossing: bool,
}
impl SingleRow {
    fn reliability(&self) -> f64 {
        if self.glyph_crossing || self.vertical_coverages.is_empty() {
            0.
        } else {
            self.vertical_coverages.iter().copied().fold(
                self.top_coverage
                    .min(self.bottom_coverage)
                    .min(self.height_ratio),
                f64::min,
            )
        }
    }
    fn diagnostic(&self) -> Value {
        let r = self.reliability();
        json!({"reliability":r,"confidence":confidence(r),"verified":r>=0.8,"top":self.top_coverage,"bottom":self.bottom_coverage,"vertical":self.vertical_coverages,"height_ratio":self.height_ratio,"glyph_crossing":self.glyph_crossing})
    }
}
#[derive(Serialize)]
struct SingleColumn {
    horizontal_coverages: Vec<f64>,
    left_coverage: f64,
    right_coverage: f64,
    minimum_height_ratio: f64,
    glyph_crossing: bool,
}
impl SingleColumn {
    fn reliability(&self) -> f64 {
        if self.glyph_crossing || self.horizontal_coverages.is_empty() {
            0.
        } else {
            self.horizontal_coverages.iter().copied().fold(
                self.left_coverage
                    .min(self.right_coverage)
                    .min(self.minimum_height_ratio),
                f64::min,
            )
        }
    }
    fn diagnostic(&self) -> Value {
        let r = self.reliability();
        json!({"reliability":r,"confidence":confidence(r),"verified":r>=0.8,"horizontal":self.horizontal_coverages,"left":self.left_coverage,"right":self.right_coverage,"minimum_height_ratio":self.minimum_height_ratio,"glyph_crossing":self.glyph_crossing})
    }
}
fn confidence(r: f64) -> f64 {
    if r < 0.8 {
        0.
    } else {
        (0.95 + 0.05 * (r - 0.8) / (1. - 0.8)).min(1.)
    }
}
fn reject(mut d: Map<String, Value>, gate: &str, topology: Value) -> Value {
    d.insert("first_rejection_gate".into(), json!(gate));
    json!({"candidate":null,"html":null,"topology":topology,"diagnostics":d})
}
impl Tracks {
    fn validate(&self) -> Result<(), Error> {
        if self.rows == 0
            || self.cols == 0
            || self.rows > rapiddoc_core::resources::limit(199)
            || self.cols > rapiddoc_core::resources::limit(199)
            || self.rows * self.cols > rapiddoc_core::resources::limit(10000)
            || self.rules.len() > rapiddoc_core::resources::limit(5000)
        {
            return Err(limit());
        }
        if [
            self.snap_tolerance,
            self.local_width,
            self.narrow_empty_threshold,
        ]
        .iter()
        .any(|n| !n.is_finite() || *n < 0. || *n > 1e9)
        {
            return Err(invalid());
        }
        if self.is_single_row_shape && (self.rows != 1 || self.cols < 2)
            || self.is_single_column_shape && (self.cols != 1 || self.rows < 2)
        {
            return Err(invalid());
        }
        for (tracks, coords, n) in [
            (&self.canonical_x_tracks, &self.x_tracks, self.cols),
            (&self.canonical_y_tracks, &self.y_tracks, self.rows),
        ] {
            if tracks.len() != n + 1
                || coords.len() != n + 1
                || !coords.windows(2).all(|p| p[1] > p[0])
            {
                return Err(invalid());
            }
            for (t, c) in tracks.iter().zip(coords) {
                if t.coordinate != *c
                    || t.aliases.is_empty()
                    || t.aliases.len() > 200
                    || t.aliases
                        .iter()
                        .chain([&t.coordinate])
                        .any(|n| !n.is_finite() || n.abs() > 1e9)
                    || !t.aliases.windows(2).all(|p| p[1] > p[0])
                {
                    return Err(invalid());
                }
            }
        }
        if self.rules.iter().any(|r| {
            !["horizontal", "vertical"].contains(&r.orientation.as_str())
                || [r.coordinate, r.start, r.end]
                    .iter()
                    .any(|n| !n.is_finite() || n.abs() > 1e9)
        }) {
            return Err(invalid());
        }
        Ok(())
    }
    fn cov(
        &self,
        orientation: &str,
        aliases: &[f64],
        start: f64,
        end: f64,
        work: &mut Work,
    ) -> Result<f64, Error> {
        work.charge(self.rules.len() * aliases.len())?;
        coverage(
            self.rules
                .iter()
                .filter(|r| {
                    r.orientation == orientation
                        && aliases
                            .iter()
                            .any(|a| (r.coordinate - a).abs() <= self.snap_tolerance)
                })
                .map(|r| (r.start, r.end))
                .collect(),
            start,
            end,
        )
    }
    fn rows_evidence(&self, text: &Text, work: &mut Work) -> Result<Vec<PhysicalRow>, Error> {
        let (left, right) = (self.x_tracks[0], *self.x_tracks.last().unwrap());
        let mut out = Vec::new();
        for (row, pair) in self.canonical_y_tracks.windows(2).enumerate() {
            let (top, bottom) = (pair[0].coordinate, pair[1].coordinate);
            let a = self.cov("horizontal", &pair[0].aliases, left, right, work)?;
            let b = self.cov("horizontal", &pair[1].aliases, left, right, work)?;
            let mut l = self.cov(
                "vertical",
                &self.canonical_x_tracks[0].aliases,
                top,
                bottom,
                work,
            )?;
            let mut r = self.cov(
                "vertical",
                &self.canonical_x_tracks.last().unwrap().aliases,
                top,
                bottom,
                work,
            )?;
            if left <= 2. * self.snap_tolerance {
                l = l.max(a.min(b))
            }
            if self.local_width - right <= 2. * self.snap_tolerance {
                r = r.max(a.min(b))
            }
            work.charge(text.glyphs.len())?;
            let crossing = text.glyphs.iter().any(|g| {
                g.bbox[1] < bottom
                    && g.bbox[3] > top
                    && !(top <= (g.bbox[1] + g.bbox[3]) / 2.
                        && (g.bbox[1] + g.bbox[3]) / 2. <= bottom)
            });
            out.push(PhysicalRow {
                row,
                top_coverage: a,
                bottom_coverage: b,
                left_coverage: l,
                right_coverage: r,
                height_ratio: ((bottom - top) / self.narrow_empty_threshold.max(0.1)).min(1.),
                glyph_crossing: crossing,
            });
        }
        Ok(out)
    }
    fn single_row(&self, text: &Text, work: &mut Work) -> Result<SingleRow, Error> {
        let (top, bottom) = (self.y_tracks[0], self.y_tracks[1]);
        let (left, right) = (self.x_tracks[0], *self.x_tracks.last().unwrap());
        let top_coverage = self.cov(
            "horizontal",
            &self.canonical_y_tracks[0].aliases,
            left,
            right,
            work,
        )?;
        let bottom_coverage = self.cov(
            "horizontal",
            &self.canonical_y_tracks[1].aliases,
            left,
            right,
            work,
        )?;
        let vertical_coverages = self
            .canonical_x_tracks
            .iter()
            .map(|t| self.cov("vertical", &t.aliases, top, bottom, work))
            .collect::<Result<Vec<_>, _>>()?;
        work.charge(text.glyphs.len() * self.cols)?;
        let glyph_crossing = text.glyphs.iter().any(|g| {
            self.x_tracks[1..self.cols]
                .iter()
                .any(|x| g.bbox[0] < *x && *x < g.bbox[2])
        });
        Ok(SingleRow {
            top_coverage,
            bottom_coverage,
            vertical_coverages,
            height_ratio: ((bottom - top) / self.narrow_empty_threshold.max(0.1)).min(1.),
            glyph_crossing,
        })
    }
    fn single_column(&self, text: &Text, work: &mut Work) -> Result<SingleColumn, Error> {
        let (left, right) = (self.x_tracks[0], self.x_tracks[1]);
        let (top, bottom) = (self.y_tracks[0], *self.y_tracks.last().unwrap());
        let horizontal_coverages = self
            .canonical_y_tracks
            .iter()
            .map(|t| self.cov("horizontal", &t.aliases, left, right, work))
            .collect::<Result<Vec<_>, _>>()?;
        let left_coverage = self.cov(
            "vertical",
            &self.canonical_x_tracks[0].aliases,
            top,
            bottom,
            work,
        )?;
        let right_coverage = self.cov(
            "vertical",
            &self.canonical_x_tracks[1].aliases,
            top,
            bottom,
            work,
        )?;
        let minimum_height_ratio = self
            .y_tracks
            .windows(2)
            .map(|p| ((p[1] - p[0]) / self.narrow_empty_threshold.max(0.1)).min(1.))
            .fold(f64::INFINITY, f64::min);
        work.charge(text.glyphs.len() * self.rows)?;
        let glyph_crossing = text.glyphs.iter().any(|g| {
            self.y_tracks[1..self.rows]
                .iter()
                .any(|y| g.bbox[1] < *y && *y < g.bbox[3])
        });
        Ok(SingleColumn {
            horizontal_coverages,
            left_coverage,
            right_coverage,
            minimum_height_ratio,
            glyph_crossing,
        })
    }
}
fn occupied(text: &Text, y: &[f64], work: &mut Work) -> Result<BTreeSet<usize>, Error> {
    work.charge(text.rows.len() * y.len())?;
    Ok(text
        .rows
        .iter()
        .filter_map(|r| {
            let cy = (r.bbox[1] + r.bbox[3]) / 2.;
            y.windows(2).position(|p| p[0] <= cy && cy <= p[1])
        })
        .collect())
}
fn stability(
    text: &Text,
    t: &Tracks,
    physical: &BTreeSet<usize>,
    work: &mut Work,
) -> Result<(f64, f64), Error> {
    let rows = occupied(text, &t.y_tracks, work)?.union(physical).count();
    let mut cols = BTreeSet::new();
    for row in &text.rows {
        for token in &row.tokens {
            work.charge(t.x_tracks.len())?;
            let center = (token.bbox[0] + token.bbox[2]) / 2.;
            if let Some(col) = t
                .x_tracks
                .windows(2)
                .position(|p| p[0] <= center && center <= p[1])
            {
                cols.insert(col);
            }
        }
    }
    let rd = t.rows.min(text.rows.len().max(1));
    let cd = t
        .cols
        .min(text.rows.iter().map(|r| r.tokens.len()).max().unwrap_or(1));
    // 旧实现对非空行但全无token会ZeroDivisionError；显式无效输入，不伪造稳定度。
    if cd == 0 {
        return Err(invalid());
    }
    Ok((
        (rows as f64 / rd as f64).min(1.),
        (cols.len() as f64 / cd as f64).min(1.),
    ))
}
pub(super) fn validate_text(v: &Value) -> Result<Text, Error> {
    let text = table_native_candidate::text(v)?;
    for row in &text.rows {
        if row.bbox.iter().any(|n| !n.is_finite() || n.abs() > 1e9)
            || row
                .tokens
                .iter()
                .any(|t| t.bbox.iter().any(|n| !n.is_finite() || n.abs() > 1e9))
        {
            return Err(invalid());
        }
    }
    Ok(text)
}
pub fn build(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    build_with_work(v, &mut Work(0))
}
pub(super) fn build_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    let t: Tracks = serde::Deserialize::deserialize(&v["tracks"]).map_err(|_| invalid())?;
    t.validate()?;
    let text = validate_text(&v["text"])?;
    let label = v["evidence_label"]
        .as_str()
        .filter(|s| ["line_grid", "rect_grid"].contains(s))
        .ok_or_else(invalid)?;
    let mut d = match v.get("diagnostics") {
        None => Map::new(),
        Some(d) => d.as_object().cloned().ok_or_else(invalid)?,
    };
    let mut edges = Vec::new();
    let mut decisions = Vec::new();
    let mut ambiguous = 0;
    let (mut xr, mut yr) = (0usize, 0usize);
    let mut affected = BTreeSet::new();
    for row in 0..t.rows {
        for boundary in 1..t.cols {
            let track = &t.canonical_x_tracks[boundary];
            let strict = t.cov(
                "vertical",
                &[track.coordinate],
                t.y_tracks[row],
                t.y_tracks[row + 1],
                work,
            )?;
            let c = t.cov(
                "vertical",
                &track.aliases,
                t.y_tracks[row],
                t.y_tracks[row + 1],
                work,
            )?;
            if track.aliases.len() > 1 && strict <= 1. - 0.8 && c >= 0.8 {
                xr += 1;
                affected.insert(row);
            }
            decisions.push(c.max(1. - c));
            if c <= 1. - 0.8 {
                edges.push((row * t.cols + boundary - 1, row * t.cols + boundary))
            } else if c < 0.8 {
                ambiguous += 1
            }
        }
    }
    for boundary in 1..t.rows {
        for col in 0..t.cols {
            let track = &t.canonical_y_tracks[boundary];
            let strict = t.cov(
                "horizontal",
                &[track.coordinate],
                t.x_tracks[col],
                t.x_tracks[col + 1],
                work,
            )?;
            let c = t.cov(
                "horizontal",
                &track.aliases,
                t.x_tracks[col],
                t.x_tracks[col + 1],
                work,
            )?;
            if track.aliases.len() > 1 && strict <= 1. - 0.8 && c >= 0.8 {
                yr += 1;
                affected.insert(boundary - 1);
                affected.insert(boundary);
            }
            decisions.push(c.max(1. - c));
            if c <= 1. - 0.8 {
                edges.push(((boundary - 1) * t.cols + col, boundary * t.cols + col))
            } else if c < 0.8 {
                ambiguous += 1
            }
        }
    }
    let ratio = if decisions.is_empty() {
        0.
    } else {
        ambiguous as f64 / decisions.len() as f64
    };
    d.insert("grid".into(), json!({"rows":t.rows,"cols":t.cols}));
    d.insert("ambiguous_separator_ratio".into(), json!(ratio));
    d.insert("alias_separator_recoveries".into(), json!(xr));
    d.insert("y_alias_separator_recoveries".into(), json!(yr));
    d.insert("alias_affected_rows".into(), json!(affected));
    if ratio > 0.05 {
        return Ok(reject(d, "ambiguous_separator", Value::Null));
    }
    let single_row = if t.is_single_row_shape {
        let e = t.single_row(&text, work)?;
        d.insert("single_row_evidence".into(), e.diagnostic());
        if e.reliability() < 0.8 {
            return Ok(reject(d, "single_row_physical_evidence", Value::Null));
        }
        Some(e)
    } else {
        None
    };
    let single_column = if t.is_single_column_shape {
        let e = t.single_column(&text, work)?;
        d.insert("single_column_evidence".into(), e.diagnostic());
        if e.reliability() < 0.8 {
            return Ok(reject(d, "single_column_physical_evidence", Value::Null));
        }
        Some(e)
    } else {
        None
    };
    let Some(regions) = rapiddoc_native::table_component_specs(t.rows, t.cols, edges)? else {
        return Ok(reject(d, "nonrectangular_topology", Value::Null));
    };
    work.charge(regions.len() * (t.rows + t.cols))?;
    let max_row = (0..t.rows)
        .map(|row| {
            regions
                .iter()
                .filter(|s| s.row <= row && row < s.row + s.row_span)
                .count()
        })
        .max()
        .unwrap();
    let max_col = (0..t.cols)
        .map(|col| {
            regions
                .iter()
                .filter(|s| s.column <= col && col < s.column + s.column_span)
                .count()
        })
        .max()
        .unwrap();
    if max_row < 2 && !t.is_single_column_shape || max_col < 2 && !t.is_single_row_shape {
        return Ok(reject(d, "degenerate_grid", Value::Null));
    }
    let specs:Vec<_>=regions.iter().map(|s|json!({"row":s.row,"col":s.column,"rowspan":s.row_span,"colspan":s.column_span,"bbox":[t.x_tracks[s.column],t.y_tracks[s.row],t.x_tracks[s.column+s.column_span],t.y_tracks[s.row+s.row_span]]})).collect();
    let topology = json!({"specs":specs,"separator_decisions":decisions,"ambiguous_ratio":ratio,"alias_separator_recoveries":xr,"y_alias_separator_recoveries":yr,"alias_affected_rows":affected,"single_row_evidence":single_row,"single_column_evidence":single_column});
    let mut decisive = if decisions.is_empty() {
        1.
    } else {
        rapiddoc_native::table_decisiveness_mean(&decisions)?
    };
    let row_conf = single_row
        .as_ref()
        .map_or(1., |e| confidence(e.reliability()));
    let col_conf = single_column
        .as_ref()
        .map_or(1., |e| confidence(e.reliability()));
    if single_row.is_some() {
        decisive = decisive.max(row_conf)
    }
    if single_column.is_some() {
        decisive = decisive.max(col_conf)
    }
    let structure = decisive
        .min((t.rules.len() as f64 / (t.rows + t.cols).max(1) as f64).min(1.))
        .min(1. - ratio)
        .min(row_conf)
        .min(col_conf);
    let mut verified = BTreeSet::new();
    if t.is_line_grid {
        let evidence = t.rows_evidence(&text, work)?;
        verified = evidence
            .iter()
            .filter(|e| e.reliability() >= 0.8)
            .map(|e| e.row)
            .collect();
        d.insert(
            "physical_rows".into(),
            json!(evidence
                .iter()
                .map(PhysicalRow::diagnostic)
                .collect::<Vec<_>>()),
        );
    }
    let (row_stability, column_stability) = stability(&text, &t, &verified, work)?;
    let collapsed_x = t
        .canonical_x_tracks
        .iter()
        .filter(|t| t.aliases.len() > 1)
        .count();
    let collapsed_y = t
        .canonical_y_tracks
        .iter()
        .filter(|t| t.aliases.len() > 1)
        .count();
    let alias_span = t
        .canonical_x_tracks
        .iter()
        .chain(&t.canonical_y_tracks)
        .filter(|t| t.aliases.len() > 1)
        .map(|t| t.aliases.last().unwrap() - t.aliases[0])
        .fold(0f64, f64::max);
    let occupied = occupied(&text, &t.y_tracks, work)?;
    let blanks = (0..t.rows)
        .filter(|r| !occupied.contains(r) && verified.contains(r))
        .map(|r| r.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let fmt = |r: Option<f64>| r.map_or("n/a".into(), |r| format!("{r:.4}"));
    let issues = vec![
        format!("evidence={label}"),
        format!("ambiguous_separator_ratio={ratio:.4}"),
        format!("collapsed_x_tracks={collapsed_x}"),
        format!("collapsed_y_tracks={collapsed_y}"),
        format!("alias_max_span={alias_span:.4}"),
        format!("alias_separator_recoveries={xr}"),
        format!("y_alias_separator_recoveries={yr}"),
        format!("single_row_line_grid={}", t.is_single_row_shape),
        format!("single_column_line_grid={}", t.is_single_column_shape),
        format!(
            "single_row_reliability={}",
            fmt(single_row.as_ref().map(SingleRow::reliability))
        ),
        format!(
            "single_row_confidence={}",
            fmt(single_row.as_ref().map(|_| row_conf))
        ),
        format!(
            "single_column_reliability={}",
            fmt(single_column.as_ref().map(SingleColumn::reliability))
        ),
        format!(
            "single_column_confidence={}",
            fmt(single_column.as_ref().map(|_| col_conf))
        ),
        format!("physical_blank_rows={blanks}"),
    ];
    // 两个旧阶段共用工作预算的保守上界；下游实际落格/分组仍独立校验。
    work.charge(text.glyphs.len() * specs.len() * 3 + text.glyphs.len() * text.glyphs.len() * 2)?;
    let result = table_native_candidate::build(
        &json!({"source":"vector_grid","rows":t.rows,"cols":t.cols,"specs":specs,"text":v["text"],"structure_support":structure,"row_stability":row_stability,"column_stability":column_stability,"issues":issues,"allow_single_row":t.is_single_row_shape,"allow_single_column":t.is_single_column_shape,"use_grid_index":true,"diagnostics":d}),
    )?;
    d = result["diagnostics"]
        .as_object()
        .cloned()
        .ok_or_else(invalid)?;
    let candidate = &result["candidate"];
    if candidate.is_null() {
        let gate = d
            .get("candidate_rejection_gate")
            .and_then(Value::as_str)
            .unwrap_or("candidate_hard_gate")
            .to_owned();
        return Ok(reject(d, &gate, topology));
    }
    if t.is_single_row_shape
        && (number(&candidate["text_capture"])? < 1.
            || number(&candidate["order_consistency"])? < 1.)
    {
        return Ok(reject(d, "single_row_text_integrity", topology));
    }
    if t.is_single_column_shape
        && (number(&candidate["text_capture"])? < 1.
            || number(&candidate["order_consistency"])? < 1.)
    {
        return Ok(reject(d, "single_column_text_integrity", topology));
    }
    let cells = candidate["cells"].as_array().unwrap();
    work.charge(t.rows * cells.len())?;
    let empty: BTreeSet<_> = (0..t.rows)
        .filter(|row| {
            !cells.iter().any(|c| {
                let r = c["row"].as_u64().unwrap() as usize;
                let span = c["rowspan"].as_u64().unwrap() as usize;
                r <= *row
                    && *row < r + span
                    && !c["content"]
                        .as_str()
                        .unwrap()
                        .trim_matches(python_space)
                        .is_empty()
            })
        })
        .collect();
    if !empty.is_empty()
        && text.rows.len() >= 2
        && (!t.is_line_grid || !empty.is_disjoint(&affected) || !empty.is_subset(&verified))
    {
        d.insert("empty_rows".into(), json!(empty));
        return Ok(reject(d, "empty_row", topology));
    }
    d.insert("first_rejection_gate".into(), Value::Null);
    d.insert("score".into(), candidate["score"].clone());
    d.insert("empty_rows".into(), json!(empty));
    Ok(json!({"candidate":candidate,"html":result["html"],"topology":topology,"diagnostics":d}))
}
