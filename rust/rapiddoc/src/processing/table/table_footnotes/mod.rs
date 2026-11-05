//! 旧完整自动表注收集；显式表注和表内辅助标记都从当前输入判定。
use super::{
    path_graphics::{array, bbox, integer, num, Box4},
    rule_annotation_preparation as prep,
};
use super::table_members::Lookup;
use crate::legacy_page::{eq, invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{python_printable, text_boundary::python_space};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, OnceLock},
};
use unicode_normalization::UnicodeNormalization;
fn optional<'a>(v: &'a Value, k: &str, max: usize) -> Result<&'a [Value], Error> {
    v.get(k)
        .filter(|v| !v.is_null())
        .map_or(Ok(&[]), |v| array(v, max))
}
fn normalize(s: &str) -> String {
    super::table_text_styles::preparation::fold(&s.nfkc().collect::<String>())
}
fn category(c: char) -> bool {
    static R: OnceLock<Vec<(u32, u32)>> = OnceLock::new();
    let r = R.get_or_init(|| serde_json::from_str(include_str!("lns.json")).unwrap());
    let n = c as u32;
    let i = r.partition_point(|(a, _)| *a <= n);
    i > 0 && n <= r[i - 1].1
}
fn marker(text: &str) -> Option<String> {
    let c: Vec<_> = text.chars().collect();
    let mut first = 0;
    while c.get(first).is_some_and(|c| python_space(*c)) {
        first += 1;
    }
    let prefix = c.get(first).is_some_and(|c| "([{（［【".contains(*c));
    for start in if prefix {
        vec![first + 1, first]
    } else {
        vec![first]
    } {
        for len in (1..=3).rev() {
            if start + len > c.len() {
                continue;
            }
            let raw = &c[start..start + len];
            if raw
                .iter()
                .any(|c| python_space(*c) || ")]｝}）］】.:：、".contains(*c))
            {
                continue;
            }
            let end = start + len;
            let closing = c.get(end).is_some_and(|c| ")]}）］】".contains(*c));
            for mut at in if closing {
                vec![end + 1, end]
            } else {
                vec![end]
            } {
                while c.get(at).is_some_and(|c| ".:：、)]".contains(*c)) {
                    at += 1;
                }
                let space = at;
                while c.get(at).is_some_and(|c| python_space(*c)) {
                    at += 1;
                }
                if at == space || at == c.len() {
                    continue;
                }
                // Python '.' permits a final LF via '$', but no embedded LF.
                if c[at..]
                    .iter()
                    .enumerate()
                    .any(|(i, ch)| *ch == '\n' && at + i + 1 < c.len())
                {
                    continue;
                }
                let value = normalize(&raw.iter().collect::<String>());
                if !value.is_empty() && value.chars().all(category) {
                    return Some(value);
                }
            }
        }
    }
    None
}
fn median(mut values: Vec<f64>, work: &mut Work) -> Result<f64, Error> {
    if values.is_empty() {
        return Err(invalid());
    }
    work.charge(values.len() * 16 + 1)?;
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    Ok(if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.
    })
}
fn rotate(b: &Box4, p: [f64; 2], angle: i64) -> [f64; 4] {
    let [a, c, d, e] = b.n;
    match angle {
        270 => [p[1] - e, a, p[1] - c, d],
        90 => [c, p[0] - d, e, p[0] - a],
        180 => [p[0] - d, p[1] - e, p[0] - a, p[1] - c],
        _ => b.n,
    }
}
fn height(l: &Value, b: [f64; 4]) -> Result<f64, Error> {
    let repaired = l
        .get("style_scale_repaired")
        .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))?;
    let em = l.get("em_height").map_or(Ok(0.), num)?;
    let effective = l.get("effective_height").map_or(Ok(0.), num)?;
    Ok(if repaired && em > 0. {
        em
    } else if effective != 0. {
        effective
    } else {
        b[3] - b[1]
    }
    .max(0.1))
}
fn compact(text: &str, m: &str, work: &mut Work) -> Result<bool, Error> {
    work.charge(text.len() * 4 + 1)?;
    let s = normalize(text);
    if s.chars().filter(|c| !python_space(*c)).count() > 12 {
        return Ok(false);
    }
    let tokens: Vec<_> = s
        .split(|c| !category(c))
        .filter(|s| !s.is_empty())
        .collect();
    Ok(tokens.len() <= 4 && tokens.contains(&m))
}
fn superscript(
    l: &Value,
    m: &str,
    p: [f64; 2],
    angle: i64,
    work: &mut Work,
) -> Result<bool, Error> {
    let glyphs = marker_glyphs(l, p, angle, work)?;
    superscript_glyphs(&glyphs, m, work)
}
fn marker_glyphs(
    l: &Value,
    p: [f64; 2],
    angle: i64,
    work: &mut Work,
) -> Result<Vec<(String, [f64; 4])>, Error> {
    let mut glyphs = Vec::new();
    for c in optional(l, "chars", 20000)? {
        let s = c
            .get("char")
            .filter(|v| !v.is_null())
            .map_or(Ok(""), |v| v.as_str().ok_or_else(invalid))?;
        work.charge(s.len() * 4 + 1)?;
        if s.is_empty() || !s.chars().all(python_printable) || s.chars().all(python_space) {
            continue;
        }
        let Some(v) = c.get("bbox").filter(|v| !v.is_null()) else {
            continue;
        };
        let v = v.get("bbox").unwrap_or(v);
        let a = bbox(v)?;
        let mut b = a.clone();
        b.n = [
            a.n[0].min(a.n[2]),
            a.n[1].min(a.n[3]),
            a.n[0].max(a.n[2]),
            a.n[1].max(a.n[3]),
        ];
        if b.n[0] >= b.n[2] || b.n[1] >= b.n[3] {
            continue;
        }
        glyphs.push((normalize(s), rotate(&b, p, angle)));
    }
    Ok(glyphs)
}
fn superscript_glyphs(
    glyphs: &[(String, [f64; 4])],
    m: &str,
    work: &mut Work,
) -> Result<bool, Error> {
    if glyphs.len() < 2 {
        return Ok(false);
    }
    for start in 0..glyphs.len() {
        let mut combined = String::new();
        for end in start..glyphs.len() {
            work.charge(glyphs[end].0.len() + 1)?;
            combined.push_str(&glyphs[end].0);
            if !m.starts_with(&combined) {
                break;
            }
            if combined != m {
                continue;
            }
            work.charge(glyphs.len() + 1)?;
            let ordinary: Vec<_> = glyphs
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < start || *i > end)
                .map(|(_, (_, b))| *b)
                .collect();
            if ordinary.is_empty() {
                continue;
            }
            let normal = median(ordinary.iter().map(|b| b[3] - b[1]).collect(), work)?;
            if normal <= 0. {
                continue;
            }
            let base: Vec<_> = ordinary
                .iter()
                .filter(|b| b[3] - b[1] >= 0.9 * normal)
                .map(|b| (b[1] + b[3]) / 2.)
                .collect();
            let mh = median(
                glyphs[start..=end]
                    .iter()
                    .map(|(_, b)| b[3] - b[1])
                    .collect(),
                work,
            )?;
            let mc = median(
                glyphs[start..=end]
                    .iter()
                    .map(|(_, b)| (b[1] + b[3]) / 2.)
                    .collect(),
                work,
            )?;
            let nc = median(base, work)?;
            if mh <= 0.85 * normal && nc - mc >= 0.12 * normal {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
struct Prepared {
    row: prep::Row,
    ids: BTreeSet<i64>,
    height: f64,
    fonts: Vec<Value>,
    explicit: bool,
    marker: Option<String>,
}
struct PreparedRows {
    rows: Vec<Prepared>,
    geometry: Option<rapiddoc_native::TableNoteRowGeometry>,
    possible_notes: bool,
}
impl PreparedRows {
    fn new(rows: Vec<Prepared>) -> Self {
        let geometry = rapiddoc_native::TableNoteRowGeometry::new(
            rows.iter().map(|row| row.row.b.n).collect(),
        );
        let possible_notes = rows.iter().any(|row| row.explicit || row.marker.is_some());
        Self {
            rows,
            geometry,
            possible_notes,
        }
    }
    fn first_after(&self, bottom: f64) -> usize {
        if !bottom.is_finite() {
            return 0;
        }
        self.geometry
            .as_ref()
            .map_or(0, |geometry| geometry.first_after(bottom))
    }
}
pub fn collect(v: &Value) -> Result<Value, Error> {
    collect_with_work(v, &mut Work(0))
}
pub(crate) fn collect_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    collect_with_inputs(v, None, None, work)
}
pub(super) fn collect_with_inputs(
    v: &Value,
    rows_override: Option<&[prep::Row]>,
    lines_override: Option<&[&Value]>,
    work: &mut Work,
) -> Result<Value, Error> {
    collect_cached(v, rows_override, lines_override, None, work)
}
#[derive(Default)]
struct MarkerState {
    hits: Vec<i64>,
}
#[derive(Default)]
pub(super) struct FootnoteCache {
    rows: std::collections::HashMap<(u64, u64, u64, i64), PreparedRows>,
    markers: std::collections::HashMap<String, MarkerState>,
    body_metrics: Option<BodyMetrics>,
    marker_lines: Option<Vec<MarkerLine>>,
    geometry: AnnotationGeometryCache,
    timings: crate::processing::NativeStageTotals,
}
#[derive(Default)]
struct AnnotationGeometryCache {
    entries: std::collections::HashMap<(Vec<usize>, Vec<i64>), Arc<Value>>,
    weight: usize,
}
struct MarkerLine {
    source: i64,
    glyphs: Vec<(String, [f64; 4])>,
    compact_tokens: Vec<String>,
}
impl MarkerLine {
    fn prepare(line: &Value, page: [f64; 2], angle: i64, work: &mut Work) -> Result<Self, Error> {
        let glyphs = marker_glyphs(line, page, angle, work)?;
        let text = normalize(line["text"].as_str().ok_or_else(invalid)?);
        let mut compact_tokens = Vec::new();
        if text.chars().filter(|c| !python_space(*c)).count() <= 12 {
            compact_tokens = text
                .split(|c| !category(c))
                .filter(|s| !s.is_empty())
                .map(str::to_owned)
                .collect();
            if compact_tokens.len() > 4 {
                compact_tokens.clear();
            }
        }
        Ok(Self {
            source: integer(line, "source_index", 0)?,
            glyphs,
            compact_tokens,
        })
    }
    fn matches(&self, marker: &str, work: &mut Work) -> Result<bool, Error> {
        Ok(superscript_glyphs(&self.glyphs, marker, work)?
            || self.compact_tokens.iter().any(|token| token == marker))
    }
}
struct BodyMetrics {
    items: Vec<(i64, f64, f64)>,
    index: rapiddoc_native::TableNoteHeightIndex,
}
impl BodyMetrics {
    fn new(mut items: Vec<(i64, f64, f64)>) -> Self {
        let index = rapiddoc_native::TableNoteHeightIndex::new(&items);
        items.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        Self { items, index }
    }
    fn height(
        &self,
        top: f64,
        bottom: f64,
        core: &dyn Lookup,
        fallback: f64,
        work: &mut Work,
    ) -> Result<f64, Error> {
        let left = self.items.partition_point(|item| item.1 < top);
        let right = self.items.partition_point(|item| item.1 <= bottom);
        let outside = || self.items[..left].iter().chain(&self.items[right..]);
        // The index excludes the closed vertical zone. It is exact whenever
        // none of the core sources also occurs outside that zone (including
        // duplicate source IDs). Otherwise keep the original filtered median.
        if !outside().any(|item| core.contains(&item.0)) {
            if let Some(height) = self.index.height(top, bottom, fallback) {
                return Ok(height);
            }
        }
        let mut heights = outside()
            .filter(|item| !core.contains(&item.0))
            .map(|item| item.2)
            .collect::<Vec<_>>();
        if heights.len() < 4 {
            return Ok(fallback);
        }
        heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        median(
            heights[heights.len() - heights.len().div_ceil(4)..].to_vec(),
            work,
        )
    }
}
// Cache lifetime is one immutable page/angle context, never shared between parses.
pub(super) fn collect_cached(
    v: &Value,
    rows_override: Option<&[prep::Row]>,
    lines_override: Option<&[&Value]>,
    cache: Option<&mut FootnoteCache>,
    work: &mut Work,
) -> Result<Value, Error> {
    collect_cached_core(v, rows_override, lines_override, cache, None, work)
}
pub(super) fn collect_cached_core(
    v: &Value,
    rows_override: Option<&[prep::Row]>,
    lines_override: Option<&[&Value]>,
    cache: Option<&mut FootnoteCache>,
    core_override: Option<&dyn Lookup>,
    work: &mut Work,
) -> Result<Value, Error> {
    collect_output(
        v,
        rows_override,
        lines_override,
        cache,
        core_override,
        false,
        work,
    )
    .map(|value| value.as_ref().clone())
}
pub(super) fn collect_core_geometry(
    v: &Value,
    rows: &[prep::Row],
    lines: &[&Value],
    cache: &mut FootnoteCache,
    core: &dyn Lookup,
    work: &mut Work,
) -> Result<Arc<Value>, Error> {
    collect_output(
        v,
        Some(rows),
        Some(lines),
        Some(cache),
        Some(core),
        true,
        work,
    )
}
fn collect_output(
    v: &Value,
    rows_override: Option<&[prep::Row]>,
    lines_override: Option<&[&Value]>,
    cache: Option<&mut FootnoteCache>,
    core_override: Option<&dyn Lookup>,
    geometry_only: bool,
    work: &mut Work,
) -> Result<Arc<Value>, Error> {
    validate_tree(v)?;
    let em = num(&v["median_height"])?;
    if em <= 0. {
        return Err(invalid());
    }
    let rule = bbox(&v["rule_bbox"])?;
    let page = array(&v["page_size"], 2)?;
    if page.len() != 2 {
        return Err(invalid());
    }
    let p = [num(&page[0])?, num(&page[1])?];
    let angle = integer(v, "angle", 0)?;
    let owned_core;
    let core: &dyn Lookup = if let Some(core) = core_override {
        core
    } else {
        let raw_ids = array(&v["core_line_indices"], 20000)?;
        work.charge(raw_ids.len() * 16 + 1)?;
        owned_core = raw_ids
            .iter()
            .map(|v| v.as_i64().ok_or_else(invalid))
            .collect::<Result<BTreeSet<_>, _>>()?;
        &owned_core
    };
    let owned_lines;
    let lines = if let Some(lines) = lines_override {
        lines
    } else {
        owned_lines = array(&v["lines"], 4096)?.iter().collect::<Vec<_>>();
        &owned_lines
    };
    let (
        row_cache,
        mut marker_cache,
        mut metric_cache,
        mut marker_lines,
        mut geometry_cache,
        mut timings,
    ) = match cache {
        Some(cache) => {
            cache.timings.begin();
            (
                Some(&mut cache.rows),
                Some(&mut cache.markers),
                Some(&mut cache.body_metrics),
                Some(&mut cache.marker_lines),
                Some(&mut cache.geometry),
                Some(&mut cache.timings),
            )
        }
        None => (None, None, None, None, None, None),
    };
    let owned_prepared;
    let prepared = if let Some(cache) = row_cache {
        let key = (
            rule.n[0].to_bits(),
            rule.n[2].to_bits(),
            em.to_bits(),
            angle,
        );
        match cache.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => entry.insert(prepare_rows(
                v,
                rows_override,
                lines,
                &rule,
                em,
                p,
                angle,
                work,
            )?),
        }
    } else {
        owned_prepared = prepare_rows(v, rows_override, lines, &rule, em, p, angle, work)?;
        &owned_prepared
    };
    if let Some(timings) = timings.as_deref_mut() {
        timings.mark("note_corridor_preparation");
    }
    // Python prepares the same corridor once and skips all per-candidate note
    // work when neither an explicit note nor an auxiliary marker exists.
    if crate::legacy_page::budget::current().host
        && crate::legacy_page::budget::current().work == usize::MAX
        && !prepared.possible_notes
    {
        return Ok(Arc::new(json!({"footnote_rows":[],"annotation":null})));
    }
    let mut output = Vec::new();
    let mut bottom = rule.n[3];
    let mut added = BTreeSet::new();
    let mut started = false;
    let (mut note_left, mut note_height) = (0., 0.);
    let mut note_fonts: Vec<Value> = Vec::new();
    let mut body_height = None;
    // Skip only the prefix the original loop would immediately continue over.
    // The upstream geometry index preserves order even for overlapping rows.
    let start = if crate::legacy_page::budget::current().host
        && crate::legacy_page::budget::current().work == usize::MAX
    {
        prepared.first_after(bottom)
    } else {
        0
    };
    for r in &prepared.rows[start..] {
        work.charge(r.ids.len() * 16 + 1)?;
        if r.row.b.n[3] <= bottom {
            continue;
        }
        if r.ids
            .iter()
            .all(|id| core.contains(id) || added.contains(id))
        {
            bottom = bottom.max(r.row.b.n[3]);
            continue;
        }
        let gap = (r.row.b.n[1] - bottom).max(0.);
        if started && r.row.b.n[3] - rule.n[3] > 10. * em {
            break;
        }
        let mut auxiliary = false;
        if let Some(m) = &r.marker {
            if let Some(cache) = marker_cache.as_deref_mut() {
                if !cache.contains_key(m) {
                    let mut hits = Vec::new();
                    let prepared_lines = marker_lines.as_deref_mut().unwrap();
                    if prepared_lines.is_none() {
                        *prepared_lines = Some(
                            lines
                                .iter()
                                .map(|line| MarkerLine::prepare(line, p, angle, work))
                                .collect::<Result<Vec<_>, _>>()?,
                        );
                    }
                    for line in prepared_lines.as_ref().unwrap() {
                        if line.matches(m, work)? {
                            hits.push(line.source);
                        }
                    }
                    cache.insert(m.clone(), MarkerState { hits });
                }
                auxiliary = cache[m].hits.iter().any(|id| core.contains(id));
            } else {
                for l in lines {
                    work.charge(1)?;
                    if core.contains(&integer(l, "source_index", 0)?)
                        && (superscript(l, m, p, angle, work)?
                            || compact(l["text"].as_str().unwrap(), m, work)?)
                    {
                        auxiliary = true;
                        break;
                    }
                }
            }
        }
        let first_gap = if auxiliary && !r.explicit { 0.75 } else { 1.25 };
        if gap > if started { em } else { first_gap * em } {
            break;
        }
        if !started {
            if !r.explicit && !auxiliary {
                break;
            }
            let compatible = if r.explicit {
                super::graphic_captions::axis(&r.row.b, &rule, 0) >= 0.35
                    && (r.row.b.n[0] - rule.n[0]).abs() <= 2. * em
                    && r.height <= 1.15 * em
            } else {
                if body_height.is_none() {
                    let mut heights = Vec::new();
                    if let Some(metrics) = metric_cache.as_deref_mut() {
                        if metrics.is_none() {
                            let mut prepared = Vec::new();
                            for l in lines {
                                if l.get("angle").map_or(Ok(0.), num)? != angle as f64 {
                                    continue;
                                }
                                let b = rotate(&bbox(&l["bbox"])?, p, angle);
                                prepared.push((
                                    integer(l, "source_index", 0)?,
                                    (b[1] + b[3]) / 2.,
                                    height(l, b)?,
                                ));
                            }
                            *metrics = Some(BodyMetrics::new(prepared));
                        }
                        body_height = Some(metrics.as_ref().unwrap().height(
                            rule.n[1] - 3. * em,
                            rule.n[3] + 10. * em,
                            core,
                            1.25 * em,
                            work,
                        )?);
                    } else {
                        for l in lines {
                            work.charge(1)?;
                            if l.get("angle").map_or(Ok(0.), num)? != angle as f64
                                || core.contains(&integer(l, "source_index", 0)?)
                            {
                                continue;
                            }
                            let b = rotate(&bbox(&l["bbox"])?, p, angle);
                            let center = (b[1] + b[3]) / 2.;
                            if rule.n[1] - 3. * em <= center && center <= rule.n[3] + 10. * em {
                                continue;
                            }
                            heights.push(height(l, b)?);
                        }
                    }
                    let h = if heights.len() < 4 {
                        1.25 * em
                    } else {
                        work.charge(heights.len() * 16 + 1)?;
                        heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
                        median(
                            heights[heights.len() - heights.len().div_ceil(4)..].to_vec(),
                            work,
                        )?
                    };
                    if body_height.is_none() {
                        body_height = Some(h);
                    }
                }
                super::graphic_captions::axis(&r.row.b, &rule, 0) >= 0.5
                    && (r.row.b.n[0] - rule.n[0]).abs() <= 3. * em
                    && r.height <= 1.05 * em
                    && r.height <= 0.9 * body_height.unwrap()
            };
            if !compatible {
                break;
            }
            note_left = r.row.b.n[0];
            note_height = r.height.max(0.1);
            note_fonts = r.fonts.clone();
        } else {
            work.charge(
                note_fonts
                    .iter()
                    .map(|f| f.to_string().len())
                    .sum::<usize>()
                    * r.fonts.len()
                    + r.fonts.iter().map(|f| f.to_string().len()).sum::<usize>() * note_fonts.len()
                    + 1,
            )?;
            if (r.row.b.n[0] - note_left).abs() > 1.5 * em
                || !(0.75..=1.25).contains(&(r.height / note_height))
                || (!note_fonts.is_empty()
                    && !r.fonts.is_empty()
                    && !note_fonts.iter().any(|f| r.fonts.iter().any(|r| eq(f, r))))
                || super::graphic_captions::axis(&r.row.b, &rule, 0) < 0.35
            {
                break;
            }
        }
        added.extend(r.ids.iter().copied());
        bottom = bottom.max(r.row.b.n[3]);
        started = true;
        output.push(&r.row);
    }
    if let Some(timings) = timings.as_deref_mut() {
        timings.mark("note_row_selection");
    }
    // Footnote rows are immutable corridor rows. Their geometry depends only
    // on the selected row sequence and the core sources excluded from it.
    // Keep all semantic selection above; reuse only the resulting geometry.
    let geometry_key = if geometry_only
        && crate::legacy_page::budget::current().host
        && crate::legacy_page::budget::current().work == usize::MAX
    {
        let excluded = output
            .iter()
            .flat_map(|row| &row.fragments)
            .filter(|fragment| core.contains(&fragment.id))
            .map(|fragment| fragment.id)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        Some((
            output
                .iter()
                .map(|row| *row as *const prep::Row as usize)
                .collect::<Vec<_>>(),
            excluded,
        ))
    } else {
        None
    };
    if let Some(timings) = timings.as_deref_mut() {
        timings.mark("note_geometry_key");
    }
    if let (Some(cache), Some(key)) = (geometry_cache.as_deref_mut(), geometry_key.as_ref()) {
        if let Some(value) = cache.entries.get(key) {
            return Ok(value.clone());
        }
    }
    let annotation = prep::annotation_refs("footnote", &output, core, None, work)?;
    let rows = output
        .iter()
        .map(|r| {
            if geometry_only {
                value_object!({"bbox":r.b.raw})
            } else {
                prep::output(r)
            }
        })
        .collect::<Vec<_>>();
    let out = Arc::new(
        value_object!({"footnote_rows":Value::Array(rows),"annotation":annotation.unwrap_or(Value::Null)}),
    );
    if let (Some(cache), Some(key)) = (geometry_cache.as_deref_mut(), geometry_key) {
        let weight = key.0.len()
            + key.1.len()
            + out["annotation"]["line_indices"]
                .as_array()
                .map_or(0, Vec::len)
                * 2;
        // Same cache bounds as upstream annotation preparation. Eviction only
        // repeats geometry work; it never restricts document/candidate size.
        if weight <= 16384 {
            if cache.entries.len() >= 128 || cache.weight + weight > 16384 {
                cache.entries.clear();
                cache.weight = 0;
            }
            cache.weight += weight;
            cache.entries.insert(key, out.clone());
        }
    }
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(out)
}
fn prepare_rows(
    v: &Value,
    rows_override: Option<&[prep::Row]>,
    lines: &[&Value],
    rule: &Box4,
    em: f64,
    p: [f64; 2],
    angle: i64,
    work: &mut Work,
) -> Result<PreparedRows, Error> {
    let mut by_id = BTreeMap::new();
    let mut glyph_count = 0;
    for (i, l) in lines.iter().enumerate() {
        let text = l["text"].as_str().ok_or_else(invalid)?;
        work.charge(text.len() + 1)?;
        if text.chars().count() > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        bbox(&l["bbox"])?;
        num(l.get("angle").unwrap_or(&json!(0)))?;
        glyph_count += optional(l, "chars", 20000)?.len();
        if glyph_count > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        by_id.insert(integer(l, "source_index", 0)?, i);
    }
    let mut total = 0;
    let mut prepared = Vec::new();
    let owned_rows;
    let rows = if let Some(rows) = rows_override {
        rows
    } else {
        owned_rows = array(&v["rows"], 4096)?
            .iter()
            .map(|r| prep::row(r, &mut total))
            .collect::<Result<Vec<_>, _>>()?;
        &owned_rows
    };
    for r in rows {
        let Some(row) = prep::clipped(r, &rule, 2. * em, work)? else {
            continue;
        };
        let ids: BTreeSet<_> = row.fragments.iter().map(|f| f.id).collect();
        work.charge(ids.len() * 16 + 1)?;
        let mut heights = Vec::new();
        let mut fonts = Vec::new();
        for id in &ids {
            if let Some(i) = by_id.get(id) {
                let l = &lines[*i];
                heights.push(height(l, rotate(&bbox(&l["bbox"])?, p, angle))?);
                if let Some(f) = l.get("font_signature").filter(|v| !v.is_null()) {
                    let coverage = l.get("font_coverage").map_or(Ok(0.), num)?;
                    if coverage >= 0.75 {
                        let signature = array(f, 2)?;
                        if signature.len() != 2 {
                            return Err(invalid());
                        }
                        let name = signature[0].as_str().ok_or_else(invalid)?;
                        if name.len() > rapiddoc_core::resources::limit(4096) {
                            return Err(limit());
                        }
                        if !signature[1].is_i64() && !signature[1].is_boolean() {
                            return Err(invalid());
                        }
                        work.charge((name.len() + 32) * (fonts.len() + 1))?;
                        if !fonts.iter().any(|v| eq(v, f)) {
                            fonts.push(f.clone());
                        }
                    }
                }
            }
        }
        let rh = if heights.is_empty() {
            row.b.n[3] - row.b.n[1]
        } else {
            median(heights, work)?
        };
        let mut words = Vec::new();
        for f in &row.fragments {
            let text = f.raw["text"].as_str().ok_or_else(invalid)?;
            work.charge(text.len() * 4 + 1)?;
            let text = text.trim_matches(python_space);
            if !text.is_empty() {
                words.push(text);
            }
        }
        let text = words.join(" ");
        prepared.push(Prepared {
            row,
            ids,
            height: rh,
            fonts,
            explicit: super::table_block::note(&text),
            marker: marker(&text),
        });
    }
    Ok(PreparedRows::new(prepared))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn note_row_index_skips_only_the_original_prefix() {
        // Deliberately unsorted bottoms: a tall earlier row must not be skipped
        // merely because later rows finish above the table bottom.
        for bottoms in [
            vec![],
            vec![10., 30., 20., 30., 5., 60.],
            vec![0., -0., -4.],
        ] {
            let rows = bottoms.iter().enumerate().map(|(index, bottom)| {
                let value = json!({"bbox":[0.,bottom-2.,100.,bottom],"center_y":bottom-1.,
                    "fragments":[{"line_index":index,"text":"text","bbox":[0.,bottom-2.,100.,bottom],"local_bbox":[0.,bottom-2.,100.,bottom]}]});
                Prepared { row: prep::row(&value, &mut 0).unwrap(), ids: BTreeSet::new(), height: 2., fonts: vec![], explicit: false, marker: None }
            }).collect();
            let prepared = PreparedRows::new(rows);
            for bottom in [-10., -4., -0., 0., 5., 10., 20., 30., 60., 100.] {
                assert_eq!(
                    prepared.first_after(bottom),
                    bottoms
                        .iter()
                        .position(|value| *value > bottom)
                        .unwrap_or(bottoms.len())
                );
            }
            assert_eq!(prepared.first_after(f64::NAN), 0);
        }
    }
    #[test]
    fn annotation_geometry_cache_distinguishes_core_exclusions() {
        let _scope = crate::legacy_page::budget::host_page();
        let row = json!({"bbox":[0,11,100,19],"center_y":15,
            "fragments":[{"line_index":5,"text":"注：","bbox":[0,11,25,19],"local_bbox":[0,11,25,19]},
                {"line_index":6,"text":"说明","bbox":[25,11,100,19],"local_bbox":[25,11,100,19]}]});
        let rows = [prep::row(&row, &mut 0).unwrap()];
        let lines = [
            json!({"source_index":5,"text":"注：","bbox":[0,11,25,19],"chars":[]}),
            json!({"source_index":6,"text":"说明","bbox":[25,11,100,19],"chars":[]}),
        ];
        let refs = lines.iter().collect::<Vec<_>>();
        let request =
            json!({"median_height":10,"rule_bbox":[0,0,100,10],"page_size":[200,300],"angle":0});
        let mut cache = FootnoteCache::default();
        let mut first_result = None;
        for core in [
            BTreeSet::from([5]),
            BTreeSet::from([6]),
            BTreeSet::new(),
            BTreeSet::from([5]),
        ] {
            let actual =
                collect_core_geometry(&request, &rows, &refs, &mut cache, &core, &mut Work(0))
                    .unwrap();
            if core == BTreeSet::from([5]) {
                if let Some(first) = &first_result {
                    assert!(Arc::ptr_eq(first, &actual));
                } else {
                    first_result = Some(Arc::clone(&actual));
                }
            }
            let expected = collect_core_geometry(
                &request,
                &rows,
                &refs,
                &mut FootnoteCache::default(),
                &core,
                &mut Work(0),
            )
            .unwrap();
            assert_eq!(actual, expected);
            assert_eq!(
                actual["annotation"]["line_indices"],
                json!(BTreeSet::from([5, 6])
                    .difference(&core)
                    .copied()
                    .collect::<Vec<_>>())
            );
        }
        assert_eq!(cache.geometry.entries.len(), 3);
    }
    #[test]
    fn prepared_marker_lines_preserve_glyph_and_compact_rules() {
        for text in [
            "a",
            "a 1",
            "Ａ α 1",
            "1 2 3 4 5",
            "a long ordinary body paragraph",
            "",
        ] {
            for angle in [0, 90, 180, 270] {
                let chars = text.chars().enumerate().map(|(i, c)| {
                    let (top, bottom) = if i == 0 { (1., 5.) } else { (4., 14.) };
                    json!({"char":c.to_string(),"bbox":[i as f64 * 5.,top,i as f64 * 5.+4.,bottom]})
                }).collect::<Vec<_>>();
                let line = json!({"source_index":-99,"text":text,"chars":chars});
                let page = [200., 300.];
                let prepared = MarkerLine::prepare(&line, page, angle, &mut Work(0)).unwrap();
                for marker in ["a", "Ａ", "α", "1", "2", "a1", "unknown"] {
                    assert_eq!(
                        prepared.matches(marker, &mut Work(0)).unwrap(),
                        superscript(&line, marker, page, angle, &mut Work(0)).unwrap()
                            || compact(text, marker, &mut Work(0)).unwrap(),
                        "{text:?} {marker:?} {angle}"
                    );
                }
            }
        }
    }
    #[test]
    fn indexed_body_height_preserves_core_exclusion_and_boundaries() {
        let mut items = (0..120)
            .map(|i| (i / 2, i as f64, (i % 17 + 1) as f64))
            .collect::<Vec<_>>();
        items.push((999, 20., 7.));
        items.push((999, 100., 11.));
        for entries in [&items[..], &[][..]] {
            let metrics = BodyMetrics::new(entries.to_vec());
            for (top, bottom) in [(-10., 130.), (20., 100.), (20., 20.), (45., 63.)] {
                for core in [
                    BTreeSet::new(),
                    BTreeSet::from([0, 10, 20, 999]),
                    (0..60).collect(),
                ] {
                    let mut values = entries
                        .iter()
                        .filter(|(id, y, _)| !(top <= *y && *y <= bottom) && !core.contains(id))
                        .map(|item| item.2)
                        .collect::<Vec<_>>();
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let expected = if values.len() < 4 {
                        12.5
                    } else {
                        median(
                            values[values.len() - values.len().div_ceil(4)..].to_vec(),
                            &mut Work(0),
                        )
                        .unwrap()
                    };
                    assert_eq!(
                        metrics
                            .height(top, bottom, &core, 12.5, &mut Work(0))
                            .unwrap(),
                        expected
                    );
                }
            }
        }
    }
    #[test]
    fn old_table_notes() {
        for name in ["native-table-footnotes", "native-table-footnote-boundaries"] {
            let f: Value = serde_json::from_slice(
                &std::fs::read(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join(format!("../../build/migration/{name}.json")),
                )
                .unwrap(),
            )
            .unwrap();
            for c in f["cases"].as_array().unwrap() {
                assert_eq!(
                    collect(&c["request"]).unwrap(),
                    c["expected"],
                    "{}",
                    c["label"]
                );
                let _scope = crate::legacy_page::budget::host_page();
                let mut cache = FootnoteCache::default();
                for _ in 0..2 {
                    assert_eq!(
                        collect_cached(&c["request"], None, None, Some(&mut cache), &mut Work(0))
                            .unwrap(),
                        c["expected"],
                        "cached {}",
                        c["label"]
                    );
                }
            }
            if let Some(markers) = f.get("markers") {
                for c in markers.as_array().unwrap() {
                    assert_eq!(
                        marker(c["text"].as_str().unwrap()),
                        c["expected"].as_str().map(str::to_owned)
                    );
                }
            }
        }
    }

    #[test]
    fn shared_budget() {
        let q = json!({"rows":[],"lines":[],"core_line_indices":[],"rule_bbox":[0,0,100,100],"median_height":10,"page_size":[400,600]});
        assert!(collect_with_work(&q, &mut Work(10_000_000)).is_err());
        assert!(collect(&q).is_ok());
    }
}
