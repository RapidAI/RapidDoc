//! Complete existing Python text_assembly: row boundaries, footnotes and all block refinements.
mod annotations;
mod assembly;
mod block_common;
mod block_merging;
mod breaks;
mod caption_wrap;
mod captions;
mod connections;
mod footnotes;
mod grouping;
mod local_lists;
mod math_blocks;
mod member_blocks;
mod prose;
use super::{
    native_marginals::local,
    native_titles::reference_height,
    page_containers::{clip, coerce, combined, optional, size},
    path_graphics::{array, bbox, integer as strict_integer, num, Box4},
};
use crate::legacy_page::{eq, invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::{python_alnum, python_space};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
#[derive(Clone)]
struct Row {
    index: usize,
    b: Box4,
}
#[derive(Clone)]
struct Lane {
    left: f64,
    right: f64,
    span: bool,
    rows: Vec<Row>,
}
struct State {
    lines: Vec<Value>,
    page: [f64; 2],
    tables: Vec<Box4>,
    rules: Vec<Value>,
}
fn integer(v: &Value, k: &str, default: i64) -> Result<i64, Error> {
    if k == "visual_row_id" {
        if let Some(b) = v[k].as_bool() {
            return Ok(i64::from(b));
        }
    }
    strict_integer(v, k, default)
}
impl State {
    fn validate(&mut self, w: &mut Work) -> Result<(), Error> {
        for l in &mut self.lines {
            let raw = l.as_object().ok_or_else(invalid)?.clone();
            let mut normalized = super::native_lines::defaults();
            normalized.as_object_mut().unwrap().extend(raw);
            *l = normalized;
            let t = l["text"].as_str().ok_or_else(invalid)?;
            w.charge(t.len() + 1)?;
            bbox(&l["bbox"])?;
            if l["source_bbox"].is_null() {
                l["source_bbox"] = l["bbox"].clone();
            }
            for k in ["source_index", "angle", "run_index"] {
                integer(l, k, 0)?;
            }
            if !l["visual_row_id"].is_null() {
                integer(l, "visual_row_id", 0)?;
            }
            for k in ["font_coverage", "effective_height", "em_height"] {
                if !l[k].is_null() {
                    num(&l[k])?;
                }
            }
            if !l["font_signature"].is_null() {
                let f = array(&l["font_signature"], 2)?;
                if f.len() != 2 || !f[0].is_string() || !(f[1].is_i64() || f[1].is_boolean()) {
                    return Err(invalid());
                }
            }
            for k in ["chars", "native_math_words", "inline_math_regions"] {
                array(&l[k], 20000)?;
            }
        }
        Ok(())
    }
    fn l(&self, r: &Row) -> &Value {
        &self.lines[r.index]
    }
    fn t(&self, r: &Row) -> Result<&str, Error> {
        self.l(r)["text"].as_str().ok_or_else(invalid)
    }
    fn id(&self, r: &Row) -> Result<i64, Error> {
        integer(self.l(r), "source_index", 0)
    }
    fn h(&self, r: &Row) -> Result<f64, Error> {
        reference_height(self.l(r), &r.b)
    }
    fn coverage(&self, r: &Row) -> Result<f64, Error> {
        self.l(r).get("font_coverage").map_or(Ok(0.), num)
    }
    fn raw(&self, r: &Row) -> Result<Box4, Error> {
        bbox(&self.l(r)["bbox"])
    }
    fn geometry(&self, angle: i64) -> Result<Vec<Row>, Error> {
        self.lines
            .iter()
            .enumerate()
            .filter(|(_, l)| integer(l, "angle", 0).ok() == Some(angle))
            .map(|(index, l)| {
                Ok(Row {
                    index,
                    b: local(&bbox(&l["bbox"])?, self.page, angle),
                })
            })
            .collect()
    }
    fn lanes(&self, rs: &[Row], width: f64, h: f64, w: &mut Work) -> Result<Vec<Lane>, Error> {
        let raw = super::native_layout::infer(
            &rs.iter()
                .map(|r| {
                    let mut l = self.l(r).clone();
                    if let Some(b) = l["visual_row_id"].as_bool() {
                        l["visual_row_id"] = json!(i64::from(b));
                    }
                    if let Some(b) = l["font_signature"][1].as_bool() {
                        l["font_signature"][1] = json!(i64::from(b));
                    }
                    (l, r.b.clone())
                })
                .collect::<Vec<_>>(),
            width,
            h,
            true,
            w,
        )?;
        raw.iter()
            .map(|l| {
                Ok(Lane {
                    left: num(&l["left"])?,
                    right: num(&l["right"])?,
                    span: l["is_span"] == true,
                    rows: array(&l["line_indices"], 4096)?
                        .iter()
                        .map(|i| {
                            rs.get(i.as_u64().ok_or_else(invalid)? as usize)
                                .cloned()
                                .ok_or_else(invalid)
                        })
                        .collect::<Result<Vec<_>, _>>()?,
                })
            })
            .collect()
    }
    fn gap(&self, a: &Row, b: &Row) -> Result<f64, Error> {
        let h = self.h(a)?;
        Ok(if self.l(a)["restored_inline_cluster"] == true {
            (b.b.n[1] - a.b.n[3]).max(-0.25 * h)
        } else {
            b.b.n[1] - (a.b.n[1] + h)
        })
    }
    fn body_gap(&self, a: &Row, b: &Row) -> Result<f64, Error> {
        let (al, bl) = (self.l(a), self.l(b));
        if !al["baseline"].is_null() && !bl["baseline"].is_null() {
            let pitch = num(&bl["baseline"])? - num(&al["baseline"])?;
            let ah = self.h(a)?;
            let bh = self.h(b)?;
            if pitch > 0. && (0.5 * ah.min(bh)..=3. * ah.max(bh)).contains(&pitch) {
                return Ok(pitch - ah);
            }
        }
        self.gap(a, b)
    }
    fn ordered(&self, rs: &mut [Row], source: bool) {
        rs.sort_by(|a, b| {
            a.b.n[1]
                .total_cmp(&b.b.n[1])
                .then(a.b.n[0].total_cmp(&b.b.n[0]))
                .then(if source {
                    self.id(a).unwrap().cmp(&self.id(b).unwrap())
                } else {
                    std::cmp::Ordering::Equal
                })
        })
    }
    fn rule_barrier(&self, a: &Row, b: &Row, lane: &Lane) -> Result<bool, Error> {
        if b.b.n[1] <= a.b.n[3] {
            return Ok(false);
        }
        let width = (lane.right - lane.left).max(0.1);
        for rule in &self.rules {
            if rule["orientation"] != "horizontal" {
                continue;
            }
            let rb = bbox(&rule["bbox"])?;
            if a.b.n[3] <= cy(&rb)
                && cy(&rb) <= b.b.n[1]
                && (rb.n[2].min(lane.right) - rb.n[0].max(lane.left)).max(0.) / width >= 0.6
            {
                return Ok(true);
            }
        }
        Ok(false)
    }
    fn table_barrier(&self, a: &Row, b: &Row) -> Result<bool, Error> {
        let a = self.raw(a)?;
        let b = self.raw(b)?;
        let connector = coerce(&json!([
            cx(&a).min(cx(&b)) - 0.1,
            cy(&a).min(cy(&b)) - 0.1,
            cx(&a).max(cx(&b)) + 0.1,
            cy(&a).max(cy(&b)) + 0.1
        ]))?;
        Ok(connector.is_some_and(|c| self.tables.iter().any(|t| intersects(&c, t))))
    }
    fn transform_rules(&mut self, raw: &[Value], angle: i64) -> Result<(), Error> {
        self.rules = raw
            .iter()
            .map(|r| {
                let mut out = r.clone();
                let b = local(&bbox(&r["bbox"])?, self.page, angle);
                out["bbox"] = json!(b.raw);
                if angle == 90 || angle == 270 {
                    out["orientation"] = if r["orientation"] == "horizontal" {
                        json!("vertical")
                    } else {
                        json!("horizontal")
                    }
                }
                Ok(out)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(())
    }
    fn lane_gap(&self, lane: &mut Lane, w: &mut Work) -> Result<(f64, f64), Error> {
        self.ordered(&mut lane.rows, true);
        w.charge(lane.rows.len() + 1)?;
        rapiddoc_native::geometry_plan::statistics::lane_gap(
            lane.rows.iter().map(|r| r.b.n).collect(),
            lane.rows
                .iter()
                .map(|r| self.h(r))
                .collect::<Result<Vec<_>, _>>()?,
            lane.rows
                .iter()
                .map(|r| self.l(r)["restored_inline_cluster"] == true)
                .collect(),
            lane.rows
                .windows(2)
                .map(|p| {
                    eq(
                        &self.l(&p[0])["visual_row_id"],
                        &self.l(&p[1])["visual_row_id"],
                    ) && (self.l(&p[0])["split_from_row"] == true
                        || self.l(&p[1])["split_from_row"] == true)
                })
                .collect(),
        )
        .ok_or_else(invalid)
    }
}
fn cx(b: &Box4) -> f64 {
    (b.n[0] + b.n[2]) * 0.5
}
fn median(v: Vec<f64>) -> Result<f64, Error> {
    if v.is_empty() {
        Err(invalid())
    } else {
        Ok(super::page_containers::median(v))
    }
}
fn cy(b: &Box4) -> f64 {
    (b.n[1] + b.n[3]) * 0.5
}
fn intersects(a: &Box4, b: &Box4) -> bool {
    a.n[2].min(b.n[2]) > a.n[0].max(b.n[0]) && a.n[3].min(b.n[3]) > a.n[1].max(b.n[1])
}
fn axis(a: &Box4, b: &Box4, k: usize) -> f64 {
    let d = (a.n[k + 2] - a.n[k]).min(b.n[k + 2] - b.n[k]);
    if d <= 0. {
        0.
    } else {
        (a.n[k + 2].min(b.n[k + 2]) - a.n[k].max(b.n[k])).max(0.) / d
    }
}
fn words(t: &str) -> usize {
    t.split(python_space).filter(|s| !s.is_empty()).count()
}
fn regex(s: &str) -> regex::Regex {
    static CACHE: std::sync::OnceLock<std::sync::Mutex<BTreeMap<String, regex::Regex>>> =
        std::sync::OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    cache
        .entry(s.to_owned())
        .or_insert_with(|| {
            regex::Regex::new(
                &s.replace("\\s", SPACE)
                    .replace("\\S", &format!("[^{}]", &SPACE[1..SPACE.len() - 1])),
            )
            .unwrap()
        })
        .clone()
}
fn fold(t: &str) -> String {
    t.chars()
        .map(|c| match c {
            'İ' | 'ı' => 'i',
            'ſ' => 's',
            'K' => 'k',
            _ => c,
        })
        .collect()
}
fn width(r: &Row) -> f64 {
    r.b.n[2] - r.b.n[0]
}
fn ordinary(s: &State, r: &Row) -> bool {
    s.l(r)["semantic_type"].is_null()
}
fn clean(s: &State, r: &Row) -> bool {
    ordinary(s, r) && s.l(r)["paragraph_group"].is_null() && s.l(r)["caption_start"] != true
}
fn math(s: &State, r: &Row) -> bool {
    s.l(r)["formula_candidate_only"] == true || s.l(r)["compact_formula_cluster"] == true
}
fn ordered(s: &State, l: &Lane, body: bool, source: bool) -> Vec<Row> {
    let mut rs = l
        .rows
        .iter()
        .filter(|r| !body || ordinary(s, r))
        .cloned()
        .collect::<Vec<_>>();
    s.ordered(&mut rs, source);
    rs
}
fn cjk(t: &str) -> usize {
    t.chars()
        .filter(|c| ('\u{3400}'..='\u{9fff}').contains(c))
        .count()
}
fn bounded_words(t: &str) -> usize {
    regex(r"[A-Za-z]{2,}")
        .find_iter(t)
        .filter(|m| {
            let word = |c: char| c == '_' || python_alnum(c);
            !t[..m.start()].chars().next_back().is_some_and(word)
                && !t[m.end()..].chars().next().is_some_and(word)
        })
        .count()
}
fn number(t: &str) -> Result<i64, Error> {
    t.chars().try_fold(0i64, |n, c| {
        n.checked_mul(10)
            .and_then(|n| rapiddoc_native::python_decimal(c).map(|d| n + i64::from(d)))
            .ok_or_else(invalid)
    })
}
const SPACE: &str = r"[\u{9}-\u{d}\u{1c}-\u{20}\u{85}\u{a0}\u{1680}\u{2000}-\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}]";
fn norm(t: &str) -> String {
    let t = super::native_formula_components::sanitize_text(t, false);
    regex(r"[\t\f\v ]+")
        .replace_all(&t, " ")
        .trim_matches(python_space)
        .to_owned()
}
fn join_text(ts: &[String]) -> Result<String, Error> {
    rapiddoc_native::text_boundary::merge(
        &ts.iter()
            .map(|t| norm(t))
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>(),
    )
}
fn hyphen(t: &str) -> bool {
    regex(r"[A-Za-z][-\u{ad}‐‑⁃]\s*$").is_match(t)
}
fn family(f: &Value) -> Result<Option<String>, Error> {
    if f.is_null() {
        return Ok(None);
    }
    Ok(super::table_text_styles::preparation::family(&Some((
        f[0].as_str().ok_or_else(invalid)?.to_owned(),
        f[1].clone(),
    ))))
}
fn share(a: &Value, b: &Value) -> Result<bool, Error> {
    Ok(family(a)?.zip(family(b)?).is_some_and(|(a, b)| a == b))
}
fn emphasis(a: &Value, b: &Value) -> Result<bool, Error> {
    let re =
        regex(r"(?:semibold|demibold|regular|regu|roman|bold|light|medium|italic|ital|oblique)+$");
    let a = family(a)?.unwrap_or_default();
    let b = family(b)?.unwrap_or_default();
    let a = re.replace(&a, "");
    let b = re.replace(&b, "");
    Ok(!a.is_empty() && a == b)
}
fn conflict(a: &Value, b: &Value) -> Result<bool, Error> {
    if a["dominant_font_weight"].is_null() || b["dominant_font_weight"].is_null() {
        return Ok(false);
    }
    let a = num(&a["dominant_font_weight"])?;
    let b = num(&b["dominant_font_weight"])?;
    Ok((a - b).abs() >= 100. && a.max(b) >= 1.15 * a.min(b))
}
fn fonts(a: &Value, b: &Value) -> Result<bool, Error> {
    let covered = |l: &Value| l.get("font_coverage").map_or(Ok(0.), num);
    Ok((a["font_signature"].is_null()
        || b["font_signature"].is_null()
        || covered(a)? < 0.75
        || covered(b)? < 0.75
        || eq(&a["font_signature"], &b["font_signature"]))
        && !conflict(a, b)?)
}
fn flags(l: &Value) -> Result<i64, Error> {
    if l["font_signature"].is_null() {
        Ok(0)
    } else {
        l["font_signature"][1]
            .as_i64()
            .or_else(|| l["font_signature"][1].as_bool().map(i64::from))
            .ok_or_else(invalid)
    }
}
fn tight(l: &Value, p: [f64; 2]) -> Result<Option<Box4>, Error> {
    let Some(b) = coerce(&l["ink_bbox"])? else {
        return Ok(None);
    };
    clip(
        &bbox(&json!([b.n[0] - 1., b.n[1] - 1., b.n[2] + 1., b.n[3] + 1.]))?,
        p,
    )
}
/// Diagnostic for the complete existing row-connection rules, before block orchestration.
pub fn inspect_connections(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let mut s = State {
        lines: array(&v["lines"], 4096)?.to_vec(),
        page: size(&v["page_size"])?,
        tables: optional(v, "table_bboxes", 5000)?
            .iter()
            .map(bbox)
            .collect::<Result<Vec<_>, _>>()?,
        rules: vec![],
    };
    let angle = integer(v, "angle", 0)?;
    s.transform_rules(optional(v, "drawing_lines", 5000)?, angle)?;
    let rs = s.geometry(angle)?;
    let lane = Lane {
        left: num(&v["left"])?,
        right: num(&v["right"])?,
        span: v["is_span"] == true,
        rows: rs.clone(),
    };
    let mut result = Vec::new();
    for pair in array(&v["pairs"], 10000)? {
        let pair = array(pair, 2)?;
        if pair.len() != 2 {
            return Err(invalid());
        }
        let a = rs
            .get(pair[0].as_u64().ok_or_else(invalid)? as usize)
            .ok_or_else(invalid)?;
        let b = rs
            .get(pair[1].as_u64().ok_or_else(invalid)? as usize)
            .ok_or_else(invalid)?;
        result.push(json!({"semantic":connections::semantic(&s,a,b,&lane,num(&v["regular_gap"])?)?,"text":connections::text(&s,a,b,&lane,num(&v["regular_gap"])?,num(&v["gap_mad"])?)?,"gap":s.gap(a,b)?,"body_gap":s.body_gap(a,b)?,"rule_barrier":s.rule_barrier(a,b,&lane)?,"table_barrier":s.table_barrier(a,b)?}));
    }
    Ok(json!(result))
}

/// Complete original row-boundary and grouping results, with stable local-lane identities.
pub fn inspect_rows(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let mut s = State {
        lines: array(&v["lines"], 4096)?.to_vec(),
        page: size(&v["page_size"])?,
        tables: optional(v, "table_bboxes", 5000)?
            .iter()
            .map(bbox)
            .collect::<Result<Vec<_>, _>>()?,
        rules: vec![],
    };
    s.validate(&mut Work(0))?;
    let angle = integer(v, "angle", 0)?;
    s.transform_rules(optional(v, "drawing_lines", 5000)?, angle)?;
    let rs = s.geometry(angle)?;
    let lane = Lane {
        left: num(&v["left"])?,
        right: num(&v["right"])?,
        span: v["is_span"] == true,
        rows: rs.clone(),
    };
    let regular = num(&v["regular_gap"])?;
    let mad = num(&v["gap_mad"])?;
    let page_h = if angle == 90 || angle == 270 {
        s.page[0]
    } else {
        s.page[1]
    };
    let visuals = optional(v, "visual_bboxes", 5000)?
        .iter()
        .map(|b| Ok(local(&bbox(b)?, s.page, angle)))
        .collect::<Result<Vec<_>, Error>>()?;
    let page = if v["page_index"].is_null() {
        None
    } else {
        Some(integer(v, "page_index", 0)?)
    };
    let mut out = json!({"top":breaks::top(&s,&lane,page_h)?,"bullets":breaks::bullets(&s,&rs)?,"italic":breaks::italic(&s,&lane)?,"caption":breaks::caption(&s,&lane)?,"structured":breaks::structured(&s,&lane,regular,mad)?,"isolated":breaks::isolated(&s,&lane,regular,mad)?,"visual":breaks::visual(&s,&lane,&visuals,page_h)?,"typography":prose::typography(&s,&lane,regular,mad)?,"formula":breaks::formula(&s,&lane)?,"front":breaks::front(&s,&lane,page_h,page)?,"explicit":prose::explicit(&s,&lane)?,"prose":prose::prose(&s,&lane,regular,mad)?,"cjk_prose":prose::cjk_prose(&s,&lane,regular,mad)?,"cjk_entries":prose::cjk_entries(&s,&lane)?,"emphasized":grouping::emphasized(&s,&rs)?,"hanging":grouping::hanging(&s,&lane)?});
    let (lanes, map) = grouping::local_lanes(&s, &lane)?;
    let mut local_map = serde_json::Map::new();
    for (id, i) in map {
        let l = &lanes[i];
        local_map.insert(id.to_string(),json!({"left":l.left,"right":l.right,"is_span":l.span,"members":l.rows.iter().map(|r|r.index).collect::<Vec<_>>() }));
    }
    out["local_lanes"] = json!(local_map);
    out["starts_reference"] = json!(rs
        .windows(2)
        .map(|p| grouping::starts_reference(&s, &p[0], &p[1]))
        .collect::<Result<Vec<_>, _>>()?);
    let mut changed = false;
    let mut boxes = vec![];
    for r in &rs {
        let b = tight(s.l(r), s.page)?;
        changed |= b.is_some();
        boxes.push(local(&b.unwrap_or(s.raw(r)?), s.page, angle).raw);
    }
    out["tight"] = json!([boxes, changed]);
    Ok(out)
}
pub fn build_grouped_footnotes(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let s = State {
        lines: array(&v["lines"], 4096)?.to_vec(),
        page: size(&v["page_size"])?,
        tables: vec![],
        rules: vec![],
    };
    let (blocks, consumed) = footnotes::build(&s, optional(v, "page_footnote_groups", 4096)?)?;
    Ok(json!({"blocks":blocks,"consumed":consumed}))
}
pub fn build_text_blocks(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    build_with_work(v, &mut Work(0))
}
pub(super) fn build_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let p = size(&v["page_size"])?;
    let blocks = assembly::components(v, w)?;
    let mut grouped = blocks
        .iter()
        .filter(|b| !b["_reference_group"].is_null())
        .cloned()
        .collect::<Vec<_>>();
    let ordinary = blocks
        .into_iter()
        .filter(|b| b["_reference_group"].is_null())
        .collect();
    let mut blocks = block_merging::prefix(ordinary, p)?;
    blocks = block_merging::spatial(blocks, p, w)?;
    blocks = block_merging::list_intro(blocks)?;
    blocks = block_merging::unterminated(blocks, w)?;
    blocks = block_merging::overlapping(blocks, p)?;
    blocks = math_blocks::fragments(blocks, p, w)?;
    grouped.extend(math_blocks::formula_context(blocks, p, w)?);
    let out = json!(grouped);
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        Err(limit())
    } else {
        Ok(out)
    }
}
/// Standalone replay of original block refinements; not an alternate parser.
pub fn refine_blocks(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let blocks = array(&v["blocks"], 4096)?.to_vec();
    for b in &blocks {
        b.as_object().ok_or_else(invalid)?;
        block_common::angle(b)?;
        block_common::rows(b)?;
        block_common::heights(b, false)?;
        if !b["bbox"].is_null() {
            bbox(&b["bbox"])?;
        }
        for f in optional(b, "_font_signatures", 4096)? {
            let f = array(f, 2)?;
            if f.len() != 2 || !f[0].is_string() || !(f[1].is_i64() || f[1].is_boolean()) {
                return Err(invalid());
            }
        }
        for field in [
            "_text_lines",
            "_inline_math_regions",
            "_visual_row_ids",
            "_local_output_line_bboxes",
        ] {
            optional(b, field, 20000)?;
        }
    }
    let p = size(&v["page_size"])?;
    let w = &mut Work(0);
    match v["kind"].as_str().ok_or_else(invalid)? {
        "prefix" => Ok(json!(block_merging::prefix(blocks, p)?)),
        "spatial" => Ok(json!(block_merging::spatial(blocks, p, w)?)),
        "list" => Ok(json!(block_merging::list_intro(blocks)?)),
        "unterminated" => Ok(json!(block_merging::unterminated(blocks, w)?)),
        "overlap" => Ok(json!(block_merging::overlapping(blocks, p)?)),
        "math" => Ok(json!(math_blocks::fragments(blocks, p, w)?)),
        "context" => Ok(json!(math_blocks::formula_context(blocks, p, w)?)),
        _ => Err(invalid()),
    }
}
pub fn refine_page_blocks(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let out = refine_page_blocks_with_work(v, &mut Work(0))?;
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        Err(limit())
    } else {
        Ok(out)
    }
}
pub(super) fn python_truth(v: &Value) -> bool {
    member_blocks::truth(v)
}
pub(super) fn merge_internal_blocks(bs: &[Value], ids: &[usize]) -> Result<Value, Error> {
    for &i in ids {
        block_common::key(bs.get(i).ok_or_else(invalid)?)?;
    }
    block_common::merge(bs, ids, false)
}
pub(super) fn merge_internal_blocks_mode(
    bs: &[Value],
    ids: &[usize],
    spaces: bool,
) -> Result<Value, Error> {
    if !spaces {
        for &i in ids {
            block_common::key(bs.get(i).ok_or_else(invalid)?)?;
        }
    }
    block_common::merge(bs, ids, spaces)
}
pub(super) fn reassemble_members(ls: &[Value], previous: &str) -> Result<String, Error> {
    member_blocks::reassemble(ls, previous)
}
pub(super) fn refine_page_blocks_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let bs = array(&v["blocks"], 4096)?.to_vec();
    for b in &bs {
        b.as_object().ok_or_else(invalid)?;
        c_validate_block(b)?;
    }
    let p = size(&v["page_size"])?;
    let images = optional(v, "image_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let grids = optional(v, "numeric_grid_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let paths = optional(v, "bullet_paths", 5000)?;
    let page = if v["page_index"].is_null() {
        None
    } else {
        Some(integer(v, "page_index", 0)?)
    };
    match v["kind"].as_str().ok_or_else(invalid)? {
        "members" => Ok(json!(member_blocks::merge(bs, p, w)?)),
        "titles" => Ok(json!(annotations::titles(bs)?)),
        "front" => Ok(json!(annotations::front(
            bs,
            p,
            if v["page_index"].is_null() {
                None
            } else {
                Some(integer(v, "page_index", 0)?)
            }
        )?)),
        "headers" => Ok(json!(annotations::headers(bs)?)),
        "compact" => Ok(json!(annotations::compact(bs, p)?)),
        "captions" => Ok(json!(captions::merge(bs, &images, &grids, w)?)),
        "wrap" => Ok(json!(caption_wrap::restore(bs, &images, p, w)?)),
        "notes" => Ok(json!(captions::split_notes(bs, &images)?)),
        "lists" => Ok(json!(local_lists::split(bs, paths, w)?)),
        "sequence" => {
            let mut bs = member_blocks::merge(bs, p, w)?;
            bs = annotations::titles(bs)?;
            bs = annotations::front(bs, p, page)?;
            bs = captions::merge(bs, &images, &grids, w)?;
            bs = annotations::headers(bs)?;
            bs = caption_wrap::restore(bs, &images, p, w)?;
            bs = annotations::compact(bs, p)?;
            bs = captions::split_notes(bs, &images)?;
            bs = local_lists::split(bs, paths, w)?;
            Ok(json!(bs))
        }
        _ => Err(invalid()),
    }
}
fn c_validate_block(b: &Value) -> Result<(), Error> {
    if !b["bbox"].is_null() {
        bbox(&b["bbox"])?;
    }
    let rows = block_common::rows(b)?;
    block_common::heights(b, false)?;
    block_common::angle(b)?;
    if ["text", "paragraph_title", "doc_title", "header", "footer"]
        .iter()
        .any(|t| b["type"] == *t)
        && rows.is_empty()
    {
        return Err(invalid());
    }
    for f in optional(b, "_font_signatures", 4096)? {
        let f = array(f, 2)?;
        if f.len() != 2 || !f[0].is_string() || !(f[1].is_i64() || f[1].is_boolean()) {
            return Err(invalid());
        }
    }
    for l in member_blocks::lines(b)? {
        l.as_object().ok_or_else(invalid)?;
        l["text"].as_str().ok_or_else(invalid)?;
        bbox(&l["bbox"])?;
        if !l["ink_bbox"].is_null() {
            bbox(&l["ink_bbox"])?;
        }
        integer(l, "source_index", 0)?;
        integer(l, "angle", 0)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn existing_python_caption_block_rules() {
        for name in [
            "native-caption-block-rules-canonical",
            "native-page-block-contracts",
        ] {
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../build/migration/{name}.json"));
            let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
            for c in f["cases"].as_array().unwrap() {
                let got = refine_page_blocks(&c["request"]);
                if c.get("error").is_some() {
                    assert!(got.is_err(), "{}", c["label"])
                } else {
                    let got = got.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
                    if let Some(d) = difference(&got, &c["expected"], "") {
                        panic!("{}: {d}", c["label"])
                    }
                }
            }
        }
    }
    #[test]
    fn existing_python_page_block_rules() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-page-block-rules.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let got = refine_page_blocks(&c["request"]);
            if c.get("error").is_some() {
                assert!(got.is_err(), "{}", c["label"])
            } else {
                let got = got.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
                if let Some(d) = difference(&got, &c["expected"], "") {
                    panic!("{}: {d}", c["label"])
                }
            }
        }
    }
    #[test]
    fn existing_python_text_set_types() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-text-type-contracts.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                build_text_blocks(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
    #[test]
    fn existing_python_text_row_edges() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-text-row-edges.json");
        let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                inspect_rows(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
    #[test]
    fn existing_python_final_text_block_chain() {
        for (name, document) in [
            ("native-final-text-blocks", false),
            ("native-final-text-blocks-chain", true),
            ("native-final-refined-blocks-paths", false),
            ("native-final-refined-blocks-chain-paths", true),
        ] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../build/migration/{name}.json"));
            let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            for c in f["cases"].as_array().unwrap() {
                let got = if document {
                    super::super::native_document::prepare_text_blocks(&c["request"])
                } else {
                    super::super::native_finalization::prepare_blocks(&c["request"])
                };
                if c.get("error").is_some() {
                    assert!(got.is_err(), "{}", c["label"])
                } else {
                    let got = got.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
                    if let Some(d) = difference(&got, &c["expected"], "") {
                        panic!("{}: {d}", c["label"])
                    }
                }
            }
        }
    }
    #[test]
    fn existing_python_text_connections() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-text-connections.json");
        let fixture: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for c in fixture["cases"].as_array().unwrap() {
            let got = inspect_connections(&c["request"]);
            if c.get("error").is_some() {
                assert!(got.is_err(), "{}", c["label"]);
            } else {
                assert_eq!(got.unwrap(), c["expected"], "{}", c["label"]);
            }
        }
    }
    #[test]
    fn existing_python_text_row_boundaries() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-text-rows.json");
        let fixture: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for c in fixture["cases"].as_array().unwrap() {
            let got = inspect_rows(&c["request"]);
            if c.get("error").is_some() {
                assert!(got.is_err(), "{}", c["label"])
            } else {
                assert_eq!(got.unwrap(), c["expected"], "{}", c["label"])
            }
        }
    }
    #[test]
    fn existing_python_grouped_footnotes() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-text-footnotes.json");
        let fixture: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for c in fixture["cases"].as_array().unwrap() {
            let got = build_grouped_footnotes(&c["request"]);
            if c.get("error").is_some() {
                assert!(got.is_err(), "{}", c["label"])
            } else {
                assert_eq!(got.unwrap(), c["expected"], "{}", c["label"])
            }
        }
    }
    fn difference(a: &Value, b: &Value, path: &str) -> Option<String> {
        if a == b {
            return None;
        }
        match (a, b) {
            (Value::Number(a), Value::Number(b))
                if (a.as_f64().unwrap() - b.as_f64().unwrap()).abs() <= 1e-9 =>
            {
                None
            }
            (Value::Object(a), Value::Object(b)) => {
                for k in a.keys().chain(b.keys()) {
                    let (Some(a), Some(b)) = (a.get(k), b.get(k)) else {
                        return Some(format!("{path}/{k}: missing"));
                    };
                    if let Some(d) = difference(a, b, &format!("{path}/{k}")) {
                        return Some(d);
                    }
                }
                None
            }
            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Some(format!("{path}: length {} vs {}", a.len(), b.len()));
                }
                for (i, (a, b)) in a.iter().zip(b).enumerate() {
                    if let Some(d) = difference(a, b, &format!("{path}/{i}")) {
                        return Some(d);
                    }
                }
                None
            }
            _ => Some(format!("{path}: {a} vs {b}")),
        }
    }
    #[test]
    fn existing_python_text_block_assembly() {
        for (name, step) in [
            ("native-text-blocks", false),
            ("native-text-block-merges", true),
            ("native-text-block-edges", true),
            ("native-text-block-identity", true),
            ("native-text-block-ties", true),
        ] {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../build/migration/{name}.json"));
            let fixture: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
            for c in fixture["cases"].as_array().unwrap() {
                let got = if step {
                    refine_blocks(&c["request"])
                } else {
                    build_text_blocks(&c["request"])
                };
                if c.get("error").is_some() {
                    assert!(got.is_err(), "{}", c["label"])
                } else {
                    let got = got.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
                    if let Some(d) = difference(&got, &c["expected"], "") {
                        panic!("{}: {d}", c["label"])
                    }
                }
            }
        }
    }
}
