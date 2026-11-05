//! Complete original document geometry plan: risk, styles, X/Y repairs and local metrics.
mod x;
mod y;
use super::*;
use crate::legacy_model::{page_containers::median, table_native_candidate::sum};
use rapiddoc_native::{
    geometry_plan::{
        geometry as g,
        geometry_lines::{summarize, World},
        geometry_risk::Risk,
        geometry_runs::{Run, Sample as NativeSample},
        geometry_style::{Document, Prepared},
        statistics::ordered_clusters,
    },
    python_printable,
    text_boundary::{python_digit, python_space},
};
use std::{collections::BTreeSet, sync::OnceLock};
type Key = (usize, i64);
#[derive(Clone, PartialEq)]
struct RunKey {
    family: String,
    size: f64,
    flags: i64,
    weight: i64,
    angle: i64,
    script: String,
}
impl RunKey {
    fn value(&self) -> Value {
        json!([
            self.family,
            self.size,
            self.flags,
            self.weight,
            self.angle,
            self.script
        ])
    }
    fn cmp(&self, b: &Self) -> std::cmp::Ordering {
        self.family
            .cmp(&b.family)
            .then(self.size.total_cmp(&b.size))
            .then(self.flags.cmp(&b.flags))
            .then(self.weight.cmp(&b.weight))
            .then(self.angle.cmp(&b.angle))
            .then(self.script.cmp(&b.script))
    }
    fn style(&self, b: &Self) -> bool {
        self.family == b.family
            && self.size == b.size
            && self.flags == b.flags
            && self.weight == b.weight
            && self.angle == b.angle
    }
}
#[derive(Clone)]
struct Sample {
    page: usize,
    line: usize,
    position: usize,
    index: i64,
    source: [f64; 4],
    tight: [f64; 4],
    origin: [f64; 2],
    local: [f64; 4],
    ink: [f64; 4],
    base: [f64; 2],
    run: usize,
    size: f64,
    anchor: bool,
    raw: Option<[f64; 4]>,
}
struct Page<'a> {
    lines: &'a [Value],
    size: [f64; 2],
    tight: BTreeMap<i64, Value>,
    origins: BTreeMap<i64, Value>,
    loose: BTreeMap<i64, Value>,
}
#[derive(Default)]
struct Plan {
    chars: BTreeMap<Key, Value>,
    lines: BTreeMap<Key, Value>,
    scales: BTreeMap<Key, f64>,
    local: BTreeSet<Key>,
    inks: BTreeMap<Key, [f64; 4]>,
    baselines: BTreeMap<Key, f64>,
    inflated: BTreeSet<usize>,
    diagnostics: Vec<Value>,
}
impl Plan {
    fn value(&self, keys: &[RunKey]) -> Value {
        let rows = |m: &BTreeMap<Key, Value>, char_key: bool| {
            m.iter()
                .map(|(&(p, i), v)| {
                    let mut v = v.clone();
                    v["page_index"] = json!(p);
                    v[if char_key { "char_idx" } else { "source_index" }] = json!(i);
                    v
                })
                .collect::<Vec<_>>()
        };
        let mut inflated = self.inflated.iter().map(|&i| &keys[i]).collect::<Vec<_>>();
        inflated.sort_by(|a, b| a.cmp(b));
        json!({"char_repairs":rows(&self.chars,true),"line_repairs":rows(&self.lines,false),"line_style_scales":self.scales.iter().map(|(&(p,i),s)|json!({"page_index":p,"source_index":i,"scale":s})).collect::<Vec<_>>(),"local_metric_repair_sources":self.local.iter().map(|&(p,i)|json!({"page_index":p,"source_index":i})).collect::<Vec<_>>(),"line_ink_bboxes":self.inks.iter().map(|(&(p,i),b)|json!({"page_index":p,"source_index":i,"bbox":b})).collect::<Vec<_>>(),"line_baselines":self.baselines.iter().map(|(&(p,i),b)|json!({"page_index":p,"source_index":i,"baseline":b})).collect::<Vec<_>>(),"style_inflated_runs":inflated.iter().map(|k|k.value()).collect::<Vec<_>>(),"document_style_anomaly":!self.inflated.is_empty(),"run_diagnostics":self.diagnostics})
    }
}
fn category(c: char) -> u64 {
    static R: OnceLock<Vec<[u64; 3]>> = OnceLock::new();
    let ranges = R.get_or_init(|| {
        let v: Value = serde_json::from_str(include_str!("unicode.json")).unwrap();
        serde_json::from_value(v["ranges"].clone()).unwrap()
    });
    let n = c as u64;
    let i = ranges.partition_point(|r| r[0] <= n);
    if i > 0 && n <= ranges[i - 1][1] {
        ranges[i - 1][2]
    } else {
        0
    }
}
fn anchor(s: &str) -> bool {
    !s.is_empty()
        && s.chars().all(python_printable)
        && !s.chars().all(python_space)
        && s.chars().next().is_some_and(|c| category(c) > 0)
}
fn script(s: &str) -> String {
    if !s.is_empty() && s.chars().all(|c| c.is_ascii_alphabetic()) {
        return "latin".into();
    }
    if !s.is_empty() && s.chars().all(python_digit) {
        return "digit".into();
    }
    let c = s.chars().next().unwrap_or('\0');
    let n = c as u32;
    if (0x3400..=0x9fff).contains(&n)
        || (0xf900..=0xfaff).contains(&n)
        || (0x3040..=0x30ff).contains(&n)
        || (0xac00..=0xd7af).contains(&n)
    {
        "cjk"
    } else {
        match category(c) {
            1 => "letter",
            2 => "number",
            _ => "other",
        }
    }
    .into()
}
fn float(v: &Value) -> Option<f64> {
    crate::legacy_model::table_text_styles::numeric(Some(v))
}
fn int(v: &Value) -> i64 {
    v.as_i64()
        .or_else(|| v.as_bool().map(i64::from))
        .or_else(|| v.as_f64().map(|n| n.trunc() as i64))
        .or_else(|| {
            v.as_str()
                .and_then(|s| {
                    crate::legacy_model::table_native_text::numeric_string(s)
                        .ok()
                        .flatten()
                })
                .and_then(|s| s.parse().ok())
        })
        .unwrap_or(0)
}
fn font(c: &Value, a: i64, text: &str) -> Result<(RunKey, f64), Error> {
    let f = &c["font"];
    if !f.is_null() && !f.is_object() && crate::legacy_page::truth(f) {
        return Err(invalid());
    }
    let name = super::super::table_native_multiline::spatial::raw_string(f.get("name"), true);
    let family = preparation::family(&Some((
        if name.is_empty() {
            "<unknown>".into()
        } else {
            name
        },
        json!(0),
    )))
    .unwrap_or_else(|| "<unknown>".into());
    let size = float(&f["size"]).unwrap_or(0.);
    let weight = float(&f["weight"]).unwrap_or(0.);
    if size.abs() > 1e9 || weight.abs() > 1e9 {
        return Err(limit());
    }
    for k in ["size", "weight"] {
        if let Some(raw) = f[k].as_str() {
            let parsed = crate::legacy_model::table_native_text::numeric_string(raw)?
                .and_then(|s| s.parse::<f64>().ok());
            if parsed.is_some_and(|n| n.is_infinite() || k == "size" && n.is_nan()) {
                return Err(invalid());
            }
        }
    }
    Ok((
        RunKey {
            family,
            size: (size * 4.).round_ties_even() / 4.,
            flags: int(&f["flags"]),
            weight: (weight / 100.).round_ties_even() as i64 * 100,
            angle: a,
            script: script(text),
        },
        size,
    ))
}
fn maps(v: &Value, key: &str, value: &str) -> Result<BTreeMap<i64, Value>, Error> {
    let mut out = BTreeMap::new();
    for r in optional(v, key, 20000)? {
        out.insert(
            r["char_idx"].as_i64().ok_or_else(invalid)?,
            r[value].clone(),
        );
    }
    Ok(out)
}
fn rect(v: &Value) -> Result<Option<[f64; 4]>, Error> {
    Ok(coerce(v)?.map(|b| b.n))
}
fn origin(v: &Value) -> Result<Option<[f64; 2]>, Error> {
    let Some(a) = v.as_array().filter(|a| a.len() == 2) else {
        return Ok(None);
    };
    let Some(x) = float(&a[0]) else {
        return Ok(None);
    };
    let Some(y) = float(&a[1]) else {
        return Ok(None);
    };
    if x.abs() > 1e9 || y.abs() > 1e9 {
        return Err(limit());
    }
    Ok(Some([x, y]))
}
fn union(v: impl IntoIterator<Item = [f64; 4]>) -> [f64; 4] {
    let mut it = v.into_iter();
    let mut b = it.next().unwrap();
    for c in it {
        for i in 0..4 {
            if i < 2 && c[i] < b[i] || i >= 2 && c[i] > b[i] {
                b[i] = c[i]
            }
        }
    }
    b
}
fn quantile(mut v: Vec<f64>, q: f64) -> f64 {
    if v.is_empty() {
        return 0.;
    }
    v.sort_by(f64::total_cmp);
    let p = q.clamp(0., 1.) * (v.len() - 1) as f64;
    let lo = p.floor() as usize;
    let hi = p.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        let w = p - lo as f64;
        v[lo] * (1. - w) + v[hi] * w
    }
}
fn med(v: Vec<f64>) -> f64 {
    if v.is_empty() {
        0.
    } else {
        median(v)
    }
}
fn clipped(b: [f64; 4], p: [f64; 2]) -> Option<[f64; 4]> {
    g::clip(Some(b), p)
}
fn world(b: [f64; 4], p: [f64; 2], a: i64) -> [f64; 4] {
    match a {
        90 => [p[0] - b[3], b[0], p[0] - b[1], b[2]],
        180 => [p[0] - b[2], p[1] - b[3], p[0] - b[0], p[1] - b[1]],
        270 => [b[1], p[1] - b[2], b[3], p[1] - b[0]],
        _ => b,
    }
}
fn physical(samples: &[Sample]) -> BTreeMap<(usize, usize), Vec<usize>> {
    let mut groups = BTreeMap::new();
    for (i, s) in samples.iter().enumerate() {
        groups
            .entry((s.page, s.line))
            .or_insert_with(Vec::new)
            .push(i);
    }
    groups
}
fn skip(l: &Value) -> bool {
    l["formula_candidate_only"] == true
        || l["compact_formula_cluster"] == true
        || l["restored_inline_cluster"] == true
}
fn collect(
    pages: &[Page],
    keys: &mut Vec<RunKey>,
    anchors: bool,
    w: &mut Work,
) -> Result<(Vec<Sample>, Vec<(Key, Vec<usize>)>), Error> {
    let mut samples = Vec::new();
    let mut by_line: Vec<(Key, Vec<usize>)> = Vec::new();
    for (p, page) in pages.iter().enumerate() {
        for (li, l) in page.lines.iter().enumerate() {
            let a = integer(l, "angle", 0)?;
            let angle = i32::try_from(a).map_err(|_| invalid())?;
            for (pos, c) in optional(l, "chars", 20000)?.iter().enumerate() {
                let text =
                    super::super::table_native_multiline::spatial::raw_string(c.get("char"), true);
                w.charge(text.len() + keys.len() + 1)?;
                if if anchors {
                    !anchor(&text)
                } else {
                    text.is_empty()
                        || !text.chars().all(python_printable)
                        || text.chars().all(python_space)
                } {
                    continue;
                }
                let Some(index) = c["char_idx"].as_i64() else {
                    continue;
                };
                let raw = rect(&c["bbox"])?;
                let tight = page.tight.get(&index).map(rect).transpose()?.flatten();
                let side = page.loose.get(&index).map(rect).transpose()?.flatten();
                let rotation = float(&c["rotation"]).unwrap_or(if c["rotation"].is_null() {
                    0.
                } else {
                    f64::NAN
                });
                let source = if rotation.is_finite() && rotation.abs() <= 1e-9 {
                    raw
                } else {
                    match (raw, side) {
                        (Some(r), Some(s)) => Some(
                            if s[2] - s[0]
                                > 1.6
                                    * (r[2] - r[0])
                                        .max(tight.map_or(0., |t| t[2] - t[0]))
                                        .max(0.1)
                            {
                                r
                            } else {
                                s
                            },
                        ),
                        (r, s) => s.or(r),
                    }
                };
                let Some(source) = g::clip(source, page.size) else {
                    continue;
                };
                let Some(tight) = g::clip(tight, page.size) else {
                    continue;
                };
                let Some(origin) = page.origins.get(&index).map(origin).transpose()?.flatten()
                else {
                    continue;
                };
                let (k, size) = font(c, a, &text)?;
                let run = keys.iter().position(|x| *x == k).unwrap_or_else(|| {
                    keys.push(k);
                    keys.len() - 1
                });
                let i = samples.len();
                samples.push(Sample {
                    page: p,
                    line: li,
                    position: pos,
                    index,
                    source,
                    tight,
                    origin,
                    local: g::rotate(source, page.size, angle),
                    ink: g::rotate(tight, page.size, angle),
                    base: g::rotate_point(origin, page.size, angle),
                    run,
                    size,
                    anchor: anchor(&text),
                    raw,
                });
                let k = (p, integer(l, "source_index", 0)?);
                if let Some((_, g)) = by_line.iter_mut().find(|(k1, _)| *k1 == k) {
                    g.push(i)
                } else {
                    by_line.push((k, vec![i]))
                }
                if samples.len() > rapiddoc_core::resources::limit(100000) {
                    return Err(limit());
                }
            }
        }
    }
    Ok((samples, by_line))
}
fn line_repair(
    source: [f64; 4],
    layout: [f64; 4],
    ink: Option<[f64; 4]>,
    baseline: Option<f64>,
    em: f64,
) -> Value {
    json!({"source_bbox":source,"layout_bbox":layout,"ink_bbox":ink,"baseline":baseline,"em_height":em,"state":"healthy","confidence":1.,"split_y_candidate":false,"repaired_char_count":0,"y_intrusion_ratio":0.,"run_key":null})
}
fn diagnostics(prepared: &Prepared, keys: &[RunKey]) -> Vec<Value> {
    let mut ids = (0..keys.len()).collect::<Vec<_>>();
    ids.sort_by(|&a, &b| keys[a].cmp(&keys[b]));
    ids.into_iter().map(|i|{let r=&prepared.runs[i];let n=r.ratios.len();json!({"run_key":keys[i].value(),"sample_count":r.members.len(),"reliable_pair_count":n,"median_ratio":if n>0{json!(med(r.ratios.clone()))}else{Value::Null},"p90_ratio":if n>0{json!(quantile(r.ratios.clone(),0.9))}else{Value::Null},"ratio_gt_1_35_share":if n>0{r.ratios.iter().filter(|v|**v>1.35).count()as f64/n as f64}else{0.},"next_tight_overlap_share":if n>0{r.overlaps.iter().filter(|v|**v).count()as f64/n as f64}else{0.},"strong_x_bad":r.strong,"sibling_x_bad":r.sibling,"style_y_bad":prepared.style.inflated.contains(&i)})}).collect()
}
pub(super) fn build(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let mut pages = Vec::new();
    for p in array(&v["pages"], 1000)? {
        let source = &p["source"];
        let ls = array(&source["lines"], 4096)?;
        for l in ls {
            bbox(&l["bbox"])?;
            integer(l, "source_index", 0)?;
        }
        let geometry = &p["geometry"];
        pages.push(Page {
            lines: ls,
            size: size(&source["page_size"])?,
            tight: maps(geometry, "tight_bboxes", "bbox")?,
            origins: maps(geometry, "origins", "origin")?,
            loose: maps(geometry, "loose_bboxes", "bbox")?,
        });
    }
    let mut plan = Plan::default();
    let mut keys = Vec::new();
    let original_loose = pages
        .iter()
        .map(|p| {
            p.loose
                .iter()
                .map(|(&i, b)| json!({"char_idx":i,"bbox":b}))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    if !pages
        .iter()
        .any(|p| !p.tight.is_empty() && !p.origins.is_empty())
    {
        return Ok(json!({"plan":plan.value(&keys),"remaining_loose_bboxes":original_loose}));
    }
    let (anchors, _) = collect(&pages, &mut keys, true, w)?;
    let mut risk = Risk::default();
    for ((p, li), ids) in &physical(&anchors) {
        w.charge(ids.len() * 20 + 1)?;
        let l = &pages[*p].lines[*li];
        let entries = ids
            .iter()
            .map(|&i| {
                let s = &anchors[i];
                (s.run, s.local, s.ink, s.base, s.size)
            })
            .collect();
        if risk
            .add_line(
                *p,
                integer(l, "source_index", 0)?,
                l.get("effective_height").map_or(Ok(0.), num)?,
                integer(l, "angle", 0)? != 0 || skip(l),
                entries,
            )
            .ok_or_else(invalid)?
        {
            break;
        }
    }
    let (layout, style) = risk.finish();
    if !layout && !style {
        return Ok(
            json!({"plan":plan.value(&keys),"remaining_loose_bboxes":pages.iter().map(|_|Vec::<Value>::new()).collect::<Vec<_>>()}),
        );
    }
    keys.clear();
    let (mut samples, by_line) = collect(&pages, &mut keys, false, w)?;
    let mut doc = Document::default();
    for (&(page, li), ids) in &physical(&samples) {
        let line = &pages[page].lines[li];
        doc.append(
            page as i64,
            integer(line, "source_index", 0)?,
            line.get("effective_height").map_or(Ok(0.), num)?,
            ids.iter()
                .map(|&i| {
                    let s = &samples[i];
                    NativeSample {
                        run: s.run,
                        position: s.position as i64,
                        source: s.local,
                        tight: s.ink,
                        origin: s.base,
                        size: s.size,
                        anchor: s.anchor,
                    }
                })
                .collect(),
        );
    }
    let mut families = Vec::new();
    let mut names = Vec::new();
    for k in &keys {
        let i = names
            .iter()
            .position(|n| n == &k.family)
            .unwrap_or_else(|| {
                names.push(k.family.clone());
                names.len() - 1
            });
        families.push(i)
    }
    w.charge(samples.len().saturating_mul(30))?;
    let legacy = layout.then(|| {
        samples
            .iter()
            .map(|s| (s.raw, pages[s.page].size, keys[s.run].angle as i32))
            .collect()
    });
    let prepared = doc.prepare(&families, legacy).ok_or_else(invalid)?;
    for (i, s) in samples.iter_mut().enumerate() {
        s.anchor = prepared.samples[i].anchor;
        if let Some(Some(b)) = prepared.restored.get(i) {
            s.source = *b;
            s.local = prepared.samples[i].source;
        }
    }
    plan.inflated
        .extend(prepared.style.inflated.iter().copied());
    for &(i, scale) in &prepared.style.scales {
        let (p, source) = prepared.line_keys[i];
        plan.scales.insert((p as usize, source), scale);
    }
    plan.diagnostics = diagnostics(&prepared, &keys);
    if layout {
        let world = samples
            .iter()
            .map(|s| {
                let l = &pages[s.page].lines[s.line];
                World {
                    tight: s.tight,
                    canonical: l["formula_candidate_only"] != true
                        && l["compact_formula_cluster"] != true,
                    y_eligible: integer(l, "angle", 0).unwrap_or(0) == 0 && !skip(l),
                }
            })
            .collect::<Vec<_>>();
        let (inks, baselines, y_risk) = summarize(&prepared, &world, true).ok_or_else(invalid)?;
        for ((p, i), b) in inks {
            plan.inks.insert((p as usize, i), b);
        }
        for ((p, i), b) in baselines {
            plan.baselines.insert((p as usize, i), b);
        }
        x::repair(
            &mut plan,
            &samples,
            &by_line,
            &prepared.runs,
            &keys,
            &pages,
            w,
        )?;
        if y_risk {
            y::repair(
                &mut plan,
                &samples,
                &by_line,
                &keys,
                &pages,
                &prepared
                    .runs
                    .iter()
                    .enumerate()
                    .filter_map(|(i, r)| (r.strong || r.sibling).then_some(i))
                    .collect(),
                w,
            )?;
            y::extreme(&mut plan, &samples, &by_line, &keys, &pages, w)?;
        }
    }
    Ok(
        json!({"plan":plan.value(&keys),"remaining_loose_bboxes":pages.iter().map(|_|Vec::<Value>::new()).collect::<Vec<_>>()}),
    )
}
