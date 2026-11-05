//! 旧二维inline簇连接、宿主选择、物化与多行cell脚本聚合。
use super::super::table_native_candidate::sum;
use super::{lanes, preparation};
use crate::legacy_page::{eq, invalid, limit, truth, validate_tree, Work};
use lanes::{flag, median, num, number, rect};
use rapiddoc_core::Error;
use rapiddoc_native::{
    python_decimal,
    text_boundary::{python_alnum, python_space},
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};
type Box4 = [f64; 4];
type ClusterDecision = (bool, Option<bool>);
fn defaults() -> &'static Value {
    static D: OnceLock<Value> = OnceLock::new();
    D.get_or_init(|| serde_json::from_str(include_str!("cluster_defaults.json")).unwrap())
}
fn integer(v: &Value) -> Option<i64> {
    v.as_i64().or_else(|| v.as_bool().map(i64::from))
}
fn source(v: &Value) -> i64 {
    v["source_index"].as_i64().unwrap()
}
fn angle(v: &Value) -> i64 {
    v["angle"].as_i64().unwrap()
}
fn rotate(b: Box4, p: [f64; 2], a: i64) -> Box4 {
    match a {
        90 => [b[1], p[0] - b[2], b[3], p[0] - b[0]],
        180 => [p[0] - b[2], p[1] - b[3], p[0] - b[0], p[1] - b[1]],
        270 => [p[1] - b[3], b[0], p[1] - b[1], b[2]],
        _ => b,
    }
}
fn union(a: Box4, b: Box4) -> Box4 {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}
fn overlap(a: Box4, b: Box4, k: usize) -> f64 {
    (a[k + 2].min(b[k + 2]) - a[k].max(b[k])).max(0.) / (a[k + 2] - a[k]).min(b[k + 2] - b[k])
}
fn intersects(a: Box4, b: Box4) -> bool {
    a[2].min(b[2]) > a[0].max(b[0]) && a[3].min(b[3]) > a[1].max(b[1])
}
fn eff(v: &Value, b: Box4) -> Result<f64, Error> {
    let h = num(v, "effective_height", 0.)?;
    let em = num(v, "em_height", 0.)?;
    Ok(0.1f64.max(if flag(v, "style_scale_repaired")? && em > 0. {
        em
    } else if h != 0. {
        h
    } else {
        b[3] - b[1]
    }))
}
fn page(v: &Value) -> Result<[f64; 2], Error> {
    let a = v.as_array().filter(|a| a.len() == 2).ok_or_else(invalid)?;
    let p = [number(&a[0])?, number(&a[1])?];
    if p.iter().any(|&n| n <= 0.) {
        return Err(invalid());
    }
    Ok(p)
}
fn canonical(v: &Value) -> Result<Value, Error> {
    canonical_bounded(v, 1500)
}
fn canonical_bounded(v: &Value, maximum: usize) -> Result<Value, Error> {
    let maximum = crate::legacy_page::budget::records(maximum);
    let m = v.as_object().ok_or_else(invalid)?;
    let mut out = defaults().clone();
    for k in ["text", "bbox", "angle", "source_index"] {
        out[k] = m.get(k).cloned().ok_or_else(invalid)?;
    }
    for (k, x) in m {
        if defaults().get(k).is_some() {
            out[k] = x.clone()
        }
    }
    if !out["text"].is_string()
        || out["source_index"].as_i64().is_none()
        || out["angle"].as_i64().is_none()
    {
        return Err(invalid());
    }
    rect(&out["bbox"])?;
    if out["source_bbox"].is_null() {
        out["source_bbox"] = out["bbox"].clone()
    }
    for k in ["source_bbox", "ink_bbox"] {
        if !out[k].is_null() {
            rect(&out[k])?;
        }
    }
    for k in [
        "caption_start",
        "paragraph_formula_context",
        "split_from_row",
        "preserve_split_boundary",
        "restored_inline_cluster",
        "compact_formula_cluster",
        "formula_candidate_only",
        "style_scale_repaired",
        "paragraph_terminal",
        "structural_title",
        "explicit_section_title",
        "title_suppressed",
    ] {
        flag(&out, k)?;
    }
    for k in ["effective_height", "em_height", "font_coverage"] {
        num(&out, k, 0.)?;
    }
    for k in [
        "visual_row_id",
        "run_index",
        "paragraph_group",
        "title_band_id",
    ] {
        if !out[k].is_null() && out[k].as_i64().is_none() {
            return Err(invalid());
        }
    }
    if !out["semantic_type"].is_null() && !out["semantic_type"].is_string() {
        return Err(invalid());
    }
    if !out["chars"]
        .as_array()
        .is_some_and(|a| a.len() <= maximum && a.iter().all(Value::is_object))
    {
        return Err(limit());
    }
    let regions = out["inline_math_regions"]
        .as_array()
        .filter(|a| a.len() <= rapiddoc_core::resources::limit(2000))
        .ok_or_else(limit)?;
    for b in regions {
        rect(b)?;
    }
    let words = out["native_math_words"].as_array().ok_or_else(invalid)?;
    let mut ordered = BTreeSet::new();
    for s in words {
        ordered.insert(s.as_str().ok_or_else(invalid)?.to_owned());
    }
    out["native_math_words"] = json!(ordered);
    Ok(out)
}
fn connection(
    a: &Value,
    b: &Value,
    bounds: (Box4, Box4),
    h: f64,
    width: f64,
    tables: &[Box4],
    work: &mut Work,
) -> Result<bool, Error> {
    let (ab, bb) = bounds;
    work.charge(1)?;
    if angle(a) != angle(b)
        || a["formula_candidate_only"] != b["formula_candidate_only"]
        || a["semantic_type"] != b["semantic_type"]
        || overlap(ab, bb, 1) < 0.55
    {
        return Ok(false);
    }
    let ap = rect(&a["bbox"])?;
    let bp = rect(&b["bbox"])?;
    let (acx, acy) = ((ap[0] + ap[2]) / 2., (ap[1] + ap[3]) / 2.);
    let (bcx, bcy) = ((bp[0] + bp[2]) / 2., (bp[1] + bp[3]) / 2.);
    let connector = [
        acx.min(bcx) - 0.1,
        acy.min(bcy) - 0.1,
        acx.max(bcx) + 0.1,
        acy.max(bcy) + 0.1,
    ];
    for &t in tables {
        work.charge(1)?;
        if intersects(connector, t) {
            return Ok(false);
        }
    }
    let ag = if truth(&a["source_bbox"]) {
        rect(&a["source_bbox"])?
    } else {
        ab
    };
    let bg = if truth(&b["source_bbox"]) {
        rect(&b["source_bbox"])?
    } else {
        bb
    };
    let (l, r) = if ag[0] <= bg[0] { (ag, bg) } else { (bg, ag) };
    if (flag(a, "style_scale_repaired")? || flag(b, "style_scale_repaired")?)
        && l[2] <= 0.5 * width
        && r[0] >= 0.5 * width
        && r[0] - l[2] >= 0.02 * width
    {
        return Ok(false);
    }
    let gap = (ab[0] - bb[2]).max(bb[0] - ab[2]).max(0.);
    if a["visual_row_id"] == b["visual_row_id"]
        && (flag(a, "split_from_row")? || flag(b, "split_from_row")?)
    {
        return Ok(gap <= 3. * h);
    }
    Ok(overlap(ab, bb, 0) > 0. || gap <= 1.5 * eff(a, ab)?.max(eff(b, bb)?))
}
fn prose(text: &str) -> bool {
    if text
        .chars()
        .filter(|&c| matches!(c as u32, 0x3400..=0x9fff))
        .count()
        >= 2
    {
        return true;
    }
    let cs: Vec<_> = text.chars().collect();
    let mut i = 0;
    let mut words = 0;
    while i < cs.len() {
        if !cs[i].is_ascii_alphabetic() {
            i += 1;
            continue;
        }
        let start = i;
        while i < cs.len() && cs[i].is_ascii_alphabetic() {
            i += 1
        }
        let word = |c: char| python_alnum(c) || c == '_';
        if i - start >= 2 && (start == 0 || !word(cs[start - 1])) && (i == cs.len() || !word(cs[i]))
        {
            words += 1;
            if words >= 2 {
                return true;
            }
        }
    }
    false
}
fn paired_host(
    ids: &[usize],
    lines: &[Value],
    local: &[Box4],
    h: f64,
    work: &mut Work,
) -> Result<bool, Error> {
    if ids.len() != 2
        || ids.iter().any(|&i| {
            angle(&lines[i]) != 0
                || truth(&lines[i]["preserve_split_boundary"])
                || truth(&lines[i]["semantic_type"])
        })
    {
        return Ok(false);
    }
    let mut host = ids[0];
    for &i in &ids[1..] {
        if local[i][2] - local[i][0] > local[host][2] - local[host][0] {
            host = i
        }
    }
    if !prose(lines[host]["text"].as_str().unwrap()) || local[host][2] - local[host][0] < 4. * h {
        return Ok(false);
    }
    let mut owned = Vec::new();
    for &i in ids {
        for c in lines[i]["chars"].as_array().unwrap() {
            let Some(id) = integer(&c["char_idx"]) else {
                return Ok(false);
            };
            owned.push((id, c, i));
        }
    }
    if owned.is_empty() {
        return Ok(false);
    }
    owned.sort_by_key(|(id, _, _)| *id);
    work.charge(owned.len().saturating_mul(owned.len()))?;
    let mut tight = serde_json::Map::new();
    let mut origins = serde_json::Map::new();
    for (id, c, _) in &owned {
        for (k, map) in [("tight_bbox", &mut tight), ("origin", &mut origins)] {
            if !c[k].is_null() {
                map.insert(id.to_string(), c[k].clone());
            }
        }
    }
    let chars: Vec<_> = owned.iter().map(|(_, c, _)| *c).collect();
    let paired = rapiddoc_native::script_geometry::refine::paired_indices(
        &json!({"chars":chars,"tight_bboxes":tight,"origins":origins}),
    )?;
    Ok(paired.len() == 2 && owned[paired[0]].2 != owned[paired[1]].2)
}
fn kind(
    ids: &[usize],
    lines: &[Value],
    local: &[Box4],
    lane_width: f64,
    h: f64,
    tables: &[Box4],
    work: &mut Work,
) -> Result<Option<ClusterDecision>, Error> {
    if ids.len() < 2 {
        return Ok(None);
    }
    for &i in ids {
        if flag(&lines[i], "caption_start")? {
            let seed = rect(&lines[i]["bbox"])?;
            let height = eff(&lines[i], seed)?;
            for &j in ids {
                work.charge(1)?;
                let other = rect(&lines[j]["bbox"])?;
                if j != i && other[2] <= seed[0] + 0.5 * height && other[0] < seed[0] - height {
                    return Ok(None);
                }
            }
        }
    }
    let rows: BTreeSet<_> = ids
        .iter()
        .filter_map(|&i| lines[i]["visual_row_id"].as_i64())
        .collect();
    if rows.len() < 2 {
        return Ok(None);
    }
    for &i in ids {
        for &t in tables {
            work.charge(1)?;
            if intersects(rect(&lines[i]["bbox"])?, t) {
                return Ok(None);
            }
        }
    }
    let all = ids.iter().map(|&i| local[i]).reduce(union).unwrap();
    if all[3] - all[1] > 3. * h {
        return Ok(None);
    }
    let mut fragment = false;
    let mut body = false;
    for &i in ids {
        let e = eff(&lines[i], local[i])?;
        let cov = num(&lines[i], "font_coverage", 0.)?;
        fragment |= e <= 0.88 * h || cov < 0.75;
        body |= local[i][2] - local[i][0] >= (4. * h).max(0.35 * lane_width)
            && e >= 0.8 * h
            && cov >= 0.75;
    }
    if !fragment {
        return Ok(None);
    }
    if body {
        return Ok(Some((false, None)));
    }
    let paired = paired_host(ids, lines, local, h, work)?;
    if paired {
        return Ok(Some((false, Some(true))));
    }
    if ids.len() >= 3 && rows.len() >= 3 && all[2] - all[0] <= 0.6 * lane_width {
        return Ok(Some((true, Some(false))));
    }
    Ok(None)
}
fn label(text: &str) -> bool {
    let text = text.trim_matches(python_space);
    let cs: Vec<_> = text.chars().collect();
    if cs.len() < 2 || !matches!(cs[cs.len() - 1], '.' | ')') {
        return false;
    }
    let a = &cs[..cs.len() - 1];
    (a.len() == 1 && a[0].is_ascii_alphabetic())
        || ((1..=2).contains(&a.len()) && a.iter().all(|&c| python_decimal(c).is_some()))
}
fn materialize(
    ids: &[usize],
    lines: &[Value],
    local: &[Box4],
    p: [f64; 2],
    h: f64,
    decision: ClusterDecision,
    work: &mut Work,
    full: bool,
) -> Result<Value, Error> {
    let (compact, paired) = decision;
    work.charge(ids.len().saturating_mul(ids.len()))?;
    let all = ids.iter().map(|&i| local[i]).reduce(union).unwrap();
    let center = (all[1] + all[3]) / 2.;
    let mut host = ids[0];
    let mut best = None;
    for &i in ids {
        let b = local[i];
        let vertical: Vec<_> = ids
            .iter()
            .filter(|&&j| j != i)
            .map(|&j| (b[3].min(local[j][3]) - b[1].max(local[j][1])).max(0.))
            .collect();
        let same = ids
            .iter()
            .filter(|&&j| {
                j != i
                    && !lines[i]["visual_row_id"].is_null()
                    && lines[i]["visual_row_id"] == lines[j]["visual_row_id"]
            })
            .count();
        let score = [
            f64::from(u8::from(
                eff(&lines[i], b)? >= 0.8 * h && num(&lines[i], "font_coverage", 0.)? >= 0.75,
            )),
            same as f64,
            sum(&vertical),
            b[2] - b[0],
            -((b[1] + b[3]) / 2. - center).abs(),
            -(source(&lines[i]) as f64),
        ];
        if best.is_none_or(|old: [f64; 6]| score > old) {
            best = Some(score);
            host = i;
        }
    }
    let mut ordered = ids.to_vec();
    ordered.sort_by_key(|&i| source(&lines[i]));
    let separator = if match paired {
        Some(value) => value,
        None => paired_host(ids, lines, local, h, work)?,
    } {
        ""
    } else {
        " "
    };
    let text = ordered
        .iter()
        .filter_map(|&i| {
            let t = lines[i]["text"]
                .as_str()
                .unwrap()
                .trim_matches(python_space);
            (!t.is_empty()).then_some(t)
        })
        .collect::<Vec<_>>()
        .join(separator);
    let mut out = defaults().clone();
    out["text"] = json!(text);
    out["bbox"] = json!(ordered
        .iter()
        .map(|&i| rect(&lines[i]["bbox"]).unwrap())
        .reduce(union)
        .unwrap());
    out["source_bbox"] = json!(ordered
        .iter()
        .map(|&i| rect(if truth(&lines[i]["source_bbox"]) {
            &lines[i]["source_bbox"]
        } else {
            &lines[i]["bbox"]
        })
        .unwrap())
        .reduce(union)
        .unwrap());
    out["ink_bbox"] = ordered
        .iter()
        .filter(|&&i| !lines[i]["ink_bbox"].is_null())
        .map(|&i| rect(&lines[i]["ink_bbox"]).unwrap())
        .reduce(union)
        .map_or(Value::Null, |b| json!(b));
    out["source_index"] = json!(source(&lines[ordered[0]]));
    for k in [
        "angle",
        "baseline",
        "visual_row_id",
        "run_index",
        "effective_height",
        "font_signature",
        "font_coverage",
        "dominant_font_weight",
        "median_glyph_width",
        "semantic_type",
    ] {
        out[k] = lines[host][k].clone();
    }
    out["em_height"] = if truth(&lines[host]["em_height"]) {
        lines[host]["em_height"].clone()
    } else {
        lines[host]["effective_height"].clone()
    };
    for k in ["leading_emphasis_width", "leading_typography_width"] {
        out[k] = lines[ordered[0]][k].clone();
    }
    out["paragraph_terminal"] = lines[*ordered.last().unwrap()]["paragraph_terminal"].clone();
    for k in [
        "caption_start",
        "paragraph_formula_context",
        "split_from_row",
        "preserve_split_boundary",
        "style_scale_repaired",
    ] {
        out[k] = json!(ordered.iter().any(|&i| truth(&lines[i][k])));
    }
    out["restored_inline_cluster"] = json!(true);
    out["compact_formula_cluster"] = json!(compact);
    out["formula_candidate_only"] = json!(ordered
        .iter()
        .all(|&i| truth(&lines[i]["formula_candidate_only"])));
    out["chars"] = json!(ordered
        .iter()
        .flat_map(|&i| lines[i]["chars"].as_array().unwrap())
        .collect::<Vec<_>>());
    let words: BTreeSet<_> = ordered
        .iter()
        .flat_map(|&i| lines[i]["native_math_words"].as_array().unwrap())
        .map(|s| s.as_str().unwrap().to_owned())
        .collect();
    out["native_math_words"] = json!(words);
    let mut regions: Vec<_> = ordered
        .iter()
        .flat_map(|&i| {
            lines[i]["inline_math_regions"]
                .as_array()
                .unwrap()
                .iter()
                .cloned()
        })
        .collect();
    regions.extend(
        ordered
            .iter()
            .filter(|&&i| compact || i != host)
            .map(|&i| lines[i]["bbox"].clone()),
    );
    if regions.len() > rapiddoc_core::resources::limit(2000) {
        return Err(limit());
    }
    out["inline_math_regions"] = json!(regions);
    let mut visual = ordered.clone();
    visual.sort_by(|&i, &j| {
        rect(&lines[i]["bbox"]).unwrap()[0]
            .partial_cmp(&rect(&lines[j]["bbox"]).unwrap()[0])
            .unwrap()
    });
    if visual.len() == 2
        && label(lines[visual[0]]["text"].as_str().unwrap())
        && eq(
            &lines[visual[0]]["font_signature"],
            &lines[visual[1]]["font_signature"],
        )
    {
        out["native_title_label_left"] = lines[visual[1]]["native_title_label_left"].clone()
    }
    for k in ["paragraph_group", "title_band_id"] {
        let value = &lines[ordered[0]][k];
        if !value.is_null() && ordered.iter().all(|&i| eq(&lines[i][k], value)) {
            out[k] = value.clone();
            if k == "title_band_id" {
                for f in ["structural_title", "explicit_section_title"] {
                    out[f] = json!(ordered.iter().all(|&i| truth(&lines[i][f])));
                }
            }
        }
    }
    out["title_suppressed"] = json!(ordered
        .iter()
        .all(|&i| truth(&lines[i]["title_suppressed"])));
    if !out["chars"].as_array().unwrap().is_empty() {
        let m = if full {
            preparation::page_metrics(&out, p)?
        } else {
            preparation::metrics(&out, p)?
        };
        out.as_object_mut()
            .unwrap()
            .extend(m.as_object().unwrap().clone());
    }
    Ok(out)
}
/// Materialize members already proved by another original classifier, without
/// rerunning the overlapping-component acceptance rules.
pub(in crate::processing) fn materialize_members(
    members: &[Value],
    p: [f64; 2],
    h: f64,
    compact: bool,
    work: &mut Work,
) -> Result<Value, Error> {
    if members.is_empty() || members.len() > rapiddoc_core::resources::limit(512) {
        return Err(limit());
    }
    let lines = members
        .iter()
        .map(canonical)
        .collect::<Result<Vec<_>, _>>()?;
    let local = lines
        .iter()
        .map(|l| rect(&l["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let ids = (0..lines.len()).collect::<Vec<_>>();
    materialize(&ids, &lines, &local, p, h, (compact, None), work, false)
}
/// Original cluster constructor with explicit upright member geometry for page formula recovery.
pub(in crate::processing) fn materialize_page_members(
    members: &[Value],
    geometry: &[Value],
    p: [f64; 2],
    h: f64,
    compact: bool,
    work: &mut Work,
) -> Result<Value, Error> {
    if members.is_empty()
        || members.len() > rapiddoc_core::resources::limit(4096)
        || members.len() != geometry.len()
    {
        return Err(limit());
    }
    let lines = members
        .iter()
        .map(|l| canonical_bounded(l, 20000))
        .collect::<Result<Vec<_>, _>>()?;
    let local = geometry.iter().map(rect).collect::<Result<Vec<_>, _>>()?;
    let ids = (0..lines.len()).collect::<Vec<_>>();
    let result = materialize(&ids, &lines, &local, p, h, (compact, None), work, true)?;
    let mut out = super::super::native_lines::defaults();
    out.as_object_mut()
        .unwrap()
        .extend(result.as_object().unwrap().clone());
    Ok(out)
}
fn root(parents: &mut [usize], mut i: usize) -> usize {
    while parents[i] != i {
        parents[i] = parents[parents[i]];
        i = parents[i]
    }
    i
}
pub(super) fn merge_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    merge_bounded(v, work, false)
}
pub(in crate::processing) fn merge_page(v: &Value, work: &mut Work) -> Result<Value, Error> {
    merge_bounded(v, work, true)
}
pub(in crate::processing) fn merge_page_parts(
    raw: &[Value],
    page_size: &Value,
    table_bboxes: Option<&Value>,
    work: &mut Work,
) -> Result<Value, Error> {
    merge_parts(raw, page_size, table_bboxes, work, true)
}
fn merge_bounded(v: &Value, work: &mut Work, full: bool) -> Result<Value, Error> {
    page(&v["page_size"])?;
    let raw = v["lines"]
        .as_array()
        .filter(|a| a.len() <= crate::legacy_page::budget::records(if full { 4096 } else { 512 }))
        .ok_or_else(limit)?;
    merge_parts(raw, &v["page_size"], v.get("table_bboxes"), work, full)
}
fn merge_parts(
    raw: &[Value],
    page_size: &Value,
    table_bboxes: Option<&Value>,
    work: &mut Work,
    full: bool,
) -> Result<Value, Error> {
    let p = page(page_size)?;
    if raw.len() > crate::legacy_page::budget::records(if full { 4096 } else { 512 }) {
        return Err(limit());
    }
    let mut sources = BTreeSet::new();
    let mut chars = 0;
    let mut bytes = 0;
    let maximum_bytes = rapiddoc_core::resources::limit(8_000_000);
    let mut lines = Vec::new();
    for l in raw {
        if maximum_bytes != usize::MAX {
            bytes += l.to_string().len();
            if bytes > maximum_bytes {
                return Err(limit());
            }
        }
        let mut c = canonical_bounded(l, if full { 20000 } else { 1500 })?;
        if full {
            let mut d = super::super::native_lines::defaults();
            for (k, v) in l.as_object().ok_or_else(invalid)? {
                // Canonical fields already own their validated values. Copy
                // only the additional native fields, then transfer that map.
                if c.get(k).is_none() && d.get(k).is_some() {
                    d[k] = v.clone();
                }
            }
            d.as_object_mut()
                .unwrap()
                .extend(std::mem::take(c.as_object_mut().unwrap()));
            c = d;
        }
        let l = c;
        chars += l["chars"].as_array().unwrap().len();
        if chars > crate::legacy_page::budget::records(if full { 20000 } else { 1500 }) {
            return Err(limit());
        }
        if !sources.insert(source(&l)) {
            return Err(invalid());
        }
        lines.push(l);
    }
    if lines.len() < 2 {
        return Ok(
            json!({"lines":lines,"groups":(0..raw.len()).map(|i|vec![i]).collect::<Vec<_>>()}),
        );
    }
    let tables = table_bboxes.map_or(Ok(Vec::new()), |v| {
        v.as_array()
            .filter(|a| a.len() <= rapiddoc_core::resources::limit(5000))
            .ok_or_else(limit)?
            .iter()
            .map(rect)
            .collect::<Result<Vec<_>, _>>()
    })?;
    let angles: BTreeSet<_> = lines.iter().map(angle).collect();
    let mut consumed = BTreeSet::new();
    let mut merged = Vec::new();
    for a in angles {
        let indices: Vec<_> = lines
            .iter()
            .enumerate()
            .filter_map(|(i, l)| (angle(l) == a).then_some(i))
            .collect();
        if indices.len() < 2 {
            continue;
        }
        let local: Vec<_> = lines
            .iter()
            .map(|l| rotate(rect(&l["bbox"]).unwrap(), p, a))
            .collect();
        let h = median(
            indices
                .iter()
                .map(|&i| eff(&lines[i], local[i]).unwrap())
                .collect(),
        );
        let width = if a == 90 || a == 270 { p[1] } else { p[0] };
        let bounds: Vec<Value> = indices.iter().map(|&i| json!(local[i])).collect();
        let geometry: Vec<_> = indices
            .iter()
            .zip(&bounds)
            .map(|(&i, b)| (&lines[i], b))
            .collect();
        let lanes = lanes::infer_layout_rows(
            &geometry,
            &json!({"local_page_width":width,"median_height":h}),
            work,
            full,
        )?;
        for lane in lanes.as_array().unwrap() {
            if lane["is_span"] == true {
                continue;
            }
            let mut members: Vec<_> = lane["line_indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| indices[x.as_u64().unwrap() as usize])
                .collect();
            if members.len() < 2 {
                continue;
            }
            members.sort_by(|&i, &j| {
                local[i][1]
                    .partial_cmp(&local[j][1])
                    .unwrap()
                    .then_with(|| local[i][0].partial_cmp(&local[j][0]).unwrap())
                    .then(source(&lines[i]).cmp(&source(&lines[j])))
            });
            let h = median(
                members
                    .iter()
                    .map(|&i| eff(&lines[i], local[i]).unwrap())
                    .collect(),
            );
            let mut parents: Vec<_> = (0..members.len()).collect();
            for i in 0..members.len() {
                for j in i + 1..members.len() {
                    let (x, y) = (members[i], members[j]);
                    if connection(
                        &lines[x],
                        &lines[y],
                        (local[x], local[y]),
                        h,
                        width,
                        &tables,
                        work,
                    )? {
                        let x = root(&mut parents, i);
                        let y = root(&mut parents, j);
                        if x != y {
                            parents[y] = x;
                        }
                    }
                }
            }
            let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
            for (i, &id) in members.iter().enumerate() {
                let r = root(&mut parents, i);
                if let Some(g) = groups.iter_mut().find(|g| g.0 == r) {
                    g.1.push(id)
                } else {
                    groups.push((r, vec![id]));
                }
            }
            for (_, ids) in groups {
                let lane_width =
                    (lane["right"].as_f64().unwrap() - lane["left"].as_f64().unwrap()).max(0.1);
                if let Some((compact, paired)) =
                    kind(&ids, &lines, &local, lane_width, h, &tables, work)?
                {
                    let mut out =
                        materialize(&ids, &lines, &local, p, h, (compact, paired), work, full)?;
                    if full {
                        let mut d = super::super::native_lines::defaults();
                        d.as_object_mut()
                            .unwrap()
                            .extend(out.as_object().unwrap().clone());
                        super::super::native_lines::markers(&mut d);
                        out = d;
                    }
                    for &i in &ids {
                        consumed.insert(source(&lines[i]));
                    }
                    merged.push((out, ids));
                }
            }
        }
    }
    let mut output: Vec<_> = lines
        .into_iter()
        .enumerate()
        .filter_map(|(i, l)| (!consumed.contains(&source(&l))).then_some((l, vec![i])))
        .collect();
    output.extend(merged);
    output.sort_by(|(a, _), (b, _)| {
        let x = rotate(rect(&a["bbox"]).unwrap(), p, angle(a));
        let y = rotate(rect(&b["bbox"]).unwrap(), p, angle(b));
        angle(a)
            .cmp(&angle(b))
            .then_with(|| x[1].partial_cmp(&y[1]).unwrap())
            .then_with(|| x[0].partial_cmp(&y[0]).unwrap())
            .then(source(a).cmp(&source(b)))
    });
    let (lines, groups): (Vec<_>, Vec<_>) = output.into_iter().unzip();
    let mut out = json!({"groups":groups});
    out["lines"] = Value::Array(lines);
    Ok(out)
}
pub fn merge(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    cardinal_angles(v)?;
    merge_with_work(v, &mut Work(0))
}
fn cardinal_angles(v: &Value) -> Result<(), Error> {
    if v["lines"].as_array().is_some_and(|lines| {
        lines
            .iter()
            .any(|l| !matches!(l["angle"].as_i64(), Some(0 | 90 | 180 | 270)))
    }) {
        return Err(invalid());
    }
    Ok(())
}
pub fn cell_roles(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    cardinal_angles(v)?;
    cell_roles_with_work(v, &mut Work(0))
}
pub(super) fn cell_roles_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    let mut request = v.clone();
    request["table_bboxes"] = json!([]);
    let merged = merge_with_work(&request, work)?;
    let mut output = BTreeMap::new();
    for line in merged["lines"].as_array().unwrap() {
        let n = line["chars"].as_array().unwrap().len();
        work.charge(n.saturating_mul(n))?;
        let args = json!({"line":line,"page_size":v["page_size"],"tight_bboxes":v.get("tight_bboxes").cloned().unwrap_or(json!({})),"origins":v.get("origins").cloned().unwrap_or(json!({})),"fraction_members":v.get("fraction_members").cloned().unwrap_or(json!([]))});
        let (chars, roles) = preparation::line_roles(&args)?;
        preparation::accumulate_roles(&mut output, &chars, roles);
    }
    Ok(preparation::roles_value(output))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_cluster_keeps_all_characters_in_production() {
        let line = json!({"text":"a", "bbox":[0.,0.,10.,10.], "angle":0,
            "source_index":0, "chars":vec![json!({"char":"a"}); 20001]});
        assert!(canonical_bounded(&line, 20000).is_err());
        let _scope = crate::legacy_page::budget::host_page();
        assert_eq!(
            canonical_bounded(&line, 20000).unwrap()["chars"]
                .as_array()
                .unwrap()
                .len(),
            20001
        );
        let mut malformed = line;
        malformed["chars"][0] = Value::Null;
        assert!(canonical_bounded(&malformed, 20000).is_err());
    }
    #[test]
    fn old_empty_and_control_cluster_boundaries() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-inline-clusters-empty.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        let extra = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/script-empty-printable.json");
        let extra: Value = serde_json::from_slice(&std::fs::read(extra).unwrap()).unwrap();
        for c in extra["fractions"].as_array().unwrap() {
            assert_eq!(
                rapiddoc_native::script_geometry::refine::fraction_members(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
        for c in extra["runs"].as_array().unwrap() {
            assert_eq!(
                rapiddoc_native::script_geometry::refine::classify_runs(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                merge(&c["request"]).unwrap()["lines"],
                c["expected_lines"],
                "{}",
                c["label"]
            );
            assert_eq!(
                cell_roles(&c["role_request"]).unwrap(),
                c["expected_roles"],
                "{}",
                c["label"]
            );
            assert_eq!(
                rapiddoc_native::script_geometry::refine::classify_line(&c["line_request"])
                    .unwrap(),
                c["expected_line"],
                "{}",
                c["label"]
            );
        }
    }
    #[test]
    fn old_multi_line_cell_html_chains() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-inline-clusters.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        for c in f["cases"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|c| c.get("render_request").is_some())
        {
            let roles = cell_roles(&c["role_request"]).unwrap();
            let mut request = c["render_request"].clone();
            request["roles"] = roles;
            assert_eq!(
                super::super::render_cell(&request).unwrap(),
                c["expected_html"].as_str().unwrap(),
                "{}",
                c["label"]
            );
        }
    }
    #[test]
    fn old_complete_inline_clusters() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-inline-clusters.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                merge(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
            if let Some(roles) = c.get("expected_roles") {
                assert_eq!(
                    cell_roles(&c["role_request"]).unwrap(),
                    *roles,
                    "{}",
                    c["label"]
                );
            }
        }
    }
}
