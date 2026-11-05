//! Original image/page footnotes and parallel table-end field protection.
mod groups;
mod lane;
use super::{
    page_containers::{coerce, combined, height, median, optional, size},
    path_graphics::{array, bbox, integer, num, Box4},
    table_text_styles::lanes,
};
use crate::legacy_page::{eq, invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{
    python_decimal,
    text_boundary::{python_digit, python_space},
};
use serde_json::{json, Value};
use std::{
    borrow::Cow,
    collections::{BTreeMap, BTreeSet},
};
#[derive(Clone)]
struct Row<'a> {
    l: Cow<'a, Value>,
    b: Box4,
}
struct Rule {
    b: Box4,
    original: Box4,
    width: f64,
    horizontal: bool,
}
fn local(b: Box4, p: [f64; 2], a: i64) -> Box4 {
    super::native_marginals::local(&b, p, a)
}
fn text(l: &Value) -> Result<&str, Error> {
    l["text"].as_str().ok_or_else(invalid)
}
fn id(l: &Value) -> Result<i64, Error> {
    integer(l, "source_index", 0)
}
fn h(r: &Row) -> Result<f64, Error> {
    height(&r.l, &r.b)
}
fn cy(b: &Box4) -> f64 {
    (b.n[1] + b.n[3]) * 0.5
}
fn cx(b: &Box4) -> f64 {
    (b.n[0] + b.n[2]) * 0.5
}
fn axis(a: &Box4, b: &Box4, k: usize) -> f64 {
    let d = (a.n[k + 2] - a.n[k]).min(b.n[k + 2] - b.n[k]);
    if d > 0. {
        (a.n[k + 2].min(b.n[k + 2]) - a.n[k].max(b.n[k])).max(0.) / d
    } else {
        0.
    }
}
fn overlap(a: &Box4, b: &Box4) -> bool {
    a.n[2].min(b.n[2]) > a.n[0].max(b.n[0]) && a.n[3].min(b.n[3]) > a.n[1].max(b.n[1])
}
fn expanded(b: &Box4, d: f64) -> Result<Box4, Error> {
    bbox(&json!([b.n[0] - d, b.n[1] - d, b.n[2] + d, b.n[3] + d]))
}
fn gap(a: &Row, b: &Row) -> Result<f64, Error> {
    let em = h(a)?;
    Ok(if a.l["restored_inline_cluster"] == true {
        (b.b.n[1] - a.b.n[3]).max(-0.25 * em)
    } else {
        b.b.n[1] - a.b.n[1] - em
    })
}
fn weight(l: &Value) -> Result<f64, Error> {
    if l["dominant_font_weight"].is_null() {
        Ok(400.)
    } else {
        let n = num(&l["dominant_font_weight"])?;
        Ok(if n == 0. { 400. } else { n })
    }
}
fn sorted<'a>(mut rows: Vec<Row<'a>>, source_tie: bool) -> Vec<Row<'a>> {
    rows.sort_by(|a, b| {
        a.b.n[1]
            .total_cmp(&b.b.n[1])
            .then(a.b.n[0].total_cmp(&b.b.n[0]))
            .then(if source_tie {
                id(&a.l).unwrap().cmp(&id(&b.l).unwrap())
            } else {
                std::cmp::Ordering::Equal
            })
    });
    rows
}
fn geometry<'a>(
    ls: &'a [Value],
    sz: [f64; 2],
    w: &mut Work,
) -> Result<Option<(Vec<Row<'a>>, i64, [f64; 2])>, Error> {
    let mut support = BTreeMap::<i64, f64>::new();
    for l in ls.iter().filter(|l| l["semantic_type"].is_null()) {
        w.charge(text(l)?.len() + 1)?;
        let a = integer(l, "angle", 0)?;
        let b = local(bbox(&l["bbox"])?, sz, a);
        *support.entry(a).or_default() += (b.n[2] - b.n[0]).max(0.1) * height(l, &b)?;
    }
    let mut best = None;
    for (&a, &s) in &support {
        if best.is_none_or(|(_, v)| s > v) {
            best = Some((a, s))
        }
    }
    let Some((a, _)) = best else { return Ok(None) };
    let mut rows = Vec::new();
    for l in ls.iter().filter(|l| l["semantic_type"].is_null()) {
        if integer(l, "angle", 0)? == a {
            rows.push(Row {
                l: Cow::Borrowed(l),
                b: local(bbox(&l["bbox"])?, sz, a),
            })
        }
    }
    let p = if a == 90 || a == 270 {
        [sz[1], sz[0]]
    } else {
        sz
    };
    Ok(Some((rows, a, p)))
}
fn rules(raw: &[Value], sz: [f64; 2], a: i64) -> Result<Vec<Rule>, Error> {
    raw.iter()
        .map(|r| {
            let original = bbox(&r["bbox"])?;
            let b = local(original.clone(), sz, a);
            let horizontal = b.n[2] - b.n[0] >= b.n[3] - b.n[1];
            Ok(Rule {
                b,
                original,
                width: r.get("width").map_or(Ok(1.), num)?,
                horizontal,
            })
        })
        .collect()
}
fn table_rules(rs: &[Rule], tables: &[Box4], w: &mut Work) -> Result<Vec<usize>, Error> {
    let mut ids = Vec::new();
    for (i, r) in rs.iter().enumerate() {
        w.charge(tables.len() + 1)?;
        if r.horizontal {
            let b = expanded(&r.original, r.width.max(0.5))?;
            if tables.iter().any(|t| overlap(&b, t)) {
                ids.push(i)
            }
        }
    }
    Ok(ids)
}
fn belongs(r: &Rule, rs: &[Rule], table: &[usize], width: f64) -> bool {
    table.iter().any(|&i| {
        let t = &rs[i];
        (cy(&r.b) - cy(&t.b)).abs() <= 1f64.max(r.width).max(t.width)
            && (r.b.n[0] - t.b.n[2]).max(t.b.n[0] - r.b.n[2]).max(0.) <= 0.04 * width
    })
}
fn image(
    ls: &mut [Value],
    sz: [f64; 2],
    images: &[Box4],
    tables: &[Box4],
    draw: &[Value],
    reference: Option<f64>,
    w: &mut Work,
) -> Result<(), Error> {
    if images.is_empty() || draw.is_empty() {
        return Ok(());
    }
    let Some((rows, a, p)) = geometry(ls, sz, w)? else {
        return Ok(());
    };
    if rows.is_empty() || p[0] <= 0. || p[1] <= 0. {
        return Ok(());
    }
    let rows = sorted(rows, true);
    let em = if let Some(n) = reference.filter(|&n| n > 0.) {
        n.max(0.1)
    } else {
        let mut hs = rows
            .iter()
            .filter(|r| r.b.n[2] - r.b.n[0] >= 0.2 * p[0])
            .map(h)
            .collect::<Result<Vec<_>, _>>()?;
        if hs.is_empty() {
            hs = rows.iter().map(h).collect::<Result<Vec<_>, _>>()?
        }
        median(hs).max(0.1)
    };
    let images = images
        .iter()
        .map(|b| local(b.clone(), sz, a))
        .collect::<Vec<_>>();
    let rs = rules(draw, sz, a)?;
    let tr = table_rules(&rs, tables, w)?;
    let mut matched = BTreeSet::new();
    for image in &images {
        w.charge(images.len() + rs.len() * (tr.len() + 1) + rows.len() + 1)?;
        let width = (image.n[2] - image.n[0]).max(0.1);
        let bottom = images
            .iter()
            .filter(|b| axis(image, b, 1) >= 0.5)
            .map(|b| b.n[3])
            .reduce(f64::max)
            .ok_or_else(invalid)?;
        let rule = rs
            .iter()
            .filter(|r| {
                r.horizontal
                    && (0.75 * width..=1.3 * width).contains(&(r.b.n[2] - r.b.n[0]))
                    && (r.b.n[2].min(image.n[2]) - r.b.n[0].max(image.n[0])).max(0.) >= 0.85 * width
                    && (0. ..=(0.01 * p[1]).max(0.75 * em)).contains(&(r.b.n[1] - bottom))
                    && !belongs(r, &rs, &tr, p[0])
            })
            .min_by(|r, s| {
                (r.b.n[1] - bottom)
                    .max(0.)
                    .total_cmp(&(s.b.n[1] - bottom).max(0.))
                    .then(r.b.n[1].total_cmp(&s.b.n[1]))
            });
        let Some(rule) = rule else { continue };
        let mut first = None;
        for (i, r) in rows.iter().enumerate() {
            if (-0.25 * em..=(0.025 * p[1]).max(2. * em)).contains(&(r.b.n[1] - rule.b.n[3]))
                && r.b.n[0] >= rule.b.n[0] - 0.5 * em
                && r.b.n[2] <= rule.b.n[2] + 0.5 * em
                && h(r)? <= 0.9 * em
            {
                first = Some(i);
                break;
            }
        }
        let Some(first) = first else { continue };
        matched.insert(id(&rows[first].l)?);
        let continuation = (1.25 * h(&rows[first])?).max(0.01 * p[1]);
        let mut previous = first;
        for (i, r) in rows.iter().enumerate() {
            if i == first
                || r.b.n[1] < rows[first].b.n[1]
                || r.b.n[0] < rule.b.n[0] - 0.5 * em
                || r.b.n[2] > rule.b.n[2] + 0.5 * em
                || h(r)? > 0.95 * em
            {
                continue;
            }
            if gap(&rows[previous], r)? > continuation {
                break;
            }
            matched.insert(id(&r.l)?);
            previous = i;
        }
    }
    for l in ls {
        if l["semantic_type"].is_null() && matched.contains(&id(l)?) {
            l["semantic_type"] = json!("footnote")
        }
    }
    Ok(())
}
fn parallel(ls: &mut [Value], tables: &[Box4], w: &mut Work) -> Result<(), Error> {
    let re = constant_regex!(r"^(?:[0-9*†‡]|[一二三四五六七八九十]+[、.])");
    for table in tables {
        let mut short = Vec::new();
        for (i, l) in ls.iter().enumerate() {
            w.charge(text(l)?.len() + 1)?;
            let b = bbox(&l["bbox"])?;
            let t = text(l)?;
            if l["semantic_type"].is_null()
                && integer(l, "angle", 0)? == 0
                && b.n[1] > table.n[3] + 2. * l.get("effective_height").map_or(Ok(0.), num)?.max(1.)
                && (2..=14).contains(&t.chars().filter(|c| !python_space(*c)).count())
                && t.chars().any(|c| ('\u{3400}'..='\u{9fff}').contains(&c))
                && !re.is_match(&mapped(t.trim_matches(python_space)))
            {
                short.push((i, b))
            }
        }
        short.sort_by(|a, b| {
            cy(&a.1)
                .total_cmp(&cy(&b.1))
                .then(a.1.n[0].total_cmp(&b.1.n[0]))
        });
        let mut rows = Vec::<Vec<(usize, Box4)>>::new();
        for item in short {
            let em = ls[item.0].get("effective_height").map_or(Ok(0.), num)?;
            if rows
                .last()
                .is_none_or(|r| (cy(&item.1) - cy(&r[0].1)).abs() > 0.35 * em)
            {
                rows.push(vec![item])
            } else {
                rows.last_mut().unwrap().push(item)
            }
        }
        let mut bands = Vec::new();
        for pair in rows.windows(2) {
            w.charge(pair[0].len() + pair[1].len() + 1)?;
            let (a, b) = (&pair[0], &pair[1]);
            if a.len() < 2 || b.len() < 2 {
                continue;
            }
            let em = a
                .iter()
                .chain(b)
                .map(|(i, _)| ls[*i].get("effective_height").map_or(Ok(0.), num))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .reduce(f64::max)
                .unwrap();
            let dy = b[0].1.n[1] - a[0].1.n[3];
            if dy > 0.
                && dy < 1.5 * em
                && a.iter()
                    .take(2)
                    .all(|(_, x)| b.iter().any(|(_, y)| (x.n[0] - y.n[0]).abs() < 0.5 * em))
            {
                bands.push(a.iter().chain(b).map(|(i, _)| *i).collect::<Vec<_>>())
            }
        }
        if bands.len() >= 2 {
            for i in bands.into_iter().flatten() {
                ls[i]["semantic_type"] = json!("text");
                ls[i]["title_suppressed"] = json!(true)
            }
        }
    }
    Ok(())
}
fn mapped(s: &str) -> String {
    s.chars()
        .map(|c| python_decimal(c).map_or(c, |d| (b'0' + d) as char))
        .collect()
}
pub fn classify(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    classify_with_work(v, &mut Work(0))
}
pub(super) fn prepared_rules(p: &Value, a: i64, w: &mut Work) -> Result<Vec<Value>, Error> {
    let sz = size(p.get("page_size").unwrap_or(&p["source"]["page_size"]))?;
    let rs = rules(
        p.get("drawing_lines")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(optional(&p["source"], "drawing_lines", 5000)?),
        sz,
        a,
    )?;
    let tables = optional(p, "table_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let ts = table_rules(&rs, &tables, w)?;
    let width = if a == 90 || a == 270 { sz[1] } else { sz[0] };
    let mut out = Vec::new();
    for r in &rs {
        w.charge(ts.len() + 1)?;
        out.push(json!({"bbox":r.b.raw,"original_bbox":r.original.raw,"width":r.width,"horizontal":r.horizontal,"table":belongs(r,&rs,&ts,width)}));
    }
    Ok(out)
}
pub(super) fn deferred_image(p: &Value, body: f64, w: &mut Work) -> Result<Value, Error> {
    let mut ls = array(&p["remaining_lines"], 4096)?.to_vec();
    let sz = size(p.get("page_size").unwrap_or(&p["source"]["page_size"]))?;
    let images = optional(p, "fixed_blocks", 4000)?
        .iter()
        .filter(|b| b["type"] == "image")
        .map(|b| bbox(&b["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let tables = optional(p, "table_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    image(
        &mut ls,
        sz,
        &images,
        &tables,
        p.get("drawing_lines")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(optional(&p["source"], "drawing_lines", 5000)?),
        Some(body),
        w,
    )?;
    Ok(json!(ls))
}
pub(super) fn auxiliary(v: &Value, w: &mut Work) -> Result<Value, Error> {
    auxiliary_lines_owned(array(&v["lines"], 4096)?.to_vec(), v, w)
}
pub(super) fn auxiliary_lines_owned(
    mut ls: Vec<Value>,
    v: &Value,
    w: &mut Work,
) -> Result<Value, Error> {
    let sz = size(&v["page_size"])?;
    let available = ls
        .iter()
        .filter(|l| l["semantic_type"].is_null())
        .collect::<Vec<_>>();
    if available.iter().filter(|l| l["angle"] == 0).count() >= 4 {
        let mut support = BTreeMap::<i64, f64>::new();
        for l in &available {
            let a = integer(l, "angle", 0)?;
            let b = local(bbox(&l["bbox"])?, sz, a);
            *support.entry(a).or_default() += (b.n[2] - b.n[0]).max(0.1) * height(l, &b)?;
        }
        let total = support.values().sum::<f64>();
        if total > 0.
            && support.get(&0).copied().unwrap_or(0.) / total >= 0.8
            && sz[0] > 0.
            && sz[1] > 0.
        {
            for l in &mut ls {
                w.charge(1)?;
                if l["semantic_type"].is_null() && matches!(integer(l, "angle", 0)?, 90 | 270) {
                    let b = bbox(&l["bbox"])?;
                    if b.n[2] - b.n[0] <= 0.08 * sz[0]
                        && b.n[3] - b.n[1] >= 0.15 * sz[1]
                        && (b.n[2] <= 0.12 * sz[0] || b.n[0] >= 0.88 * sz[0])
                    {
                        l["semantic_type"] = json!("aside_text")
                    }
                }
            }
        }
    }
    let images = optional(v, "visual_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let tables = optional(v, "table_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let draw = optional(v, "drawing_lines", 5000)?;
    image(&mut ls, sz, &images, &tables, draw, None, w)?;
    let groups = groups::page(&mut ls, sz, &images, &tables, draw, &[], w)?;
    let mut out = json!({"page_footnote_groups":groups});
    out["lines"] = Value::Array(ls);
    Ok(out)
}
pub(super) fn classify_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    classify_lines_owned(array(&v["lines"], 4096)?.to_vec(), v, w)
}
pub(super) fn classify_lines_owned(
    ls: Vec<Value>,
    v: &Value,
    w: &mut Work,
) -> Result<Value, Error> {
    classify_lines_with_reference(ls, v, None, w)
}
pub(super) fn classify_lines_with_reference(
    mut ls: Vec<Value>,
    v: &Value,
    reference_lines: Option<&[Value]>,
    w: &mut Work,
) -> Result<Value, Error> {
    let sz = size(&v["page_size"])?;
    let images = optional(v, "visual_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let tables = optional(v, "table_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let draw = optional(v, "drawing_lines", 5000)?;
    let reference = v
        .get("reference_body_height")
        .filter(|v| !v.is_null())
        .map(num)
        .transpose()?;
    image(&mut ls, sz, &images, &tables, draw, reference, w)?;
    parallel(&mut ls, &tables, w)?;
    let groups = groups::page(
        &mut ls,
        sz,
        &images,
        &tables,
        draw,
        match reference_lines {
            Some(rows) => {
                if rows.len() > rapiddoc_core::resources::limit(4096) {
                    return Err(limit());
                }
                rows
            }
            None => optional(v, "reference_lines", 4096)?,
        },
        w,
    )?;
    let mut out = json!({"page_footnote_groups":groups});
    out["lines"] = Value::Array(ls);
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        Err(limit())
    } else {
        Ok(out)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_early_footnotes() {
        for name in [
            "native-early-footnotes-canonical",
            "native-early-footnote-edges",
        ] {
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../build/migration/{name}.json"));
            let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
            for c in f["cases"].as_array().unwrap() {
                let out = classify(&c["request"]).unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
                if let Some(d) =
                    super::super::native_document::tests::diff(&out, &c["expected"], "")
                {
                    panic!("{}: {d}", c["label"])
                }
            }
        }
    }
    #[test]
    fn old_document_footnotes() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-footnote-document.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let out = super::super::native_document::prepare_pages(&c["request"]);
            if !c["error"].is_null() {
                assert!(out.is_err(), "{}", c["label"]);
                continue;
            }
            let out = out.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = super::super::native_document::tests::diff(&out, &c["expected"], "") {
                panic!("{}: {d}", c["label"])
            }
        }
    }
}
