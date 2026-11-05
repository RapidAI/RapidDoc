//! Original whole-document marginal sequence and document body evidence.
mod borders;
mod profiles;
mod tails;
use super::{
    native_marginals::local,
    page_containers::{clip, coerce, combined, height, median, optional, size},
    path_graphics::{array, bbox, integer, num, Box4},
};
use crate::legacy_page::{eq, invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
fn sz(p: &Value) -> Result<[f64; 2], Error> {
    size(p.get("page_size").unwrap_or(&p["source"]["page_size"]))
}
fn ls(p: &Value) -> Result<&[Value], Error> {
    array(&p["remaining_lines"], 4096)
}
fn bs(l: &Value, p: [f64; 2], a: i64, ink: bool) -> Result<Box4, Error> {
    Ok(local(
        &bbox(if ink && !l["ink_bbox"].is_null() {
            &l["ink_bbox"]
        } else {
            &l["bbox"]
        })?,
        p,
        a,
    ))
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
fn intersects(a: &Box4, b: &Box4) -> bool {
    a.n[2].min(b.n[2]) > a.n[0].max(b.n[0]) && a.n[3].min(b.n[3]) > a.n[1].max(b.n[1])
}
fn canonical(l: &Value, b: &Box4) -> Result<f64, Error> {
    let em = l.get("em_height").map_or(Ok(0.), num)?;
    let h = l.get("effective_height").map_or(Ok(0.), num)?;
    Ok((if em > 0. {
        em
    } else if h != 0. {
        h
    } else {
        b.n[3] - b.n[1]
    })
    .max(0.1))
}
fn text(l: &Value) -> Result<&str, Error> {
    l["text"].as_str().ok_or_else(invalid)
}
fn local_size(p: [f64; 2], a: i64) -> [f64; 2] {
    if a == 90 || a == 270 {
        [p[1], p[0]]
    } else {
        p
    }
}
fn dominant(lines: &[Value], p: [f64; 2]) -> Result<Option<i64>, Error> {
    let mut supports = BTreeMap::<i64, f64>::new();
    for l in lines {
        let a = integer(l, "angle", 0)?;
        let b = bs(l, p, a, false)?;
        *supports.entry(a).or_default() += (b.n[2] - b.n[0]).max(0.1) * height(l, &b)?;
    }
    let mut best = None;
    for (a, v) in supports {
        if best.is_none_or(|(_, old)| v > old) {
            best = Some((a, v))
        }
    }
    Ok(best.map(|(a, _)| a))
}
fn row_gap(a: &Value, ab: &Box4, bb: &Box4) -> Result<f64, Error> {
    let h = height(a, ab)?;
    Ok(if a["restored_inline_cluster"] == true {
        (bb.n[1] - ab.n[3]).max(-0.25 * h)
    } else {
        bb.n[1] - ab.n[1] - h
    })
}
fn visual_headers(pages: &mut [Value], w: &mut Work) -> Result<(), Error> {
    let mut candidates = Vec::new();
    for (i, p) in pages.iter().enumerate().skip(1) {
        let s = sz(p)?;
        if s[0] <= 0. || s[1] <= 0. {
            continue;
        }
        for (j, b) in optional(p, "fixed_blocks", 4000)?.iter().enumerate() {
            if b["type"] != "image" {
                continue;
            }
            let Some(bbox) = coerce(&b["bbox"])?
                .map(|b| clip(&b, s))
                .transpose()?
                .flatten()
            else {
                continue;
            };
            if bbox.n[3] > 0.12 * s[1] {
                continue;
            }
            let mut n = bbox.n;
            n[0] /= s[0];
            n[2] /= s[0];
            n[1] /= s[1];
            n[3] /= s[1];
            let angle = if b["angle"].is_null() {
                0
            } else {
                super::table_native_text::number(&b["angle"])?.ok_or_else(invalid)? as i64
            }
            .rem_euclid(360);
            candidates.push((i, j, n, angle));
        }
    }
    if candidates.len() < 3 {
        return Ok(());
    }
    let mut parents = (0..candidates.len()).collect::<Vec<_>>();
    fn root(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i]
        }
        i
    }
    for (i, (pi, _, a, aa)) in candidates.iter().enumerate() {
        for (j, (pj, _, b, ba)) in candidates.iter().enumerate().skip(i + 1) {
            w.charge(1)?;
            let delta = pj - pi;
            if delta > 2 {
                break;
            }
            if delta == 0 || aa != ba {
                continue;
            }
            let (aw, ah, bw, bh) = (a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]);
            if aw.min(ah).min(bw).min(bh) <= 0.
                || aw.max(bw) / aw.min(bw) > 1.1
                || ah.max(bh) / ah.min(bh) > 1.1
            {
                continue;
            }
            let area = (a[2].min(b[2]) - a[0].max(b[0])).max(0.)
                * (a[3].min(b[3]) - a[1].max(b[1])).max(0.);
            let union = aw * ah + bw * bh - area;
            if union > 0. && area / union >= 0.9 {
                let x = root(&mut parents, i);
                let y = root(&mut parents, j);
                if x != y {
                    parents[y] = x
                }
            }
        }
    }
    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    for i in 0..parents.len() {
        groups.entry(root(&mut parents, i)).or_default().push(i);
    }
    for ids in groups.values() {
        if ids
            .iter()
            .map(|&i| candidates[i].0)
            .collect::<BTreeSet<_>>()
            .len()
            >= 3
        {
            for &i in ids {
                let (p, b, _, _) = candidates[i];
                pages[p]["fixed_blocks"][b]["type"] = json!("header");
            }
        }
    }
    Ok(())
}
fn split_compound(pages: &mut [Value], w: &mut Work) -> Result<(), Error> {
    let single = pages.len() == 1;
    for p in pages {
        let s = sz(p)?;
        let mut lines = ls(p)?.to_vec();
        let mut groups = Vec::<((i64, i64), Vec<usize>)>::new();
        for (i, l) in lines.iter().enumerate() {
            if l["visual_row_id"].is_null() || l["split_from_row"] != true {
                continue;
            }
            let key = (integer(l, "angle", 0)?, integer(l, "visual_row_id", 0)?);
            if let Some(g) = groups.iter_mut().find(|g| g.0 == key) {
                g.1.push(i)
            } else {
                groups.push((key, vec![i]))
            }
        }
        for ((a, _), ids) in &groups {
            w.charge(ids.len() + 1)?;
            let bs = ids
                .iter()
                .map(|&i| bs(&lines[i], s, *a, false))
                .collect::<Result<Vec<_>, _>>()?;
            let y = super::table_native_candidate::sum(&bs.iter().map(cy).collect::<Vec<_>>())
                / bs.len() as f64;
            let h = local_size(s, *a)[1];
            let target = if y <= 0.1 * h {
                "header"
            } else if y >= 0.9 * h {
                "footer"
            } else {
                continue;
            };
            if ids.iter().any(|&i| {
                lines[i]["semantic_type"] == target || lines[i]["semantic_type"] == "page_number"
            }) {
                for &i in ids {
                    if lines[i]["semantic_type"].is_null()
                        && lines[i]["marginal_body_context"] != true
                    {
                        lines[i]["semantic_type"] = json!(target)
                    }
                }
            }
        }
        if single && s[0] > 0. && s[1] > 0. {
            for ((a, _), ids) in &groups {
                let ids = ids
                    .iter()
                    .copied()
                    .filter(|&i| lines[i]["semantic_type"].is_null())
                    .collect::<Vec<_>>();
                if ids.len() < 2 {
                    continue;
                }
                let lp = local_size(s, *a);
                let boxes = ids
                    .iter()
                    .map(|&i| bs(&lines[i], s, *a, false))
                    .collect::<Result<Vec<_>, _>>()?;
                let bounds = combined(boxes.iter())?;
                if bounds.n[1] < 0. || bounds.n[3] > 0.05 * lp[1] {
                    continue;
                }
                let mut body = Vec::new();
                for l in &lines {
                    w.charge(ids.len() + 1)?;
                    if !l["semantic_type"].is_null()
                        || integer(l, "angle", 0)? != *a
                        || ids.iter().any(|&i| eq(l, &lines[i]))
                    {
                        continue;
                    }
                    let b = bs(l, s, *a, false)?;
                    if b.n[1] >= 0.05 * lp[1]
                        && b.n[2] - b.n[0] >= 0.3 * lp[0]
                        && axis(&bounds, &b, 0) >= 0.2
                    {
                        body.push((l, b))
                    }
                }
                if body.len() < 3 {
                    continue;
                }
                let em = median(
                    body.iter()
                        .map(|(l, b)| height(l, b))
                        .collect::<Result<Vec<_>, _>>()?,
                );
                let row = ids
                    .iter()
                    .zip(&boxes)
                    .map(|(&i, b)| height(&lines[i], b))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .reduce(f64::max)
                    .unwrap();
                if row > 0.85 * em {
                    continue;
                }
                let right = median(body.iter().map(|(_, b)| b.n[2]).collect());
                if boxes.iter().any(|b| {
                    b.n[2] - b.n[0] <= (4. * row).max(0.12 * lp[0])
                        && (b.n[2] - right).abs() <= 3f64.max(0.02 * lp[0])
                }) {
                    for i in ids {
                        lines[i]["semantic_type"] = json!("header")
                    }
                }
            }
        }
        p["remaining_lines"] = json!(lines);
    }
    Ok(())
}
pub fn prepare(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_with_work(v, &mut Work(0))
}
pub(super) fn prepare_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    prepare_owned(v.clone(), w)
}
pub(super) fn prepare_owned(mut v: Value, w: &mut Work) -> Result<Value, Error> {
    array(&v["pages"], 1000)?;
    let Value::Array(mut pages) = v["pages"].take() else {
        return Err(invalid());
    };
    drop(v);
    visual_headers(&mut pages, w)?;
    super::native_marginals::repeat_prepared_in_place(&mut pages, w)?;
    split_compound(&mut pages, w)?;
    borders::classify(&mut pages, w)?;
    tails::classify(&mut pages, w)?;
    profiles::restore(&mut pages, w)?;
    let body = profiles::body(&pages, false, w)?;
    let canonical = profiles::body(&pages, true, w)?;
    if !body.is_null() {
        let h = num(&body["body_height"])?;
        for p in &mut pages {
            p["remaining_lines"] = super::native_footnotes::deferred_image(p, h, w)?;
        }
    }
    let mut out = json!({"body_profile":body,"canonical_body_profile":canonical});
    out["pages"] = Value::Array(pages);
    if crate::legacy_page::budget::output_exceeds(&out) {
        Err(limit())
    } else {
        Ok(out)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_document_auxiliary_sequence() {
        let f: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-document-auxiliary-canonical.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            let out = prepare(&c["request"]);
            if !c["error"].is_null() {
                assert!(out.is_err(), "{}", c["label"]);
                continue;
            }
            let out = out.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = super::super::native_document::tests::diff(&out, &c["expected"], "") {
                panic!("{}: {d}", c["label"]);
            }
        }
    }
    #[test]
    fn old_document_auxiliary_proofs() {
        check("native-document-auxiliary-proofs", prepare)
    }
    #[test]
    fn old_document_auxiliary_chain() {
        check(
            "native-document-auxiliary-chain",
            super::super::native_document::prepare_auxiliary,
        )
    }
    fn check(name: &str, f: fn(&Value) -> Result<Value, Error>) {
        let data: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join(format!("../../build/migration/{name}.json")),
            )
            .unwrap(),
        )
        .unwrap();
        for c in data["cases"].as_array().unwrap() {
            let out = f(&c["request"]);
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
