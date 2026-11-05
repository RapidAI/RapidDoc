//! Complete original final-page grouped ordering, using unchanged XYCut++ recursion.
mod panels;
mod tail;
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_python_visual_order_contracts() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-visual-order-contracts.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let got = if c["kind"] == "bind" {
                super::super::native_visual_annotations::bind(&c["request"])
            } else {
                run(&c["request"])
            }
            .unwrap();
            if let Some(d) = super::super::native_document::tests::diff(&got, &c["expected"], "") {
                panic!("{}: {d}", c["label"])
            }
        }
    }
    #[test]
    fn original_python_grouped_order() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-grouped-order.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let got = run(&c["request"]);
            if c.get("error").is_some() {
                assert!(got.is_err(), "{}", c["label"])
            } else {
                let got = got.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
                if let Some(d) =
                    super::super::native_document::tests::diff(&got, &c["expected"], "")
                {
                    panic!("{}: {d}", c["label"])
                }
            }
        }
    }
    #[test]
    fn original_python_xycut() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-xycut.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let got = xy(c["request"].as_array().unwrap(), &mut Work(0));
            if c.get("error").is_some() {
                assert!(got.is_err(), "{}", c["label"])
            } else {
                assert_eq!(json!(got.unwrap()), c["expected"], "{}", c["label"])
            }
        }
    }
}
use super::{
    native_marginals::local,
    page_containers::{combined, median, optional, size},
    path_graphics::{array, bbox, num, Box4},
};
use crate::legacy_page::{eq, invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::python_space;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
fn angle(b: &Value) -> Result<i64, Error> {
    super::native_final_output::block_angle(b)
}
fn bb(b: &Value) -> Result<Box4, Error> {
    bbox(&b["bbox"])
}
fn truth(v: &Value) -> bool {
    super::native_text_assembly::python_truth(v)
}
fn cx(b: &Box4) -> f64 {
    (b.n[0] + b.n[2]) * 0.5
}
fn cy(b: &Box4) -> f64 {
    (b.n[1] + b.n[3]) * 0.5
}
fn axis(a: &Box4, b: &Box4, k: usize) -> f64 {
    let min = (a.n[k + 2] - a.n[k]).min(b.n[k + 2] - b.n[k]);
    if min > 0. {
        (a.n[k + 2].min(b.n[k + 2]) - a.n[k].max(b.n[k])).max(0.) / min
    } else {
        0.
    }
}
fn text(b: &Value) -> &str {
    b["content"].as_str().unwrap_or("")
}
fn regex(t: &str) -> regex::Regex {
    regex::Regex::new(&t.replace("\\s",r"[\u{9}-\u{d}\u{1c}-\u{20}\u{85}\u{a0}\u{1680}\u{2000}-\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}]")).unwrap()
}
fn rowid(v: &Value) -> Option<i64> {
    v.as_i64().or_else(|| v.as_bool().map(i64::from))
}
fn heights(b: &Value) -> Result<Vec<f64>, Error> {
    optional(b, "_line_heights", 4096)?
        .iter()
        .filter(|v| v.is_number() || v.is_boolean())
        .map(|v| v.as_bool().map_or_else(|| num(v), |b| Ok(f64::from(b))))
        .filter(|r| !r.as_ref().is_ok_and(|&n| n <= 0.))
        .collect()
}
fn xy(bs: &[Value], w: &mut Work) -> Result<Vec<usize>, Error> {
    super::native_visual_annotations::sort_proxies(bs, w)
}
fn union(bs: &[Value], ids: &[usize]) -> Result<Box4, Error> {
    let boxes = ids
        .iter()
        .map(|&i| bb(&bs[i]))
        .collect::<Result<Vec<_>, _>>()?;
    combined(boxes.iter())
}
fn inline(bs: &[Value], excluded: &BTreeSet<usize>) -> Result<Vec<(i64, Vec<usize>)>, Error> {
    let mut images: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
    for (i, b) in bs.iter().enumerate() {
        if excluded.contains(&i) || b["type"] != "image" {
            continue;
        }
        if let Some(id) = rowid(&b["_inline_visual_row_id"]) {
            images.entry(id).or_default().push(i)
        }
    }
    let mut used: BTreeSet<usize> = BTreeSet::new();
    let mut out = vec![];
    for (id, mut ids) in images {
        let mut texts = vec![];
        for (i, b) in bs.iter().enumerate() {
            if excluded.contains(&i) || used.contains(&i) || b["type"] != "text" {
                continue;
            }
            if optional(b, "_visual_row_ids", 4096)?
                .iter()
                .any(|v| eq(v, &json!(id)))
            {
                texts.push(i)
            }
        }
        if texts.is_empty() {
            continue;
        }
        used.extend(&texts);
        ids.extend(texts);
        out.push((id, ids))
    }
    Ok(out)
}
fn inline_key(b: &Value, p: [f64; 2]) -> Result<(f64, f64, i32), Error> {
    let rows = optional(b, "_local_line_bboxes", 4096)?;
    let b_local = if b["type"] == "text" && !rows.is_empty() {
        bbox(&rows[0])?
    } else {
        local(&bb(b)?, p, angle(b)?)
    };
    Ok((
        b_local.n[0],
        b_local.n[1],
        if b["type"] == "image" { 0 } else { 1 },
    ))
}
fn marginal(ids: &[usize], bs: &[Value], p: [f64; 2], w: &mut Work) -> Result<Vec<usize>, Error> {
    let notes = ids
        .iter()
        .copied()
        .filter(|&i| bs[i]["type"] == "page_footnote")
        .collect::<Vec<_>>();
    if notes.len() >= 2 && notes.len() == ids.len() {
        let proxies = notes.iter().map(|&i| bs[i].clone()).collect::<Vec<_>>();
        return Ok(xy(&proxies, w)?.into_iter().map(|i| notes[i]).collect());
    }
    if !notes.is_empty() && notes.len() < ids.len() {
        let ordinary = ids
            .iter()
            .copied()
            .filter(|&i| bs[i]["type"] != "page_footnote")
            .collect::<Vec<_>>();
        let mut out = marginal(&notes, bs, p, w)?;
        out.extend(marginal(&ordinary, bs, p, w)?);
        return Ok(out);
    }
    if ids.len() < 2 {
        return Ok(ids.to_vec());
    }
    let mut geometry = ids
        .iter()
        .map(|&i| Ok((i, local(&bb(&bs[i])?, p, angle(&bs[i])?))))
        .collect::<Result<Vec<_>, Error>>()?;
    let mut hs = ids
        .iter()
        .map(|&i| heights(&bs[i]))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    hs.sort_by(f64::total_cmp);
    let h = if hs.is_empty() { 1. } else { hs[hs.len() / 2] };
    geometry.sort_by(|a, b| cy(&a.1).total_cmp(&cy(&b.1)));
    let mut rows: Vec<Vec<(usize, Box4)>> = vec![];
    for item in geometry {
        let target = rows.iter().position(|r| {
            (cy(&item.1) - r.iter().map(|(_, b)| cy(b)).sum::<f64>() / r.len() as f64).abs()
                <= 0.75 * h
        });
        if let Some(i) = target {
            rows[i].push(item)
        } else {
            rows.push(vec![item])
        }
    }
    let center =
        |r: &Vec<(usize, Box4)>| r.iter().map(|(_, b)| cy(b)).sum::<f64>() / r.len() as f64;
    rows.sort_by(|a, b| center(a).total_cmp(&center(b)));
    let mut out = vec![];
    for mut row in rows {
        row.sort_by(|a, b| a.1.n[0].total_cmp(&b.1.n[0]));
        out.extend(row.into_iter().map(|(i, _)| i))
    }
    Ok(out)
}
pub fn run(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let p = size(&v["page_size"])?;
    let mut bs = array(&v["blocks"], 4096)?.to_vec();
    let regions = optional(v, "regions", 4096)?
        .iter()
        .map(|r| {
            array(r, 4096)?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .map(|n| n as usize)
                        .filter(|&i| i < bs.len())
                        .ok_or_else(invalid)
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let order = sort(&mut bs, p, &regions, &mut Work(0))?;
    let out = json!({"blocks":bs,"order":order});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        Err(limit())
    } else {
        Ok(out)
    }
}
pub(super) fn sort(
    bs: &mut [Value],
    p: [f64; 2],
    regions: &[Vec<usize>],
    w: &mut Work,
) -> Result<Vec<usize>, Error> {
    let (mut top, mut bottom, mut body_ids) = (vec![], vec![], vec![]);
    for (i, b) in bs.iter().enumerate() {
        let ty = b["type"].as_str().unwrap_or("");
        if ty == "header" || ty == "page_number" && cy(&bb(b)?) <= 0.5 * p[1] {
            top.push(i)
        } else if ["footer", "page_footnote", "page_number"].contains(&ty) {
            bottom.push(i)
        } else {
            body_ids.push(i)
        }
    }
    let mut body = body_ids.iter().map(|&i| bs[i].clone()).collect::<Vec<_>>();
    let by_identity = body_ids
        .iter()
        .enumerate()
        .map(|(i, &old)| (old, i))
        .collect::<BTreeMap<_, _>>();
    let mut region_groups = vec![];
    let mut region_used = BTreeSet::new();
    for region in regions {
        let ids = region
            .iter()
            .filter_map(|i| by_identity.get(i).copied())
            .collect::<Vec<_>>();
        if ids.len() < 2 || ids.iter().any(|i| region_used.contains(i)) {
            continue;
        }
        for &i in &ids {
            body[i]["_visual_annotation_region_member"] = json!(true)
        }
        region_used.extend(&ids);
        region_groups.push(ids)
    }
    region_groups = panels::numbered_regions(region_groups, &body, w)?;
    for g in panels::parallel(&body, &region_used, w)? {
        region_used.extend(&g);
        region_groups.push(g)
    }
    for g in panels::steps(&body, &region_used, w)? {
        region_used.extend(&g);
        region_groups.push(g)
    }
    let inline = inline(&body, &region_used)?;
    let inline_used = inline
        .iter()
        .flat_map(|(_, g)| g.iter().copied())
        .collect::<BTreeSet<_>>();
    let mut grouped: Vec<(i64, Vec<usize>)> = vec![];
    for (i, b) in body.iter().enumerate() {
        if region_used.contains(&i) || inline_used.contains(&i) {
            continue;
        }
        if let Some(id) = rowid(&b["_single_run_row_id"]) {
            if let Some((_, ids)) = grouped.iter_mut().find(|(k, _)| *k == id) {
                ids.push(i)
            } else {
                grouped.push((id, vec![i]))
            }
        }
    }
    let mut used = region_used;
    used.extend(inline_used);
    let mut lines = vec![];
    for b in &body {
        let ls = optional(b, "_text_lines", 4096)?;
        if b["type"] == "text" && ls.len() >= 3 {
            lines.extend_from_slice(ls)
        }
    }
    let layout = super::native_layout::build(&lines, p, 0, &[], w)?;
    let mut virtuals: Vec<(Value, Vec<usize>, u8)> = vec![];
    for members in region_groups {
        let region = union(&body, &members)?;
        let mut sorting = region.clone();
        if layout.lanes.len() == 1 {
            let lane = &layout.lanes[0];
            let l = num(&lane["left"])?;
            let r = num(&lane["right"])?;
            let mut wraps = false;
            for (i, b) in body.iter().enumerate() {
                if members.iter().any(|&j| eq(&body[j], b)) {
                    continue;
                }
                w.charge(1)?;
                if ["text", "paragraph_title", "image", "table", "code"]
                    .iter()
                    .any(|t| b["type"] == *t)
                    && axis(&bb(b)?, &region, 1) >= 0.1
                {
                    wraps = true;
                    break;
                }
                let _ = i;
            }
            if l <= region.n[0] && region.n[2] <= r && !wraps {
                sorting = bbox(&json!([l, region.n[1], r, region.n[3]]))?;
            }
        }
        virtuals.push((json!({"type":"_xycut_visual_annotation_region","bbox":sorting.raw,"angle":body[members[0]].get("angle").cloned().unwrap_or(json!(0)),"content":""}),members,1));
    }
    for (_, members) in inline {
        let b = union(&body, &members)?;
        virtuals.push((json!({"type":"_xycut_visual_row_group","bbox":b.raw,"angle":body[members[0]].get("angle").cloned().unwrap_or(json!(0)),"content":""}),members,2));
    }
    for (_, members) in grouped {
        if members.len() < 2 {
            continue;
        }
        used.extend(&members);
        let b = union(&body, &members)?;
        virtuals.push((json!({"type":"_xycut_visual_row_group","bbox":b.raw,"angle":body[members[0]].get("angle").cloned().unwrap_or(json!(0)),"content":""}),members,0));
    }
    let mut sortable = vec![];
    let mut mappings = vec![];
    for (i, b) in body.iter().enumerate() {
        if !used.contains(&i) {
            sortable.push(b.clone());
            mappings.push((vec![i], 3));
        }
    }
    for (b, ids, kind) in virtuals {
        sortable.push(b);
        mappings.push((ids, kind))
    }
    let mut output = vec![];
    for idx in xy(&sortable, w)? {
        let (mut members, kind) = mappings[idx].clone();
        if kind == 2 {
            members.sort_by(|&i, &j| {
                inline_key(&body[i], p)
                    .unwrap()
                    .partial_cmp(&inline_key(&body[j], p).unwrap())
                    .unwrap()
            })
        } else if kind == 0 {
            let a = angle(&sortable[idx])?;
            members.sort_by(|&i, &j| {
                let a = local(&bb(&body[i]).unwrap(), p, a);
                let b = local(&bb(&body[j]).unwrap(), p, a_angle(&sortable[idx]));
                a.n[0].total_cmp(&b.n[0]).then(a.n[1].total_cmp(&b.n[1]))
            });
        }
        output.extend(members)
    }
    tail::stabilize(&mut output, &body, p, w)?;
    output = tail::leading(output, &body, p, w)?;
    output = tail::equations(output, &body, w)?;
    for (i, &old) in body_ids.iter().enumerate() {
        bs[old] = body[i].clone()
    }
    let mut out = marginal(&top, bs, p, w)?;
    out.extend(output.into_iter().map(|i| body_ids[i]));
    out.extend(marginal(&bottom, bs, p, w)?);
    Ok(out)
}
fn a_angle(b: &Value) -> i64 {
    angle(b).unwrap()
}
