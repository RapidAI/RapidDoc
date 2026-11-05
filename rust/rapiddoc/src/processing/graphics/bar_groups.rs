//! 旧柱图前段：真实表框/柱体准备→共同零轴/间隔簇→数字证据和标签锚点。
use super::path_graphics::{array, bbox, num, overlap_first, union, Box4};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{python_decimal, text_boundary::python_space};
use serde_json::{json, Value};
pub(super) fn numeric(s: &str) -> bool {
    let mut it = s.trim_matches(python_space).chars().peekable();
    if it.peek().is_some_and(|c| matches!(c, '+' | '-' | '−')) {
        it.next();
    }
    let mut count = 0;
    while it.peek().is_some_and(|c| python_decimal(*c).is_some()) {
        it.next();
        count += 1;
    }
    if count == 0 {
        return false;
    }
    if it.peek().is_some_and(|c| matches!(c, ',' | '.')) {
        it.next();
        let mut digits = 0;
        while it.peek().is_some_and(|c| python_decimal(*c).is_some()) {
            it.next();
            digits += 1;
        }
        if digits == 0 {
            return false;
        }
    }
    if it.peek() == Some(&'%') {
        it.next();
    }
    it.next().is_none()
}
fn distance(a: &Box4, b: &Box4) -> f64 {
    let dx = (a.n[0] - b.n[2]).max(b.n[0] - a.n[2]).max(0.);
    let dy = (a.n[1] - b.n[3]).max(b.n[1] - a.n[3]).max(0.);
    rapiddoc_pdf::legacy_vectors::hypot(dx, dy)
}
struct Group {
    horizontal: bool,
    baseline: Value,
    members: Vec<usize>,
    numeric: Vec<usize>,
    core: Box4,
}
pub fn prepare(v: &Value) -> Result<Value, Error> {
    prepare_with_work(v, &mut Work(0))
}
pub(crate) fn prepare_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_source(&v["source"], v, work)
}
pub(super) fn prepare_source(source: &Value, v: &Value, work: &mut Work) -> Result<Value, Error> {
    crate::legacy_page::validate_source_request(source, v)?;
    source.as_object().ok_or_else(invalid)?;
    let em = num(&v["em"])?;
    if em <= 0. {
        return Err(invalid());
    }
    let paths = source
        .get("path_infos")
        .map_or(Ok(&[][..]), |v| array(v, 5000))?;
    let rules = source
        .get("drawing_lines")
        .map_or(Ok(&[][..]), |v| array(v, 5000))?;
    let lines = source
        .get("lines")
        .map_or(Ok(&[][..]), |v| array(v, 4096))?;
    let tables = super::tiled_tables::detect_with_work(
        &json!({"source":source,"excluded_bboxes":[]}),
        work,
    )?;
    let table_boxes: Vec<_> = tables["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["bbox"].clone())
        .collect();
    let admitted = super::bar_geometry::prepare_with_work(
        &json!({"path_infos":paths,"drawing_lines":[],"em":em,"table_bboxes":table_boxes}),
        work,
    )?["admitted_bars"]
        .as_array()
        .unwrap()
        .clone();
    let boxes = admitted
        .iter()
        .map(|p| bbox(&p["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut numeric_lines = None;
    let mut groups = Vec::new();
    for horizontal in [true, false] {
        let bars: Vec<_> = admitted
            .iter()
            .enumerate()
            .filter(|(_, b)| {
                b[if horizontal { "horizontal" } else { "vertical" }]
                    .as_bool()
                    .unwrap()
            })
            .map(|(i, _)| i)
            .collect();
        if bars.len() > rapiddoc_core::resources::limit(5000) {
            return Err(limit());
        }
        let mut pending = super::tiled_tables::pending::Pending::new(bars.len());
        loop {
            work.charge(pending.capacity() + 1)?;
            let order = pending.ordered();
            let Some(&seed) = order.iter().min() else {
                break;
            };
            let edges = if horizontal { [0, 2] } else { [3, 1] };
            work.charge(order.len() * 6 + 1)?;
            let mut baseline = boxes[bars[seed]].n[edges[0]];
            let mut edge = edges[0];
            let mut support = 0;
            for e in edges {
                let value = boxes[bars[seed]].n[e];
                let count = order
                    .iter()
                    .filter(|i| {
                        edges
                            .iter()
                            .any(|e| (boxes[bars[**i]].n[*e] - value).abs() <= 0.35 * em)
                    })
                    .count();
                if count > support {
                    support = count;
                    baseline = value;
                    edge = e;
                }
            }
            let mut group: Vec<_> = order
                .into_iter()
                .filter(|i| {
                    edges
                        .iter()
                        .any(|e| (boxes[bars[*i]].n[*e] - baseline).abs() <= 0.35 * em)
                })
                .collect();
            pending.difference(&group);
            work.charge(group.len() * 16 + pending.capacity())?;
            let category = if horizontal { 1 } else { 0 };
            let length = 1 - category;
            group.sort_by(|a, b| {
                boxes[bars[*a]].n[category]
                    .partial_cmp(&boxes[bars[*b]].n[category])
                    .unwrap()
            });
            let mut clusters: Vec<Vec<usize>> = vec![Vec::new()];
            for i in group {
                let b = &boxes[bars[i]];
                let current = clusters.last_mut().unwrap();
                if let Some(&j) = current.last() {
                    let prior = &boxes[j];
                    if b.n[category] - prior.n[category + 2]
                        > (5. * em).max(2. * (b.n[category + 2] - b.n[category]))
                    {
                        clusters.push(Vec::new());
                    }
                }
                clusters.last_mut().unwrap().push(bars[i]);
            }
            for members in clusters {
                if members.is_empty() {
                    continue;
                }
                work.charge(members.len() * 32 + 1)?;
                // CPython round(x,1) rounds the exact input double, not x*10 rounded first.
                let mut positions: Vec<f64> = members
                    .iter()
                    .map(|i| {
                        format!("{:.1}", boxes[*i].n[category] / em)
                            .parse::<f64>()
                            .unwrap()
                    })
                    .collect();
                positions.sort_by(|a, b| a.partial_cmp(b).unwrap());
                positions.dedup_by(|a, b| *a == *b);
                let lengths: Vec<_> = members
                    .iter()
                    .map(|i| boxes[*i].n[length + 2] - boxes[*i].n[length])
                    .collect();
                if positions.len() < 2
                    || lengths.iter().copied().reduce(f64::max).unwrap()
                        - lengths.iter().copied().reduce(f64::min).unwrap()
                        < 0.5 * em
                {
                    continue;
                }
                let mut intervals: Vec<_> = members
                    .iter()
                    .map(|i| (boxes[*i].n[category], boxes[*i].n[category + 2]))
                    .collect();
                intervals.sort_by(|a, b| {
                    a.0.partial_cmp(&b.0)
                        .unwrap()
                        .then_with(|| a.1.partial_cmp(&b.1).unwrap())
                });
                intervals.dedup_by(|a, b| *a == *b);
                if !intervals.windows(2).any(|p| p[1].0 - p[0].1 > 0.1 * em) {
                    continue;
                }
                let mut core = boxes[members[0]].clone();
                for i in &members[1..] {
                    core = union(core, &boxes[*i]);
                }
                if numeric_lines.is_none() {
                    let mut prepared = Vec::new();
                    for (i, l) in lines.iter().enumerate() {
                        let text = l["text"].as_str().ok_or_else(invalid)?;
                        work.charge(text.len() + 1)?;
                        if text.chars().count() > rapiddoc_core::resources::limit(20000) {
                            return Err(limit());
                        }
                        if numeric(text) {
                            prepared.push((i, bbox(&l["bbox"])?));
                        }
                    }
                    numeric_lines = Some(prepared);
                }
                let candidates = numeric_lines.as_ref().unwrap();
                work.charge(
                    candidates
                        .len()
                        .checked_mul(members.len() + 1)
                        .ok_or_else(limit)?,
                )?;
                let numbers: Vec<_> = candidates
                    .iter()
                    .filter(|(_, b)| {
                        members.iter().all(|i| overlap_first(b, &boxes[*i]) < 0.1)
                            && distance(b, &core) <= 5. * em
                    })
                    .map(|(i, _)| *i)
                    .collect();
                if numbers.len() < 2
                    || core.n[2] - core.n[0] < 6. * em
                    || core.n[3] - core.n[1] < 3. * em
                {
                    continue;
                }
                groups.push(Group {
                    horizontal,
                    baseline: boxes[bars[seed]].raw[edge].clone(),
                    members,
                    numeric: numbers,
                    core,
                });
                if groups.len() > rapiddoc_core::resources::limit(5000) {
                    return Err(limit());
                }
            }
        }
    }
    let queries: Vec<_> = groups
        .iter()
        .map(|g| json!({"bounds":g.core.raw,"horizontal":g.horizontal,"baseline":g.baseline}))
        .collect();
    let axes = super::bar_geometry::prepare_with_work(
        &json!({"path_infos":[],"drawing_lines":rules,"em":em,"table_bboxes":[],"axis_queries":queries}),
        work,
    )?;
    let mut out_groups = Vec::new();
    for (i, g) in groups.into_iter().enumerate() {
        let mut bounds = g.core.clone();
        for b in axes["zero_axis_bboxes"][i].as_array().unwrap() {
            work.charge(1)?;
            bounds = union(bounds, &bbox(b)?);
        }
        let mut anchor = bounds.clone();
        for j in &g.numeric {
            work.charge(1)?;
            anchor = union(anchor, &bbox(&lines[*j]["bbox"])?);
        }
        out_groups.push(json!({"horizontal":g.horizontal,"baseline":g.baseline,"bar_indices":g.members,"numeric_line_indices":g.numeric,"core_bbox":g.core.raw,"bbox":bounds.raw,"label_anchor":anchor.raw}));
    }
    let out = json!({"table_bboxes":table_boxes,"admitted_bars":admitted,"groups":out_groups});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_bar_groups() {
        for name in ["native-bar-groups", "native-bar-group-boundaries"] {
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
                    prepare(&c["request"]).unwrap(),
                    c["expected"],
                    "{}",
                    c["label"]
                );
            }
            if let Some(rounding) = f.get("rounding") {
                for r in rounding.as_array().unwrap() {
                    let v = r["value"].as_f64().unwrap();
                    assert_eq!(
                        format!("{v:.1}").parse::<f64>().unwrap(),
                        r["expected"].as_f64().unwrap(),
                        "round {v}"
                    );
                }
            }
        }
    }

    #[test]
    fn shared_budget() {
        let v = json!({"source":{"page_size":[400,600]},"em":10});
        assert!(prepare_with_work(&v, &mut Work(10_000_000)).is_err());
        assert!(prepare(&v).is_ok());
    }
}
