//! 旧完整表格替换仲裁：交集并面积不足时拒绝替代候选，不重复累计覆盖。
use super::path_graphics::{array, bbox, overlap_smaller, Box4};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
fn bounds(v: &Value) -> Result<Box4, Error> {
    v.as_object().ok_or_else(invalid)?;
    bbox(
        v.get("core_bbox")
            .filter(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
            .unwrap_or(&v["bbox"]),
    )
}
pub fn prefer(v: &Value) -> Result<Value, Error> {
    prefer_with_work(v, &mut Work(0))
}
pub(crate) fn prefer_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    validate_tree(v)?;
    let base = array(&v["base"], 5000)?;
    let recovered = array(&v["recovered"], 5000)?;
    let replacements = array(&v["replacements"], 5000)?;
    let bs = base.iter().map(bounds).collect::<Result<Vec<_>, _>>()?;
    let rs = recovered
        .iter()
        .map(bounds)
        .collect::<Result<Vec<_>, _>>()?;
    let new = replacements
        .iter()
        .map(bounds)
        .collect::<Result<Vec<_>, _>>()?;
    work.charge(bs.len() + rs.len() + new.len() + 1)?;
    let mut rejected = vec![false; new.len()];
    for old in bs.iter().chain(&rs) {
        work.charge(new.len() + 1)?;
        let overlapping: Vec<_> = new
            .iter()
            .enumerate()
            .filter(|(_, b)| overlap_smaller(old, b) >= 0.9)
            .map(|(i, _)| i)
            .collect();
        if overlapping.is_empty() {
            continue;
        }
        let pieces: Vec<_> = overlapping
            .iter()
            .map(|i| {
                let b = &new[*i];
                [
                    old.n[0].max(b.n[0]),
                    old.n[1].max(b.n[1]),
                    old.n[2].min(b.n[2]),
                    old.n[3].min(b.n[3]),
                ]
            })
            .collect();
        let mut edges: Vec<_> = pieces.iter().flat_map(|b| [b[0], b[2]]).collect();
        work.charge(edges.len() * 16 + 1)?;
        edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
        edges.dedup_by(|a, b| *a == *b);
        let mut area = 0.;
        for edge in edges.windows(2) {
            work.charge(pieces.len() + 1)?;
            let mut spans: Vec<_> = pieces
                .iter()
                .filter(|b| b[0] <= edge[0] && edge[1] <= b[2])
                .map(|b| (b[1], b[3]))
                .collect();
            work.charge(spans.len() * 16 + 1)?;
            spans.sort_by(|a, b| {
                a.0.partial_cmp(&b.0)
                    .unwrap()
                    .then(a.1.partial_cmp(&b.1).unwrap())
            });
            let (mut end, mut height) = (f64::NEG_INFINITY, 0.);
            for (top, bottom) in spans {
                height += (bottom - top.max(end)).max(0.);
                end = end.max(bottom);
            }
            area += (edge[1] - edge[0]) * height;
        }
        // Keep the original multiplication order and sequential float additions.
        if area < 0.9 * (old.n[2] - old.n[0]) * (old.n[3] - old.n[1]) {
            for i in overlapping {
                rejected[i] = true;
            }
        }
    }
    let retained_new: Vec<_> = new
        .iter()
        .enumerate()
        .filter(|(i, _)| !rejected[*i])
        .collect();
    let mut keep = |raw: &[Value], boxes: &[Box4]| -> Result<Vec<Value>, Error> {
        let mut output = Vec::new();
        for (v, b) in raw.iter().zip(boxes) {
            work.charge(retained_new.len() + 1)?;
            if !retained_new
                .iter()
                .any(|(_, n)| overlap_smaller(b, n) >= 0.9)
            {
                output.push(v.clone());
            }
        }
        Ok(output)
    };
    let out = json!({"base":keep(base,&bs)?,"recovered":keep(recovered,&rs)?,"replacements":replacements.iter().enumerate().filter(|(i,_)|!rejected[*i]).map(|(_,v)|v.clone()).collect::<Vec<_>>()});
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
    fn old_replacements() {
        let f: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-table-replacements.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                prefer(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
    #[test]
    fn shared_budget() {
        let v = json!({"base":[],"recovered":[],"replacements":[]});
        assert!(prefer_with_work(&v, &mut Work(10_000_000)).is_err());
        assert!(prefer(&v).is_ok());
    }
}
