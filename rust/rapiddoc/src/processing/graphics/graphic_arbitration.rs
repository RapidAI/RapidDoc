//! 旧强图形候选尾阶段：有字多列网格排除、包含拒绝、重叠合并和最终稳定排序。
use super::{
    graphic_grid,
    path_graphics::{area, array, bbox, num, overlap_first, overlap_smaller, union, Box4},
};
use crate::legacy_page::{limit, validate_tree, Work};
use num_rational::BigRational;
use rapiddoc_core::Error;
use serde_json::{json, Value};
pub(super) fn area_key(b: &Box4) -> BigRational {
    let ints: Option<Vec<_>> = b.raw.iter().map(Value::as_i64).collect();
    if let Some(v) = ints {
        BigRational::from_integer(
            ((v[2] as i128 - v[0] as i128).max(0) * (v[3] as i128 - v[1] as i128).max(0)).into(),
        )
    } else {
        BigRational::from_float(area(b)).unwrap()
    }
}
pub fn select(v: &Value) -> Result<Value, Error> {
    select_with_work(v, &mut Work(0))
}
pub(crate) fn select_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    validate_tree(v)?;
    select_source(&v["source"], v, work)
}
pub(super) fn select_source(source: &Value, v: &Value, work: &mut Work) -> Result<Value, Error> {
    crate::legacy_page::validate_source_request(source, v)?;
    let em = num(&v["median_height"])?;
    let p = graphic_grid::prepare(source, em, work)?;
    let mut candidates = array(&v["candidate_bboxes"], 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    work.charge(candidates.len() * 16)?;
    candidates.sort_by_cached_key(|b| std::cmp::Reverse(area_key(b)));
    let mut output: Vec<Box4> = Vec::new();
    let mut rejected = Vec::new();
    let mut proof_count = 0usize;
    for (i, b) in candidates.into_iter().enumerate() {
        let grids = graphic_grid::grids(&p, &b, work)?;
        if !grids.is_empty() {
            proof_count += grids.len();
            if proof_count > rapiddoc_core::resources::limit(20000) {
                return Err(limit());
            }
            rejected.push(json!({"sorted_candidate_index":i,"grid_bboxes":grids.into_iter().map(|b|b.raw).collect::<Vec<_>>()}));
            continue;
        }
        work.charge(output.len() * 2 + 1)?;
        if output.iter().any(|a| overlap_first(&b, a) >= 0.9) {
            continue;
        }
        let mut merged = None;
        let mut retained = Vec::new();
        for a in output {
            if overlap_smaller(&b, &a) >= 0.75 {
                merged = Some(match merged {
                    None => union(b.clone(), &a),
                    Some(x) => union(x, &a),
                });
            } else {
                retained.push(a)
            }
        }
        retained.push(merged.unwrap_or(b));
        output = retained;
    }
    work.charge(output.len() * 16)?;
    output.sort_by(|a, b| {
        a.n[1]
            .partial_cmp(&b.n[1])
            .unwrap()
            .then(a.n[0].partial_cmp(&b.n[0]).unwrap())
            .then(a.n[3].partial_cmp(&b.n[3]).unwrap())
            .then(a.n[2].partial_cmp(&b.n[2]).unwrap())
    });
    let out = json!({"graphic_bboxes":output.into_iter().map(|b|b.raw).collect::<Vec<_>>(),"grid_rejections":rejected});
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
    fn old_graphic_arbitration() {
        for name in [
            "native-graphic-arbitration",
            "native-graphic-arbitration-edges",
        ] {
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../build/migration/{name}.json"));
            let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
            for c in f["cases"].as_array().unwrap() {
                assert_eq!(
                    select(&c["request"]).unwrap(),
                    c["expected"],
                    "{}",
                    c["label"]
                );
            }
        }
    }
    #[test]
    fn shared_budget() {
        let q = json!({"source":{"page_size":[400,600]},"median_height":10,"candidate_bboxes":[[10,10,100,100]]});
        assert!(select_with_work(&q, &mut Work(10_000_000)).is_err());
        assert!(select(&q).is_ok());
    }
}
