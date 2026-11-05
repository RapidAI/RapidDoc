//! 旧绘图线连通分量/统计及含二维复杂路径的图形候选。
use super::path_graphics::{array, bbox, integer, num, overlap_smaller, union, Box4};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
use std::collections::BTreeMap;
fn find(parents: &mut [usize], mut i: usize) -> usize {
    while parents[i] != i {
        parents[i] = parents[parents[i]];
        i = parents[i];
    }
    i
}
fn distance(a: &Box4, b: &Box4) -> f64 {
    let x = (a.n[0] - b.n[2]).max(b.n[0] - a.n[2]).max(0.);
    let y = (a.n[1] - b.n[3]).max(b.n[1] - a.n[3]).max(0.);
    rapiddoc_pdf::legacy_vectors::hypot(x, y)
}
pub fn detect(v: &Value) -> Result<Value, Error> {
    detect_with_work(v, &mut Work(0))
}
pub(crate) fn detect_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    validate_tree(v)?;
    let size = array(&v["page_size"], 2)?;
    if size.len() != 2 {
        return Err(invalid());
    }
    let size = [num(&size[0])?, num(&size[1])?];
    let em = num(&v["median_height"])?;
    let tolerance = 2f64.max(0.75 * em);
    let mut lines = Vec::new();
    let mut horizontal = Vec::new();
    for l in array(&v["drawing_lines"], 5000)? {
        l.as_object().ok_or_else(invalid)?;
        lines.push(bbox(&l["bbox"])?);
        let direction = l["orientation"].as_str().ok_or_else(invalid)?;
        if !matches!(direction, "horizontal" | "vertical") {
            return Err(invalid());
        }
        horizontal.push(direction == "horizontal");
    }
    let mut complex = Vec::new();
    let mut complex_indices = Vec::new();
    for (i, p) in array(&v["path_infos"], 5000)?.iter().enumerate() {
        let b = bbox(&p["bbox"])?;
        if integer(p, "form_depth", 0)? == 0
            && integer(p, "segment_count", 0)? >= 6
            && b.n[2] - b.n[0] >= 1.5 * em
            && b.n[3] - b.n[1] >= 1.5 * em
        {
            complex.push(b);
            complex_indices.push(i);
        }
    }
    work.charge(lines.len() * (log(lines.len()) + 2) + complex.len())?;
    let mut parents: Vec<_> = (0..lines.len()).collect();
    let mut order = parents.clone();
    order.sort_by(|a, b| lines[*a].n[0].partial_cmp(&lines[*b].n[0]).unwrap());
    for (pos, &i) in order.iter().enumerate() {
        let end = lines[i].n[2] + tolerance;
        for &j in &order[pos + 1..] {
            work.charge(1)?;
            if lines[j].n[0] > end {
                break;
            }
            if distance(&lines[i], &lines[j]) <= tolerance {
                let a = find(&mut parents, i);
                let b = find(&mut parents, j);
                if a != b {
                    parents[b] = a;
                }
            }
        }
    }
    // Python dict以原始行顺序首次遇到的root插入，不能按root数值重排分量。
    let mut positions = BTreeMap::new();
    let mut components: Vec<Vec<usize>> = Vec::new();
    for i in 0..lines.len() {
        let root = find(&mut parents, i);
        let next = components.len();
        let position = *positions.entry(root).or_insert(next);
        if position == components.len() {
            components.push(Vec::new())
        }
        components[position].push(i);
    }
    let mut summaries = Vec::new();
    let mut output = Vec::new();
    for ids in components {
        work.charge(ids.len() + 1)?;
        let core = ids
            .iter()
            .map(|i| lines[*i].clone())
            .reduce(|a, b| union(a, &b))
            .unwrap();
        let h = ids.iter().filter(|i| horizontal[**i]).count();
        let vert = ids.len() - h;
        let (w, height) = (core.n[2] - core.n[0], core.n[3] - core.n[1]);
        if ids.len() >= 4
            && h >= 2
            && vert >= 2
            && w >= 4. * em
            && height >= 3. * em
            && w <= 0.65 * size[0]
            && height <= 0.5 * size[1]
        {
            work.charge(complex.len())?;
            let mut matched = None;
            for p in &complex {
                if overlap_smaller(p, &core) >= 0.2 {
                    matched = Some(match matched {
                        None => p.clone(),
                        Some(b) => union(b, p),
                    });
                }
            }
            if let Some(b) = matched {
                output.push(union(core.clone(), &b).raw);
            }
        }
        summaries.push(
            json!({"line_indices":ids,"bbox":core.raw,"horizontal_count":h,"vertical_count":vert}),
        );
    }
    let out = json!({"tolerance":tolerance,"complex_path_indices":complex_indices,"components":summaries,"graphic_bboxes":output});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(out)
}
fn log(n: usize) -> usize {
    if n < 2 {
        1
    } else {
        usize::BITS as usize - (n - 1).leading_zeros() as usize + 1
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_drawing_components() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-drawing-graphics.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                detect(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
    #[test]
    fn shared_budget() {
        let q = json!({"page_size":[400,600],"median_height":10,"drawing_lines":[{"bbox":[0,0,100,0],"orientation":"horizontal"}],"path_infos":[]});
        assert!(detect_with_work(&q, &mut Work(10_000_000)).is_err());
        assert!(detect(&q).is_ok());
    }
}
