//! 旧无文字矢量图体和小栅格判定；按旧增量合并次序返回独立候选。
use super::path_graphics::{
    area, array, bbox, integer, num, overlap_first, overlap_smaller, union, Box4,
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
use std::collections::BTreeSet;
fn optional<'a>(v: &'a Value, k: &str, max: usize) -> Result<&'a [Value], Error> {
    v.get(k).map_or(Ok(&[]), |v| array(v, max))
}
fn boolean(v: &Value, k: &str) -> Result<bool, Error> {
    v.get(k)
        .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.
    }
}
fn distance(a: &Box4, b: &Box4) -> f64 {
    rapiddoc_pdf::legacy_vectors::hypot(
        (a.n[0] - b.n[2]).max(b.n[0] - a.n[2]).max(0.),
        (a.n[1] - b.n[3]).max(b.n[1] - a.n[3]).max(0.),
    )
}
struct Path {
    b: Box4,
    segments: i64,
    fill: bool,
    color: Option<[u32; 4]>,
    index: usize,
}
pub fn detect(v: &Value) -> Result<Value, Error> {
    detect_with_work(v, &mut Work(0))
}
pub(crate) fn detect_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    validate_tree(v)?;
    detect_source(&v["source"], v, work)
}
pub(super) fn detect_source(source: &Value, v: &Value, work: &mut Work) -> Result<Value, Error> {
    crate::legacy_page::validate_source_request(source, v)?;
    source.as_object().ok_or_else(invalid)?;
    let size = array(&source["page_size"], 2)?;
    if size.len() != 2 {
        return Err(invalid());
    }
    let size = [num(&size[0])?, num(&size[1])?];
    let page_area = size[0] * size[1];
    let mut lines = Vec::new();
    let mut heights = Vec::new();
    for l in optional(source, "lines", 4096)? {
        let b = bbox(&l["bbox"])?;
        if l.get("angle").map_or(Ok(0.), num)? == 0. {
            let em = l.get("em_height").map_or(Ok(0.), num)?;
            let effective = l.get("effective_height").map_or(Ok(0.), num)?;
            let h = if boolean(l, "style_scale_repaired")? && em > 0. {
                em
            } else if effective != 0. {
                effective
            } else {
                b.n[3] - b.n[1]
            };
            heights.push(h.max(0.1));
        }
        lines.push(b);
    }
    work.charge(lines.len() * (log(lines.len()) + 1))?;
    let has_height = !heights.is_empty();
    let em = if has_height { median(heights) } else { 10. };
    let rasters = optional(source, "image_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let mut paths = Vec::new();
    for (index, p) in optional(source, "path_infos", 5000)?.iter().enumerate() {
        let b = bbox(&p["bbox"])?;
        let fill = boolean(p, "fill_visible")?;
        let stroke = boolean(p, "stroke_visible")?;
        let depth = integer(p, "form_depth", 0)?;
        let segments = integer(p, "segment_count", 0)?;
        let rectangles = optional(p, "rectangle_bboxes", 20000)?;
        let a = area(&b);
        if depth != 0
            || !(fill || stroke)
            || !rectangles.is_empty()
            || segments < 8
            || a <= 0.
            || a > 0.06 * page_area
        {
            continue;
        }
        let color = match p.get("fill_rgba") {
            None | Some(Value::Null) => None,
            Some(v) => {
                let a = array(v, 4)?;
                if a.len() != 4 {
                    return Err(invalid());
                }
                let mut c = [0; 4];
                for (i, v) in a.iter().enumerate() {
                    c[i] = v.as_u64().filter(|v| *v <= 255).ok_or_else(invalid)? as u32;
                }
                Some(c)
            }
        };
        paths.push(Path {
            b,
            segments,
            fill,
            color,
            index,
        });
    }
    work.charge(paths.len() + rasters.len())?;
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in 0..paths.len() {
        let mut retained = Vec::new();
        let mut merged = vec![i];
        for group in groups {
            work.charge(1)?;
            let mut connected = false;
            for &j in &group {
                work.charge(1)?;
                if distance(&paths[i].b, &paths[j].b) <= 0.35 * em {
                    connected = true;
                    break;
                }
            }
            if connected {
                work.charge(group.len())?;
                merged.extend(group);
            } else {
                retained.push(group);
            }
        }
        retained.push(merged);
        groups = retained;
    }
    let mut output = Vec::new();
    for group in &groups {
        work.charge(group.len() * (log(group.len()) + 3) + 1)?;
        let b = group
            .iter()
            .map(|i| paths[*i].b.clone())
            .reduce(|a, b| union(a, &b))
            .unwrap();
        let a = area(&b) / page_area;
        let segments = group.iter().try_fold(0i64, |sum, i| {
            sum.checked_add(paths[*i].segments).ok_or_else(limit)
        })?;
        if !(0.0005..=0.06).contains(&a)
            || (b.n[2] - b.n[0]).min(b.n[3] - b.n[1]) < 2.5 * em
            || segments < 30
            || !group.iter().any(|i| {
                paths[*i].segments >= 20
                    && (paths[*i].b.n[2] - paths[*i].b.n[0])
                        .min(paths[*i].b.n[3] - paths[*i].b.n[1])
                        >= 1.5 * em
            })
        {
            continue;
        }
        let edges = [
            b.n[0] <= 0.2 * em,
            b.n[1] <= 0.2 * em,
            b.n[2] >= size[0] - 0.2 * em,
            b.n[3] >= size[1] - 0.2 * em,
        ]
        .into_iter()
        .filter(|b| *b)
        .count();
        let colors: BTreeSet<_> = group
            .iter()
            .filter(|i| paths[**i].fill)
            .map(|i| paths[*i].color)
            .collect();
        if edges >= 2 && (group.len() < 8 || colors.len() < 3) {
            continue;
        }
        work.charge(lines.len() + rasters.len())?;
        if lines.iter().any(|l| overlap_first(l, &b) > 0.1)
            || rasters.iter().any(|r| overlap_smaller(&b, r) > 0.2)
        {
            continue;
        }
        output.push(b.raw);
    }
    let mut meaningful = Vec::new();
    if has_height {
        for (i, b) in rasters.iter().enumerate() {
            work.charge(1)?;
            if (b.n[2] - b.n[0]).min(b.n[3] - b.n[1]) >= 3. * em
                && area(b) >= 0.0005 * size[0] * size[1]
            {
                work.charge(lines.len())?;
                if !lines.iter().any(|l| overlap_first(l, b) > 0.01) {
                    meaningful.push(i)
                }
            }
        }
    }
    let path_groups: Vec<_> = groups
        .into_iter()
        .map(|g| g.into_iter().map(|i| paths[i].index).collect::<Vec<_>>())
        .collect();
    let out = json!({"em":em,"path_groups":path_groups,"vector_bboxes":output,"meaningful_raster_indices":meaningful});
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
    fn old_isolated_graphics() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-isolated-graphics.json");
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
        let q = json!({"source":{"page_size":[400,600],"lines":[{"bbox":[0,0,10,10]}]}});
        assert!(detect_with_work(&q, &mut Work(10_000_000)).is_err());
        assert!(detect(&q).is_ok());
    }
}
