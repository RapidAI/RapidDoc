//! 旧编号图题/百分比标签证明的同心轮廓图；候选随后仍需共用网格仲裁。
use super::path_graphics::{array, bbox, integer, num, overlap_first, overlap_smaller, union, Box4};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{python_decimal, text_boundary::python_space};
use serde_json::{json, Value};
fn optional<'a>(v: &'a Value, k: &str, max: usize) -> Result<&'a [Value], Error> {
    v.get(k).map_or(Ok(&[]), |v| array(v, max))
}
fn boolean(v: &Value, k: &str) -> Result<bool, Error> {
    v.get(k)
        .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))
}
fn prefix(c: &[char], offset: usize, s: &str) -> bool {
    s.chars().enumerate().all(|(j, x)| {
        c.get(offset + j).is_some_and(|c| {
            if x == 'i' {
                matches!(c, 'i' | 'I' | 'İ' | 'ı')
            } else {
                *c == x || *c == x.to_ascii_uppercase()
            }
        })
    })
}
pub(super) fn caption(s: &str) -> bool {
    let c: Vec<_> = s.trim_start_matches(python_space).chars().collect();
    if !prefix(&c, 0, "fig") {
        return false;
    }
    let mut i = 3;
    if prefix(&c, i, "ure") {
        i += 3
    }
    if c.get(i) == Some(&'.') {
        i += 1
    }
    while c.get(i).is_some_and(|c| matches!(c, ' ' | '\t')) {
        i += 1
    }
    c.get(i).is_some_and(|c| python_decimal(*c).is_some())
}
fn percentage(s: &str) -> bool {
    let c: Vec<_> = s.trim_matches(python_space).chars().collect();
    let mut i = 0;
    while c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
        i += 1
    }
    if i == 0 {
        return false;
    }
    if c.get(i).is_some_and(|c| matches!(c, '.' | ',')) {
        i += 1;
        let start = i;
        while c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
            i += 1
        }
        if i == start {
            return false;
        }
    }
    while c.get(i).is_some_and(|c| python_space(*c)) {
        i += 1
    }
    i + 1 == c.len() && c[i] == '%'
}
fn axis_overlap(a: &Box4, b: &Box4) -> f64 {
    let short = (a.n[2] - a.n[0]).min(b.n[2] - b.n[0]);
    if short > 0. {
        (a.n[2].min(b.n[2]) - a.n[0].max(b.n[0])).max(0.) / short
    } else {
        0.
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
    depth: i64,
    segments: i64,
    stroke: bool,
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
    let em = num(&v["em"])?;
    let mut captions = Vec::new();
    let mut percents = Vec::new();
    for l in optional(source, "lines", 4096)? {
        let text = l["text"].as_str().ok_or_else(invalid)?;
        work.charge(text.len() + 1)?;
        if text.chars().count() > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        let b = bbox(&l["bbox"])?;
        if l.get("angle").map_or(Ok(0.), num)? == 0. && caption(text) {
            captions.push(b.clone())
        }
        if percentage(text) {
            percents.push(b)
        }
    }
    let mut paths = Vec::new();
    for p in optional(source, "path_infos", 5000)? {
        paths.push(Path {
            b: bbox(&p["bbox"])?,
            depth: integer(p, "form_depth", 0)?,
            segments: integer(p, "segment_count", 0)?,
            stroke: boolean(p, "stroke_visible")?,
        });
    }
    work.charge(paths.len())?;
    let eligible: Vec<_> = paths
        .iter()
        .enumerate()
        .filter(|(_, p)| {
            let (w, h) = (p.b.n[2] - p.b.n[0], p.b.n[3] - p.b.n[1]);
            p.depth == 0
                && p.stroke
                && p.segments >= 7
                && w.min(h) >= 4. * em
                && (0.7..=1.3).contains(&(w / h.max(0.1)))
        })
        .map(|(i, _)| i)
        .collect();
    let mut output: Vec<Box4> = Vec::new();
    if !captions.is_empty() {
        for &i in &eligible {
            let outer = &paths[i].b;
            let diameter = (outer.n[2] - outer.n[0]).max(outer.n[3] - outer.n[1]);
            if diameter > 0.4 * size[0] || diameter > 0.4 * size[1] {
                continue;
            }
            work.charge(
                eligible.len() * 3 + captions.len() + percents.len() + paths.len() + output.len(),
            )?;
            let peers: Vec<_> = eligible
                .iter()
                .copied()
                .filter(|j| *j != i && overlap_smaller(&paths[*j].b, outer) >= 0.9)
                .collect();
            // 两个center先分别求和/除二，与旧Python求值顺序一致。
            let peers: Vec<_> = peers
                .into_iter()
                .filter(|j| {
                    (((paths[*j].b.n[0] + paths[*j].b.n[2]) / 2.)
                        - ((outer.n[0] + outer.n[2]) / 2.))
                        .abs()
                        <= 0.08 * diameter
                        && (((paths[*j].b.n[1] + paths[*j].b.n[3]) / 2.)
                            - ((outer.n[1] + outer.n[3]) / 2.))
                            .abs()
                            <= 0.08 * diameter
                })
                .collect();
            if peers.is_empty()
                || !captions.iter().any(|c| {
                    0. <= outer.n[1] - c.n[3]
                        && outer.n[1] - c.n[3] <= 10. * em
                        && axis_overlap(c, outer) >= 0.5
                })
            {
                continue;
            }
            if percents
                .iter()
                .filter(|b| distance(b, outer) <= 0.5 * diameter)
                .count()
                < 2
            {
                continue;
            }
            let mut complete = outer.clone();
            for j in peers {
                complete = union(complete, &paths[j].b)
            }
            for p in &paths {
                if p.depth == 0
                    && p.stroke
                    && (2..=5).contains(&p.segments)
                    && overlap_smaller(&p.b, outer) >= 0.1
                    && p.b.n[2] - p.b.n[0] <= 0.7 * diameter
                    && p.b.n[3] - p.b.n[1] <= 0.8 * diameter
                {
                    complete = union(complete, &p.b);
                }
            }
            if !output.iter().any(|p| overlap_first(&complete, p) >= 0.9) {
                output.push(complete)
            }
        }
    }
    let out = json!({"concentric_bboxes":output.into_iter().map(|b|b.raw).collect::<Vec<_>>()});
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
    fn old_concentric_graphics() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-concentric-graphics.json");
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
        let q = json!({"source":{"page_size":[400,600],"lines":[{"text":"Figure 1","bbox":[0,0,100,10]}]},"em":10});
        assert!(detect_with_work(&q, &mut Work(10_000_000)).is_err());
        assert!(detect(&q).is_ok());
    }
}
