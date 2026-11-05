//! 旧二维复杂路径、填充容器及相交坐标轴候选；最终图形/表格仲裁在后续阶段。
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
#[derive(Clone)]
pub(super) struct Box4 {
    pub(super) n: [f64; 4],
    pub(super) raw: [Value; 4],
}
pub(super) fn num(v: &Value) -> Result<f64, Error> {
    let n = v.as_f64().filter(|n| n.is_finite()).ok_or_else(invalid)?;
    if n.abs() > 1e9 {
        return Err(limit());
    }
    Ok(n)
}
pub(super) fn array(v: &Value, max: usize) -> Result<&[Value], Error> {
    let a = v.as_array().ok_or_else(invalid)?;
    let max = if max > 8 {
        crate::legacy_page::budget::records(max)
    } else {
        max
    };
    if a.len() > max {
        return Err(limit());
    }
    Ok(a)
}
pub(super) fn bbox(v: &Value) -> Result<Box4, Error> {
    let a = array(v, 4)?;
    if a.len() != 4 {
        return Err(invalid());
    }
    Ok(Box4 {
        n: [num(&a[0])?, num(&a[1])?, num(&a[2])?, num(&a[3])?],
        raw: std::array::from_fn(|i| a[i].clone()),
    })
}
pub(super) fn union(mut a: Box4, b: &Box4) -> Box4 {
    for i in 0..4 {
        if (i < 2 && b.n[i] < a.n[i]) || (i >= 2 && b.n[i] > a.n[i]) {
            a.n[i] = b.n[i];
            a.raw[i] = b.raw[i].clone();
        }
    }
    a
}
pub(super) fn area(b: &Box4) -> f64 {
    (b.n[2] - b.n[0]).max(0.) * (b.n[3] - b.n[1]).max(0.)
}
fn intersection(a: &Box4, b: &Box4) -> f64 {
    (a.n[2].min(b.n[2]) - a.n[0].max(b.n[0])).max(0.)
        * (a.n[3].min(b.n[3]) - a.n[1].max(b.n[1])).max(0.)
}
pub(super) fn overlap_first(a: &Box4, b: &Box4) -> f64 {
    let area = area(a);
    if area > 0. {
        intersection(a, b) / area
    } else {
        0.
    }
}
pub(super) fn overlap_smaller(a: &Box4, b: &Box4) -> f64 {
    let area = ((a.n[2] - a.n[0]) * (a.n[3] - a.n[1])).min((b.n[2] - b.n[0]) * (b.n[3] - b.n[1]));
    if area > 0. {
        intersection(a, b) / area
    } else {
        0.
    }
}
fn boolean(v: &Value, k: &str) -> Result<bool, Error> {
    v.get(k)
        .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))
}
pub(super) fn integer(v: &Value, k: &str, default: i64) -> Result<i64, Error> {
    v.get(k)
        .map_or(Ok(default), |v| v.as_i64().ok_or_else(invalid))
}
struct Path {
    b: Box4,
    depth: i64,
    segments: i64,
    source: i64,
    fill: bool,
    stroke: bool,
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
    let page = [num(&size[0])?, num(&size[1])?];
    let em = num(&v["median_height"])?;
    let mut paths = Vec::new();
    for p in array(&v["path_infos"], 5000)? {
        p.as_object().ok_or_else(invalid)?;
        paths.push(Path {
            b: bbox(&p["bbox"])?,
            depth: integer(p, "form_depth", 0)?,
            segments: integer(p, "segment_count", 0)?,
            source: integer(p, "source_index", 0)?,
            fill: boolean(p, "fill_visible")?,
            stroke: boolean(p, "stroke_visible")?,
        });
    }
    work.charge(paths.len())?;
    let complex: Vec<_> = paths
        .iter()
        .enumerate()
        .filter(|(_, p)| {
            p.depth == 0
                && p.segments >= 6
                && p.b.n[2] - p.b.n[0] >= 1.5 * em
                && p.b.n[3] - p.b.n[1] >= 1.5 * em
        })
        .map(|(i, _)| i)
        .collect();
    let page_area = (page[0] * page[1]).max(0.1);
    let mut containers = Vec::new();
    for p in &paths {
        let b = &p.b;
        let ratio = area(b) / page_area;
        if p.depth != 0
            || !p.fill
            || !(0.005..=0.5).contains(&ratio)
            || b.n[2] - b.n[0] < 4. * em
            || b.n[3] - b.n[1] < 4. * em
        {
            continue;
        }
        work.charge(paths.len())?;
        let inner: Vec<_> = paths
            .iter()
            .enumerate()
            .filter(|(_, q)| {
                q.source != p.source && overlap_first(&q.b, b) >= 0.9 && area(&q.b) < 0.95 * area(b)
            })
            .map(|(i, _)| i)
            .collect();
        work.charge(inner.len())?;
        if inner.len() >= 4 && inner.iter().any(|i| complex.binary_search(i).is_ok()) {
            containers.push(b.raw.clone());
            if containers.len() > rapiddoc_core::resources::limit(5000) {
                return Err(limit());
            }
        }
    }
    let thin = 1f64.max(0.5 * em);
    let length = 6. * em;
    let tolerance = 2f64.max(em);
    let horizontal: Vec<_> = paths
        .iter()
        .filter(|p| {
            p.depth == 0 && p.stroke && p.b.n[3] - p.b.n[1] <= thin && p.b.n[2] - p.b.n[0] >= length
        })
        .collect();
    let vertical: Vec<_> = paths
        .iter()
        .filter(|p| {
            p.depth == 0 && p.stroke && p.b.n[2] - p.b.n[0] <= thin && p.b.n[3] - p.b.n[1] >= length
        })
        .collect();
    let mut axis = Vec::new();
    for h in &horizontal {
        work.charge(vertical.len())?;
        let y = (h.b.n[1] + h.b.n[3]) / 2.;
        // 在原纵轴顺序扫描后执行原距离条件，等价于旧ULP扩展索引候选的最终筛选。
        for v in &vertical {
            let x = (v.b.n[0] + v.b.n[2]) / 2.;
            if (x - h.b.n[0]).abs().min((x - h.b.n[2]).abs()) > tolerance
                || (y - v.b.n[1]).abs().min((y - v.b.n[3]).abs()) > tolerance
            {
                continue;
            }
            let plot = union(h.b.clone(), &v.b);
            if plot.n[2] - plot.n[0] > 0.65 * page[0] || plot.n[3] - plot.n[1] > 0.5 * page[1] {
                continue;
            }
            work.charge(complex.len())?;
            let mut matched = None;
            for &i in &complex {
                if overlap_smaller(&paths[i].b, &plot) >= 0.2 {
                    matched = Some(match matched {
                        None => paths[i].b.clone(),
                        Some(b) => union(b, &paths[i].b),
                    });
                }
            }
            if let Some(b) = matched {
                axis.push(union(plot, &b).raw);
                if axis.len() + containers.len() > rapiddoc_core::resources::limit(5000) {
                    return Err(limit());
                }
            }
        }
    }
    let out =
        json!({"complex_path_indices":complex,"container_bboxes":containers,"axis_bboxes":axis});
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
    fn old_path_graphics() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-path-graphics.json");
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
        let q =
            json!({"page_size":[400,600],"median_height":10,"path_infos":[{"bbox":[0,0,100,100]}]});
        assert!(detect_with_work(&q, &mut Work(10_000_000)).is_err());
        assert!(detect(&q).is_ok());
    }
}
