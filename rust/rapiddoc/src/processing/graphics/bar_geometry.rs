//! 旧原生柱图的复合基线、柱体准入和零轴补全；上游填充表格框明确传入。
use super::path_graphics::{array, bbox, integer, num, overlap_first, Box4};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
fn optional<'a>(v: &'a Value, k: &str, max: usize) -> Result<&'a [Value], Error> {
    v.get(k).map_or(Ok(&[]), |v| array(v, max))
}
fn boolean(v: &Value, k: &str) -> Result<bool, Error> {
    v.get(k)
        .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))
}
pub fn prepare(v: &Value) -> Result<Value, Error> {
    prepare_with_work(v, &mut Work(0))
}
pub(crate) fn prepare_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    validate_tree(v)?;
    let em = num(&v["em"])?;
    let tables = array(&v["table_bboxes"], 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let paths = array(&v["path_infos"], 5000)?;
    let mut rectangles = Vec::new();
    let mut compound = Vec::new();
    let mut total = 0;
    for (i, p) in paths.iter().enumerate() {
        p.as_object().ok_or_else(invalid)?;
        let rects = optional(p, "rectangle_bboxes", 20000)?
            .iter()
            .map(bbox)
            .collect::<Result<Vec<_>, _>>()?;
        total += rects.len();
        if total > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        if boolean(p, "fill_visible")? && rects.len() > 1 {
            let count = rects.len();
            work.charge(
                count
                    .checked_mul(count)
                    .and_then(|n| n.checked_mul(4))
                    .ok_or_else(limit)?,
            )?;
            compound.push((i, rects.iter().map(|b| b.n).collect::<Vec<_>>()));
        }
        rectangles.push(rects);
    }
    let flags = rapiddoc_native::bar_geometry::compound_baselines(
        &compound.iter().map(|(_, v)| v.clone()).collect::<Vec<_>>(),
        em,
    );
    let mut all_flags = vec![Value::Null; paths.len()];
    for ((i, _), f) in compound.iter().zip(flags.iter()) {
        all_flags[*i] = json!([f.0, f.1]);
    }
    let mut admitted = Vec::new();
    for (i, p) in paths.iter().enumerate() {
        let rects = &rectangles[i];
        let fill = boolean(p, "fill_visible")?;
        let seg = integer(p, "segment_count", 0)?;
        let shared = if rects.len() <= 1 {
            [true, true]
        } else if all_flags[i].is_null() {
            [false, false]
        } else {
            [
                all_flags[i][0].as_bool().unwrap(),
                all_flags[i][1].as_bool().unwrap(),
            ]
        };
        if !fill || !(shared[0] || shared[1]) || !(rects.len() > 1 || (4..=24).contains(&seg)) {
            continue;
        }
        let Some(color) = p.get("fill_rgba").filter(|v| !v.is_null()) else {
            continue;
        };
        let color = array(color, 4)?;
        if color.len() != 4 {
            return Err(invalid());
        }
        let color = color
            .iter()
            .map(|v| v.as_u64().filter(|n| *n <= 255).ok_or_else(invalid))
            .collect::<Result<Vec<_>, _>>()?;
        if *color[..3].iter().min().unwrap() >= 230 {
            continue;
        }
        let candidates: Vec<(Option<usize>, Box4)> = if rects.len() > 1 {
            rects
                .iter()
                .cloned()
                .enumerate()
                .map(|(j, b)| (Some(j), b))
                .collect()
        } else {
            vec![(None, bbox(&p["bbox"])?)]
        };
        for (j, b) in candidates {
            work.charge(tables.len() + 1)?;
            let (w, h) = (b.n[2] - b.n[0], b.n[3] - b.n[1]);
            if w.max(h) < 2. * em || tables.iter().any(|t| overlap_first(&b, t) >= 0.95) {
                continue;
            }
            admitted.push(json!({"path_index":i,"rectangle_index":j,"bbox":b.raw,"horizontal":shared[0]&&0.2*em<=h&&h<=5.*em&&w>=1.25*h,"vertical":shared[1]&&0.2*em<=w&&w<=5.*em&&h>=1.25*w}));
            if admitted.len() > rapiddoc_core::resources::limit(20000) {
                return Err(limit());
            }
        }
    }
    let queries = optional(v, "axis_queries", 5000)?;
    let rules = optional(v, "drawing_lines", 5000)?;
    let mut axes = Vec::new();
    let mut total_axes = 0;
    for q in queries {
        q.as_object().ok_or_else(invalid)?;
        let bounds = bbox(&q["bounds"])?;
        let horizontal = q["horizontal"].as_bool().ok_or_else(invalid)?;
        let baseline = num(&q["baseline"])?;
        work.charge(rules.len() + 1)?;
        let mut boxes = Vec::new();
        for r in rules {
            let b = bbox(&r["bbox"])?;
            let (start, end, low, high, cross_start, cross_end) = if horizontal {
                (b.n[1], b.n[3], bounds.n[1], bounds.n[3], b.n[0], b.n[2])
            } else {
                (b.n[0], b.n[2], bounds.n[0], bounds.n[2], b.n[1], b.n[3])
            };
            let length = high - low;
            if r["orientation"].as_str() == Some(if horizontal { "vertical" } else { "horizontal" })
                && cross_end - cross_start <= 0.25 * em
                && ((cross_start + cross_end) / 2. - baseline).abs() <= 0.35 * em
                && 0.8 * length <= end - start
                && end - start <= 2. * length
                && (end.min(high) - start.max(low)).max(0.) >= 0.8 * length
            {
                let padding = num(&r["width"])?.max(0.) / 2.;
                let mut out = b.raw;
                if horizontal {
                    out[1] = json!(b.n[1] - padding);
                    out[3] = json!(b.n[3] + padding)
                } else {
                    out[0] = json!(b.n[0] - padding);
                    out[2] = json!(b.n[2] + padding)
                }
                boxes.push(out);
                total_axes += 1;
                if total_axes > rapiddoc_core::resources::limit(20000) {
                    return Err(limit());
                }
            }
        }
        axes.push(boxes);
    }
    let out =
        json!({"compound_baselines":all_flags,"admitted_bars":admitted,"zero_axis_bboxes":axes});
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
    fn old_bar_geometry() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-bar-geometry.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                prepare(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
    #[test]
    fn shared_budget() {
        let q = json!({"path_infos":[],"drawing_lines":[],"table_bboxes":[],"em":10,"axis_queries":[{"bounds":[0,0,100,100],"horizontal":true,"baseline":0}]});
        assert!(prepare_with_work(&q, &mut Work(10_000_000)).is_err());
        assert!(prepare(&q).is_ok());
    }
}
