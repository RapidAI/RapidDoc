//! 旧复杂路径 diagram 完整链：真实表格屏障、正文拒绝及时间轴认领。
use super::path_graphics::{
    area, array, bbox, integer, num, overlap_first, overlap_smaller, union, Box4,
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
fn optional<'a>(v: &'a Value, k: &str, max: usize) -> Result<&'a [Value], Error> {
    v.get(k)
        .filter(|v| !v.is_null())
        .map_or(Ok(&[]), |v| array(v, max))
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
fn combined(boxes: impl IntoIterator<Item = Box4>) -> Result<Box4, Error> {
    let mut it = boxes.into_iter();
    let mut b = it.next().ok_or_else(invalid)?;
    for other in it {
        b = union(b, &other);
    }
    Ok(b)
}
pub(super) fn inflated(s: &Value, w: &mut Work) -> Result<bool, Error> {
    let (mut widths, mut heights, mut inks) = (Vec::new(), Vec::new(), Vec::new());
    for ch in optional(s, "chars", 20000)? {
        let text = ch
            .get("char")
            .map_or(Ok(""), |v| v.as_str().ok_or_else(invalid))?;
        w.charge(text.len() + 1)?;
        let mut it = text.chars();
        if !it
            .next()
            .is_some_and(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
            || it.next().is_some()
        {
            continue;
        }
        let truth = |v: Option<&Value>| {
            v.is_some_and(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
        };
        if !truth(ch.get("bbox")) || !truth(ch.get("tight_bbox")) {
            continue;
        }
        let b = bbox(&ch["bbox"])?;
        let t = bbox(&ch["tight_bbox"])?;
        widths.push(b.n[2] - b.n[0]);
        heights.push(b.n[3] - b.n[1]);
        inks.push(t.n[3] - t.n[1]);
    }
    w.charge(widths.len() * 48 + 1)?;
    if widths.len() < 20 {
        return Ok(false);
    }
    let width = median(widths);
    let height = median(heights);
    let ink = median(inks);
    Ok(width > 0. && height >= 2.2 * width && 0.5 * width <= ink && ink <= 1.4 * width)
}
struct Path {
    b: Box4,
    segments: i64,
    fill: bool,
    strip: bool,
}
struct Line {
    b: Box4,
    len: usize,
}
fn prose(lines: &[Line], b: &Box4, w: &mut Work) -> Result<bool, Error> {
    w.charge(lines.len() + 1)?;
    Ok(lines
        .iter()
        .filter(|l| {
            l.len >= 25
                && l.b.n[2] - l.b.n[0] > 0.6 * (b.n[2] - b.n[0])
                && overlap_first(&l.b, b) > 0.2
        })
        .take(2)
        .count()
        >= 2)
}
pub fn detect(v: &Value) -> Result<Value, Error> {
    detect_with_work(v, &mut Work(0))
}
pub(crate) fn detect_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    validate_tree(v)?;
    detect_source(&v["source"], v, w)
}
pub(super) fn detect_source(s: &Value, v: &Value, w: &mut Work) -> Result<Value, Error> {
    crate::legacy_page::validate_source_request(s, v)?;
    s.as_object().ok_or_else(invalid)?;
    let em = num(&v["em"])?;
    if em <= 0. {
        return Err(invalid());
    }
    if !inflated(s, w)? {
        return Ok(json!({"diagram_component_bboxes":[]}));
    }
    let size = array(&s["page_size"], 2)?;
    if size.len() != 2 {
        return Err(invalid());
    }
    let width = num(&size[0])?;
    let height = num(&size[1])?;
    let page_area = width * height;
    if width <= 0. || height <= 0. {
        return Err(invalid());
    }
    let empty = json!([]);
    let mut bounds = Vec::new();
    let tiled =
        super::tiled_tables::detect_with_work(&json!({"source":s,"excluded_bboxes":[]}), w)?;
    let filled = super::filled_grid::detect_with_work(
        &json!({"path_infos":s.get("path_infos").unwrap_or(&empty),"page_size":size,"excluded_bboxes":[]}),
        w,
    )?;
    for c in tiled["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .chain(filled["candidates"].as_array().unwrap())
    {
        bounds.push(bbox(&c["bbox"])?);
    }
    let complete = super::closed_rule_pipeline::detect_complete_with_work(
        &json!({"source":s,"excluded_bboxes":[]}),
        w,
    )?;
    let cs = complete["candidates"].as_array().unwrap();
    let flags = super::closed_grid::inspect_with_work(
        &json!({"source":s,"candidates":cs,"allow_two_by_two":true}),
        w,
    )?;
    for (c, flag) in cs
        .iter()
        .zip(flags["closed_grid_flags"].as_array().unwrap())
    {
        if flag == true {
            bounds.push(bbox(
                c.get("core_bbox")
                    .filter(|v| !v.is_null())
                    .unwrap_or(&c["bbox"]),
            )?);
        }
    }
    let mut paths = Vec::new();
    let mut rectangles = 0;
    for p in optional(s, "path_infos", 5000)? {
        w.charge(bounds.len() + 1)?;
        let b = bbox(&p["bbox"])?;
        let fill = boolean(p, "fill_visible")?;
        let stroke = boolean(p, "stroke_visible")?;
        let segments = integer(p, "segment_count", 0)?;
        if integer(p, "form_depth", 0)? != 0
            || !(fill || stroke)
            || b.n[1] < 0.08 * height
            || b.n[3] > 0.93 * height
            || area(&b) >= 0.5 * page_area
            || bounds.iter().any(|t| overlap_first(&b, t) >= 0.9)
            || segments == 13 && (b.n[2] - b.n[0]).max(b.n[3] - b.n[1]) <= 1.1 * em
        {
            continue;
        }
        let rects = optional(p, "rectangle_bboxes", 20000)?;
        rectangles += rects.len();
        if rectangles > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        let color = optional(p, "fill_rgba", 4)?;
        let strip = if fill && !rects.is_empty() && !color.is_empty() {
            if color.len() != 4 {
                return Err(invalid());
            }
            let mut min = 255u64;
            for value in color {
                let n = value.as_u64().filter(|n| *n <= 255).ok_or_else(invalid)?;
                min = min.min(n);
            }
            color[..3]
                .iter()
                .map(|n| n.as_u64().unwrap())
                .min()
                .unwrap()
                < 230
        } else {
            false
        };
        paths.push(Path {
            b,
            segments,
            fill,
            strip,
        });
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in 0..paths.len() {
        let mut retained = Vec::new();
        let mut merged = vec![i];
        for group in groups {
            let mut hit = false;
            for &j in &group {
                w.charge(1)?;
                if distance(&paths[i].b, &paths[j].b) <= 0.35 * (2. * em) {
                    hit = true;
                    break;
                }
            }
            if hit {
                w.charge(group.len() + 1)?;
                merged.extend(group);
            } else {
                retained.push(group);
            }
        }
        retained.push(merged);
        groups = retained;
    }
    let mut lines = Vec::new();
    for l in optional(s, "lines", 4096)? {
        let text = l["text"].as_str().ok_or_else(invalid)?;
        if text.chars().count() > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        w.charge(text.len() + 1)?;
        lines.push(Line {
            b: bbox(&l["bbox"])?,
            len: text.chars().count(),
        });
    }
    let images = optional(s, "image_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let mut result: Vec<Box4> = Vec::new();
    for group in groups {
        w.charge(group.len() + 1)?;
        let mut b = combined(group.iter().map(|&i| paths[i].b.clone()))?;
        let segments: i128 = group.iter().map(|&i| paths[i].segments as i128).sum();
        if group.len() < 3
            || segments < 100
            || (b.n[2] - b.n[0]).min(b.n[3] - b.n[1]) < 2. * em
            || !(0.001..=0.55).contains(&(area(&b) / page_area))
            || prose(&lines, &b, w)?
        {
            continue;
        }
        for _ in 0..2 {
            w.charge(images.len() + paths.len() + 1)?;
            let photos = images.iter().filter(|i| {
                area(i) < 0.15 * page_area
                    && i.n[1] >= 0.08 * height
                    && i.n[3] <= 0.93 * height
                    && distance(i, &b) <= 3. * em
            });
            let nearby = paths
                .iter()
                .filter(|p| distance(&p.b, &b) <= 1.5 * em)
                .map(|p| &p.b);
            let expanded = combined(
                std::iter::once(b.clone())
                    .chain(photos.cloned())
                    .chain(nearby.cloned()),
            )?;
            if prose(&lines, &expanded, w)? {
                break;
            }
            b = expanded;
        }
        w.charge(result.len() + 1)?;
        let overlaps: Vec<_> = result
            .iter()
            .filter(|prior| overlap_smaller(&b, prior) >= 0.2)
            .cloned()
            .collect();
        result.retain(|p| !overlaps.iter().any(|o| o.n == p.n));
        result.push(combined(std::iter::once(b).chain(overlaps))?);
    }
    let icons: Vec<_> = result
        .iter()
        .filter(|b| {
            2. * em <= b.n[2] - b.n[0]
                && b.n[2] - b.n[0] <= 5. * em
                && 2. * em <= b.n[3] - b.n[1]
                && b.n[3] - b.n[1] <= 5. * em
        })
        .cloned()
        .collect();
    for anchor in &icons {
        w.charge(icons.len() + 1)?;
        let peers: Vec<_> = icons
            .iter()
            .filter(|b| (((b.n[1] + b.n[3]) / 2.) - ((anchor.n[1] + anchor.n[3]) / 2.)).abs() <= em)
            .cloned()
            .collect();
        if peers.len() < 4 {
            continue;
        }
        let row = combined(peers.clone())?;
        w.charge(paths.len() + 1)?;
        let strips: Vec<_> = paths
            .iter()
            .filter(|p| {
                p.fill
                    && p.strip
                    && 0. <= p.b.n[1] - row.n[3]
                    && p.b.n[1] - row.n[3] <= 4. * em
                    && 0.5 * em <= p.b.n[3] - p.b.n[1]
                    && p.b.n[3] - p.b.n[1] <= 2. * em
                    && p.b.n[2] - p.b.n[0] >= 2. * em
            })
            .map(|p| p.b.clone())
            .collect();
        if strips.is_empty() {
            continue;
        }
        let sb = combined(strips.clone())?;
        if sb.n[2] - sb.n[0] < 0.8 * (row.n[2] - row.n[0]) {
            continue;
        }
        let b = combined(std::iter::once(row.clone()).chain(strips))?;
        w.charge(lines.len() + result.len() * peers.len() + 1)?;
        let labels = lines
            .iter()
            .filter(|l| {
                l.len <= 25
                    && l.b.n[2] - l.b.n[0] < 0.2 * (b.n[2] - b.n[0])
                    && b.n[0] <= (l.b.n[0] + l.b.n[2]) / 2.
                    && (l.b.n[0] + l.b.n[2]) / 2. <= b.n[2]
                    && row.n[1] <= l.b.n[1]
                    && l.b.n[1] <= b.n[3] + 20. * em
            })
            .map(|l| l.b.clone());
        result.retain(|p| !peers.iter().any(|o| o.n == p.n));
        result.push(combined(std::iter::once(b.clone()).chain(labels))?);
        break;
    }
    let out =
        json!({"diagram_component_bboxes":result.into_iter().map(|b|b.raw).collect::<Vec<_>>()});
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
    fn old_complete_diagram_components() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../build/migration");
        for name in [
            "native-diagram-components",
            "native-diagram-component-specials",
        ] {
            let root = root.join(format!("{name}.json"));
            let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
            for c in f["cases"].as_array().unwrap() {
                assert_eq!(
                    detect(&c["request"]).unwrap(),
                    c["expected"],
                    "{}",
                    c["label"]
                );
            }
            for c in f["errors"].as_array().unwrap() {
                assert!(
                    detect(&c["request"]).is_err(),
                    "old exception {}",
                    c["label"]
                );
            }
        }
    }
    #[test]
    fn shared_work_and_recovery() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-diagram-component-specials.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let q = &c["request"];
            let mut w = Work(0);
            assert!(detect_with_work(q, &mut w).is_ok());
            assert!(detect_with_work(q, &mut Work(10_000_000 - w.0 + 1)).is_err());
            assert!(detect(q).is_ok());
        }
    }
}
