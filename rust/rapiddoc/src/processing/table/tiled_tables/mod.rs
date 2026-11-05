//! 旧相接填充单元格候选；含真实CJK尺度、推断表头网格与相邻表题转移。
use super::path_graphics::{array, bbox, integer, num, overlap_smaller, union, Box4};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::python_space;
use serde_json::{json, Value};
use std::collections::BTreeSet;
pub(super) mod pending;
fn optional<'a>(v: &'a Value, k: &str, max: usize) -> Result<&'a [Value], Error> {
    v.get(k).map_or(Ok(&[]), |v| array(v, max))
}
fn median(a: &mut [f64]) -> f64 {
    a.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = a.len();
    if n % 2 == 1 {
        a[n / 2]
    } else {
        (a[n / 2 - 1] + a[n / 2]) / 2.
    }
}
fn truth_box(v: Option<&Value>) -> Option<&Value> {
    v.filter(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
}
fn cjk(s: &str) -> bool {
    let mut it = s.chars();
    it.next()
        .is_some_and(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
        && it.next().is_none()
}
fn axis(a: &Box4, b: &Box4, i: usize) -> f64 {
    let short = (a.n[i + 2] - a.n[i]).min(b.n[i + 2] - b.n[i]);
    if short > 0. {
        (a.n[i + 2].min(b.n[i + 2]) - a.n[i].max(b.n[i])).max(0.) / short
    } else {
        0.
    }
}
fn set_coord(b: &mut Box4, i: usize, n: f64, raw: Value) {
    b.n[i] = n;
    b.raw[i] = raw;
}
struct Line {
    b: Box4,
    ink: Box4,
    id: i64,
    text_len: usize,
    caption_text: bool,
}
struct Rule {
    b: Box4,
    raw: Value,
    vertical: bool,
    horizontal: bool,
}
struct Candidate {
    b: Box4,
    ids: BTreeSet<i64>,
    score: usize,
    grid: Vec<Value>,
    annotations: Vec<Value>,
}
pub fn detect(v: &Value) -> Result<Value, Error> {
    detect_with_work(v, &mut Work(0))
}
pub(crate) fn detect_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    validate_tree(v)?;
    let s = &v["source"];
    s.as_object().ok_or_else(invalid)?;
    let size = array(&s["page_size"], 2)?;
    if size.len() != 2 {
        return Err(invalid());
    }
    let (width, height) = (num(&size[0])?, num(&size[1])?);
    let chars = optional(s, "chars", 20000)?;
    for (key, max) in [
        ("path_infos", 5000),
        ("drawing_lines", 5000),
        ("lines", 4096),
    ] {
        optional(s, key, max)?;
    }
    optional(v, "excluded_bboxes", 5000)?;
    let (mut widths, mut loose, mut inks, mut all_widths) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for ch in chars {
        let text = ch
            .get("char")
            .map_or(Ok(""), |v| v.as_str().ok_or_else(invalid))?;
        work.charge(text.len() + 1)?;
        if !cjk(text) {
            continue;
        }
        if let Some(b) = truth_box(ch.get("bbox")) {
            let b = bbox(b)?;
            all_widths.push(b.n[2] - b.n[0]);
            if let Some(t) = truth_box(ch.get("tight_bbox")) {
                let t = bbox(t)?;
                widths.push(b.n[2] - b.n[0]);
                loose.push(b.n[3] - b.n[1]);
                inks.push(t.n[3] - t.n[1]);
            }
        }
    }
    work.charge(chars.len() * 20)?;
    let inflated = if widths.len() >= 20 {
        let w = median(&mut widths);
        let h = median(&mut loose);
        let ink = median(&mut inks);
        w > 0. && h >= 2.2 * w && 0.5 * w <= ink && ink <= 1.4 * w
    } else {
        false
    };
    if !inflated {
        return Ok(json!({"candidates":[]}));
    }
    let tolerance = 0.5f64.max(width.min(height) * 0.001);
    let excluded = optional(v, "excluded_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let paths = optional(s, "path_infos", 5000)?;
    let mut rectangles = Vec::new();
    for p in paths {
        let b = bbox(&p["bbox"])?;
        let fill = p
            .get("fill_visible")
            .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))?;
        let rects = optional(p, "rectangle_bboxes", 20000)?;
        work.charge(excluded.len() + 1)?;
        if integer(p, "form_depth", 0)? == 0
            && fill
            && rects.len() == 1
            && integer(p, "segment_count", 0)? == 5
            && b.n[2] - b.n[0] >= 6. * tolerance
            && b.n[3] - b.n[1] >= 4. * tolerance
            && super::path_graphics::area(&b) < 0.65 * width * height
            && !excluded.iter().any(|e| overlap_smaller(&b, e) >= 0.5)
        {
            rectangles.push(b);
        }
    }
    let rules = optional(s, "drawing_lines", 5000)?
        .iter()
        .map(|r| {
            let b = bbox(&r["bbox"])?;
            let _ = num(&r["width"])?;
            let o = r["orientation"].as_str().ok_or_else(invalid)?;
            Ok(Rule {
                b,
                raw: r.clone(),
                vertical: o == "vertical",
                horizontal: o == "horizontal",
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let mut lines = Vec::new();
    let mut char_count = 0;
    for l in optional(s, "lines", 4096)? {
        let b = bbox(&l["bbox"])?;
        let mut ink = None;
        for c in l
            .get("chars")
            .filter(|v| !v.is_null())
            .map_or(Ok(&[][..]), |v| array(v, 20000))?
        {
            char_count += 1;
            if char_count > rapiddoc_core::resources::limit(20000) {
                return Err(limit());
            }
            if let Some(t) = truth_box(c.get("tight_bbox")) {
                let t = bbox(t)?;
                ink = Some(ink.map_or_else(|| t.clone(), |a| union(a, &t)));
            }
        }
        let text = l["text"].as_str().ok_or_else(invalid)?;
        let text_len = text.chars().count();
        if text_len > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        work.charge(text.len() * 2 + 1)?;
        let stripped = text.trim_matches(python_space);
        let caption_text = (8..=35).contains(&stripped.chars().count())
            && stripped
                .chars()
                .all(|c| ('\u{4e00}'..='\u{9fff}').contains(&c));
        lines.push(Line {
            ink: ink.unwrap_or_else(|| b.clone()),
            b,
            id: integer(l, "source_index", 0)?,
            text_len,
            caption_text,
        });
    }
    let mut pending = pending::Pending::new(rectangles.len());
    let mut result: Vec<Candidate> = Vec::new();
    let mut total_grid = 0;
    while let Some(seed) = pending.pop() {
        let mut members = vec![seed];
        let mut at = 0;
        while at < members.len() {
            let b = &rectangles[members[at]];
            let order = pending.ordered();
            work.charge(order.len() * 2 + pending.capacity() + 1)?;
            let touching: Vec<_> = order
                .into_iter()
                .filter(|i| {
                    let o = &rectangles[*i];
                    (0..2).any(|a| {
                        (b.n[a] - o.n[a + 2]).abs().min((b.n[a + 2] - o.n[a]).abs()) <= tolerance
                            && axis(b, o, 1 - a) >= 0.8
                    })
                })
                .collect();
            pending.difference(&touching);
            members.extend(touching);
            at += 1;
        }
        if members.len() < 6 {
            continue;
        }
        let mut bounds = rectangles[members[0]].clone();
        for i in &members[1..] {
            bounds = union(bounds, &rectangles[*i]);
        }
        work.charge(rules.len() * 2)?;
        let columns: Vec<_> = rules
            .iter()
            .filter(|r| {
                let x = (r.b.n[0] + r.b.n[2]) / 2.;
                r.vertical
                    && bounds.n[0] + tolerance < x
                    && x < bounds.n[2] - tolerance
                    && axis(&r.b, &bounds, 1) >= 0.75
            })
            .collect();
        let rows = rules
            .iter()
            .filter(|r| {
                let y = (r.b.n[1] + r.b.n[3]) / 2.;
                r.horizontal
                    && bounds.n[1] + tolerance < y
                    && y < bounds.n[3] - tolerance
                    && axis(&r.b, &bounds, 0) >= 0.8
            })
            .count();
        if columns.is_empty() || rows < 2 {
            continue;
        }
        let bottom = columns
            .iter()
            .max_by(|a, b| {
                a.b.n[3]
                    .partial_cmp(&b.b.n[3])
                    .unwrap()
                    .then(std::cmp::Ordering::Greater)
            })
            .unwrap();
        if bottom.b.n[3] > bounds.n[3]
            && bottom.b.n[3] - bounds.n[3] <= 0.3 * (bounds.n[3] - bounds.n[1])
        {
            set_coord(&mut bounds, 3, bottom.b.n[3], bottom.b.raw[3].clone());
        }
        let mut ids = BTreeSet::new();
        let mut above = Vec::new();
        work.charge(lines.len() + 1)?;
        for (i, l) in lines.iter().enumerate() {
            let b = &l.ink;
            let (x, y) = ((b.n[0] + b.n[2]) / 2., (b.n[1] + b.n[3]) / 2.);
            if bounds.n[0] - tolerance <= x && x <= bounds.n[2] + tolerance {
                if bounds.n[1] <= y && y <= bounds.n[3] + tolerance {
                    ids.insert(l.id);
                } else if 0. < bounds.n[1] - b.n[3]
                    && bounds.n[1] - b.n[3] <= 1.5 * (b.n[3] - b.n[1]).max(tolerance)
                    && l.text_len <= 16
                {
                    above.push(i);
                }
            }
        }
        let mut grid = Vec::new();
        if above.len() >= 2 {
            let mut heights: Vec<_> = above
                .iter()
                .map(|i| lines[*i].ink.n[3] - lines[*i].ink.n[1])
                .collect();
            work.charge(above.len() * 16)?;
            let slack = (2. * tolerance).max(0.25 * median(&mut heights));
            let top = above
                .iter()
                .min_by(|a, b| {
                    lines[**a].ink.n[1]
                        .partial_cmp(&lines[**b].ink.n[1])
                        .unwrap()
                })
                .unwrap();
            let max = above
                .iter()
                .map(|i| lines[*i].ink.n[1])
                .fold(f64::NEG_INFINITY, f64::max);
            if max - lines[*top].ink.n[1] <= slack {
                let body_top = bounds.raw[1].clone();
                let t = &lines[*top].ink;
                set_coord(&mut bounds, 1, t.n[1], t.raw[1].clone());
                for i in &above {
                    ids.insert(lines[*i].id);
                }
                grid.push(json!({"bbox":[bounds.raw[0],bounds.n[1]-tolerance/2.,bounds.raw[2],bounds.n[1]+tolerance/2.],"width":tolerance,"orientation":"horizontal"}));
                // Python set literal inserts the two original edges before the float centres.
                let mut tracks = vec![
                    (bounds.n[0], bounds.raw[0].clone()),
                    (bounds.n[2], bounds.raw[2].clone()),
                ];
                tracks.extend(columns.iter().map(|r| {
                    let x = (r.b.n[0] + r.b.n[2]) / 2.;
                    (x, json!(x))
                }));
                work.charge(tracks.len() * 16 + rules.len())?;
                tracks.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                tracks.dedup_by(|a, b| a.0 == b.0);
                for (x, _) in tracks {
                    grid.push(json!({"bbox":[json!(x-tolerance/2.),bounds.raw[1],json!(x+tolerance/2.),body_top],"width":tolerance,"orientation":"vertical"}));
                }
                for r in &rules {
                    let (x, y) = ((r.b.n[0] + r.b.n[2]) / 2., (r.b.n[1] + r.b.n[3]) / 2.);
                    if bounds.n[0] - tolerance <= x
                        && x <= bounds.n[2] + tolerance
                        && bounds.n[1] - tolerance <= y
                        && y <= bounds.n[3] + tolerance
                    {
                        grid.push(r.raw.clone());
                    }
                }
            }
        }
        total_grid += grid.len();
        if total_grid > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        result.push(Candidate {
            b: bounds,
            ids,
            score: 130 + members.len(),
            grid,
            annotations: Vec::new(),
        });
    }
    result.sort_by(|a, b| {
        a.b.n[1]
            .partial_cmp(&b.b.n[1])
            .unwrap()
            .then_with(|| a.b.n[0].partial_cmp(&b.b.n[0]).unwrap())
    });
    let em = if all_widths.is_empty() {
        0.
    } else {
        median(&mut all_widths)
    };
    for i in 0..result.len().saturating_sub(1) {
        let (before, tail) = result.split_at_mut(i + 1);
        let before = &mut before[i];
        let after = &mut tail[0];
        let gap = after.b.n[1] - before.b.n[3];
        if !(0. <= gap && gap <= 2. * em) {
            continue;
        }
        work.charge(lines.len() * 3)?;
        let mut caption: Option<usize> = None;
        for (j, l) in lines.iter().enumerate() {
            if !before.ids.contains(&l.id) {
                continue;
            }
            if l.caption_text
                && (l.b.n[0] - before.b.n[0]).abs() <= em
                && l.b.n[2] - l.b.n[0] < 0.6 * (before.b.n[2] - before.b.n[0])
                && before.b.n[3] - 2. * em <= l.b.n[1]
                && l.b.n[1] < after.b.n[1]
                && caption.is_none_or(|k| l.b.n[1] > lines[k].b.n[1])
            {
                caption = Some(j);
            }
        }
        if let Some(j) = caption {
            let c = &lines[j];
            let bottom = lines
                .iter()
                .enumerate()
                .filter(|(k, l)| *k != j && before.ids.contains(&l.id) && l.b.n[1] < c.b.n[1])
                .map(|(_, l)| l.b.n[3])
                .reduce(f64::max)
                .ok_or_else(invalid)?
                + tolerance;
            let y = bottom.min(c.b.n[1] - tolerance);
            set_coord(&mut before.b, 3, y, json!(y));
            before.ids.remove(&c.id);
            let mut boxes = serde_json::Map::new();
            boxes.insert(c.id.to_string(), json!(c.b.raw));
            after.annotations.push(
                json!({"kind":"caption","bbox":c.b.raw,"line_indices":[c.id],"line_bboxes":boxes}),
            );
        }
    }
    let out = json!({"candidates":result.into_iter().map(|c|json!({"bbox":c.b.raw,"local_bbox":c.b.raw,"angle":0,"score":c.score,"core_bbox":c.b.raw,"line_indices":c.ids,"annotations":c.annotations,"inferred_grid":c.grid,"preserve_inline_font_styles":false,"inferred_grid_authoritative":!c.grid.is_empty()})).collect::<Vec<_>>()});
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
    fn old_tiled_tables() {
        let f: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-tiled-tables.json"),
            )
            .unwrap(),
        )
        .unwrap();
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
    fn shared_work() {
        let s = json!({"source":{"page_size":[400,600],"chars":[{"char":"中","bbox":[0,0,10,30],"tight_bbox":[0,0,10,10]}]}});
        assert!(detect_with_work(&s, &mut Work(10_000_000)).is_err());
        assert!(detect(&s).is_ok());
    }
}
