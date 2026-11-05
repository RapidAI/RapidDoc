//! 旧完整原生柱图候选；含图题行副作用、两轮扩框和实际有字网格排除。
use super::path_graphics::{area, array, bbox, integer, num, overlap_smaller, union, Box4};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::python_space;
use serde_json::{json, Value};
fn optional<'a>(v: &'a Value, key: &str, max: usize) -> Result<&'a [Value], Error> {
    v.get(key).map_or(Ok(&[]), |v| array(v, max))
}
fn distance(a: &Box4, b: &Box4) -> f64 {
    rapiddoc_pdf::legacy_vectors::hypot(
        (a.n[0] - b.n[2]).max(b.n[0] - a.n[2]).max(0.),
        (a.n[1] - b.n[3]).max(b.n[1] - a.n[3]).max(0.),
    )
}
struct Line {
    b: Box4,
    id: i64,
    angle: f64,
    note: bool,
    english_note: bool,
    short: bool,
    height: f64,
    caption: bool,
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
    let em = num(&v["em"])?;
    if em <= 0. {
        return Err(invalid());
    }
    optional(source, "image_bboxes", 5000)?;
    optional(source, "form_bboxes", 5000)?;
    let groups = super::bar_groups::prepare_source(source, v, work)?;
    let mut output_source = source.clone();
    let mut lines = optional(source, "lines", 4096)?.to_vec();
    let mut output = Vec::new();
    if !groups["groups"].as_array().unwrap().is_empty() {
        let size = array(&source["page_size"], 2)?;
        if size.len() != 2 {
            return Err(invalid());
        }
        let page_width = num(&size[0])?;
        let page = page_width * num(&size[1])?;
        let admitted = groups["admitted_bars"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| bbox(&p["bbox"]))
            .collect::<Result<Vec<_>, _>>()?;
        let mut complex = Vec::new();
        for p in optional(source, "path_infos", 5000)? {
            work.charge(1)?;
            if p.get("fill_visible")
                .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))?
                && integer(p, "segment_count", 0)? > 24
            {
                complex.push(bbox(&p["bbox"])?);
            }
        }
        let images = optional(source, "image_bboxes", 5000)?
            .iter()
            .map(bbox)
            .collect::<Result<Vec<_>, _>>()?;
        let mut meta = Vec::new();
        for l in &lines {
            let text = l["text"].as_str().ok_or_else(invalid)?;
            work.charge(text.len() * 3 + 1)?;
            let b = bbox(&l["bbox"])?;
            let repaired = l
                .get("style_scale_repaired")
                .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))?;
            let em_height = l.get("em_height").map_or(Ok(0.), num)?;
            let effective = l.get("effective_height").map_or(Ok(0.), num)?;
            let height = if repaired && em_height > 0. {
                em_height
            } else if effective != 0. {
                effective
            } else {
                b.n[3] - b.n[1]
            };
            meta.push(Line {
                b,
                id: integer(l, "source_index", 0)?,
                angle: l.get("angle").map_or(Ok(0.), num)?,
                note: super::graphic_captions::source_note(text, true),
                english_note: super::graphic_captions::source_note(text, false),
                short: text.split(python_space).filter(|s| !s.is_empty()).count() <= 8,
                height: height.max(0.1),
                caption: super::concentric_graphics::caption(text),
            });
        }
        let mut grid = None;
        for g in groups["groups"].as_array().unwrap() {
            let mut bounds = bbox(&g["bbox"])?;
            let anchor = bbox(&g["label_anchor"])?;
            let protected =
                super::graphic_captions::protect(&mut lines, &[anchor.clone()], em, work)?;
            for _ in 0..2 {
                work.charge(
                    admitted.len()
                        + complex
                            .len()
                            .checked_mul(meta.len() + 1)
                            .ok_or_else(limit)?
                        + images.len()
                        + meta.len()
                        + 1,
                )?;
                let additions: Vec<_> = admitted
                    .iter()
                    .filter(|b| overlap_smaller(b, &bounds) >= 0.25)
                    .collect();
                let outlines: Vec<_> = complex
                    .iter()
                    .filter(|b| {
                        b.n[2] - b.n[0] <= 0.4 * (bounds.n[2] - bounds.n[0])
                            && b.n[3] - b.n[1] <= 6. * em
                            && super::graphic_captions::axis(b, &anchor, 0) >= 0.5
                            && 0. <= b.n[1] - anchor.n[3]
                            && b.n[1] - anchor.n[3] <= 2. * em
                            && !meta.iter().any(|l| {
                                l.english_note && anchor.n[3] <= l.b.n[1] && l.b.n[1] <= b.n[3]
                            })
                    })
                    .collect();
                let image_parts: Vec<_> = images
                    .iter()
                    .filter(|b| {
                        area(b) < 0.4 * page
                            && (overlap_smaller(b, &bounds) >= 0.8
                                || (super::graphic_captions::axis(b, &bounds, 0) >= 0.3
                                    && distance(b, &bounds) <= 1.5 * em))
                    })
                    .collect();
                let labels: Vec<_> = meta
                    .iter()
                    .enumerate()
                    .filter(|(i, l)| {
                        lines[*i].get("semantic_type").is_none_or(|v| v.is_null())
                            && !protected.contains(&l.id)
                            && !l.note
                            && l.short
                            && l.b.n[2] - l.b.n[0] <= 0.45 * page_width
                            && ((super::graphic_captions::axis(&l.b, &bounds, 1) >= 0.35
                                && distance(&l.b, &bounds) <= 1.5 * em)
                                || (![0., 90., 180., 270.].contains(&l.angle)
                                    && super::graphic_captions::axis(&l.b, &bounds, 0) >= 0.5
                                    && 0. <= l.b.n[1] - bounds.n[3]
                                    && l.b.n[1] - bounds.n[3] <= 4. * em))
                            && l.height <= 1.25 * em
                            && !l.caption
                    })
                    .map(|(_, l)| &l.b)
                    .collect();
                for b in additions
                    .into_iter()
                    .chain(outlines)
                    .chain(image_parts)
                    .chain(labels)
                {
                    work.charge(1)?;
                    bounds = union(bounds, b);
                }
                for i in g["numeric_line_indices"].as_array().unwrap() {
                    work.charge(1)?;
                    bounds = union(bounds, &meta[i.as_u64().unwrap() as usize].b);
                }
            }
            if grid.is_none() {
                grid = Some(super::graphic_grid::prepare(source, em, work)?);
            }
            if !super::graphic_grid::grids(grid.as_ref().unwrap(), &bounds, work)?.is_empty() {
                continue;
            }
            output.push(bounds.raw);
            if output.len() > rapiddoc_core::resources::limit(5000) {
                return Err(limit());
            }
        }
    }
    if source.get("lines").is_some() {
        output_source
            .as_object_mut()
            .unwrap()
            .insert("lines".into(), json!(lines));
    }
    let out = json!({"source":output_source,"bar_bboxes":output});
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
    fn old_full_bar_graphics() {
        let f: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-bar-graphics.json"),
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
    fn shared_budget() {
        let v = json!({"source":{"page_size":[400,600]},"em":10});
        assert!(detect_with_work(&v, &mut Work(10_000_000)).is_err());
        assert!(detect(&v).is_ok());
    }
}
