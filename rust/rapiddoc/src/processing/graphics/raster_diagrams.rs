//! 旧栅格标签连接图和普通文字框证明，复用原组件及有字网格。
use super::{
    drawing_graphics, graphic_grid,
    path_graphics::{area, array, bbox, integer, num, overlap_first, union},
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
fn optional<'a>(v: &'a Value, k: &str, max: usize) -> Result<&'a [Value], Error> {
    v.get(k).map_or(Ok(&[]), |v| array(v, max))
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
    let size = array(&source["page_size"], 2)?;
    if size.len() != 2 {
        return Err(invalid());
    }
    let page = [num(&size[0])?, num(&size[1])?];
    let drawings = optional(source, "drawing_lines", 5000)?;
    let images = optional(source, "image_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let frames = optional(v, "frame_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    work.charge(drawings.len() + images.len() + frames.len())?;
    let mut output = Vec::new();
    let mut flags = Vec::new();
    if (images.len() >= 4 && drawings.len() >= 12) || !frames.is_empty() {
        let components = drawing_graphics::detect_with_work(
            &json!({"drawing_lines":drawings,"path_infos":[],"page_size":page,"median_height":em}),
            work,
        )?;
        if images.len() >= 4 && drawings.len() >= 12 {
            let grid = graphic_grid::prepare(source, em, work)?;
            for c in components["components"].as_array().unwrap() {
                let ids = c["line_indices"].as_array().unwrap();
                let b = bbox(&c["bbox"])?;
                if ids.len() < 12
                    || c["horizontal_count"].as_u64().unwrap() < 4
                    || c["vertical_count"].as_u64().unwrap() < 4
                    || b.n[2] - b.n[0] < 8. * em
                    || b.n[3] - b.n[1] < 5. * em
                    || area(&b) > 0.65 * page[0] * page[1]
                {
                    continue;
                }
                work.charge(images.len() + ids.len())?;
                let contained: Vec<_> = images
                    .iter()
                    .filter(|i| overlap_first(i, &b) >= 0.95)
                    .collect();
                if contained.len() < 4 || !graphic_grid::grids(&grid, &b, work)?.is_empty() {
                    continue;
                }
                let mut interior = 0;
                for id in ids {
                    let d = &drawings[id.as_u64().unwrap() as usize];
                    let line = bbox(&d["bbox"])?;
                    if d["orientation"].as_str() == Some("horizontal")
                        && line.n[0] > b.n[0] + em
                        && line.n[2] < b.n[2] - em
                    {
                        interior += 1
                    }
                }
                if interior >= 3 {
                    work.charge(contained.len())?;
                    output.push(contained.into_iter().fold(b, |a, b| union(a, b)).raw);
                }
            }
        }
        if !frames.is_empty() {
            let lines = optional(source, "lines", 4096)?
                .iter()
                .map(|l| bbox(&l["bbox"]))
                .collect::<Result<Vec<_>, _>>()?;
            let paths = optional(source, "path_infos", 5000)?
                .iter()
                .map(|p| Ok((bbox(&p["bbox"])?, integer(p, "segment_count", 0)?)))
                .collect::<Result<Vec<_>, Error>>()?;
            for b in frames {
                work.charge(
                    components["components"].as_array().unwrap().len()
                        + images.len()
                        + paths.len()
                        + lines.len(),
                )?;
                let mut simple = false;
                for c in components["components"].as_array().unwrap() {
                    if c["line_indices"].as_array().unwrap().len() == 4
                        && c["horizontal_count"] == 2
                        && c["vertical_count"] == 2
                    {
                        let cb = bbox(&c["bbox"])?;
                        if overlap_first(&b, &cb) >= 0.98 && overlap_first(&cb, &b) >= 0.98 {
                            simple = true;
                            break;
                        }
                    }
                }
                flags.push(
                    simple
                        && !images.iter().any(|i| overlap_first(i, &b) >= 0.9)
                        && !paths
                            .iter()
                            .any(|(p, n)| *n > 8 && overlap_first(p, &b) >= 0.9)
                        && lines.iter().any(|l| overlap_first(l, &b) >= 0.95),
                );
            }
        }
    }
    let out = json!({"raster_diagram_bboxes":output,"plain_text_frame_flags":flags});
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
    fn old_raster_diagrams() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-raster-diagrams.json");
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
        let q = json!({"source":{"page_size":[400,600],"drawing_lines":[],"image_bboxes":[]},"em":10,"frame_bboxes":[[0,0,100,100]]});
        assert!(detect_with_work(&q, &mut Work(10_000_000)).is_err());
        assert!(detect(&q).is_ok());
    }
}
