//! 旧完整强图形总编排：各类原生图体顺序、真实网格仲裁及可观察行副作用。
use super::path_graphics::{array, bbox, num};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
fn optional<'a>(v: &'a Value, k: &str, max: usize) -> Result<&'a [Value], Error> {
    v.get(k).map_or(Ok(&[]), |v| array(v, max))
}
fn append(out: &mut Vec<Value>, v: &Value, key: &str, w: &mut Work) -> Result<(), Error> {
    let a = array(&v[key], 5000)?;
    w.charge(a.len() + 1)?;
    if out.len() + a.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    out.extend_from_slice(a);
    Ok(())
}
pub fn detect(v: &Value) -> Result<Value, Error> {
    detect_with_work(v, &mut Work(0))
}
pub(crate) fn detect_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    validate_tree(v)?;
    detect_source(&v["source"], w)
}
pub(crate) fn detect_source(original: &Value, w: &mut Work) -> Result<Value, Error> {
    detect_source_owned(original.clone(), w)
}
pub(crate) fn detect_source_owned(owned: Value, w: &mut Work) -> Result<Value, Error> {
    let original = &owned;
    crate::legacy_page::validate_source_request(original, &json!({}))?;
    original.as_object().ok_or_else(invalid)?;
    let paths = optional(original, "path_infos", 5000)?;
    let images = optional(original, "image_bboxes", 5000)?;
    w.charge(1)?;
    if paths.is_empty() && images.is_empty() {
        return Ok(crate::processing::with_source(
            owned,
            json!({"graphic_bboxes":[]}),
        ));
    }
    let mut source = original.clone();
    for key in [
        "lines",
        "chars",
        "drawing_lines",
        "path_infos",
        "image_bboxes",
        "form_bboxes",
        "signature_bboxes",
    ] {
        if source.get(key).is_none() {
            source[key] = json!([]);
        }
    }
    let mut heights = Vec::new();
    for l in optional(&source, "lines", 4096)? {
        w.charge(1)?;
        if l.get("angle").map_or(Ok(0.), num)? != 0. {
            continue;
        }
        let b = bbox(&l["bbox"])?;
        let effective = l.get("effective_height").map_or(Ok(0.), num)?;
        let em = l.get("em_height").map_or(Ok(0.), num)?;
        let repaired = l
            .get("style_scale_repaired")
            .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))?;
        let h = if repaired && em > 0. {
            em
        } else if effective != 0. {
            effective
        } else {
            b.n[3] - b.n[1]
        };
        heights.push(h.max(0.1));
    }
    w.charge(heights.len() * 16 + 1)?;
    heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = heights.len();
    let fallback = if n == 0 {
        1.
    } else if n % 2 == 1 {
        heights[n / 2]
    } else {
        (heights[n / 2 - 1] + heights[n / 2]) / 2.
    };
    let (em, _) = super::stacked_graphics::measure_with_work(&source, fallback, w)?;
    let mut candidates = Vec::new();
    let isolated = super::isolated_graphics::detect_source(&source, &json!({}), w)?;
    append(&mut candidates, &isolated, "vector_bboxes", w)?;
    let raster = super::raster_diagrams::detect_source(&source, &value_object!({"em":em}), w)?;
    append(&mut candidates, &raster, "raster_diagram_bboxes", w)?;
    let diagram = super::diagram_components::detect_source(&source, &value_object!({"em":em}), w)?;
    append(&mut candidates, &diagram, "diagram_component_bboxes", w)?;
    let stacked = super::stacked_graphics::detect_source(
        &source,
        &value_object!({"em":em,"fallback_em":fallback}),
        w,
    )?;
    append(&mut candidates, &stacked, "stacked_column_bboxes", w)?;
    let axes = super::raster_charts::cores_source(&source, w)?;
    append(&mut candidates, &axes, "graphic_bboxes", w)?;
    let mut bars = super::bar_graphics::detect_source(&source, &value_object!({"em":em}), w)?;
    append(&mut candidates, &bars, "bar_bboxes", w)?;
    source = bars["source"].take();
    let concentric =
        super::concentric_graphics::detect_source(&source, &value_object!({"em":em}), w)?;
    append(&mut candidates, &concentric, "concentric_bboxes", w)?;
    let paths = super::path_graphics::detect_with_work(
        &value_object!({"path_infos":source["path_infos"],"page_size":source["page_size"],"median_height":em}),
        w,
    )?;
    append(&mut candidates, &paths, "container_bboxes", w)?;
    append(&mut candidates, &paths, "axis_bboxes", w)?;
    let drawings = super::drawing_graphics::detect_with_work(
        &value_object!({"path_infos":source["path_infos"],"drawing_lines":source["drawing_lines"],"page_size":source["page_size"],"median_height":em}),
        w,
    )?;
    append(&mut candidates, &drawings, "graphic_bboxes", w)?;
    let selected = super::graphic_arbitration::select_source(
        &source,
        &value_object!({"median_height":em,"candidate_bboxes":candidates}),
        w,
    )?;
    let updated = (source["lines"] != *original.get("lines").unwrap_or(&Value::Array(Vec::new())))
        .then(|| source["lines"].take());
    let mut output_source = owned;
    if let Some(lines) = updated {
        output_source["lines"] = lines;
    }
    let out = crate::processing::with_source(
        output_source,
        value_object!({"graphic_bboxes":selected["graphic_bboxes"]}),
    );
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
    fn borrowed_graphic_detectors_match_installed_real_sources() {
        let _scope = crate::legacy_page::budget::host_page();
        let fixture: Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!("../../../../../tests/fixtures/upstream_068/borrowed_graphics.json.gz")
                [..],
        ))
        .unwrap();
        for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
            let actual = detect_source(&case["source"], &mut Work(0)).unwrap();
            assert_eq!(actual, case["expected"], "real source {i}");
        }
    }
    #[test]
    fn shared_budget_and_recovery() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-strong-graphics-complete.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap().iter().step_by(31) {
            let q = &c["request"];
            let mut work = Work(0);
            assert!(detect_with_work(q, &mut work).is_ok());
            assert!(detect_with_work(q, &mut Work(10_000_000 - work.0 + 1)).is_err());
            assert!(detect(q).is_ok());
        }
    }
    #[test]
    fn old_complete_strong_graphics() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-strong-graphics-complete.json");
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
            assert!(detect(&c["request"]).is_err(), "{}", c["label"]);
        }
    }
}
