//! 目录预认领与 Form 选择；后续图像/文字物化仍由页准备阶段负责。
mod captions;
mod forms;
mod graphics;
mod images;
mod index;
mod late_images;
pub(super) mod media;
mod roles;
use super::path_graphics::{array, bbox, num, union, Box4};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
pub(super) fn optional<'a>(v: &'a Value, k: &str, max: usize) -> Result<&'a [Value], Error> {
    v.get(k)
        .filter(|v| !v.is_null())
        .map_or(Ok(&[]), |v| array(v, max))
}
pub(super) fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.
    }
}
pub(super) fn combined<'a>(v: impl IntoIterator<Item = &'a Box4>) -> Result<Box4, Error> {
    let mut it = v.into_iter();
    let mut b = it.next().ok_or_else(invalid)?.clone();
    for c in it {
        b = union(b, c);
    }
    Ok(b)
}
pub(super) fn height(l: &Value, b: &Box4) -> Result<f64, Error> {
    let em = l.get("em_height").map_or(Ok(0.), num)?;
    let h = l.get("effective_height").map_or(Ok(0.), num)?;
    let repaired = l
        .get("style_scale_repaired")
        .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))?;
    Ok((if repaired && em > 0. {
        em
    } else if h != 0. {
        h
    } else {
        b.n[3] - b.n[1]
    })
    .max(0.1))
}
pub(super) fn coerce(v: &Value) -> Result<Option<Box4>, Error> {
    let Some(a) = v.as_array().filter(|a| a.len() == 4) else {
        return Ok(None);
    };
    let mut n = [0.; 4];
    for (i, v) in a.iter().enumerate() {
        let Some(x) = super::table_native_text::number(v)? else {
            return Ok(None);
        };
        n[i] = x;
    }
    if n[0] > n[2] {
        n.swap(0, 2);
    }
    if n[1] > n[3] {
        n.swap(1, 3);
    }
    Ok((n[2] > n[0] && n[3] > n[1]).then(|| Box4 {
        n,
        raw: n.map(|x| json!(x)),
    }))
}
pub(super) fn clip(b: &Box4, size: [f64; 2]) -> Result<Option<Box4>, Error> {
    coerce(&json!([
        b.n[0].min(size[0]).max(0.),
        b.n[1].min(size[1]).max(0.),
        b.n[2].min(size[0]).max(0.),
        b.n[3].min(size[1]).max(0.)
    ]))
}
pub(super) fn size(v: &Value) -> Result<[f64; 2], Error> {
    let a = array(v, 2)?;
    if a.len() != 2 {
        return Err(invalid());
    }
    Ok([num(&a[0])?, num(&a[1])?])
}
fn output(v: Value) -> Result<Value, Error> {
    if crate::legacy_page::budget::output_exceeds_limit(
        &v,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        Err(limit())
    } else {
        Ok(v)
    }
}
pub fn extract_index(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    output(index::extract(v, &mut Work(0))?)
}
pub(super) fn extract_index_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    index::extract_final_page(v, w)
}
pub fn select_forms(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    output(forms::select(&v["source"], &mut Work(0))?)
}
pub(super) fn form_supersedes(form: &Box4, nested: &Box4) -> bool {
    forms::supersedes(form, nested)
}
pub fn prepare(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_with_work(v, &mut Work(0))
}
/// Geometry-stage checkpoint: split calibrated rows, then reclaim image labels.
/// Callers must supply the lines after geometry repair and resplitting.
pub fn finish_image_text(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    finish_image_text_with_work(v, &mut Work(0))
}
pub(super) fn finish_image_text_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    finish_image_state(v, w).map(|(out, _, _)| out)
}
fn finish_image_state(
    v: &Value,
    w: &mut Work,
) -> Result<(Value, Vec<Value>, std::collections::BTreeSet<i64>), Error> {
    finish_image_state_lines(v, v["remaining_lines"].clone(), w)
}
fn finish_image_state_owned(
    v: &mut Value,
    w: &mut Work,
) -> Result<(Value, Vec<Value>, std::collections::BTreeSet<i64>), Error> {
    let lines = v
        .get_mut("remaining_lines")
        .map(Value::take)
        .unwrap_or(Value::Null);
    finish_image_state_lines(v, lines, w)
}
fn finish_image_state_lines(
    v: &Value,
    lines: Value,
    w: &mut Work,
) -> Result<(Value, Vec<Value>, std::collections::BTreeSet<i64>), Error> {
    let mut image_boxes = Vec::new();
    let mut containers = optional(v, "caption_graphics", 4000)?.to_vec();
    for key in ["form_image_blocks", "graphic_blocks", "raster_image_blocks"] {
        for block in optional(v, key, 4000)? {
            image_boxes.push(block["bbox"].clone());
            containers.push(block.clone());
        }
    }
    let mut request = value_object!({"drawing_lines":v["source"]["drawing_lines"],"image_bboxes":image_boxes,"table_bboxes":v["table_bboxes"],"page_size":v["source"]["page_size"],"source_index_start":v["source_index_start"]});
    request["lines"] = lines;
    let (lines, replaced) = late_images::split_owned(request, w)?;
    let mut request = value_object!({"strong_graphic_bboxes":v["strong_graphic_bboxes"]});
    request["remaining_lines"] = lines;
    request["image_containers"] = Value::Array(containers);
    let (out, updates) = late_images::reclaim_source_owned(&v["source"], request, w)?;
    Ok((output(out)?, updates, replaced))
}
/// Resume only after line/character geometry repair and visual resplitting.
/// This preserves separate block identities even when their bboxes coincide.
pub fn finish_calibrated_page(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    finish_calibrated_page_with_work(v, &mut Work(0))
}
pub(super) fn finish_calibrated_page_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    finish_calibrated_page_with_aliases(v, w, &std::collections::BTreeSet::new())
}
pub(super) fn finish_calibrated_page_with_aliases(
    v: &Value,
    w: &mut Work,
    detached: &std::collections::BTreeSet<i64>,
) -> Result<Value, Error> {
    finish_calibrated_page_with_roles(v, w, detached, None)
}
pub(super) fn finish_calibrated_page_with_roles(
    v: &Value,
    w: &mut Work,
    detached: &std::collections::BTreeSet<i64>,
    roles: Option<(i64, bool, bool, Option<&Value>)>,
) -> Result<Value, Error> {
    finish_calibrated_page_owned(v.clone(), w, detached, roles)
}
pub(super) fn finish_calibrated_page_owned(
    mut out: Value,
    w: &mut Work,
    detached: &std::collections::BTreeSet<i64>,
    roles: Option<(i64, bool, bool, Option<&Value>)>,
) -> Result<Value, Error> {
    let mut profile = crate::processing::NativeStageProfile::new();
    let (mut images, image_updates, mut replaced) = finish_image_state_owned(&mut out, w)?;
    profile.mark("finish_image_state");
    replaced.extend(detached);
    let mut offset = 0;
    let blocks = array(&images["image_containers"], 4000)?;
    for key in [
        "caption_graphics",
        "form_image_blocks",
        "graphic_blocks",
        "raster_image_blocks",
    ] {
        let count = optional(&out, key, 4000)?.len();
        out[key] = Value::Array(blocks[offset..offset + count].to_vec());
        offset += count;
    }
    out["reclaimed_line_indices"] = images["reclaimed_line_indices"].clone();
    let (remaining, fraction_updates, fraction_detached) =
        super::native_formula_text::restore_owned_with_identities(
            images["remaining_lines"].take(),
            &out["source"]["page_size"],
            &out["source"]["drawing_lines"],
            w,
        )?;
    out["remaining_lines"] = remaining;
    profile.mark("finish_restore_formula_text");
    let mut updates = std::collections::BTreeMap::new();
    for l in image_updates.iter().chain(fraction_updates.iter()) {
        let id = crate::legacy_model::path_graphics::integer(l, "source_index", 0)?;
        if !replaced.contains(&id) && !l["median_glyph_width"].is_null() {
            updates.insert(id, l["median_glyph_width"].clone());
        }
    }
    optional(&out["source"], "lines", 4096)?;
    for line in out["source"]["lines"].as_array_mut().unwrap() {
        if let Some(width) = updates.get(&crate::legacy_model::path_graphics::integer(
            line,
            "source_index",
            0,
        )?) {
            line["median_glyph_width"] = width.clone();
        }
    }
    let mut fixed = optional(&out, "prior_fixed_blocks", 4000)?.to_vec();
    for key in [
        "rule_code_blocks",
        "annotation_blocks",
        "table_blocks",
        "code_blocks",
        "form_image_blocks",
        "graphic_blocks",
        "raster_image_blocks",
        "vector_formula_blocks",
        "caption_graphics",
        "index_blocks",
    ] {
        fixed.extend_from_slice(optional(&out, key, 4000)?);
    }
    if fixed.len() > rapiddoc_core::resources::limit(4000) {
        return Err(limit());
    }
    out["fixed_blocks"] = Value::Array(fixed);
    if let Some((page, anomaly, footnotes, complete)) = roles {
        replaced.extend(fraction_detached);
        out["canonical_formula_source_lines"] = if anomaly {
            out["remaining_lines"].clone()
        } else {
            json!([])
        };
        let remaining = array(&out["remaining_lines"], 4096)?;
        let sources = array(&out["source"]["lines"], 4096)?;
        let mut aliases = Vec::new();
        let mut positions = std::collections::BTreeMap::new();
        for (i, l) in remaining.iter().enumerate() {
            w.charge(1)?;
            positions
                .entry(super::path_graphics::integer(l, "source_index", 0)?)
                .or_insert(i);
        }
        for (reference_index, r) in sources.iter().enumerate() {
            w.charge(1)?;
            let id = super::path_graphics::integer(r, "source_index", 0)?;
            if replaced.contains(&id) {
                continue;
            }
            if let Some(&line_index) = positions.get(&id) {
                aliases.push(
                    value_object!({"reference_index":reference_index,"line_index":line_index}),
                );
            }
        }
        let mut visual = Vec::new();
        for k in [
            "caption_graphics",
            "form_image_blocks",
            "graphic_blocks",
            "raster_image_blocks",
        ] {
            visual.extend(optional(&out, k, 4000)?.iter().map(|b| b["bbox"].clone()));
        }
        let mut regions = Vec::new();
        for b in optional(&out, "vector_formula_blocks", 4000)? {
            if !optional(b, "_inline_formula_prose_sources", 4096)?.is_empty() {
                regions.push(
                    value_object!({"bbox":b["bbox"],"source_indices":b["_inline_formula_prose_sources"]}),
                );
            }
        }
        let options = value_object!({"page_size":out["source"]["page_size"],"page_index":page,"reference_aliases":aliases,"visual_bboxes":visual,"table_bboxes":out["table_bboxes"],"inline_regions":regions});
        let Value::Array(mut source) = out["source"]["lines"].take() else {
            return Err(invalid());
        };
        let Value::Array(remaining) = out["remaining_lines"].take() else {
            return Err(invalid());
        };
        let mut lines = super::native_text_roles::classify_lines_with_reference(
            remaining,
            &options,
            Some(&source),
            w,
        )?;
        profile.mark("finish_text_roles");
        if footnotes {
            let Value::Array(owned) = lines else {
                return Err(invalid());
            };
            let mut notes = super::native_footnotes::classify_lines_with_reference(
                owned,
                &value_object!({"page_size":out["source"]["page_size"],"visual_bboxes":visual,"table_bboxes":out["table_bboxes"],"drawing_lines":out["source"]["drawing_lines"]}),
                Some(&source),
                w,
            )?;
            lines = notes["lines"].take();
            out["page_footnote_groups"] = notes["page_footnote_groups"].take();
        }
        profile.mark("finish_footnotes");
        let ls = array(&lines, 4096)?;
        for a in aliases {
            source[a["reference_index"].as_u64().unwrap() as usize] =
                ls[a["line_index"].as_u64().unwrap() as usize].clone();
        }
        out["source"]["lines"] = Value::Array(source);
        out["remaining_lines"] = lines;
        if let Some(options) = complete {
            let result = super::native_page_preparation::finish_owned(out, options, &replaced, w);
            profile.mark("finish_native_page_preparation");
            return result;
        }
    }
    // Later text/formula classifiers still run before the final trailing-table query.
    output(out)
}
pub fn prepare_document(v: &Value) -> Result<Value, Error> {
    document(v, true, true)
}
pub(super) fn normalize_form_sources_owned(
    sources: Vec<Value>,
    w: &mut Work,
) -> Result<Vec<Value>, Error> {
    roles::normalize_owned(sources, w)
}
pub(super) fn layout_lanes(
    lines: &[Value],
    sz: [f64; 2],
    w: &mut Work,
) -> Result<Vec<Value>, Error> {
    Ok(roles::layout(lines.to_vec(), sz, 0, w)?.lanes)
}
pub(super) fn layout_corridor(b: &Box4, lanes: &[Value], width: f64) -> Option<(f64, f64)> {
    captions::corridor(b, lanes, width)
}
pub(super) fn strong_caption(s: &str) -> bool {
    roles::caption(s)
}
/// Frozen checkpoint before vector-formula processing.
pub fn prepare_image_document(v: &Value) -> Result<Value, Error> {
    document(v, true, false)
}
pub fn prepare_form_document(v: &Value) -> Result<Value, Error> {
    document(v, false, false)
}
fn document(v: &Value, all_images: bool, vector_formulas: bool) -> Result<Value, Error> {
    validate_tree(v)?;
    let w = &mut Work(0);
    let sources = roles::normalize(array(&v["sources"], 1000)?, w)?;
    let mut pages = Vec::new();
    for source in sources {
        pages.push(prepare_with_work(
            &value_object!({"source":source,"include_form_images":true,"include_graphic_images":all_images,"include_vector_formulas":vector_formulas}),
            w,
        )?);
    }
    output(value_object!({"pages":pages}))
}
pub(super) fn materialize_forms_owned(v: Value, w: &mut Work) -> Result<Value, Error> {
    media::forms_owned(v, w)
}
pub(super) fn materialize_graphics_owned(v: Value, w: &mut Work) -> Result<Value, Error> {
    graphics::build_owned(v, w)
}
pub(super) fn materialize_images_owned(v: Value, w: &mut Work) -> Result<Value, Error> {
    images::build_owned(v, w)
}
pub(crate) fn prepare_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    prepare_owned(v.clone(), w)
}
pub(crate) fn prepare_owned(mut v: Value, w: &mut Work) -> Result<Value, Error> {
    let mut profile = crate::processing::NativeStageProfile::new();
    let mut source = v["source"].take();
    source.as_object().ok_or_else(invalid)?;
    for k in [
        "lines",
        "chars",
        "drawing_lines",
        "path_infos",
        "image_bboxes",
        "signature_bboxes",
        "form_bboxes",
    ] {
        if source.get(k).is_none() {
            source[k] = json!([]);
        }
    }
    if v["include_precontainer_classifiers"] == true
        && super::path_graphics::integer(&v, "page_index", 0)? == 0
    {
        source = super::native_precontainers::correspondence(&source, w)?;
    }
    let mut containers = optional(&source, "image_bboxes", 5000)?.to_vec();
    containers.extend_from_slice(optional(&source, "signature_bboxes", 5000)?);
    if containers.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    let mut index_request = value_object!({"page_size":source["page_size"],"container_bboxes":containers,"require_heading":true,"require_explicit_heading":true});
    index_request["lines"] = source["lines"].take();
    let mut ix = index::extract_owned(index_request, w)?;
    source["lines"] = ix["lines"].take();
    profile.mark("containers_index");
    let mut caption = if v["include_graphic_images"] == true {
        captions::build_owned(source, w)?
    } else {
        crate::processing::with_source(
            source,
            json!({"caption_graphics":[],"claimed_line_indices":[]}),
        )
    };
    source = caption["source"].take();
    profile.mark("containers_captions");
    if v["include_precontainer_classifiers"] == true {
        let mut strong = super::strong_graphics::detect_source_owned(source, w)?;
        source = strong["source"].take();
        let mut barriers = optional(&source, "image_bboxes", 5000)?.to_vec();
        barriers.extend(
            array(&caption["caption_graphics"], 4000)?
                .iter()
                .map(|b| b["bbox"].clone()),
        );
        barriers.extend_from_slice(array(&strong["graphic_bboxes"], 5000)?);
        if barriers.len() > rapiddoc_core::resources::limit(5000) {
            return Err(limit());
        }
        source = super::native_precontainers::appendix_headings(&source, &barriers, w)?;
    }
    let mut selected = forms::select_owned(source, w)?;
    profile.mark("containers_graphics_and_forms");
    source = selected["source"].take();
    let caption_boxes = array(&caption["caption_graphics"], 4000)?
        .iter()
        .map(|b| bbox(&b["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut selected_forms = Vec::new();
    for b in array(&selected["form_bboxes"], 5000)? {
        w.charge(caption_boxes.len() + 1)?;
        let b_box = bbox(b)?;
        if !caption_boxes
            .iter()
            .any(|c| crate::legacy_model::path_graphics::overlap_smaller(&b_box, c) >= 0.8)
        {
            selected_forms.push(b.clone());
        }
    }
    let include_graphic_images = v["include_graphic_images"] == true;
    let mut claimed = optional(&v, "prior_claimed_line_indices", 20000)?.to_vec();
    claimed.extend_from_slice(array(&ix["claimed_line_indices"], 4096)?);
    claimed.extend_from_slice(array(&caption["claimed_line_indices"], 20000)?);
    let mut q = v;
    q["source"] = source;
    q["prior_claimed_line_indices"] = Value::Array(claimed);
    let mut out = super::table_page::prepare_containers_owned(
        q,
        w,
        &selected_forms,
        array(&ix["index_blocks"], 4000)?,
        array(&caption["caption_graphics"], 4000)?,
    )?;
    out["index_blocks"] = ix["index_blocks"].take();
    profile.mark("containers_tables");
    out["selected_form_bboxes"] = json!(selected_forms);
    if include_graphic_images {
        out["caption_graphics"] = caption["caption_graphics"].take();
    }
    output(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owned_late_image_text_preserves_python_split_reclaim() {
        let f: Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!(
                "../../../../../../tests/fixtures/native_late_image_owned_reference.json.gz"
            )[..],
        ))
        .unwrap();
        for host in [false, true] {
            let _scope = host.then(crate::legacy_page::budget::host_page);
            for c in f["cases"].as_array().unwrap() {
                let q = &c["request"];
                let (mut a, mut b) = (Work(0), Work(0));
                let old = finish_image_state(q, &mut a).unwrap();
                let new = finish_image_state_owned(&mut q.clone(), &mut b).unwrap();
                assert_eq!(old.0, c["expected"], "{}", c["label"]);
                assert_eq!(new, old, "{}", c["label"]);
                assert_eq!(a.0, b.0);
            }
        }
        for q in [json!(null), json!(3), json!({})] {
            let old = finish_image_state(&q, &mut Work(0)).unwrap_err();
            let new = finish_image_state_owned(&mut q.clone(), &mut Work(0)).unwrap_err();
            assert_eq!(old.code, new.code);
            assert_eq!(old.message, new.message);
        }
    }
    #[test]
    fn calibrated_input_errors() {
        let bad = json!({"lines":[{"text":"body","bbox":[0,null,10,10],"source_index":0}],"drawing_lines":[],"page_size":[400,600]});
        assert!(super::super::native_formula_text::restore_leading(&bad).is_err());
        let mut bad = bad;
        bad["image_bboxes"] = json!([]);
        bad["table_bboxes"] = json!([]);
        assert!(late_images::split_with_sources(&bad, &mut Work(0)).is_err());
    }
    #[test]
    fn old_calibrated_page_continuation() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-calibrated-pages-canonical.json");
        let fixtures: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for c in fixtures["cases"].as_array().unwrap() {
            assert_eq!(
                finish_calibrated_page(&c["request"])
                    .unwrap_or_else(|e| panic!("{}: {e}", c["label"])),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
    #[test]
    fn old_calibrated_image_text() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-late-image-text.json");
        let fixtures: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for c in fixtures["cases"].as_array().unwrap() {
            let out =
                finish_image_text(&c["request"]).unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            assert_eq!(out, c["expected"], "{}", c["label"]);
        }
    }
    #[test]
    fn old_caption_supported_images() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-caption-images.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let out = captions::build(&c["request"], &mut Work(0))
                .unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            assert_eq!(out, c["expected"], "{}", c["label"]);
        }
    }
    #[test]
    fn old_complete_image_chain() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-image-chain.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let q = &c["request"];
            let mut w = Work(0);
            let out=if q["kind"]=="graphic"{let strong=crate::legacy_model::strong_graphics::detect_with_work(&json!({"source":q["source"]}),&mut w).unwrap();graphics::build(&json!({"source":strong["source"],"strong_core_bboxes":strong["graphic_bboxes"],"excluded_bboxes":q.get("excluded_bboxes").cloned().unwrap_or(json!([])),"claimed_line_indices":q.get("claimed_line_indices").cloned().unwrap_or(json!([]))}),&mut w)}else{images::build(&json!({"source":q["source"],"container_blocks":q.get("container_blocks").cloned().unwrap_or(json!([])),"claimed_line_indices":q.get("claimed_line_indices").cloned().unwrap_or(json!([]))}),&mut w)}.unwrap_or_else(|e|panic!("{}: {e}",c["label"]));
            assert_eq!(out, c["expected"], "{}", c["label"]);
        }
    }
    #[test]
    fn old_complete_containers() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../build/migration");
        for (name, f) in [
            (
                "native-index-containers",
                extract_index as fn(&Value) -> Result<Value, Error>,
            ),
            ("native-form-selection-canonical", select_forms),
            ("native-form-map-keys", select_forms),
            ("native-index-integer-limits", extract_index),
            ("native-container-table-page-canonical", prepare),
        ] {
            let fixtures: Value =
                serde_json::from_slice(&std::fs::read(root.join(format!("{name}.json"))).unwrap())
                    .unwrap();
            for c in fixtures["cases"].as_array().unwrap() {
                let result =
                    f(&c["request"]).unwrap_or_else(|e| panic!("{name}/{}: {e}", c["label"]));
                assert_eq!(result, c["expected"], "{name}/{}", c["label"]);
            }
            for c in fixtures["errors"].as_array().unwrap() {
                assert!(f(&c["request"]).is_err(), "{}", c["label"]);
            }
        }
        let f: Value = serde_json::from_slice(
            &std::fs::read(root.join("native-page-container-boundaries.json")).unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            let result = match c["kind"].as_str().unwrap() {
                "index" => extract_index(&c["request"]),
                _ => select_forms(&c["request"]),
            }
            .unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            assert_eq!(result, c["expected"], "{}", c["label"]);
        }
    }
    #[test]
    fn shared_work_and_recovery() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../build/migration");
        for (name, f) in [
            (
                "native-index-containers",
                index::extract as fn(&Value, &mut Work) -> Result<Value, Error>,
            ),
            ("native-form-selection-canonical", |v, w| {
                forms::select(&v["source"], w)
            }),
            ("native-container-table-page-canonical", prepare_with_work),
        ] {
            let fixtures: Value =
                serde_json::from_slice(&std::fs::read(root.join(format!("{name}.json"))).unwrap())
                    .unwrap();
            for c in fixtures["cases"].as_array().unwrap().iter().step_by(17) {
                let q = &c["request"];
                let mut w = Work(0);
                assert!(f(q, &mut w).is_ok());
                if w.0 > 0 {
                    assert!(
                        f(q, &mut Work(10_000_000 - w.0 + 1)).is_err(),
                        "{}",
                        c["label"]
                    );
                }
                assert!(f(q, &mut Work(0)).is_ok());
            }
        }
    }
}
