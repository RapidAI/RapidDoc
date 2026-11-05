//! Shared document geometry, marginal/Form roles and original page statements
//! through container claims, calibrated resplitting and leading fractions.
use super::{
    page_containers::optional,
    path_graphics::{array, integer},
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Map, Value};
fn geometry_map(g: &Value, key: &str, value: &str) -> Result<Value, Error> {
    let mut out = Map::new();
    for e in optional(g, key, 20000)? {
        let id = integer(e, "char_idx", 0)?;
        out.insert(id.to_string(), e[value].clone());
    }
    Ok(Value::Object(out))
}
pub fn prepare_pages(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_pages_with_work(v, &mut Work(0))
}
fn prepare_pages_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    prepare_pages_owned(v.clone(), w)
}
fn prepare_pages_owned(v: Value, w: &mut Work) -> Result<Value, Error> {
    let chain = v["chain_table_grids"] == true;
    let post_roles = v["include_post_text_roles"] == true;
    let post_footnotes = v["include_post_footnotes"] == true;
    let prepare_page = v["prepare_page"] == true;
    let mut profile = crate::processing::NativeStageProfile::new();
    let mut prefix = super::native_marginals::prepare_document_sources_owned(v, w)?;
    profile.mark("document_source_prefix");
    let Value::Array(raw_pages) = prefix["pages"].take() else {
        return Err(invalid());
    };
    let mut pages = Vec::new();
    let mut previous_grid = Value::Null;
    for (index, mut page) in raw_pages.into_iter().enumerate() {
        let mut page_profile = crate::processing::NativeStageProfile::new();
        let g = page["geometry"].take();
        let mut chars = Map::new();
        if !optional(&prefix["plan"], "local_metric_repair_sources", 20000)?.is_empty() {
            for c in optional(&prefix["plan"], "char_repairs", 20000)? {
                w.charge(1)?;
                if integer(c, "page_index", 0)? == index as i64 && c["y_state"] == "abnormal" {
                    chars.insert(
                        integer(c, "char_idx", 0)?.to_string(),
                        c["layout_bbox"].clone(),
                    );
                }
            }
        }
        let mut source = page["source"].take();
        let tight = geometry_map(&g, "tight_bboxes", "bbox")?;
        let origins = geometry_map(&g, "origins", "origin")?;
        // The document entry follows the original sequential loop; earlier
        // per-page diagnostic endpoints retain their independent-page contract.
        if chain {
            source["page_index"] = json!(index);
            if !previous_grid.is_null() {
                source["previous_table_grid"] = previous_grid.clone();
            } else if let Some(obj) = source.as_object_mut() {
                obj.remove("previous_table_grid");
            }
        }
        let mut request = value_object!({"page_index":index,"include_precontainer_classifiers":true,"include_form_images":true,"include_graphic_images":true,"include_vector_formulas":true,"tight_bboxes":tight,"origins":origins,"calibrated_char_bboxes":chars,"table_header_separator_bboxes":prefix["header_separator_bboxes"][index]});
        request["source"] = source;
        let prepared = super::page_containers::prepare_owned(request, w)?;
        page_profile.mark("prepare_page_containers");
        let mut request = json!({"page_index":index,"include_post_text_roles":post_roles,"include_post_footnotes":post_footnotes,"prepare_page":prepare_page,"tight_bboxes":tight,"origins":origins,"style_lines":page.get("style_lines").cloned().unwrap_or(json!([])),"link_lines":page.get("link_lines").cloned().unwrap_or(json!([]))});
        request["prepared"] = prepared;
        pages.push(super::native_geometry::finish_page_owned(
            request,
            &prefix["plan"],
            w,
        )?);
        previous_grid = pages.last().unwrap()["next_table_grid"].clone();
        page_profile.mark("finish_page_geometry_and_roles");
    }
    profile.mark("prepare_document_pages");
    let mut out = value_object!({"plan":prefix["plan"].take(),"header_separator_bboxes":prefix["header_separator_bboxes"].take()});
    out["pages"] = Value::Array(pages);
    if crate::legacy_page::budget::output_exceeds(&out) {
        Err(limit())
    } else {
        Ok(out)
    }
}
/// Original preparation and document auxiliary/body passes, before title inference.
pub fn prepare_auxiliary(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_auxiliary_with_work(v, &mut Work(0))
}
fn prepare_auxiliary_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    prepare_auxiliary_owned(v.clone(), w)
}
fn prepare_auxiliary_owned(mut request: Value, w: &mut Work) -> Result<Value, Error> {
    request["prepare_page"] = json!(true);
    request["chain_table_grids"] = json!(true);
    let mut out = prepare_pages_owned(request, w)?;
    let Value::Array(pages) = &mut out["pages"] else {
        return Err(invalid());
    };
    super::native_vector_formulas::classify_repeated_in_place(pages, w)?;
    let mut request = json!({});
    request["pages"] = out["pages"].take();
    let mut classified = super::native_document_text::prepare_owned(request, w)?;
    out["pages"] = classified["pages"].take();
    out["body_profile"] = classified["body_profile"].take();
    out["canonical_body_profile"] = classified["canonical_body_profile"].take();
    if crate::legacy_page::budget::output_exceeds(&out) {
        Err(limit())
    } else {
        Ok(out)
    }
}
/// The complete original `_classify_document_text` sequence after full page preparation.
pub fn prepare_text(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_text_with_work(v, &mut Work(0))
}
fn prepare_text_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    prepare_text_owned(v.clone(), w)
}
fn prepare_text_owned(v: Value, w: &mut Work) -> Result<Value, Error> {
    let mut out = prepare_auxiliary_owned(v, w)?;
    let mut request = value_object!({"body_profile":out["body_profile"],"canonical_body_profile":out["canonical_body_profile"]});
    request["pages"] = out["pages"].take();
    let mut titles = super::native_titles::classify_document_owned(request, w)?;
    out["pages"] = titles["pages"].take();
    out["title_profile"] = titles["title_profile"].take();
    if crate::legacy_page::budget::output_exceeds(&out) {
        Err(limit())
    } else {
        Ok(out)
    }
}
/// Original document reference events/grouping and compact-title pass before final page assembly.
pub fn prepare_references(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_references_with_work(v, &mut Work(0))
}
pub(super) fn prepare_references_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    prepare_references_owned(v.clone(), w)
}
fn prepare_references_owned(v: Value, w: &mut Work) -> Result<Value, Error> {
    let mut out = prepare_text_owned(v, w)?;
    let mut request = json!({});
    request["pages"] = out["pages"].take();
    let mut refs = super::native_references::prepare_owned(request, w)?;
    out["pages"] = refs["pages"].take();
    if crate::legacy_page::budget::output_exceeds(&out) {
        Err(limit())
    } else {
        Ok(out)
    }
}
pub fn prepare_final_text(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let w = &mut Work(0);
    let mut out = prepare_references_with_work(v, w)?;
    let mut checkpoints = Vec::new();
    for (i, p) in array(&out["pages"], 1000)?.iter().enumerate() {
        checkpoints.push(super::native_finalization::prepare_lines_with_work(&value_object!({"page":p,"page_index":i,"document_body_profile":out["body_profile"],"canonical_body_profile":out["canonical_body_profile"],"document_title_profile":out["title_profile"]}),w)?)
    }
    out["text_pages"] = json!(checkpoints);
    if crate::legacy_page::budget::output_exceeds(&out) {
        Err(limit())
    } else {
        Ok(out)
    }
}
pub fn prepare_text_blocks(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_text_blocks_owned(v.clone(), true)
}
pub(super) fn finalize_owned(v: Value) -> Result<Value, Error> {
    prepare_text_blocks_owned(v, false).map(|mut out| out["finalized_pages"].take())
}
fn prepare_text_blocks_owned(v: Value, checkpoints: bool) -> Result<Value, Error> {
    validate_tree(&v)?;
    let materialize = v
        .get("materialize_inline")
        .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))?;
    let finalize = v
        .get("finalize_output")
        .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))?;
    let refine = v
        .get("refine_blocks")
        .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))?;
    let w = &mut Work(0);
    let mut spacing = vec![];
    if materialize {
        for p in array(&v["pages"], 1000)? {
            spacing.push(if let Some(lines)=p.get("spacing_lines"){json!(array(lines,4096)?)}else{super::inline_output::prepare_spacing_with_work(&value_object!({"lines":optional(&p["source"],"lines",4096)?,"space_before":p.get("space_before").cloned().unwrap_or(Value::Null)}),w)?});
        }
    }
    let mut out = prepare_references_owned(v, w)?;
    let mut pages = vec![];
    for (i, p) in array(&out["pages"], 1000)?.iter().enumerate() {
        pages.push(super::native_finalization::prepare_blocks_with_work(&value_object!({"page":p,"page_index":i,"document_body_profile":out["body_profile"],"canonical_body_profile":out["canonical_body_profile"],"document_title_profile":out["title_profile"],"refine_blocks":refine,"finalize_output":finalize||materialize}),w)?)
    }
    if checkpoints {
        out["block_pages"] = json!(pages);
    }
    if materialize {
        let prepared = array(&out["pages"], 1000)?;
        let mut inline = json!({"prepared_pages":prepared.iter().map(|p|json!({"script_lines":p.get("script_lines").cloned().unwrap_or(json!([]))})).collect::<Vec<_>>(),"page_style_lines":prepared.iter().map(|p|p.get("style_lines").cloned().unwrap_or(json!([]))).collect::<Vec<_>>(),"page_link_lines":prepared.iter().map(|p|p.get("link_lines").cloned().unwrap_or(json!([]))).collect::<Vec<_>>(),"page_sizes":prepared.iter().map(|p|p.get("page_size").cloned().unwrap_or_else(||p["source"]["page_size"].clone())).collect::<Vec<_>>(),"page_spacing_lines":spacing});
        inline["finalized_pages"] =
            Value::Array(pages.iter_mut().map(|p| p["model_list"].take()).collect());
        if !checkpoints {
            out["pages"] = Value::Null;
        }
        let mut finalized = super::native_inline_document::materialize_with_work(&inline, w)?;
        out["finalized_pages"] = finalized["pages"].take();
    }
    if crate::legacy_page::budget::output_exceeds(&out) {
        Err(limit())
    } else {
        Ok(out)
    }
}
/// Raw document preparation through references and the original final-page formula prefix.
pub fn prepare_formulas(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let w = &mut Work(0);
    let mut out = prepare_references_with_work(v, w)?;
    let mut checkpoints = Vec::new();
    for p in array(&out["pages"], 1000)? {
        checkpoints.push(super::native_formula_components::prepare_page_with_work(
            &value_object!({"page":p}),
            w,
        )?)
    }
    out["formula_pages"] = json!(checkpoints);
    if crate::legacy_page::budget::output_exceeds(&out) {
        Err(limit())
    } else {
        Ok(out)
    }
}
#[cfg(test)]
pub(super) mod tests {
    #[test]
    fn transferred_finalization_matches_previous_installed_package() {
        let fixture: Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!(
                "../../../../../tests/fixtures/upstream_068/native_final_transfer.json.gz"
            )[..],
        ))
        .unwrap();
        for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
            let actual = super::prepare_text_blocks(&case["request"]).unwrap();
            if let Some(d) = diff(&actual, &case["expected"], "final") {
                panic!("case {i}: {d}");
            }
            let actual = super::finalize_owned(case["request"].clone()).unwrap();
            if let Some(d) = diff(&actual, &case["expected"]["finalized_pages"], "owned") {
                panic!("case {i}: {d}");
            }
            let mut compact = case["request"].clone();
            for page in compact["pages"].as_array_mut().unwrap() {
                page["geometry"].as_object_mut().unwrap().remove("chars");
            }
            let actual = super::finalize_owned(compact).unwrap();
            if let Some(d) = diff(
                &actual,
                &case["expected"]["finalized_pages"],
                "no_geometry_snapshot",
            ) {
                panic!("case {i}: {d}");
            }
        }
    }
    #[test]
    fn owned_auxiliary_matches_previous_installed_package() {
        let fixture: Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!(
                "../../../../../tests/fixtures/upstream_068/native_auxiliary_owned.json.gz"
            )[..],
        ))
        .unwrap();
        for (i, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
            let input = value_object!({"pages":case["pages"]});
            let actual = super::prepare_auxiliary_owned(input, &mut Work(0)).unwrap();
            if let Some(d) = diff(&actual, &case["expected"], "auxiliary") {
                panic!("case {i}: {d}");
            }
        }
    }
    #[test]
    fn owned_native_finalization_matches_all_diagnostic_stages() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/upstream_068/native_window_owned.json.gz");
        let input: Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            std::fs::File::open(path).unwrap(),
        ))
        .unwrap();
        let _scope = crate::legacy_page::budget::document(1);
        let expected = super::prepare_text_blocks(&input).unwrap();
        let actual = super::finalize_owned(input).unwrap();
        if let Some(d) = diff(&actual, &expected["finalized_pages"], "pages") {
            panic!("{d}");
        }
    }
    use super::*;
    pub(in crate::processing) fn diff(a: &Value, b: &Value, path: &str) -> Option<String> {
        if crate::legacy_page::eq(a, b) {
            return None;
        }
        match (a, b) {
            (Value::Object(a), Value::Object(b)) => {
                for k in a.keys().chain(b.keys()) {
                    if !a.contains_key(k) || !b.contains_key(k) {
                        return Some(format!("{path}/{k}: missing"));
                    }
                    if let Some(d) = diff(&a[k], &b[k], &format!("{path}/{k}")) {
                        return Some(d);
                    }
                }
            }
            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Some(format!("{path}: length {} vs {}", a.len(), b.len()));
                }
                for (i, (a, b)) in a.iter().zip(b).enumerate() {
                    if let Some(d) = diff(a, b, &format!("{path}/{i}")) {
                        return Some(d);
                    }
                }
            }
            _ => return Some(format!("{path}: {a} vs {b}")),
        }
        None
    }
    #[test]
    fn old_document_pages() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-document-pages-canonical.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let actual = prepare_pages(&c["request"]);
            if !c["error"].is_null() {
                assert!(actual.is_err(), "{}", c["label"]);
                continue;
            }
            let actual = actual.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = diff(&actual, &c["expected"], "") {
                panic!("{}: {d}", c["label"])
            }
        }
    }
}
