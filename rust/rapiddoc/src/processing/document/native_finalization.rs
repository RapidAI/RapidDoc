//! Contiguous existing Python final-page statements through full text assembly and member/caption/list rebuilding.
//! Optional final output continues through the entire original visual and ordering tail.
use super::{
    native_formula_components as formulas, native_page_preparation as merging,
    native_titles as titles,
    page_containers::{optional, size},
    path_graphics::{array, integer},
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::python_space;
use serde_json::{json, Value};
fn refine(request: &mut Value, ls: Value, kind: &str, w: &mut Work) -> Result<Value, Error> {
    request["kind"] = json!(kind);
    request["lines"] = ls;
    Ok(titles::refine_final_titles_with_work(request, w)?["lines"].take())
}
pub fn prepare_lines(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_lines_with_work(v, &mut Work(0))
}
pub fn prepare_blocks(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_blocks_with_work(v, &mut Work(0))
}
pub(super) fn prepare_blocks_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let finalized = v
        .get("finalize_output")
        .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))?;
    let refined = v
        .get("refine_blocks")
        .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))?;
    let mut out = prepare_lines_with_work(v, w)?;
    let p = &v["page"];
    let sz = size(p.get("page_size").unwrap_or(&p["source"]["page_size"]))?;
    let mut lines = array(&out["semantic_lines"], 4096)?.to_vec();
    lines.extend_from_slice(array(&out["remaining_lines"], 4096)?);
    let drawing = if p.get("drawing_lines").is_some() {
        optional(p, "drawing_lines", 5000)?
    } else {
        optional(&p["source"], "drawing_lines", 5000)?
    };
    let visual = optional(p, "fixed_blocks", 4000)?
        .iter()
        .filter(|b| b["type"] == "image")
        .map(|b| b["bbox"].clone())
        .collect::<Vec<_>>();
    out["text_blocks"] = super::native_text_assembly::build_with_work(
        &json!({"lines":lines,"page_size":sz,"table_bboxes":optional(p,"table_bboxes",5000)?,"drawing_lines":drawing,"visual_bboxes":visual,"page_footnote_groups":out["page_footnote_groups"],"page_index":integer(v,"page_index",0)?}),
        w,
    )?;
    if refined || finalized {
        let grids = optional(p, "fixed_blocks", 4000)?
            .iter()
            .filter(|b| {
                b["type"] == "table"
                    && super::native_text_assembly::python_truth(&b["_native_numeric_grid"])
            })
            .map(|b| b["bbox"].clone())
            .collect::<Vec<_>>();
        out["refined_text_blocks"] = super::native_text_assembly::refine_page_blocks_with_work(
            &json!({"blocks":out["text_blocks"],"page_size":sz,"page_index":integer(v,"page_index",0)?,"image_bboxes":visual,"numeric_grid_bboxes":grids,"bullet_paths":optional(p,"bullet_paths",5000)?,"kind":"sequence"}),
            w,
        )?;
    }
    if finalized {
        let mut absolute = optional(p, "fixed_blocks", 4000)?.to_vec();
        absolute.extend_from_slice(array(&out["formula_blocks"], 4096)?);
        absolute.extend_from_slice(array(&out["index_blocks"], 4096)?);
        absolute.extend_from_slice(array(&out["refined_text_blocks"], 4096)?);
        out["model_list"] = finish_output(absolute, sz, drawing, p, w)?;
    }
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        Err(limit())
    } else {
        Ok(out)
    }
}
fn finish_output(
    mut bs: Vec<Value>,
    p: [f64; 2],
    drawing: &[Value],
    page: &Value,
    w: &mut Work,
) -> Result<Value, Error> {
    use super::{native_final_output as output, native_visual_annotations as visual};
    visual::recover_with_work(&mut bs, drawing, w)?;
    output::tight(&mut bs, p)?;
    let bound = visual::bind_with_work(&json!({"blocks":bs,"page_size":p}), w)?;
    bs = array(&bound["blocks"], 4096)?.to_vec();
    let regions = array(&bound["regions"], 4096)?
        .iter()
        .map(|r| {
            array(r, 4096)?
                .iter()
                .map(|i| i.as_u64().map(|i| i as usize).ok_or_else(invalid))
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let order = super::native_grouped_order::sort(&mut bs, p, &regions, w)?;
    bs = order.into_iter().map(|i| bs[i].clone()).collect();
    bs = output::narrow_order(bs, p, w)?;
    bs = output::reference_order(bs, p, page, w)?;
    let negative = |b: &Value| -> Result<bool, Error> {
        let v = &b["_reference_group"];
        if !super::native_text_assembly::python_truth(v) {
            Ok(false)
        } else {
            Ok(super::table_native_text::number(v)?.ok_or_else(invalid)? < 0.)
        }
    };
    let mut front = vec![];
    for b in &bs {
        if negative(b)? {
            front.push(b.clone())
        }
    }
    front.sort_by(|a, b| {
        a["bbox"][1]
            .as_f64()
            .unwrap()
            .total_cmp(&b["bbox"][1].as_f64().unwrap())
    });
    let mut front = front.into_iter();
    let mut normalized = vec![];
    for b in bs {
        let b = if negative(&b)? {
            front.next().ok_or_else(invalid)?
        } else {
            b
        };
        let b = output::normalize(&b, p)?;
        if !b.is_null() {
            normalized.push(b)
        }
    }
    Ok(json!(normalized))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_python_complete_final_page_output() {
        for (name, chain) in [
            ("native-final-page-output", false),
            ("native-final-page-output-chain", true),
        ] {
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../build/migration/{name}.json"));
            let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
            for c in f["cases"].as_array().unwrap() {
                let mut q = c["request"].clone();
                q["finalize_output"] = json!(true);
                let got = if chain {
                    super::super::native_document::prepare_text_blocks(&q)
                } else {
                    prepare_blocks(&q)
                };
                if c.get("error").is_some() {
                    assert!(got.is_err(), "{}", c["label"])
                } else {
                    let got = got.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
                    if let Some(d) =
                        super::super::native_document::tests::diff(&got, &c["expected"], "")
                    {
                        panic!("{}: {d}", c["label"])
                    }
                }
            }
        }
    }
    #[test]
    fn existing_python_final_line_contracts() {
        let f: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-final-line-contracts.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            let out = if c["kind"] == "structure" {
                titles::classify_page_structure(&c["request"])
            } else {
                titles::refine_final_titles(&c["request"])
            }
            .unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = super::super::native_document::tests::diff(&out, &c["expected"], "") {
                panic!("{}: {d}", c["label"])
            }
        }
    }
    #[test]
    fn existing_python_final_text_lines() {
        for name in [
            "native-final-text-lines",
            "native-final-text-lines-chain-canonical",
        ] {
            let f: Value = serde_json::from_slice(
                &std::fs::read(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join(format!("../../build/migration/{name}.json")),
                )
                .unwrap(),
            )
            .unwrap();
            let mut failures = vec![];
            for c in f["cases"].as_array().unwrap() {
                let out = if name.contains("chain") {
                    super::super::native_document::prepare_final_text(&c["request"])
                } else {
                    prepare_lines(&c["request"])
                };
                if !c["error"].is_null() {
                    assert!(out.is_err(), "{}", c["label"]);
                    continue;
                }
                match out {
                    Err(e) => failures.push(format!("{}: {e}", c["label"])),
                    Ok(out) => {
                        if let Some(d) =
                            super::super::native_document::tests::diff(&out, &c["expected"], "")
                        {
                            failures.push(format!("{}: {d}", c["label"]))
                        }
                    }
                }
            }
            assert!(
                failures.is_empty(),
                "{name}: {} differences: {}",
                failures.len(),
                failures
                    .iter()
                    .take(25)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join("\n")
            );
        }
    }
}
pub(super) fn prepare_lines_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let p = &v["page"];
    let sz = size(p.get("page_size").unwrap_or(&p["source"]["page_size"]))?;
    let page = integer(v, "page_index", 0)?;
    let mut out = formulas::prepare_page_with_work(&value_object!({"page":p}), w)?;
    let fixed = optional(p, "fixed_blocks", 4000)?;
    let title_containers = fixed
        .iter()
        .filter(|b| {
            !(b["_inline_visual_row_id"].is_i64()
                || b["_inline_visual_row_id"].is_u64()
                || b["_inline_visual_row_id"].is_boolean())
        })
        .map(|b| b["bbox"].clone())
        .collect::<Vec<_>>();
    let caption_containers = fixed
        .iter()
        .filter(|b| matches!(b["type"].as_str(), Some("image" | "code")))
        .map(|b| b["bbox"].clone())
        .collect::<Vec<_>>();
    let images = fixed
        .iter()
        .filter(|b| b["type"] == "image")
        .map(|b| b["bbox"].clone())
        .collect::<Vec<_>>();
    let mut request = json!({"page_size":sz,"page_index":page,"container_bboxes":title_containers,"image_bboxes":images,"document_body_profile":v["document_body_profile"]});
    let mut ls = out["remaining_lines"].take();
    if page == 0 {
        ls = refine(&mut request, ls, "front", w)?
    }
    for kind in ["explicit", "reset", "body"] {
        request["kind"] = json!(kind);
        request["lines"] = ls;
        ls = titles::classify_page_structure_with_work(&request, w)?["lines"].take();
    }
    request["lines"] = ls;
    request["caption_container_bboxes"] = json!(caption_containers);
    request["document_title_profile"] = v["document_title_profile"].clone();
    request["classify_page"] = json!(true);
    request["final_page_objects"] = json!(true);
    ls = titles::prepare_with_work(&request, w)?["lines"].take();
    ls = refine(&mut request, ls, "suppress", w)?;
    request["lines"] = ls;
    request["kind"] = json!("promote");
    request["title_candidate_source_indices"] = json!(array(&request["lines"], 4096)?
        .iter()
        .filter(|l| l["semantic_type"] == "paragraph_title")
        .map(|l| integer(l, "source_index", 0))
        .collect::<Result<Vec<_>, _>>()?);
    request["document_body_profile"] = if v["canonical_body_profile"].is_null() {
        v["document_body_profile"].clone()
    } else {
        v["canonical_body_profile"].clone()
    };
    ls = titles::classify_page_structure_with_work(&request, w)?["lines"].take();
    request["document_body_profile"] = v["document_body_profile"].clone();
    array(&out["semantic_lines"], 4096)?;
    let Value::Array(mut all) = out["semantic_lines"].take() else {
        return Err(invalid());
    };
    array(&ls, 4096)?;
    let Value::Array(lines) = ls else {
        return Err(invalid());
    };
    all.extend(lines);
    if page == 0 {
        for l in &mut all {
            let t = l["text"]
                .as_str()
                .ok_or_else(invalid)?
                .trim_matches(python_space)
                .chars()
                .flat_map(|c| {
                    if c == 'ſ' {
                        vec!['s']
                    } else {
                        c.to_lowercase().collect()
                    }
                })
                .collect::<String>();
            if t == "abstract" || t == "摘要" {
                l["semantic_type"] = json!("paragraph_title")
            }
        }
    }
    ls = Value::Array(all);
    for kind in [
        "runin",
        "repeated",
        "panels",
        "numbered",
        "wrapped",
        "displaced",
        "cjk",
    ] {
        ls = refine(&mut request, ls, kind, w)?
    }
    let mut merge_request = value_object!({"page_size":sz,"table_bboxes":optional(p,"table_bboxes",5000)?,"kind":"sequence"});
    merge_request["lines"] = ls;
    ls = merging::merge_final_text_with_work(&merge_request, w)?["lines"].take();
    let mut reference_page = p.clone();
    reference_page["remaining_lines"] = ls;
    super::native_references::group_with_work(&mut reference_page, w)?;
    ls = reference_page["remaining_lines"].take();
    if page == 0 {
        ls = refine(&mut request, ls, "front", w)?
    }
    out["remaining_lines"] = ls;
    out["semantic_lines"] = json!([]);
    out["page_footnote_groups"] = json!(optional(&reference_page, "page_footnote_groups", 4096)?);
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        Err(limit())
    } else {
        Ok(out)
    }
}
