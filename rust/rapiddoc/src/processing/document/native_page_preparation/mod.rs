//! Original remaining-line merge closure, script evidence and compact prepared page.
mod baseline;
mod compact;
mod semantic;
use super::{
    native_lines,
    native_marginals::local,
    page_containers::media::glyph_width,
    page_containers::{combined, height, median, optional, size},
    path_graphics::{array, bbox, integer, num, Box4},
    table_text_styles::{clusters, preparation},
};
use crate::legacy_page::{eq, invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{python_decimal, text_boundary::python_space};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
fn text(l: &Value) -> Result<&str, Error> {
    l["text"].as_str().ok_or_else(invalid)
}
pub(super) fn leading_footnote_superscript(l: &Value) -> Result<bool, Error> {
    compact::footnote(l)
}
pub(super) fn formula_math_words(l: &Value) -> Result<Value, Error> {
    compact::math_words(l)
}
pub(super) fn freeze_wrapped_title(l: &mut Value, p: [f64; 2], w: &mut Work) -> Result<(), Error> {
    compact::freeze(l, p, w)
}
fn raw_em(l: &Value) -> Result<f64, Error> {
    l.get("effective_height").map_or(Ok(0.), num)
}
pub(super) fn formula_same_baseline(a: &Box4, ah: f64, b: &Box4, bh: f64, gap: f64) -> bool {
    baseline::same_with_gap(a, ah, b, bh, gap)
}
pub(super) fn formula_dense_materialize(
    indices: &[usize],
    ls: &mut [Value],
    bs: &[Box4],
    p: [f64; 2],
    w: &mut Work,
) -> Result<Value, Error> {
    baseline::materialize_mode(indices, ls, bs, p, w, true)
}
pub(super) fn formula_merge_baseline(
    ls: &mut Vec<Value>,
    p: [f64; 2],
    tables: &[Box4],
    w: &mut Work,
) -> Result<(), Error> {
    let mut aliases = vec![None; ls.len()];
    baseline::merge(ls, &mut aliases, p, tables, w)?;
    Ok(())
}
fn cache_sources(updates: &BTreeMap<i64, Value>, sources: &mut [Value]) -> Result<(), Error> {
    for l in sources {
        if let Some(width) = updates.get(&integer(l, "source_index", 0)?) {
            l["median_glyph_width"] = width.clone();
        }
    }
    Ok(())
}
pub fn merge_final_text(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    semantic::run(v, &mut Work(0))
}
pub(super) fn merge_final_text_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    semantic::run(v, w)
}
#[cfg(test)]
mod semantic_tests {
    use super::*;
    #[test]
    fn existing_python_semantic_merges() {
        let f: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-semantic-merges.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let mut failures = vec![];
        for c in f["cases"].as_array().unwrap() {
            let out = merge_final_text(&c["request"]);
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
            "{} differences: {}",
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
pub fn merge(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    merge_with_work(v, &mut Work(0))
}
pub(super) fn merge_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let mut ls = array(&v["lines"], 4096)?.to_vec();
    let p = size(&v["page_size"])?;
    let tables = optional(v, "table_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let mut aliases = vec![None; ls.len()];
    baseline::merge(&mut ls, &mut aliases, p, &tables, w)?;
    let cluster = clusters::merge_page(
        &json!({"lines":ls,"page_size":p,"table_bboxes":v["table_bboxes"]}),
        w,
    )?;
    ls = array(&cluster["lines"], 4096)?.to_vec();
    aliases = vec![None; ls.len()];
    baseline::merge(&mut ls, &mut aliases, p, &tables, w)?;
    Ok(json!(ls))
}
pub(super) fn finish_owned(
    mut out: Value,
    options: &Value,
    detached: &BTreeSet<i64>,
    w: &mut Work,
) -> Result<Value, Error> {
    let p = size(&out["source"]["page_size"])?;
    let tables = optional(&out, "table_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    array(&out["source"]["lines"], 4096)?;
    let Value::Array(mut source) = out["source"]["lines"].take() else {
        return Err(invalid());
    };
    let ids = source
        .iter()
        .map(|l| integer(l, "source_index", 0))
        .collect::<Result<BTreeSet<_>, _>>()?;
    array(&out["remaining_lines"], 4096)?;
    let Value::Array(mut ls) = out["remaining_lines"].take() else {
        return Err(invalid());
    };
    let mut aliases = ls
        .iter()
        .map(|l| {
            let id = integer(l, "source_index", 0)?;
            Ok((ids.contains(&id) && !detached.contains(&id)).then_some(id))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let updates = baseline::merge(&mut ls, &mut aliases, p, &tables, w)?;
    cache_sources(&updates, &mut source)?;
    let mut cluster = clusters::merge_page_parts(&ls, &json!(p), out.get("table_bboxes"), w)?;
    let groups = array(&cluster["groups"], 4096)?;
    aliases = groups
        .iter()
        .map(|g| {
            let a = array(g, 4096)?;
            if a.len() == 1 {
                Ok(aliases[a[0].as_u64().ok_or_else(invalid)? as usize])
            } else {
                Ok(None)
            }
        })
        .collect::<Result<Vec<_>, Error>>()?;
    array(&cluster["lines"], 4096)?;
    let Value::Array(clustered) = cluster["lines"].take() else {
        return Err(invalid());
    };
    ls = clustered;
    let updates = baseline::merge(&mut ls, &mut aliases, p, &tables, w)?;
    cache_sources(&updates, &mut source)?;
    let mut ordinary = Vec::new();
    let mut ordinary_alias = Vec::new();
    let mut candidates = Vec::new();
    let mut candidate_alias = Vec::new();
    for (l, a) in ls.into_iter().zip(aliases) {
        if l["formula_candidate_only"] == true {
            candidates.push(l);
            candidate_alias.push(a)
        } else {
            ordinary.push(l);
            ordinary_alias.push(a)
        }
    }
    out["script_lines"] = super::inline_output::prepare_scripts_parts(
        &ordinary,
        &json!(p),
        &options["tight_bboxes"],
        &options["origins"],
        &out["source"]["chars"],
        &out["source"]["drawing_lines"],
        w,
    )?;
    compact::lines(&mut ordinary, p, w)?;
    compact::lines(&mut candidates, p, w)?;
    for (l, a) in ordinary
        .iter()
        .zip(&ordinary_alias)
        .chain(candidates.iter().zip(&candidate_alias))
    {
        if let Some(id) = a {
            if let Some(s) = source
                .iter_mut()
                .find(|s| s["source_index"].as_i64() == Some(*id))
            {
                *s = l.clone();
            }
        }
    }
    out["remaining_lines"] = Value::Array(ordinary);
    out["formula_candidate_lines"] = Value::Array(candidates);
    out["formula_ink_bboxes"] = compact::unmapped(optional(&out["source"], "chars", 20000)?, w)?;
    let mut profile = Vec::new();
    if ordinary_alias.len() <= 6 && !tables.is_empty() {
        for l in &source {
            w.charge(tables.len() + 1)?;
            let b = bbox(&l["bbox"])?;
            if tables
                .iter()
                .any(|t| super::path_graphics::overlap_first(&b, t) >= 0.9)
            {
                let mut l = l.clone();
                l["chars"] = json!([]);
                l["inline_math_regions"] = json!([]);
                profile.push(l)
            }
        }
    }
    out["table_body_profile_lines"] = json!(profile);
    out["source"]["lines"] = Value::Array(source);
    let bullet = optional(&out["source"], "path_infos", 5000)?
        .iter()
        .filter(|p| {
            p["fill_visible"] == true
                && p["stroke_visible"] != true
                && p["rectangle_bboxes"]
                    .as_array()
                    .is_none_or(|a| a.is_empty())
                && p["segment_count"]
                    .as_i64()
                    .is_some_and(|n| (8..=100).contains(&n))
        })
        .cloned()
        .collect::<Vec<_>>();
    out["bullet_paths"] = json!(bullet);
    let Value::Array(remaining) = out["remaining_lines"].take() else {
        return Err(invalid());
    };
    let mut late = super::native_footnotes::auxiliary_lines_owned(
        remaining,
        &json!({"page_size":p,"visual_bboxes":optional(&out,"fixed_blocks",4000)?.iter().filter(|b|b["type"]=="image").map(|b|b["bbox"].clone()).collect::<Vec<_>>(),"table_bboxes":out["table_bboxes"],"drawing_lines":out["source"]["drawing_lines"]}),
        w,
    )?;
    out["remaining_lines"] = late["lines"].take();
    let mut groups = optional(&out, "page_footnote_groups", 4096)?.to_vec();
    groups.extend_from_slice(array(&late["page_footnote_groups"], 4096)?);
    out["page_footnote_groups"] = json!(groups);
    array(&out["remaining_lines"], 4096)?;
    array(&out["source"]["lines"], 4096)?;
    let Value::Array(mut source) = out["source"]["lines"].take() else {
        return Err(invalid());
    };
    let lines = array(&out["remaining_lines"], 4096)?;
    for (l, a) in lines.iter().zip(ordinary_alias) {
        if let Some(id) = a {
            if let Some(s) = source
                .iter_mut()
                .find(|s| s["source_index"].as_i64() == Some(id))
            {
                *s = l.clone();
            }
        }
    }
    out["source"]["lines"] = Value::Array(source);
    out["next_table_grid"] =
        super::closed_rule_pipeline::trailing_table_grid_sources(&out["source"], &out, w)?;
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        Err(limit())
    } else {
        Ok(out)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owned_line_stages_preserve_python_reference_outputs() {
        use std::io::Read;
        let mut bytes = vec![];
        flate2::read::GzDecoder::new(
            &include_bytes!("../../../../../../tests/fixtures/native_owned_line_reference.json.gz")
                [..],
        )
        .read_to_end(&mut bytes)
        .unwrap();
        let fixture: Value = serde_json::from_slice(&bytes).unwrap();
        for stage in ["roles", "notes", "clusters", "scripts"] {
            for case in fixture[stage].as_array().unwrap() {
                let q = &case["request"];
                let lines = array(&q["lines"], 4096).unwrap().to_vec();
                let mut work = Work(0);
                let result = match stage {
                    "roles" => {
                        super::super::native_text_roles::classify_lines_owned(lines, q, &mut work)
                    }
                    "notes" => {
                        super::super::native_footnotes::classify_lines_owned(lines, q, &mut work)
                    }
                    "clusters" => clusters::merge_page_parts(
                        &lines,
                        &q["page_size"],
                        q.get("table_bboxes"),
                        &mut work,
                    )
                    .map(|mut value| value["lines"].take()),
                    _ => super::super::inline_output::prepare_scripts_parts(
                        &lines,
                        &q["page_size"],
                        &q["tight_bboxes"],
                        &q["origins"],
                        &q["all_chars"],
                        &q["drawing_lines"],
                        &mut work,
                    ),
                };
                if case.get("error").is_some() {
                    assert!(result.is_err(), "{stage} {}", case["label"]);
                    continue;
                }
                let actual = result.unwrap_or_else(|e| panic!("{stage} {}: {e}", case["label"]));
                if let Some(diff) =
                    super::super::native_document::tests::diff(&actual, &case["expected"], "")
                {
                    panic!("{stage} {}: {diff}", case["label"]);
                }
            }
        }
    }
    #[test]
    fn borrowed_script_page_context_preserves_python_reference() {
        use std::io::Read;
        let mut bytes = Vec::new();
        flate2::read::GzDecoder::new(
            &include_bytes!("../../../../../../tests/fixtures/native_owned_line_reference.json.gz")
                [..],
        )
        .read_to_end(&mut bytes)
        .unwrap();
        let fixture: Value = serde_json::from_slice(&bytes).unwrap();
        let _scope = crate::legacy_page::budget::host_page();
        for case in fixture["scripts"].as_array().unwrap() {
            let q = &case["request"];
            let actual = super::super::inline_output::prepare_scripts_parts(
                q["lines"].as_array().unwrap(),
                &q["page_size"],
                &q["tight_bboxes"],
                &q["origins"],
                &q["all_chars"],
                &q["drawing_lines"],
                &mut Work(0),
            )
            .unwrap();
            assert_eq!(actual, case["expected"], "{}", case["label"]);
        }
    }
    #[test]
    fn borrowed_reference_rows_keep_python_roles_footnotes_and_work() {
        use std::io::Read;
        let mut bytes = Vec::new();
        flate2::read::GzDecoder::new(
            &include_bytes!("../../../../../../tests/fixtures/native_owned_line_reference.json.gz")
                [..],
        )
        .read_to_end(&mut bytes)
        .unwrap();
        let fixture: Value = serde_json::from_slice(&bytes).unwrap();
        for host in [false, true] {
            let _scope = host.then(crate::legacy_page::budget::host_page);
            for stage in ["roles", "notes"] {
                for case in fixture[stage].as_array().unwrap() {
                    let q = &case["request"];
                    let lines = array(&q["lines"], 4096).unwrap().to_vec();
                    let references = optional(q, "reference_lines", 4096).unwrap();
                    let mut options = q.clone();
                    options.as_object_mut().unwrap().remove("reference_lines");
                    let (mut old_work, mut work) = (Work(0), Work(0));
                    let (old, got) = if stage == "roles" {
                        (
                            super::super::native_text_roles::classify_lines_owned(
                                lines.clone(),
                                q,
                                &mut old_work,
                            ),
                            super::super::native_text_roles::classify_lines_with_reference(
                                lines,
                                &options,
                                Some(references),
                                &mut work,
                            ),
                        )
                    } else {
                        (
                            super::super::native_footnotes::classify_lines_owned(
                                lines.clone(),
                                q,
                                &mut old_work,
                            ),
                            super::super::native_footnotes::classify_lines_with_reference(
                                lines,
                                &options,
                                Some(references),
                                &mut work,
                            ),
                        )
                    };
                    assert_eq!(work.0, old_work.0, "{stage} {}", case["label"]);
                    match (old, got) {
                        (Ok(a), Ok(b)) => {
                            assert_eq!(a, b, "{stage} {}", case["label"]);
                            assert_eq!(b, case["expected"], "{stage} {}", case["label"]);
                        }
                        (Err(a), Err(b)) => {
                            assert_eq!(a.code, b.code);
                            assert_eq!(a.message, b.message);
                        }
                        other => panic!("{stage} {}: {other:?}", case["label"]),
                    }
                }
            }
        }
    }
    #[test]
    fn original_compact_font_errors() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-prepared-font-contracts.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert!(
                super::super::native_geometry::finish_page(&c["request"]).is_err(),
                "{}",
                c["label"]
            );
        }
    }
    #[test]
    fn old_page_merging() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-page-merging.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let out = merge(&c["request"]).unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = super::super::native_document::tests::diff(&out, &c["expected"], "") {
                panic!("{}: {d}", c["label"])
            }
        }
    }
    #[test]
    fn old_complete_page_preparation() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-prepared-pages.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let out = super::super::native_document::prepare_pages(&c["request"]);
            if !c["error"].is_null() {
                assert!(out.is_err(), "{}", c["label"]);
                continue;
            }
            let out = out.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = super::super::native_document::tests::diff(&out, &c["expected"], "") {
                panic!("{}: {d}", c["label"])
            }
        }
    }
    #[test]
    fn prepared_source_identity() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-prepared-identity.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let out = super::super::native_geometry::finish_page(&c["request"]);
            if !c["error"].is_null() {
                assert!(out.is_err(), "{}", c["label"]);
                continue;
            }
            let out = out.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = super::super::native_document::tests::diff(&out, &c["expected"], "") {
                panic!("{}: {d}", c["label"])
            }
        }
    }
}
