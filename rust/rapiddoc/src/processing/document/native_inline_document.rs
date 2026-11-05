//! Original _materialize_document_inline, preserving strict zip and spacing-before-spans.
use super::{inline_output, page_containers::optional, path_graphics::array};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
pub fn materialize(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    materialize_with_work(v, &mut Work(0))
}
pub(super) fn materialize_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let mut pages = array(&v["finalized_pages"], 1000)?.to_vec();
    let prepared = array(&v["prepared_pages"], 1000)?;
    let styles = array(&v["page_style_lines"], 1000)?;
    let links = array(&v["page_link_lines"], 1000)?;
    let sizes = array(&v["page_sizes"], 1000)?;
    if [prepared.len(), styles.len(), links.len(), sizes.len()]
        .into_iter()
        .any(|n| n != pages.len())
    {
        return Err(invalid());
    }
    let spacing = optional(v, "page_spacing_lines", 1000)?;
    let mut diagnostics = v
        .get("script_diagnostics")
        .filter(|v| !v.is_null())
        .map(|v| array(v, 1000).map(|a| a.to_vec()))
        .transpose()?;
    for (i, page) in pages.iter_mut().enumerate() {
        if !spacing.is_empty() {
            *page = inline_output::apply_spacing_with_work(
                &json!({"blocks":page,"spacing_lines":spacing.get(i).ok_or_else(invalid)?,"page_size":sizes[i]}),
                w,
            )?;
        }
        let q = json!({"blocks":page,"page_size":sizes[i],"style_lines":styles[i],"link_lines":links[i],"script_lines":prepared[i].get("script_lines").cloned().unwrap_or(json!([]))});
        if let Some(diagnostics) = diagnostics.as_mut() {
            let item = diagnostics.get_mut(i).ok_or_else(invalid)?;
            item.as_object_mut().ok_or_else(invalid)?;
            let out = inline_output::trace_with_work(&q, w)?;
            *page = out["blocks"].clone();
            item["materialized_ranges"] = out["materialized_ranges"].clone();
        } else {
            *page = inline_output::apply_with_work(&q, w)?;
        }
    }
    let mut out = json!({"pages":pages});
    if let Some(diagnostics) = diagnostics {
        out["script_diagnostics"] = json!(diagnostics)
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
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_python_document_media_chain() {
        for name in [
            "native-document-media-chain",
            "native-document-media-chain-positive-canonical-source",
        ] {
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../build/migration/{name}.json"));
            let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
            for c in f["cases"].as_array().unwrap() {
                let got = super::super::native_document::prepare_text_blocks(&c["request"])
                    .unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
                if let Some(d) =
                    super::super::native_document::tests::diff(&got, &c["expected"], "")
                {
                    panic!("{}: {d}", c["label"])
                }
            }
        }
    }
    #[test]
    fn original_python_document_inline() {
        for (name, chain) in [
            ("native-document-inline", false),
            ("native-final-document-inline-chain", true),
        ] {
            let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../build/migration/{name}.json"));
            let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
            for c in f["cases"].as_array().unwrap() {
                let got = if chain {
                    super::super::native_document::prepare_text_blocks(&c["request"])
                } else {
                    materialize(&c["request"])
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
}
