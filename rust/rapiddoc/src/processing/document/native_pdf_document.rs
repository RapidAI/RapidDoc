//! File-level original native PDF collection, then complete document/inline/media processing.
use super::path_graphics::array;
use crate::legacy_page::{invalid, limit, Work};
use rapiddoc_core::{Error, Limits};
use serde_json::{json, Value};
use std::path::Path;
fn value(v: impl serde::Serialize) -> Result<Value, Error> {
    serde_json::to_value(v).map_err(|_| invalid())
}
fn pairs(g: &mut Value, key: &str, name: &str) -> Result<Value, Error> {
    g[key].as_object().ok_or_else(invalid)?;
    let Value::Object(map) = g[key].take() else {
        unreachable!()
    };
    let mut a = map
        .into_iter()
        .map(|(k, v)| Ok((k.parse::<usize>().map_err(|_| invalid())?, v)))
        .collect::<Result<Vec<_>, Error>>()?;
    a.sort_by_key(|(i, _)| *i);
    Ok(Value::Array(
        a.into_iter()
            .map(|(i, v)| {
                let mut x = value_object!({"char_idx":i});
                x[name] = v;
                x
            })
            .collect(),
    ))
}
pub fn collect(
    library: &Path,
    bytes: &[u8],
    password: Option<&str>,
    max_pixels: usize,
) -> Result<Value, Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    collect_range(
        library,
        bytes,
        password,
        max_pixels,
        rapiddoc_core::PageRange::default(),
        true,
    )
}
fn collect_range(
    library: &Path,
    bytes: &[u8],
    password: Option<&str>,
    max_pixels: usize,
    range: rapiddoc_core::PageRange,
    diagnostic_chars: bool,
) -> Result<Value, Error> {
    rapiddoc_pdf::parsed_document::with_scope(bytes, || {
        collect_range_impl(
            library,
            bytes,
            password,
            max_pixels,
            range,
            diagnostic_chars,
        )
    })
}
fn collect_range_impl(
    library: &Path,
    bytes: &[u8],
    password: Option<&str>,
    max_pixels: usize,
    range: rapiddoc_core::PageRange,
    diagnostic_chars: bool,
) -> Result<Value, Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    if max_pixels == 0 {
        return Err(invalid());
    }
    let _budget = crate::legacy_page::budget::document(1);
    let limits = Limits::pdf_host();
    let source_count = rapiddoc_pdf::page_count(library, bytes, password, &limits)?;
    let range = range.resolve(source_count)?;
    let count = range.len();
    if count > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let mut pages = vec![];
    let mut image_pages = vec![];
    let mut sizes = vec![];
    for (local_index, i) in (range.start..range.end).enumerate() {
        let mut page_profile = crate::processing::NativeStageProfile::new();
        let w = &mut Work(0);
        let mut g = value(rapiddoc_pdf::character_primitives::read_visible_geometry(
            library, bytes, i, password, true, &limits,
        )?)?;
        let images = value(rapiddoc_pdf::image_infos::read(
            library, bytes, i, password, &limits,
        )?)?;
        let mut vectors = value(rapiddoc_pdf::legacy_vectors::read(
            library, bytes, i, password, &limits,
        )?)?;
        let mut forms = value(rapiddoc_pdf::form_structure::read(
            library, bytes, i, password, &limits,
        )?)?;
        let signatures = rapiddoc_pdf::signatures::read(library, bytes, i, password, &limits)?;
        let annotations = value(rapiddoc_pdf::link_annotations::read(
            library, bytes, i, password, &limits,
        )?)?;
        page_profile.mark("collect_read_pdf_facts");
        let ls = super::native_lines::build_chars(
            &g["chars"],
            &value_object!({"page_size":g["page_size"],"page_rotation":g["rotation"]}),
        )?;
        let mut line_request = value_object!({"page_size":g["page_size"],"image_infos":images["image_infos"],"include_local_lines":true});
        line_request["drawing_lines"] = vectors["drawing_lines"].take();
        line_request["lines"] = ls;
        let mut local = super::source_lines::prepare_owned(line_request, w)?;
        page_profile.mark("collect_prepare_lines");
        let spacing =
            super::inline_output::prepare_spacing_lines(&local["local_lines"], &json!({}), w)?;
        let mut evidence = super::inline_evidence::detect_parts(
            &local["lines"],
            Some(&local["drawing_lines"]),
            Some(&annotations),
            w,
        )?;
        array(&evidence["style_lines"], 4096)?;
        let Value::Array(mut styles) = evidence["style_lines"].take() else {
            unreachable!()
        };
        page_profile.mark("collect_inline_evidence");
        let plan = super::native_glyph_weight::plan_parts(&local["lines"], &g["tight_bboxes"], w)?;
        if !array(&plan["candidates"], 4096)?.is_empty() {
            let sz = super::page_containers::size(&g["page_size"])?;
            let scale = 3f64.min(3500. / sz[0].max(sz[1]));
            let image = rapiddoc_pdf::render::render_page_rgb(
                library,
                bytes,
                i,
                scale * 72.,
                max_pixels,
                password,
                &limits,
            )?;
            styles.extend_from_slice(array(
                &super::native_glyph_weight::detect_parts(
                    &local["lines"],
                    &g["tight_bboxes"],
                    &json!({"scale":scale}),
                    Some(&image.png),
                    w,
                )?,
                4096,
            )?);
        }
        let mut source = value_object!({"page_size":g["page_size"],"page_index":local_index,"signature_bboxes":signatures});
        source["form_bboxes"] = forms["form_bboxes"].take();
        source["form_infos"] = forms["form_infos"].take();
        source["path_infos"] = vectors["path_infos"].take();
        source["lines"] = local["lines"].take();
        source["drawing_lines"] = local["drawing_lines"].take();
        source["image_bboxes"] = json!([]);
        page_profile.mark("collect_glyph_weight");
        let mut geometry = json!({});
        geometry["tight_bboxes"] = pairs(&mut g, "tight_bboxes", "bbox")?;
        geometry["loose_bboxes"] = pairs(&mut g, "loose_bboxes", "bbox")?;
        geometry["origins"] = pairs(&mut g, "origins", "origin")?;
        // Geometry planning reads the three maps, never this second flat
        // character snapshot. Keep it only for the public collection diagnostic.
        if diagnostic_chars {
            geometry["chars"] = g["chars"].clone();
        }
        source["chars"] = g["chars"].take();
        // Image-region classification reads these scalar line facts only; it
        // never needs each line's duplicated per-character geometry.
        let image_lines = array(&source["lines"], 4096)?
            .iter()
            .map(|line| {
                let mut facts = serde_json::Map::new();
                for key in [
                    "text",
                    "bbox",
                    "angle",
                    "effective_height",
                    "dominant_font_weight",
                ] {
                    if let Some(value) = line.get(key) {
                        facts.insert(key.into(), value.clone());
                    }
                }
                Value::Object(facts)
            })
            .collect::<Vec<_>>();
        image_pages.push(value_object!({"page_size":g["page_size"],"lines":image_lines,"image_infos":images["image_infos"]}));
        sizes.push(g["page_size"].clone());
        let mut page = json!({});
        page["link_lines"] = evidence["link_lines"].take();
        page["source"] = source;
        page["geometry"] = geometry;
        page["style_lines"] = Value::Array(styles);
        page["spacing_lines"] = spacing;
        pages.push(page);
    }
    let mut image_request = json!({});
    image_request["pages"] = Value::Array(image_pages);
    let regions = super::image_regions::prepare(&image_request)?;
    drop(image_request);
    for (i, p) in pages.iter_mut().enumerate() {
        let w = &mut Work(0);
        p["source"]["image_bboxes"] = regions["image_bboxes"][i].clone();
        let mut graphic = super::strong_graphics::detect_source_owned(p["source"].take(), w)?;
        let q = crate::processing::with_source(
            graphic["source"].take(),
            value_object!({"visual_bboxes":graphic["graphic_bboxes"]}),
        );
        let noise = super::text_noise::prepare_with_work(&q, w)?;
        let indices = array(&noise["requested_char_indices"], 20000)?
            .iter()
            .map(|v| v.as_i64().ok_or_else(invalid))
            .collect::<Result<Vec<_>, _>>()?;
        let paint = if array(&noise["candidates"], 4096)?.is_empty() {
            Value::Null
        } else {
            value(rapiddoc_pdf::text_paint::read(
                library,
                bytes,
                range.start + i,
                &indices,
                password,
                &limits,
            )?)?
        };
        let mut q = q;
        q["paint"] = paint;
        let mut clean = super::text_noise::apply_owned(q, w)?;
        p["source"] =
            super::raster_charts::group_source_owned(clean["source"].take(), w)?["source"].take();
    }
    // json! serializes borrowed inputs and would clone the entire collected
    // window here. Transfer its already-owned JSON pages into the envelope.
    let mut out = value_object!({"source_page_count":count,"page_sizes":sizes});
    out["pages"] = Value::Array(pages);
    if crate::legacy_page::budget::output_exceeds(&out) {
        Err(limit())
    } else {
        Ok(out)
    }
}
pub fn analyze(
    library: &Path,
    bytes: &[u8],
    password: Option<&str>,
    max_pixels: usize,
) -> Result<Value, Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    let collected = collect(library, bytes, password, max_pixels)?;
    let _budget = crate::legacy_page::budget::document(
        collected["source_page_count"]
            .as_u64()
            .ok_or_else(invalid)? as usize,
    );
    let result = super::native_document::prepare_text_blocks(
        &value_object!({"pages":collected["pages"],"materialize_inline":true}),
    )?;
    let pages = super::table_block::images::pdf(
        &value_object!({"pages":result["finalized_pages"]}),
        library,
        bytes,
        password,
        max_pixels,
    )?;
    Ok(
        value_object!({"pages":pages,"source_page_count":collected["source_page_count"],"page_sizes":collected["page_sizes"]}),
    )
}
pub fn predict(
    library: &Path,
    bytes: &[u8],
    password: Option<&str>,
    max_pixels: usize,
) -> Result<Value, Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    let mut out = analyze(library, bytes, password, max_pixels)?;
    super::native_model_normalization::normalize_with_work(&mut out["pages"], &mut Work(0))?;
    Ok(out)
}
/// Public API slicing happens before document-wide layout statistics. Keep the
/// selected document profile while reading original page handles and images.
pub fn predict_range(
    library: &Path,
    bytes: &[u8],
    password: Option<&str>,
    max_pixels: usize,
    range: rapiddoc_core::PageRange,
) -> Result<Value, Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    let start = range.start;
    let mut profile = crate::processing::NativeStageProfile::new();
    let mut collected = collect_range(library, bytes, password, max_pixels, range, false)?;
    profile.mark("collect_native_pages");
    let count = collected["source_page_count"]
        .as_u64()
        .ok_or_else(invalid)? as usize;
    let _budget = crate::legacy_page::budget::document(count);
    let mut request = value_object!({"materialize_inline":true});
    request["pages"] = collected["pages"].take();
    let page_sizes = collected["page_sizes"].take();
    drop(collected);
    let finalized = super::native_document::finalize_owned(request)?;
    profile.mark("finalize_native_document");
    let mut image_request = value_object!({"source_page_start":start});
    image_request["pages"] = finalized;
    let pages =
        super::table_block::images::pdf(&image_request, library, bytes, password, max_pixels)?;
    drop(image_request);
    profile.mark("materialize_native_assets");
    let mut out = value_object!({"source_page_count":count,"page_sizes":page_sizes});
    out["pages"] = pages;
    super::native_model_normalization::normalize_with_work(&mut out["pages"], &mut Work(0))?;
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn borrowed_collection_stages_match_python_and_work() {
        let f: Value =
            serde_json::from_reader(flate2::read::GzDecoder::new(
                &include_bytes!(
                    "../../../../../tests/fixtures/native_collection_reference.json.gz"
                )[..],
            ))
            .unwrap();
        for host in [false, true] {
            let _scope = host.then(crate::legacy_page::budget::host_page);
            for stage in ["lines", "spacing", "evidence", "glyphs"] {
                for c in f[stage].as_array().unwrap() {
                    let q = &c["request"];
                    let (mut a, mut b) = (Work(0), Work(0));
                    let (old, new) = match stage {
                        "lines" => (
                            super::super::native_lines::build(q),
                            super::super::native_lines::build_chars(&q["chars"], q),
                        ),
                        "spacing" => (
                            super::super::inline_output::prepare_spacing_with_work(q, &mut a),
                            super::super::inline_output::prepare_spacing_lines(
                                &q["lines"],
                                q,
                                &mut b,
                            ),
                        ),
                        "evidence" => (
                            super::super::inline_evidence::detect_with_work(q, &mut a),
                            super::super::inline_evidence::detect_parts(
                                &q["lines"],
                                q.get("drawing_lines"),
                                q.get("annotations"),
                                &mut b,
                            ),
                        ),
                        _ => {
                            let p = super::super::native_glyph_weight::plan_with_work(q, &mut a)
                                .unwrap();
                            let n = super::super::native_glyph_weight::plan_parts(
                                &q["lines"],
                                &q["tight_bboxes"],
                                &mut b,
                            )
                            .unwrap();
                            assert_eq!(p, n);
                            assert!(
                                super::super::native_document::tests::diff(&n, &c["plan"], "")
                                    .is_none(),
                                "glyph plan {}",
                                c["label"]
                            );
                            let png = c["png_bytes"].as_array().map(|a| {
                                a.iter()
                                    .map(|v| u8::try_from(v.as_u64().unwrap()).unwrap())
                                    .collect::<Vec<_>>()
                            });
                            (
                                super::super::native_glyph_weight::detect_with_work(
                                    q,
                                    png.as_deref(),
                                    &mut a,
                                ),
                                super::super::native_glyph_weight::detect_parts(
                                    &q["lines"],
                                    &q["tight_bboxes"],
                                    q,
                                    png.as_deref(),
                                    &mut b,
                                ),
                            )
                        }
                    };
                    assert_eq!(a.0, b.0, "{stage} {}", c["label"]);
                    let old = old.unwrap_or_else(|e| panic!("{stage} {}: {e}", c["label"]));
                    let new = new.unwrap_or_else(|e| panic!("{stage} {}: {e}", c["label"]));
                    assert_eq!(new, old, "{stage} {}", c["label"]);
                    if let Some(d) =
                        super::super::native_document::tests::diff(&new, &c["expected"], "")
                    {
                        panic!("{stage} {}: {d}", c["label"]);
                    }
                }
            }
        }
    }
    #[test]
    fn borrowed_collection_input_errors_match() {
        for q in [
            json!({}),
            json!({"lines":null,"chars":null}),
            json!({"lines":[{}],"chars":[{}],"page_size":[0,0]}),
            json!({"lines":[],"chars":[],"page_size":[100,100],"annotations":null,"drawing_lines":null}),
        ] {
            let pairs = [
                (
                    super::super::native_lines::build(&q),
                    super::super::native_lines::build_chars(&q["chars"], &q),
                ),
                (
                    super::super::inline_output::prepare_spacing_with_work(&q, &mut Work(0)),
                    super::super::inline_output::prepare_spacing_lines(
                        &q["lines"],
                        &q,
                        &mut Work(0),
                    ),
                ),
                (
                    super::super::inline_evidence::detect_with_work(&q, &mut Work(0)),
                    super::super::inline_evidence::detect_parts(
                        &q["lines"],
                        q.get("drawing_lines"),
                        q.get("annotations"),
                        &mut Work(0),
                    ),
                ),
                (
                    super::super::native_glyph_weight::plan_with_work(&q, &mut Work(0)),
                    super::super::native_glyph_weight::plan_parts(
                        &q["lines"],
                        &q["tight_bboxes"],
                        &mut Work(0),
                    ),
                ),
            ];
            for (old, new) in pairs {
                match (old, new) {
                    (Ok(a), Ok(b)) => assert_eq!(a, b),
                    (Err(a), Err(b)) => {
                        assert_eq!(a.code, b.code);
                        assert_eq!(a.message.split('（').next(), b.message.split('（').next());
                    }
                    other => panic!("{q}: {other:?}"),
                }
            }
        }
    }
    #[test]
    #[ignore = "Failure-input diagnostic, run only when explicitly requested"]
    fn native_pdf_table_diagnostic() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let f: Value = serde_json::from_slice(
            &std::fs::read(root.join("build/native-pdf-table-failure.json")).unwrap(),
        )
        .unwrap();
        let _budget = crate::legacy_page::budget::document(9);
        super::super::page_containers::prepare_with_work(&f["request"], &mut Work(0)).unwrap();
    }
    #[test]
    fn original_native_pdf_acceptance() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
        let f: Value = serde_json::from_slice(
            &std::fs::read(root.join("build/migration/native-pdf-acceptance.json")).unwrap(),
        )
        .unwrap();
        let uri = regex::Regex::new(r"data:image/png;base64,([A-Za-z0-9+/=]+)").unwrap();
        for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
            if let Ok(label) = std::env::var("RAPIDDOC_PDF_ACCEPTANCE_CASE") {
                if c["label"].as_str() != Some(label.as_str()) {
                    continue;
                }
            }
            let bytes = std::fs::read(root.join(c["pdf"].as_str().unwrap())).unwrap();
            let got = predict(&library, &bytes, None, 16_000_000)
                .unwrap_or_else(|e| panic!("{} predict: {e}", c["label"]));
            std::fs::write(
                root.join(format!("build/native-pdf-acceptance-actual-{i}.json")),
                serde_json::to_vec(&got).unwrap(),
            )
            .unwrap();
            let mut pages = got["pages"].clone();
            let mut image_index = 0;
            for page in pages.as_array_mut().unwrap() {
                for block in page.as_array_mut().unwrap() {
                    if let Some(content) = block["content"].as_str() {
                        let content = uri
                            .replace_all(content, |_: &regex::Captures| {
                                let token = format!("@PNG{image_index}@");
                                image_index += 1;
                                token
                            })
                            .into_owned();
                        block["content"] = Value::String(content);
                    }
                }
            }
            assert_eq!(
                image_index,
                c["expected"]["images"].as_array().unwrap().len()
            );
            if let Some(d) =
                super::super::native_document::tests::diff(&pages, &c["expected"]["pages"], "")
            {
                panic!("{} final: {d}", c["label"])
            }
        }
    }
    #[test]
    fn original_python_native_pdf_files() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
        let f: Value = serde_json::from_slice(
            &std::fs::read(root.join("build/migration/native-pdf-document.json")).unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            let bytes = std::fs::read(root.join(c["pdf"].as_str().unwrap())).unwrap();
            let got = collect(&library, &bytes, None, 16_000_000)
                .unwrap_or_else(|e| panic!("{} collect: {e}", c["label"]));
            if let Some(d) = super::super::native_document::tests::diff(&got, &c["collected"], "") {
                std::fs::write(
                    root.join("build/native-pdf-document-actual.json"),
                    serde_json::to_vec(&got).unwrap(),
                )
                .unwrap();
                panic!("{} collected: {d}", c["label"])
            }
            let got = analyze(&library, &bytes, None, 16_000_000)
                .unwrap_or_else(|e| panic!("{} analyze: {e}", c["label"]));
            if let Some(d) = super::super::native_document::tests::diff(&got, &c["expected"], "") {
                panic!("{} final: {d}", c["label"])
            }
        }
    }
}
