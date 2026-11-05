//! 正式页准备的表格事务链；上游非表格认领证据显式传入，不注入候选或恢复结论。
use super::{
    closed_rule_pipeline as detect,
    path_graphics::{area, array, bbox, integer, num, overlap_first, overlap_smaller, Box4},
    strong_graphics, table_block,
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{python_decimal, text_boundary::python_space};
use serde_json::{json, Map, Value};
use std::collections::BTreeSet;
fn optional<'a>(v: &'a Value, k: &str, max: usize) -> Result<&'a [Value], Error> {
    v.get(k).map_or(Ok(&[]), |v| array(v, max))
}
fn stage(name: &str, mut error: Error) -> Error {
    error.message = format!("Table page {name}: {}", error.message);
    error
}
fn ids(v: &Value) -> Result<BTreeSet<i64>, Error> {
    array(v, 20000)?
        .iter()
        .map(|v| v.as_i64().ok_or_else(invalid))
        .collect()
}
fn median(mut a: Vec<f64>) -> f64 {
    a.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = a.len();
    if n % 2 == 1 {
        a[n / 2]
    } else {
        (a[n / 2 - 1] + a[n / 2]) / 2.
    }
}
fn numeric(text: &str) -> bool {
    let c: Vec<_> = text.trim_matches(python_space).chars().collect();
    let mut i = usize::from(c.first().is_some_and(|c| "+-".contains(*c)));
    let start = i;
    while c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
        i += 1;
    }
    if start == i {
        return false;
    }
    while c.get(i).is_some_and(|c| ",.".contains(*c)) {
        i += 1;
        let start = i;
        while c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
            i += 1;
        }
        if start == i {
            return false;
        }
    }
    if c.get(i) == Some(&'%') {
        i += 1;
    }
    i == c.len()
}
pub(super) fn numeric_grid(lines: &[Value], b: &Box4, w: &mut Work) -> Result<bool, Error> {
    let mut values = Vec::new();
    let mut heights = Vec::new();
    w.charge(lines.len() + 1)?;
    for l in lines {
        let lb = bbox(&l["bbox"])?;
        if overlap_first(&lb, b) < 0.95 {
            continue;
        }
        let text = l["text"].as_str().ok_or_else(invalid)?;
        w.charge(text.len() + 1)?;
        if !numeric(text) {
            continue;
        }
        let effective = l.get("effective_height").map_or(Ok(0.), num)?;
        let em = l.get("em_height").map_or(Ok(0.), num)?;
        let repaired = l
            .get("style_scale_repaired")
            .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))?;
        heights.push(
            (if repaired && em > 0. {
                em
            } else if effective != 0. {
                effective
            } else {
                lb.n[3] - lb.n[1]
            })
            .max(0.1),
        );
        values.push(lb);
    }
    if values.len() < 12 {
        return Ok(false);
    }
    let em = median(heights);
    w.charge(values.len() * 16 + 1)?;
    values.sort_by(|a, b| {
        ((a.n[1] + a.n[3]) / 2.)
            .partial_cmp(&((b.n[1] + b.n[3]) / 2.))
            .unwrap()
            .then(a.n[0].partial_cmp(&b.n[0]).unwrap())
    });
    let mut rows: Vec<Vec<Box4>> = Vec::new();
    for b in values {
        let cy = (b.n[1] + b.n[3]) / 2.;
        if rows
            .last()
            .is_none_or(|r| (cy - (r[0].n[1] + r[0].n[3]) / 2.).abs() > 0.4 * em)
        {
            rows.push(Vec::new());
        }
        rows.last_mut().unwrap().push(b);
    }
    let dense: Vec<_> = rows.iter().filter(|r| r.len() >= 3).collect();
    if dense.len() < 4 {
        return Ok(false);
    }
    let reference = dense[0];
    let mut count = 0;
    for row in &dense {
        w.charge(reference.len() * row.len() + 1)?;
        if reference
            .iter()
            .filter(|l| row.iter().any(|other| (l.n[0] - other.n[0]).abs() <= em))
            .count()
            >= 3
        {
            count += 1;
        }
    }
    Ok(count >= 4)
}
fn annotation_maps(cs: &mut [Value], w: &mut Work) -> Result<(), Error> {
    for c in cs {
        for a in c["annotations"].as_array_mut().ok_or_else(invalid)? {
            let pairs = array(&a["line_bboxes"], 20000)?;
            w.charge(pairs.len() + 1)?;
            let mut boxes = Map::new();
            for pair in pairs {
                let pair = array(pair, 2)?;
                if pair.len() != 2 {
                    return Err(invalid());
                }
                let id = pair[0].as_i64().ok_or_else(invalid)?;
                bbox(&pair[1])?;
                boxes.insert(id.to_string(), pair[1].clone());
            }
            a["line_bboxes"] = Value::Object(boxes);
        }
    }
    Ok(())
}
pub fn prepare(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_with_work(v, &mut Work(0))
}
pub(crate) fn prepare_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    prepare_impl(v, w, false, None)
}
pub fn prepare_code_tables(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_impl(v, &mut Work(0), true, None)
}
fn code_barriers(base: &[Value], source: &Value, tables: &[Value]) -> Result<Vec<Value>, Error> {
    let mut out = base.to_vec();
    out.extend_from_slice(array(&source["image_bboxes"], 5000)?);
    out.extend_from_slice(array(&source["signature_bboxes"], 5000)?);
    out.extend(tables.iter().map(|b| b["bbox"].clone()));
    if out.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    Ok(out)
}
fn prepare_impl(
    v: &Value,
    w: &mut Work,
    code: bool,
    containers: Option<(&[Value], &[Value], &[Value])>,
) -> Result<Value, Error> {
    prepare_impl_owned(v.clone(), w, code, containers)
}
pub(crate) fn prepare_containers_owned(
    v: Value,
    w: &mut Work,
    forms: &[Value],
    indices: &[Value],
    captions: &[Value],
) -> Result<Value, Error> {
    prepare_impl_owned(v, w, true, Some((forms, indices, captions)))
}
fn prepare_impl_owned(
    mut v: Value,
    w: &mut Work,
    code: bool,
    containers: Option<(&[Value], &[Value], &[Value])>,
) -> Result<Value, Error> {
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
    array(&source["lines"], 4096)?;
    // Retain the original rows for the transaction; do not clone them again
    // into the analysis page before replacing its lines with selected rows.
    let Value::Array(lines) = source["lines"].take() else {
        unreachable!()
    };
    let mut unique = BTreeSet::new();
    for l in &lines {
        if !unique.insert(l["source_index"].as_i64().ok_or_else(invalid)?) {
            return Err(invalid());
        }
    }
    let prior_blocks = optional(&v, "prior_fixed_blocks", 4000)?;
    let prior_claimed = ids(v.get("prior_claimed_line_indices").unwrap_or(&json!([])))?;
    let mut protected = prior_claimed.clone();
    for l in &lines {
        w.charge(1)?;
        if !l["semantic_type"].is_null() {
            protected.insert(integer(l, "source_index", 0)?);
        }
    }
    let mut selected = Vec::new();
    let mut positions = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if !protected.contains(&integer(l, "source_index", 0)?) {
            positions.push(i);
            selected.push(l.clone());
        }
    }
    let mut analysis = source.clone();
    analysis["lines"] = Value::Array(selected);
    if v["include_precontainer_classifiers"] == true {
        analysis["publication_bboxes"] = json!(lines
            .iter()
            .filter(|l| l["text"]
                .as_str()
                .is_some_and(super::native_vector_formulas::publication))
            .map(|l| l["bbox"].clone())
            .collect::<Vec<_>>());
    }
    profile.mark("table_page_setup");
    let mut graphics =
        strong_graphics::detect_source_owned(analysis, w).map_err(|e| stage("graphics", e))?;
    analysis = graphics["source"].take();
    let mut updated = lines;
    for (&i, l) in positions.iter().zip(array(&analysis["lines"], 4096)?) {
        updated[i] = l.clone();
    }
    source["lines"] = Value::Array(updated);
    let numeric_evidence = detect::detect_supplemental_source(
        &analysis,
        &value_object!({"kind":"aligned_numeric","excluded_bboxes":source["image_bboxes"]}),
        w,
    )
    .map_err(|e| stage("numeric evidence", e))?;
    let numeric_boxes = numeric_evidence["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| bbox(&c["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut strong = Vec::new();
    let caption_blocks = containers.map_or(&[][..], |c| c.2);
    let mut graphic_boxes = array(&graphics["graphic_bboxes"], 5000)?.to_vec();
    graphic_boxes.extend(caption_blocks.iter().map(|b| b["bbox"].clone()));
    if graphic_boxes.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    for raw in &graphic_boxes {
        w.charge(numeric_boxes.len() + 1)?;
        let b = bbox(raw)?;
        if !numeric_boxes
            .iter()
            .any(|c| overlap_smaller(&b, c) >= 0.5 && area(&b) >= 0.25 * area(c))
        {
            strong.push(raw.clone());
        }
    }
    let mut excluded = strong.clone();
    for b in optional(&v, "excluded_bboxes", 5000)? {
        bbox(b)?;
        excluded.push(b.clone());
    }
    if excluded.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    let forms = containers.map_or(&[][..], |c| c.0);
    let mut rule_barriers = if code {
        code_barriers(&excluded, &source, &[])?
    } else {
        Vec::new()
    };
    rule_barriers.extend_from_slice(forms);
    if rule_barriers.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    let rule_code = if code {
        super::code_containers::build_source(&analysis, &json!({"kind":"ruled","excluded_bboxes":rule_barriers ,"claimed_line_indices":prior_claimed}),w).map_err(|e|stage("rule code",e))?
    } else {
        json!({"code_blocks":[],"claimed_line_indices":[]})
    };
    excluded.extend(
        array(&rule_code["code_blocks"], 4000)?
            .iter()
            .map(|b| b["bbox"].clone()),
    );
    if excluded.len() > rapiddoc_core::resources::limit(5000) {
        return Err(limit());
    }
    let separators = optional(&v, "table_header_separator_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let mut rules = Vec::new();
    for r in array(&source["drawing_lines"], 5000)? {
        w.charge(separators.len() + 1)?;
        let b = bbox(&r["bbox"])?;
        if !separators.iter().any(|s| s.n == b.n) {
            rules.push(r.clone());
        }
    }
    // Filtering changes only this one field. Borrow the existing page tree and
    // restore its original drawing list before later stages, including errors.
    let original_drawings = std::mem::replace(&mut analysis["drawing_lines"], Value::Array(rules));
    profile.mark("table_page_graphics_and_code");
    let found =
        detect::detect_complete_source(&analysis, &value_object!({"excluded_bboxes":excluded}), w);
    analysis["drawing_lines"] = original_drawings;
    let mut found = found.map_err(|e| {
        #[cfg(test)]
        {
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
            let _ = std::fs::write(
                root.join("build/native-pdf-table-failure.json"),
                serde_json::to_vec(&value_object!({"request":v,"work":w.0,"error":e.to_string()}))
                    .unwrap(),
            );
        }
        stage("detection", e)
    })?;
    profile.mark("table_page_detection");
    let Value::Array(mut candidates) = found["candidates"].take() else {
        unreachable!()
    };
    if !forms.is_empty() {
        let form_boxes = forms.iter().map(bbox).collect::<Result<Vec<_>, _>>()?;
        let mut accepted = Vec::new();
        for c in candidates {
            w.charge(form_boxes.len() + 1)?;
            let b = bbox(&c["bbox"])?;
            if !form_boxes
                .iter()
                .any(|f| super::page_containers::form_supersedes(f, &b))
            {
                accepted.push(c);
            }
        }
        candidates = accepted;
    }
    annotation_maps(&mut candidates, w)?;
    let mut committed = table_block::commit::materialize_source(
        &mut source,
        &value_object!({"candidates":candidates,"tight_bboxes":v["tight_bboxes"],"origins":v["origins"],"calibrated_char_bboxes":v["calibrated_char_bboxes"]}),
        w,
    ).map_err(|e|stage("commit",e))?;
    profile.mark("table_page_commit");
    let Value::Array(mut tables) = committed["table_blocks"].take() else {
        unreachable!()
    };
    let lines = array(&source["lines"], 4096)?;
    for block in &mut tables {
        block["_native_numeric_grid"] = json!(numeric_grid(lines, &bbox(&block["bbox"])?, w)?);
    }
    let mut claimed = prior_claimed;
    claimed.extend(ids(&committed["claimed_line_indices"])?);
    claimed.extend(ids(&rule_code["claimed_line_indices"])?);
    analysis["lines"] = Value::Array(positions.iter().map(|&i| lines[i].clone()).collect());
    let mut base = strong.clone();
    base.extend_from_slice(optional(&v, "excluded_bboxes", 5000)?);
    let table_boxes = tables
        .iter()
        .map(|b| bbox(&b["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut active_forms = Vec::new();
    for raw in forms {
        w.charge(table_boxes.len() + 1)?;
        let b = bbox(raw)?;
        if !table_boxes.iter().any(|t| overlap_smaller(&b, t) >= 0.5) {
            active_forms.push(raw.clone());
        }
    }
    base.extend_from_slice(&active_forms);
    let filled_code = if code {
        super::code_containers::build_source(&analysis, &json!({"kind":"filled","excluded_bboxes":code_barriers(&base,&source,&tables)?,"claimed_line_indices":claimed}),w).map_err(|e|stage("filled code",e))?
    } else {
        json!({"code_blocks":[],"claimed_line_indices":[]})
    };
    claimed.extend(ids(&filled_code["claimed_line_indices"])?);
    let form_images = if v["include_form_images"] == true {
        let mut out = super::page_containers::materialize_forms_owned(
            crate::processing::with_source(
                analysis,
                json!({"form_bboxes":active_forms,"claimed_line_indices":claimed}),
            ),
            w,
        )?;
        array(&source["lines"], 4096)?;
        let updates = array(&out["source"]["lines"], 4096)?;
        let updated = source["lines"].as_array_mut().ok_or_else(invalid)?;
        for (&i, l) in positions.iter().zip(updates) {
            updated[i] = l.clone();
        }
        analysis = out["source"].take();
        claimed.extend(ids(&out["claimed_line_indices"])?);
        out["form_image_blocks"].take()
    } else {
        json!([])
    };
    let graphic_images = if v["include_graphic_images"] == true {
        let mut barriers = table_boxes.iter().map(|b| json!(b.raw)).collect::<Vec<_>>();
        barriers.extend(active_forms.clone());
        for b in array(&rule_code["code_blocks"], 4000)?
            .iter()
            .chain(array(&filled_code["code_blocks"], 4000)?)
        {
            barriers.push(b["bbox"].clone());
        }
        barriers.extend_from_slice(optional(&v, "excluded_bboxes", 5000)?);
        barriers.extend(caption_blocks.iter().map(|b| b["bbox"].clone()));
        let mut out = super::page_containers::materialize_graphics_owned(
            crate::processing::with_source(
                analysis,
                json!({"excluded_bboxes":barriers,"strong_core_bboxes":strong,"claimed_line_indices":claimed}),
            ),
            w,
        )?;
        analysis = out["source"].take();
        claimed.extend(ids(&out["claimed_line_indices"])?);
        out["graphic_blocks"].take()
    } else {
        json!([])
    };
    profile.mark("table_page_forms_and_code");
    let raster_images = if v["include_graphic_images"] == true {
        let mut containers = tables.clone();
        containers.extend_from_slice(array(&rule_code["code_blocks"], 4000)?);
        containers.extend_from_slice(array(&filled_code["code_blocks"], 4000)?);
        containers.extend_from_slice(array(&form_images, 4000)?);
        containers.extend_from_slice(array(&graphic_images, 4000)?);
        containers.extend_from_slice(prior_blocks);
        containers.extend_from_slice(caption_blocks);
        let mut out = super::page_containers::materialize_images_owned(
            crate::processing::with_source(
                analysis,
                json!({"container_blocks":containers,"claimed_line_indices":claimed}),
            ),
            w,
        )?;
        analysis = out["source"].take();
        claimed.extend(ids(&out["claimed_line_indices"])?);
        out["raster_image_blocks"].take()
    } else {
        json!([])
    };
    profile.mark("table_page_raster_images");
    let vector_formulas = if v["include_vector_formulas"] == true {
        let mut containers = tables.clone();
        containers.extend_from_slice(array(&rule_code["code_blocks"], 4000)?);
        containers.extend_from_slice(array(&filled_code["code_blocks"], 4000)?);
        containers.extend_from_slice(array(&form_images, 4000)?);
        containers.extend_from_slice(array(&graphic_images, 4000)?);
        containers.extend_from_slice(array(&raster_images, 4000)?);
        containers.extend_from_slice(caption_blocks);
        let mut out = super::native_vector_formulas::build_source(
            &analysis,
            &json!({"container_blocks":containers,"claimed_line_indices":claimed}),
            w,
        )?;
        claimed.extend(ids(&out["claimed_line_indices"])?);
        out["vector_formula_blocks"].take()
    } else {
        json!([])
    };
    profile.mark("table_page_vector_formulas");
    if v["include_graphic_images"] == true {
        array(&source["lines"], 4096)?;
        let updates = array(&analysis["lines"], 4096)?;
        let updated = source["lines"].as_array_mut().ok_or_else(invalid)?;
        for (&i, l) in positions.iter().zip(updates) {
            updated[i] = l.clone();
        }
    }
    let lines = array(&source["lines"], 4096)?;
    let remaining: Vec<_> = lines
        .iter()
        .filter(|l| !claimed.contains(&integer(l, "source_index", 0).unwrap_or(0)))
        .cloned()
        .collect();
    let mut fixed = prior_blocks.to_vec();
    fixed.extend_from_slice(array(&rule_code["code_blocks"], 4000)?);
    fixed.extend_from_slice(array(&committed["annotation_blocks"], 4000)?);
    fixed.extend(tables.clone());
    fixed.extend_from_slice(array(&filled_code["code_blocks"], 4000)?);
    fixed.extend_from_slice(array(&form_images, 4000)?);
    fixed.extend_from_slice(array(&graphic_images, 4000)?);
    fixed.extend_from_slice(array(&raster_images, 4000)?);
    fixed.extend_from_slice(array(&vector_formulas, 4000)?);
    fixed.extend_from_slice(caption_blocks);
    if let Some((_, indices, _)) = containers {
        fixed.extend_from_slice(indices);
    }
    if fixed.len() > rapiddoc_core::resources::limit(4000) {
        return Err(limit());
    }
    let mut prepared = json!({});
    prepared["fixed_blocks"] = Value::Array(fixed);
    prepared["remaining_lines"] = Value::Array(remaining);
    let next = detect::trailing_table_grid_sources(&source, &prepared, w)?;
    let mut result =
        crate::processing::with_source(source, json!({"claimed_line_indices":claimed}));
    result["strong_graphic_bboxes"] = Value::Array(strong);
    result["table_bboxes"] = Value::Array(tables.iter().map(|b| b["bbox"].clone()).collect());
    result["table_blocks"] = Value::Array(tables);
    result["annotation_blocks"] = committed["annotation_blocks"].take();
    result["remaining_lines"] = prepared["remaining_lines"].take();
    result["fixed_blocks"] = prepared["fixed_blocks"].take();
    result["next_table_grid"] = next;
    if code {
        result["rule_code_blocks"] = rule_code["code_blocks"].clone();
        result["code_blocks"] = filled_code["code_blocks"].clone();
    }
    if containers.is_some() {
        result["active_form_bboxes"] = json!(active_forms);
    }
    if v["include_form_images"] == true {
        result["form_image_blocks"] = form_images;
    }
    if v["include_graphic_images"] == true {
        result["graphic_blocks"] = graphic_images;
        result["raster_image_blocks"] = raster_images;
    }
    if v["include_vector_formulas"] == true {
        result["vector_formula_blocks"] = vector_formulas;
    }
    if crate::legacy_page::budget::output_exceeds_limit(
        &result,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn owned_table_transaction_matches_python() {
        let f: Value =
            serde_json::from_reader(flate2::read::GzDecoder::new(
                &include_bytes!(
                    "../../../../../tests/fixtures/native_table_owned_reference.json.gz"
                )[..],
            ))
            .unwrap();
        for host in [false, true] {
            crate::legacy_page::budget::with_resources(
                crate::legacy_page::budget::HostResources {
                    max_work: Some(if host { usize::MAX } else { usize::MAX - 1 }),
                    ..Default::default()
                },
                || {
                    let _scope = crate::legacy_page::budget::host_page();
                    for c in f["cases"].as_array().unwrap() {
                        let mut w = Work(0);
                        let actual = prepare_impl_owned(
                            c["request"].clone(),
                            &mut w,
                            c["code"].as_bool().unwrap(),
                            None,
                        )
                        .unwrap_or_else(|e| panic!("{} host={host}: {e}", c["label"]));
                        assert_eq!(actual, c["expected"], "{} host={host}", c["label"]);
                    }
                },
            );
        }
    }
    #[test]
    fn old_code_table_order() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../build/migration");
        for name in ["native-code-table-page", "native-code-table-grid-mixtures"] {
            let f: Value =
                serde_json::from_slice(&std::fs::read(root.join(format!("{name}.json"))).unwrap())
                    .unwrap();
            for c in f["cases"].as_array().unwrap() {
                assert_eq!(
                    prepare_code_tables(&c["request"])
                        .unwrap_or_else(|e| panic!("{}: {e}", c["label"])),
                    c["expected"],
                    "{}",
                    c["label"]
                );
            }
        }
    }
    #[test]
    fn code_table_shared_budget() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-code-table-page.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap().iter().step_by(11) {
            let q = &c["request"];
            let mut w = Work(0);
            assert!(prepare_impl(q, &mut w, true, None).is_ok());
            assert!(prepare_impl(q, &mut Work(10_000_000 - w.0 + 1), true, None).is_err());
            assert!(prepare_code_tables(q).is_ok());
        }
    }
    #[test]
    fn old_table_page_commit() {
        let directory =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../build/migration");
        for name in [
            "native-table-page-commit",
            "native-table-page-commit-specials",
        ] {
            let root = directory.join(format!("{name}.json"));
            let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
            for c in f["cases"].as_array().unwrap() {
                if duplicate_sources(&c["request"]) {
                    assert!(prepare(&c["request"]).is_err());
                    continue;
                }
                assert_eq!(
                    prepare(&c["request"]).unwrap_or_else(|e| panic!("{}: {e}", c["label"])),
                    c["expected"],
                    "{}",
                    c["label"]
                );
            }
            for c in f["errors"].as_array().unwrap() {
                assert!(prepare(&c["request"]).is_err(), "{}", c["label"]);
            }
        }
    }
    fn duplicate_sources(q: &Value) -> bool {
        let mut seen = BTreeSet::new();
        q["source"]["lines"]
            .as_array()
            .unwrap()
            .iter()
            .any(|l| !seen.insert(l["source_index"].as_i64().unwrap()))
    }
    #[test]
    fn shared_budget_and_recovery() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-table-page-commit-specials.json");
        let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap().iter().step_by(4) {
            let q = &c["request"];
            if duplicate_sources(q) {
                continue;
            }
            let mut w = Work(0);
            assert!(prepare_with_work(q, &mut w).is_ok());
            assert!(prepare_with_work(q, &mut Work(10_000_000 - w.0 + 1)).is_err());
            assert!(prepare(q).is_ok());
        }
    }
}
