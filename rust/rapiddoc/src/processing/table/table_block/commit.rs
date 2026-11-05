//! 旧_materialize_table_blocks事务：恢复/投影完成后才发布整组块和行认领。
use super::*;
use crate::legacy_model::{
    table_native_engine, table_native_multiline::spatial, table_native_text as text,
    table_text_styles::materialization,
};
use rapiddoc_core::ErrorCode;
fn coerce_rules(raw: &[Value]) -> Result<Vec<Value>, Error> {
    let mut output = Vec::new();
    for r in raw {
        if let Some(b) = text::normalize_bbox(&r["bbox"])? {
            if matches!(r["orientation"].as_str(), Some("horizontal" | "vertical")) {
                let width = text::number(&r["width"])?.unwrap_or(0.).max(0.);
                output.push(json!({"bbox":b,"width":width,"orientation":r["orientation"]}));
            }
        }
    }
    Ok(output)
}
fn coerce_rectangles(raw: &[Value]) -> Result<Vec<Value>, Error> {
    let mut output = Vec::new();
    for r in raw {
        if let Some(b) = text::normalize_bbox(&r["bbox"])? {
            let Some(segments) = r.get("segment_count").map_or(Some(0), Value::as_i64) else {
                continue;
            };
            let Some(depth) = r.get("form_depth").map_or(Some(0), Value::as_i64) else {
                continue;
            };
            output.push(json!({"bbox":b,"segment_count":segments,"form_depth":depth,"fill_visible":crate::legacy_page::truth(&r["fill_visible"]),"stroke_visible":crate::legacy_page::truth(&r["stroke_visible"])}));
        }
    }
    Ok(output)
}
fn recover(
    input: &Value,
    request: &Value,
    preserve: bool,
    work: &mut Work,
) -> Result<Option<(String, Value)>, Error> {
    let mut profile = crate::processing::NativeStageProfile::new();
    let result =
        table_native_engine::recover_with_work(&value_object!({"table_input":input}), work)?;
    profile.mark("table_recover_native_engine");
    if result.is_null() {
        return Ok(None);
    }
    let content = materialization::render_with_work(
        &value_object!({"table_input":input,"result":result,"tight_bboxes":request["tight_bboxes"],"origins":request["origins"],"preserve_font_styles":preserve}),
        work,
    )?;
    profile.mark("table_recover_styles");
    Ok(Some((content, result)))
}
fn recover_html(
    source: &Value,
    input: &Value,
    request: &Value,
    preserve: bool,
    work: &mut Work,
) -> Result<(String, Vec<Value>), Error> {
    let media = media::select(source, candidate_rect(&input["table_bbox"])?, work)?;
    if !media.is_empty() {
        let masked = media::mask(input, &media, work)?;
        if let Some((content, result)) = recover(&masked, request, preserve, work)? {
            if let Some(attached) = media::attach(&content, &result, input, &media, work)? {
                return Ok(attached);
            }
        }
    }
    Ok(recover(input, request, preserve, work)?
        .map_or((String::new(), Vec::new()), |(html, _)| (html, Vec::new())))
}
fn overlap_smaller(a: Box4, b: Box4) -> f64 {
    media::overlap_first(a, b).max(media::overlap_first(b, a))
}
pub fn materialize(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    materialize_with_work(v, &mut Work(0))
}
pub(crate) fn materialize_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    let mut source = v["source"].clone();
    let mut out = materialize_source(&mut source, v, work)?;
    out["lines"] = source["lines"].take();
    Ok(out)
}
/// The page transaction already owns its source. Mutate its line state directly
/// instead of copying the full character tree into and out of an envelope.
pub(crate) fn materialize_source(
    source: &mut Value,
    v: &Value,
    work: &mut Work,
) -> Result<Value, Error> {
    let mut profile = crate::processing::NativeStageProfile::new();
    let mut candidates = array(&v["candidates"], 128)?.clone();
    parse_lines(&source)?;
    array(&source["chars"], 20000)?;
    let page = array(&source["page_size"], 2)?;
    if page.len() != 2 || number(&page[0])? <= 0. || number(&page[1])? <= 0. {
        return Err(invalid());
    }
    for c in &mut candidates {
        number(&c["score"])?;
        c["line_indices"] = json!(ids(&c["line_indices"])?);
        for a in array(&c["annotations"], 2000)? {
            validate_annotation(a)?;
        }
        for a in c["annotations"].as_array_mut().unwrap() {
            a["line_indices"] = json!(ids(&a["line_indices"])?);
        }
    }
    let mut order = (0..candidates.len()).collect::<Vec<_>>();
    order.sort_by(|&a, &b| {
        candidates[b]["score"]
            .as_f64()
            .partial_cmp(&candidates[a]["score"].as_f64())
            .unwrap()
    });
    let rules = coerce_rules(array(&source["drawing_lines"], 5000)?)?;
    let rectangles = coerce_rectangles(
        source
            .get("path_infos")
            .map_or(Ok(&[][..]), |a| array(a, 5000).map(Vec::as_slice))?,
    )?;
    let mut tables = Vec::new();
    let mut annotations = Vec::new();
    let mut claimed = BTreeSet::new();
    let mut accepted = Vec::new();
    let mut cached_chars = None::<Vec<Value>>;
    let mut cached_boxes = None::<Map<String, Value>>;
    let mut output_bytes = 0usize;
    profile.mark("table_commit_source_setup");
    for index in order {
        work.charge(array(&source["chars"], 20000)?.len() + array(&source["lines"], 4096)?.len())?;
        let request = value_object!({"source":source,"candidate":candidates[index],"calibrated_char_bboxes":v["calibrated_char_bboxes"]});
        // 旧顺序先分离注释再去重，重复候选仍保留该副作用；只在接纳后记去重框。
        let lines = parse_lines(&source)?;
        separate(
            &lines,
            array(&source["drawing_lines"], 5000)?,
            &mut candidates[index],
            work,
        )?;
        let bbox = candidate_rect(&candidates[index]["bbox"])?;
        let mut duplicate = false;
        for b in &accepted {
            work.charge(1)?;
            if overlap_smaller(bbox, *b) >= 0.5 {
                duplicate = true;
                break;
            }
        }
        if duplicate {
            continue;
        }
        let mut request = request;
        request["candidate"] = candidates[index].clone();
        let cached = cached_chars.as_deref().zip(cached_boxes.as_ref());
        profile.mark("table_commit_request");
        let prepared = prepare_cached(&request, work, true, cached, true)?;
        profile.mark("table_commit_prepare");
        if cached_chars.is_none() {
            cached_chars = Some(prepared["native_chars"].as_array().unwrap().clone());
            cached_boxes = Some(
                prepared["calibrated_char_bboxes"]
                    .as_object()
                    .unwrap()
                    .clone(),
            );
        }
        candidates[index] = prepared["candidate"].clone();
        let c = &candidates[index];
        let body = candidate_rect(&prepared["body_bbox"])?;
        let mut native_rules = if flag(c, "inferred_grid_authoritative")? {
            Vec::new()
        } else {
            rules.clone()
        };
        native_rules.extend(coerce_rules(
            c.get("inferred_grid")
                .map_or(Ok(&[][..]), |a| array(a, 5000).map(Vec::as_slice))?,
        )?);
        let input = value_object!({"table_bbox":prepared["body_bbox"],"page_size":source["page_size"],"angle":c["angle"],"chars":prepared["native_chars"],"drawing_lines":Value::Array(native_rules),"rectangles":if flag(c,"inferred_grid_authoritative")?{json!([])}else{Value::Array(rectangles.clone())}});
        let mut preserve = flag(c, "preserve_inline_font_styles")?;
        for b in prepared["calibrated_char_bboxes"]
            .as_object()
            .unwrap()
            .values()
        {
            work.charge(1)?;
            preserve |= inside(center(rect(b)?), body);
        }
        profile.mark("table_commit_recovery_input");
        let (mut content, images) = match recover_html(&source, &input, v, preserve, work) {
            Ok(value) => value,
            Err(error) if error.code == ErrorCode::ResourceLimit => return Err(error),
            Err(_) => (String::new(), Vec::new()),
        };
        profile.mark("table_commit_recover_html");
        if content.is_empty() {
            let restored = output::restore_with_work(
                &value_object!({"source":source,"bbox":prepared["body_bbox"]}),
                work,
            )?;
            source["lines"] = restored["lines"].clone();
            if restored["restore"] == true {
                continue;
            }
        }
        if content.is_empty() {
            let chars = array(&source["chars"], 20000)?.iter().collect::<Vec<_>>();
            let a = text::angle(&c["angle"])?;
            match spatial::project(&chars, &prepared["body_bbox"], i64::from(a), work) {
                Ok(text) => content = text,
                Err(error) if error.code == ErrorCode::ResourceLimit => return Err(error),
                Err(_) => continue,
            }
        }
        if content.trim_matches(python_space).is_empty() {
            continue;
        }
        let mut block = json!({"type":"table","bbox":prepared["body_bbox"],"angle":c["angle"],"content":content});
        if !images.is_empty() {
            block["_table_images"] = json!(images);
        }
        output_bytes += block.to_string().len() + prepared["annotation_blocks"].to_string().len();
        if output_bytes > rapiddoc_core::resources::limit(8_000_000)
            || tables.len()
                + annotations.len()
                + prepared["annotation_blocks"].as_array().unwrap().len()
                >= rapiddoc_core::resources::limit(4000)
        {
            return Err(limit());
        }
        tables.push(block);
        annotations.extend(
            prepared["annotation_blocks"]
                .as_array()
                .unwrap()
                .iter()
                .cloned(),
        );
        accepted.push(bbox);
        claimed.extend(ids(&prepared["projection_line_indices"])?);
        claimed.extend(ids(&prepared["externalized_line_indices"])?);
    }
    for blocks in [&mut tables, &mut annotations] {
        blocks.sort_by(|a, b| {
            a["bbox"][1]
                .as_f64()
                .partial_cmp(&b["bbox"][1].as_f64())
                .unwrap()
                .then(
                    a["bbox"][0]
                        .as_f64()
                        .partial_cmp(&b["bbox"][0].as_f64())
                        .unwrap(),
                )
        });
    }
    let mut out = json!({"claimed_line_indices":claimed});
    out["table_blocks"] = Value::Array(tables);
    out["annotation_blocks"] = Value::Array(annotations);
    out["candidates"] = Value::Array(candidates);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_complete_block_transactions() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-table-block-commit.json");
        let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                materialize(&c["request"]).unwrap_or_else(|e| panic!("{}: {e}", c["label"])),
                c["expected"],
                "{}",
                c["label"]
            );
        }
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-table-block-commit-cache.json");
        let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(materialize(&c["request"]).unwrap(), c["expected"]);
        }
    }
}
