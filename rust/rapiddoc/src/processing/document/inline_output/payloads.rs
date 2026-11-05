use super::*;
fn union_values(values: &[&Value]) -> Result<Value, Error> {
    let mut out = array(values[0], 4)?.clone();
    if out.len() != 4 {
        return Err(invalid());
    }
    for b in values {
        let b = array(b, 4)?;
        if b.len() != 4 {
            return Err(invalid());
        }
        for i in 0..4 {
            let a = number(&out[i])?;
            let n = number(&b[i])?;
            if if i < 2 { n < a } else { n > a } {
                out[i] = b[i].clone()
            }
        }
    }
    Ok(Value::Array(out))
}
pub(super) fn prepare(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_with_work(v, &mut Work(0))
}
pub(super) fn prepare_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    let input = array(&v["lines"], 4096)?;
    prepare_parts(
        input,
        &v["page_size"],
        &v["tight_bboxes"],
        &v["origins"],
        &v["all_chars"],
        &v["drawing_lines"],
        work,
    )
}
pub(super) fn prepare_parts(
    input: &[Value],
    page: &Value,
    tight: &Value,
    origins: &Value,
    all: &Value,
    drawings: &Value,
    work: &mut Work,
) -> Result<Value, Error> {
    if input.len() > rapiddoc_core::resources::limit(4096) {
        return Err(limit());
    }
    let empty = json!({});
    let tight = if tight.is_null() { &empty } else { tight };
    let origins = if origins.is_null() { &empty } else { origins };
    let empty_array = json!([]);
    let all = if all.is_null() { &empty_array } else { all };
    let drawings = if drawings.is_null() {
        &empty_array
    } else {
        drawings
    };
    let n = array(all, 20000)?.len();
    let rules = array(drawings, 5000)?.len();
    let mut fractions = BTreeMap::new();
    let mut output = Vec::new();
    for line in input {
        let angle = line["angle"].as_i64().unwrap_or(0).rem_euclid(360);
        if !fractions.contains_key(&angle) {
            work.charge(n.saturating_mul(rules))?;
            let members = if rules == 0 {
                json!([])
            } else {
                rapiddoc_native::script_geometry::refine::fraction_members_page(
                    &value_object!({"chars":all,"drawing_lines":drawings,"page_size":page,"angle":angle,"tight_bboxes":tight}),
                )?
            };
            fractions.insert(angle, members);
        }
        let count = array(&line["chars"], 1500)?.len();
        work.charge(count.saturating_mul(count).saturating_add(count * 5))?;
        let classified = if crate::legacy_page::budget::current().host {
            rapiddoc_native::script_geometry::refine::classify_line_parts(
                line,
                page,
                tight,
                origins,
                &fractions[&angle],
            )?
        } else {
            // Keep the finite diagnostic tree-validation contract unchanged.
            super::super::table_text_styles::classify_line(
                &value_object!({"line":line,"page_size":page,"tight_bboxes":tight,"origins":origins,"fraction_members":fractions[&angle]}),
            )?
        };
        let chars = classified["chars"].as_array().ok_or_else(invalid)?;
        if chars.is_empty() {
            continue;
        }
        let mut text = String::new();
        let mut roles = Vec::new();
        let mut boxes = Vec::new();
        let mut counts = Vec::new();
        let mut flags = Vec::new();
        for (i, c) in chars.iter().enumerate() {
            let fragment =
                super::super::inline_evidence::compact(c["char"].as_str().ok_or_else(invalid)?);
            let m = fragment.chars().count();
            let index = c["char_idx"]
                .as_i64()
                .or_else(|| c["char_idx"].as_bool().map(i64::from));
            let b = index
                .and_then(|i| tight.get(i.to_string()))
                .filter(|v| !v.is_null());
            for _ in 0..m {
                roles.push(classified["roles"][i].as_str().ok_or_else(invalid)?);
                boxes.push(b);
                counts.push(classified["body_counts"][i].as_u64().ok_or_else(invalid)?);
                flags.push(
                    classified["formula_flags"][i]
                        .as_bool()
                        .ok_or_else(invalid)?,
                )
            }
            text.push_str(&fragment);
        }
        if text.is_empty() {
            continue;
        }
        let mut ranges = Vec::new();
        let mut start = 0;
        while start < roles.len() {
            let role = roles[start];
            let mut end = start + 1;
            while end < roles.len() && roles[end] == role {
                end += 1
            }
            if matches!(role, "sup" | "sub") {
                let boxes: Vec<_> = boxes[start..end].iter().flatten().copied().collect();
                if !boxes.is_empty() {
                    ranges.push(value_object!({"start":start,"end":end,"style":if role=="sup"{"superscript"}else{"subscript"},"bbox":union_values(&boxes)?,"stable_body_count":counts[start..end].iter().copied().max().unwrap_or(0),"formula_region":flags[start..end].iter().any(|v|*v)}))
                }
            }
            start = end;
        }
        output.push(value_object!({"bbox":line["bbox"],"text":text,"script_ranges":ranges,"source_index":line["source_index"].as_i64().unwrap_or(0),"angle":angle}));
    }
    let result = Value::Array(output);
    if crate::legacy_page::budget::output_exceeds_limit(
        &result,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(result)
}
