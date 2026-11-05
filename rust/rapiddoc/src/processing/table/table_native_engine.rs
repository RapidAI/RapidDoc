//! 旧engine.py完整原始字符→候选编排/仲裁/普通HTML，尚不从PDF提取原语。
use super::{
    table_native_aligned_numeric as aligned, table_native_banded_numeric as banded,
    table_native_candidate, table_native_engine_policy as policy,
    table_native_multiline as multiline, table_native_rule as arbitration,
    table_native_rule_band as rule_band, table_native_sparse_hybrid as hybrid,
    table_native_text as text, table_native_text_grid as text_grid, table_native_vector as vector,
};
use super::{table_native_hierarchical as hierarchical, table_native_text_quality as quality};
use crate::legacy_page::{invalid, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::python_space;
use serde_json::{json, Value};
use std::collections::BTreeSet;
struct Evaluation {
    primitive_count: Value,
    text: Value,
    generated: Vec<Value>,
    retained: Vec<usize>,
    selected: Option<usize>,
    physical: bool,
    gate: Option<&'static str>,
    text_quality: Vec<Value>,
}
fn threshold(c: &Value) -> f64 {
    match c["source"].as_str().unwrap() {
        "sparse_hybrid" | "sparse_multiline" => 0.98,
        _ => 0.95,
    }
}
fn verified(c: &Value) -> bool {
    c["score"].as_f64().unwrap() >= threshold(c)
}
fn priority(c: &Value) -> usize {
    match c["source"].as_str().unwrap() {
        "vector_grid" => 6,
        "sparse_hybrid" => 5,
        "sparse_grid" => 4,
        "key_value" => 3,
        "text_grid" => 2,
        _ => 1,
    }
}
fn issues(c: &Value, s: &str) -> bool {
    c["issues"].as_array().unwrap().iter().any(|i| i == s)
}
fn verified_rect(c: &Value) -> bool {
    c["source"] == "vector_grid"
        && c["score"].as_f64().unwrap() >= 0.95
        && issues(c, "evidence=rect_grid")
        && issues(c, "ambiguous_separator_ratio=0.0000")
}
fn count(v: &Value, k: &str) -> usize {
    v.get(k).and_then(Value::as_array).map_or(0, Vec::len)
}
fn facts(
    input: &Value,
    txt: &Value,
    attempts: &[Value],
    candidates: &[Value],
    work: &mut Work,
) -> Result<Value, Error> {
    work.charge(
        count(input, "drawing_lines")
            + count(input, "rectangles")
            + candidates
                .iter()
                .map(|c| count(c, "cells") * 16)
                .sum::<usize>()
            + attempts.len() * 8,
    )?;
    policy::inspect(
        &value_object!({"table_input":input,"text":txt,"vector_attempts":attempts.to_vec(),"candidates":candidates.to_vec()}),
    )
}
fn select(candidates: &mut Vec<Value>, remove: bool, work: &mut Work) -> Result<Value, Error> {
    work.charge(
        candidates
            .iter()
            .map(|c| count(c, "cells") * (candidates.len() + 1))
            .sum::<usize>(),
    )?;
    // Keep the same JSON wrapper and validator. Transfer the array temporarily
    // instead of deep-copying every candidate for each arbitration pass.
    let mut request = json!({"remove_undercounted":remove});
    request["candidates"] = Value::Array(std::mem::take(candidates));
    struct Restore<'a> {
        original: &'a mut Vec<Value>,
        request: Value,
    }
    impl Drop for Restore<'_> {
        fn drop(&mut self) {
            *self.original = std::mem::take(self.request["candidates"].as_array_mut().unwrap());
        }
    }
    let guard = Restore {
        original: candidates,
        request,
    };
    // Restore on both Result::Err and unwinding; only the temporary wrapper owns
    // the array while the unchanged read-only arbiter runs.
    arbitration::select(&guard.request)
}
fn values(v: &mut Value, k: &str) -> Result<Vec<Value>, Error> {
    v.get_mut(k)
        .and_then(Value::as_array_mut)
        .map(std::mem::take)
        .ok_or_else(invalid)
}
fn retain(candidates: Vec<Value>, minimum: u64) -> Vec<Value> {
    candidates
        .into_iter()
        .filter(|c| c["cols"].as_u64().unwrap() >= minimum)
        .collect()
}
fn ensure_attempts(
    attempts: &mut Vec<Value>,
    request: &Value,
    work: &mut Work,
) -> Result<(), Error> {
    if attempts.is_empty() {
        *attempts = values(
            &mut vector::generate_with_work(request, work)?,
            "diagnostics",
        )?;
    }
    Ok(())
}
fn covers(a: &Value, b: &Value) -> bool {
    a["rows"].as_f64().unwrap() >= 1.5 * b["rows"].as_f64().unwrap()
        || a["cols"].as_u64().unwrap() > b["cols"].as_u64().unwrap()
}
fn header_undercount(selected: &Value, aligned: &Value, work: &mut Work) -> Result<bool, Error> {
    let old = selected["cells"].as_array().unwrap();
    let new = aligned["cells"].as_array().unwrap();
    work.charge(old.len() * new.len() * 2)?;
    let first = old
        .iter()
        .take(selected["cols"].as_u64().unwrap() as usize * 2)
        .any(|cell| {
            cell["colspan"].as_u64().unwrap() >= 3
                && new
                    .iter()
                    .filter(|n| {
                        let (a, b) = (&cell["bbox"], &n["bbox"]);
                        !n["content"].as_str().unwrap().is_empty()
                            && a[0].as_f64().unwrap() <= b[0].as_f64().unwrap()
                            && b[2].as_f64().unwrap() <= a[2].as_f64().unwrap()
                            && b[1].as_f64().unwrap() < a[3].as_f64().unwrap()
                            && a[1].as_f64().unwrap() < b[3].as_f64().unwrap()
                    })
                    .count()
                    >= 2
        });
    Ok(first
        || new.iter().any(|n| {
            n["row"].as_u64().unwrap() < 2
                && n["colspan"].as_u64().unwrap() >= 2
                && !n["content"]
                    .as_str()
                    .unwrap()
                    .trim_matches(python_space)
                    .is_empty()
                && old.iter().any(|o| {
                    o["row"].as_u64().unwrap() < 2
                        && o["colspan"] == 1
                        && o["content"] == n["content"]
                })
        }))
}
fn evaluate(input: &Value, work: &mut Work) -> Result<Evaluation, Error> {
    evaluate_prepared(input, None, work)
}
fn evaluate_prepared(
    input: &Value,
    prepared: Option<text::Text>,
    work: &mut Work,
) -> Result<Evaluation, Error> {
    let empty = json!({"glyphs":[],"rows":[],"median_glyph_width":1.,"median_glyph_height":1.});
    let initial = facts(input, &empty, &[], &[], work)?;
    let primitive = initial["primitive_count"].clone();
    let early = |gate| Evaluation {
        primitive_count: primitive.clone(),
        text: Value::Null,
        generated: Vec::new(),
        retained: Vec::new(),
        selected: None,
        physical: false,
        gate: Some(gate),
        text_quality: Vec::new(),
    };
    if initial["input_rejection_gate"] == "input_geometry" {
        return Ok(early("input_geometry"));
    }
    if initial["input_rejection_gate"] == "primitive_limit" {
        return Ok(early("primitive_limit"));
    }
    let input_quality = quality::input(input)?;
    if input_quality.rejected() {
        let mut result = early("text_quality");
        result.text_quality.push(input_quality.record());
        return Ok(result);
    }
    let Some(txt) = (if prepared.is_some() {
        prepared
    } else {
        text::prepare_with_work(input, work)?
    }) else {
        return Ok(early("native_text"));
    };
    let nested = hierarchical::find(input, &txt, work)?;
    let hierarchy = hierarchical::build(input, &txt, &nested, work)?;
    let mut txt = json!(txt);
    let request = value_object!({"table_input":input,"text":txt});
    let mut output = vector::generate_with_work(&request, work)?;
    let mut attempts = values(&mut output, "diagnostics")?;
    let vector_all = values(&mut output, "candidates")?;
    let p = facts(input, &txt, &attempts, &vector_all, work)?;
    let minimum = p["minimum_columns"].as_u64().unwrap();
    let mut vectors = retain(vector_all, minimum);
    let mut aligned_output = aligned::generate_with_work(&request, work)?;
    let mut aligned = aligned_output["candidate"].take();
    if !aligned.is_null()
        && (aligned["cols"].as_u64().unwrap() < minimum
            || !hierarchical::preserves(&aligned, &nested))
    {
        aligned = Value::Null;
    }
    let replaces = !aligned.is_null() && vectors.iter().any(|c| covers(&aligned, c));
    if !aligned.is_null() {
        vectors.retain(|c| !covers(&aligned, c));
    }
    let mut physical = false;
    if vectors.iter().any(verified_rect) {
        let f = facts(input, &txt, &attempts, &vectors, work)?;
        physical = f["attempted_line_rect_grid_conflict"].as_bool().unwrap();
        if physical && !f["resolved_rect_undercount"].is_null() {
            physical = false;
        }
    }
    let mut eligible_vectors: Vec<_> = vectors
        .iter()
        .filter(|c| hierarchical::preserves(c, &nested))
        .cloned()
        .collect();
    let vector_selected =
        !physical && !select(&mut eligible_vectors, false, work)?["selected"].is_null();
    let rows = count(&txt, "rows");
    let mut sparse_allowed = rows >= 2 && !vector_selected && !physical;
    if sparse_allowed {
        ensure_attempts(&mut attempts, &request, work)?;
        if p["alias_affected_physical_blank_row"] == true {
            sparse_allowed = false;
        }
    }
    let rule = if sparse_allowed {
        values(
            &mut rule_band::generate_with_work(&request, work)?,
            "candidates",
        )?
    } else {
        Vec::new()
    };
    let hybrids = if sparse_allowed
        && rule
            .iter()
            .filter(|c| c["cols"].as_u64().unwrap() >= minimum)
            .count()
            == 0
    {
        values(
            &mut hybrid::generate_with_work(&request, work)?,
            "candidates",
        )?
    } else {
        Vec::new()
    };
    let sparse_selected = !select(
        &mut retain(rule.iter().chain(&hybrids).cloned().collect(), minimum),
        false,
        work,
    )?["selected"]
        .is_null();
    let mut texts = if rows >= 2 && !vector_selected && !sparse_selected {
        values(
            &mut text_grid::generate_with_work(&request, work)?,
            "candidates",
        )?
    } else {
        Vec::new()
    };
    if !texts.is_empty() && (vectors.is_empty() || physical) {
        ensure_attempts(&mut attempts, &request, work)?;
        if p["alias_affected_physical_blank_row"] == true || p["physical_row_undercount"] == true {
            texts.clear();
        }
    }
    let mut generated = vectors;
    generated.extend(hierarchy);
    generated.extend(rule);
    generated.extend(hybrids);
    generated.extend(texts);
    let existing_selection = select(&mut generated, true, work)?;
    let existing: Vec<_> = existing_selection["kept"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i.as_u64().unwrap() as usize)
        .filter(|i| {
            generated[*i]["cols"].as_u64().unwrap() >= minimum
                && hierarchical::preserves(&generated[*i], &nested)
        })
        .collect();
    let mut multiline_allowed =
        rows >= 2 && !physical && !existing.iter().any(|i| verified(&generated[*i]));
    if multiline_allowed {
        ensure_attempts(&mut attempts, &request, work)?;
        if p["alias_affected_physical_blank_row"] == true {
            multiline_allowed = false;
        }
    }
    let multi = if multiline_allowed {
        values(
            &mut multiline::generate_with_work(&request, work)?,
            "candidates",
        )?
    } else {
        Vec::new()
    };
    let banded = if multiline_allowed
        && multi
            .iter()
            .filter(|c| c["cols"].as_u64().unwrap() >= minimum)
            .count()
            == 0
    {
        banded::generate_with_work(&request, work)?["candidate"].clone()
    } else {
        Value::Null
    };
    generated.extend(multi);
    if !banded.is_null() {
        generated.push(banded);
    }
    let removed = select(&mut generated, true, work)?;
    let mut retained: Vec<_> = removed["kept"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i.as_u64().unwrap() as usize)
        .filter(|i| {
            generated[*i]["cols"].as_u64().unwrap() >= minimum
                && hierarchical::preserves(&generated[*i], &nested)
        })
        .collect();
    let mut candidates: Vec<_> = retained.iter().map(|i| generated[*i].clone()).collect();
    let mut selected = if physical {
        None
    } else {
        select(&mut candidates, false, work)?["selected"]
            .as_u64()
            .map(|i| retained[i as usize])
    };
    let header = if let Some(i) = selected {
        if aligned.is_null() {
            false
        } else {
            header_undercount(&generated[i], &aligned, work)?
        }
    } else {
        false
    };
    if !aligned.is_null()
        && !physical
        && (replaces
            || existing.is_empty()
            || header
            || selected.is_some_and(|i| covers(&aligned, &generated[i])))
    {
        let i = generated.len();
        generated.push(aligned);
        retained.push(i);
        selected = Some(i);
        txt = aligned_output["text"].clone();
    }
    if sparse_allowed {
        let rowspan = super::table_native_banded_rowspan::candidate(
            input,
            &request["text"],
            &aligned_output["text"],
            work,
        )?;
        if !rowspan.is_null()
            && rowspan["cols"].as_u64().unwrap() >= minimum
            && hierarchical::preserves(&rowspan, &nested)
        {
            let replace = selected.is_none_or(|i| {
                generated[i]["cols"] == rowspan["cols"]
                    && generated[i]["rows"].as_u64() < rowspan["rows"].as_u64()
            });
            let index = generated.len();
            generated.push(rowspan);
            retained.push(index);
            if replace {
                selected = Some(index);
                txt = aligned_output["text"].clone();
            }
        }
    }
    let gate = if selected.is_some() {
        None
    } else if generated.is_empty() {
        Some("candidate_generation")
    } else if retained.is_empty() {
        Some(if !nested.is_empty() {
            "nested_cell_undercount"
        } else if minimum > 0
            && generated
                .iter()
                .any(|c| c["cols"].as_u64().unwrap() < minimum)
        {
            "physical_column_undercount"
        } else {
            "undercount_guard"
        })
    } else if !retained.iter().any(|i| verified(&generated[*i])) {
        Some("verified_threshold")
    } else {
        Some("topology_conflict")
    };
    let mut text_quality = vec![input_quality.record()];
    let mut gate = gate;
    if let Some(index) = selected {
        let output = quality::measure(
            generated[index]["cells"]
                .as_array()
                .unwrap()
                .iter()
                .map(|cell| cell["content"].as_str().unwrap()),
            "output",
        );
        if output.rejected() {
            selected = None;
            gate = Some("text_quality");
        }
        text_quality.push(output.record());
    }
    Ok(Evaluation {
        primitive_count: primitive,
        text: txt,
        generated,
        retained,
        selected,
        physical,
        gate,
        text_quality,
    })
}
impl Evaluation {
    fn record(&self) -> Value {
        json!({"primitive_count":self.primitive_count,"text":self.text,"generated_candidates":self.generated,"candidates":self.retained.iter().map(|i|&self.generated[*i]).collect::<Vec<_>>(),"selected":self.selected.map(|i|&self.generated[i]),"physical_topology_conflict":self.physical,"first_rejection_gate":self.gate,"text_quality":self.text_quality.iter().map(|q|json!({"stage":q["stage"],"text_chars":q["text_chars"],"abnormal_chars":q["abnormal_chars"]})).collect::<Vec<_>>()})
    }
    fn recover(&self) -> Result<Value, Error> {
        let Some(i) = self.selected else {
            return Ok(Value::Null);
        };
        let c = &self.generated[i];
        let mut diagnostics = vec![format!("candidate={}", c["source"].as_str().unwrap())];
        for key in [
            "score",
            "text_capture",
            "structure_support",
            "row_stability",
            "column_stability",
            "order_consistency",
        ] {
            diagnostics.push(format!("{key}={:.4}", c[key].as_f64().unwrap()));
        }
        diagnostics.extend(
            c["issues"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap().to_owned()),
        );
        Ok(
            json!({"html":table_native_candidate::render(c)?,"rows":c["rows"],"cols":c["cols"],"cells":c["cells"],"text":self.text,"source":c["source"],"confidence":c["score"],"diagnostics":diagnostics}),
        )
    }
}
fn candidate_record(c: &Value) -> Value {
    json!({"source":c["source"],"rows":c["rows"],"cols":c["cols"],"tracks":{"x":c["cols"].as_u64().unwrap()+1,"y":c["rows"].as_u64().unwrap()+1},"score":c["score"],"verified":verified(c),"score_components":{"text_capture":c["text_capture"],"structure_support":c["structure_support"],"row_stability":c["row_stability"],"column_stability":c["column_stability"],"order_consistency":c["order_consistency"]},"span_signature":c["cells"].as_array().unwrap().iter().filter(|s|s["rowspan"].as_u64().unwrap()>1||s["colspan"].as_u64().unwrap()>1).map(|s|json!([s["row"],s["col"],s["rowspan"],s["colspan"]])).collect::<Vec<_>>(),"issues":c["issues"]})
}
fn diagnose(input: &Value, e: &Evaluation, work: &mut Work) -> Result<Value, Error> {
    let (mut vector_attempts, mut text_attempts, mut sparse_attempts, mut multi_attempts) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    if !e.text.is_null() {
        let request = value_object!({"table_input":input,"text":e.text});
        vector_attempts = values(
            &mut vector::generate_with_work(&request, work)?,
            "diagnostics",
        )?;
        text_attempts = values(
            &mut text_grid::generate_with_work(&request, work)?,
            "diagnostics",
        )?;
        sparse_attempts = values(
            &mut hybrid::generate_with_work(&request, work)?,
            "diagnostics",
        )?;
        if count(&e.text, "rows") >= 4 {
            sparse_attempts
                .push(banded::generate_with_work(&request, work)?["diagnostics"].clone());
        }
        multi_attempts = values(
            &mut rule_band::generate_with_work(&request, work)?,
            "diagnostics",
        )?;
        multi_attempts.extend(values(
            &mut multiline::generate_with_work(&request, work)?,
            "diagnostics",
        )?);
    }
    let mut gate = e.gate.map(str::to_owned);
    if e.gate == Some("candidate_generation") {
        let text = text_attempts.iter().find(|a| {
            [
                "dense_row_ambiguity",
                "header_requires_rowspan",
                "token_split",
            ]
            .iter()
            .any(|g| a["first_rejection_gate"] == *g)
        });
        let line = vector_attempts
            .iter()
            .find(|a| a["evidence"] == "line_grid");
        let rejection = |a: &&Value| {
            a["first_rejection_gate"]
                .as_str()
                .is_some_and(|s| !s.is_empty())
        };
        if let Some(a) = text {
            gate = Some(format!(
                "text_{}",
                a["first_rejection_gate"].as_str().unwrap()
            ));
        } else if let Some(a) = sparse_attempts.iter().find(rejection) {
            gate = Some(format!(
                "sparse_hybrid_{}",
                a["first_rejection_gate"].as_str().unwrap()
            ));
        } else if let Some(a) = line.filter(|a| rejection(a)) {
            gate = Some(format!(
                "vector_{}",
                a["first_rejection_gate"].as_str().unwrap()
            ));
        } else if let Some(a) = multi_attempts.iter().find(rejection) {
            gate = Some(format!(
                "sparse_multiline_{}",
                a["first_rejection_gate"].as_str().unwrap()
            ));
        }
    }
    let mut candidates: Vec<_> = e.retained.iter().map(|i| e.generated[*i].clone()).collect();
    let conflict = select(&mut candidates, false, work)?["physical_topology_conflict"]
        .as_bool()
        .unwrap();
    let retained: BTreeSet<_> = e.retained.iter().copied().collect();
    let mut best = None;
    for (i, c) in e.generated.iter().enumerate() {
        let key = (c["score"].as_f64().unwrap(), priority(c));
        if best.is_none_or(|j: usize| {
            key > (
                e.generated[j]["score"].as_f64().unwrap(),
                priority(&e.generated[j]),
            )
        }) {
            best = Some(i);
        }
    }
    Ok(
        json!({"primitive_count":e.primitive_count,"glyph_count":count(&e.text,"glyphs"),"visual_text_rows":count(&e.text,"rows"),"first_rejection_gate":gate,"text_quality":e.text_quality,"line_rect_topology_conflict":e.physical||conflict,"vector_attempts":vector_attempts,"sparse_hybrid_attempts":sparse_attempts,"sparse_multiline_attempts":multi_attempts,"text_attempts":text_attempts,"generated_candidates":e.generated.iter().map(candidate_record).collect::<Vec<_>>(),"removed_by_undercount":e.generated.iter().enumerate().filter(|(i,_)|!retained.contains(i)).map(|(_,c)|candidate_record(c)).collect::<Vec<_>>(),"counterfactual_best":best.map(|i|candidate_record(&e.generated[i])),"adopted":e.selected.map(|i|candidate_record(&e.generated[i]))}),
    )
}
pub fn evaluate_json(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let e = evaluate(&v["table_input"], &mut Work(0))?;
    Ok(e.record())
}
pub fn recover(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    recover_with_work(v, &mut Work(0))
}
pub(crate) fn recover_prepared(input: &Value, text: text::Text) -> Result<Value, Error> {
    evaluate_prepared(input, Some(text), &mut Work(0))?.recover()
}
pub(super) fn recover_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    evaluate(&v["table_input"], work)?.recover()
}
pub fn inspect(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let mut work = Work(0);
    let e = evaluate(&v["table_input"], &mut work)?;
    diagnose(&v["table_input"], &e, &mut work)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn engine_diagnostic_replay_cannot_reset_shared_work_budget() {
        let mut lines = Vec::new();
        for y in [0., 30., 59.9] {
            lines.push(json!({"bbox":[0.,y,100.,y+0.1],"width":0.1,"orientation":"horizontal"}));
        }
        for x in [0., 50., 99.9] {
            lines.push(json!({"bbox":[x,0.,x+0.1,60.],"width":0.1,"orientation":"vertical"}));
        }
        let chars: Vec<_> = (0..1000)
            .map(|i| {
                let cell = i / 250;
                let x = 10. + 50. * (cell % 2) as f64;
                let y = 10. + 30. * (cell / 2) as f64;
                json!({"char_idx":i,"char":"A","bbox":[x,y,x+8.,y+8.]})
            })
            .collect();
        let input = json!({"table_bbox":[0.,0.,100.,60.],"page_size":[200.,200.],"angle":0,"drawing_lines":lines,"rectangles":[],"chars":chars});
        let e = evaluate(&input, &mut Work(0)).unwrap();
        assert!(e.selected.is_some());
        let request = json!({"table_input":input,"text":e.text});
        assert!(vector::generate(&request).is_ok());
        assert!(hybrid::generate(&request).is_ok());
        assert!(text_grid::generate(&request).is_ok());
        assert!(inspect(&json!({"table_input":input})).is_err());
    }
    #[test]
    fn upstream_068_pdf_references() {
        fn difference(a: &Value, b: &Value, path: &str) -> Option<String> {
            match (a, b) {
                (Value::Number(a), Value::Number(b)) if a.as_f64() == b.as_f64() => None,
                (Value::Object(a), Value::Object(b)) => {
                    if a.keys().ne(b.keys()) {
                        return Some(format!("{path}: keys {:?} != {:?}", a.keys(), b.keys()));
                    }
                    for (k, v) in a {
                        if let Some(diff) = difference(v, &b[k], &format!("{path}.{k}")) {
                            return Some(diff);
                        }
                    }
                    None
                }
                (Value::Array(a), Value::Array(b)) => {
                    if a.len() != b.len() {
                        return Some(format!("{path}: lengths {} != {}", a.len(), b.len()));
                    }
                    for (i, (a, b)) in a.iter().zip(b).enumerate() {
                        if let Some(diff) = difference(a, b, &format!("{path}[{i}]")) {
                            return Some(diff);
                        }
                    }
                    None
                }
                _ => (a != b).then(|| format!("{path}: {a} != {b}")),
            }
        }
        let compressed = std::fs::File::open(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../tests/fixtures/upstream_068/pdf_reference.json.gz"),
        )
        .unwrap();
        let f: Value = serde_json::from_reader(flate2::read::GzDecoder::new(compressed)).unwrap();
        for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
            let actual = if c["kind"] == "table" {
                let e = super::evaluate_json(&c["request"])
                    .unwrap_or_else(|e| panic!("case {i} eval: {e:?}"));
                json!({"evaluation":e,"result":super::recover(&c["request"]).unwrap(),"diagnostics":super::inspect(&c["request"]).unwrap()})
            } else if c["kind"] == "render" {
                json!(
                    crate::legacy_model::table_text_styles::materialization::render(&c["request"])
                        .unwrap()
                )
            } else {
                crate::legacy_model::inline_output::trace(&c["request"])
                    .unwrap_or_else(|e| panic!("case {i} inline: {e:?}"))
            };
            if let Some(diff) = difference(&actual, &c["expected"], "result") {
                panic!("case {i} {}: {diff}", c["kind"]);
            }
        }
    }
    #[test]
    fn old_raw_character_engine_evaluation_recovery_and_diagnostics() {
        let f: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-table-engine.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for (i, c) in f["cases"].as_array().unwrap().iter().enumerate() {
            let e = evaluate(&c["request"]["table_input"], &mut Work(0))
                .unwrap_or_else(|e| panic!("eval {i}: {e:?}"));
            assert_eq!(
                e.record(),
                c["expected"]["evaluation"],
                "eval {i}: {}",
                c["label"]
            );
            assert_eq!(e.recover().unwrap(), c["expected"]["result"], "recover {i}");
            assert_eq!(
                inspect(&c["request"]).unwrap_or_else(|e| panic!("diag {i}: {e:?}")),
                c["expected"]["diagnostics"],
                "diag {i}"
            );
        }
    }
}
