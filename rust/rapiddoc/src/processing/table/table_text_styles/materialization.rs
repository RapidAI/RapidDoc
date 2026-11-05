//! 旧整表脚本样式渲染及 recover_table_result；阶段之间共享工作预算。
use super::{
    clusters, preparation, render_cell_parts_with_work, source_value, spacing_char, CellRender,
};
use crate::legacy_model::{
    table_native_candidate, table_native_engine, table_native_multiline::spatial,
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

fn original(result: &Value) -> Result<String, Error> {
    let html = result["html"].as_str().ok_or_else(invalid)?;
    if html.len() > rapiddoc_core::resources::limit(4_000_000) {
        return Err(limit());
    }
    Ok(html.to_owned())
}
pub fn render(v: &Value) -> Result<String, Error> {
    validate_tree(v)?;
    render_with_work(v, &mut Work(0))
}
pub(in crate::processing) fn render_with_work(v: &Value, work: &mut Work) -> Result<String, Error> {
    let result = &v["result"];
    let tight = &v["tight_bboxes"];
    let origins = &v["origins"];
    for map in [tight, origins] {
        if !map.is_null() && !map.is_object() {
            return Err(invalid());
        }
        if map
            .as_object()
            .is_some_and(|m| m.len() > crate::legacy_page::budget::records(20000))
        {
            return Err(limit());
        }
    }
    // 与旧入口一致：保留字体也不能绕过缺少侧映射时的原 HTML 回退。
    if [tight, origins]
        .iter()
        .any(|m| m.as_object().is_none_or(Map::is_empty))
        || ![
            "glyphs",
            "rows",
            "median_glyph_width",
            "median_glyph_height",
        ]
        .iter()
        .all(|k| result["text"].get(k).is_some())
    {
        return original(result);
    }
    let preserve = v
        .get("preserve_font_styles")
        .map_or(Ok(false), |b| b.as_bool().ok_or_else(invalid))?;
    let prepared = preparation::prepare_with_work(v, work)?;
    let input = &v["table_input"];
    let chars = input["chars"].as_array().ok_or_else(invalid)?;
    let mut by_source = Map::new();
    for (i, c) in chars.iter().enumerate() {
        if let Some(id) = c.get("char_idx").map_or(Some(i as i64), Value::as_i64) {
            by_source.entry(id.to_string()).or_insert_with(|| c.clone());
        }
    }
    let by_source = Value::Object(by_source);
    let mut cells = BTreeMap::new();
    let mut changed = preserve;
    for cell in prepared["cells"].as_array().unwrap() {
        work.charge(
            chars.len() + tight.as_object().unwrap().len() + origins.as_object().unwrap().len(),
        )?;
        let roles = clusters::cell_roles_with_work(
            &value_object!({"lines":cell["visual_lines"],"page_size":input["page_size"],"tight_bboxes":super::cell_side_map(tight,cell["chars"].as_array().unwrap()),"origins":super::cell_side_map(origins,cell["chars"].as_array().unwrap()),"fraction_members":cell["fraction_members"]}),
            work,
        )?;
        changed |= !roles.as_object().ok_or_else(invalid)?.is_empty();
        for pair in cell["glyphs"].as_array().unwrap().windows(2) {
            work.charge(1)?;
            let left = pair[0]["source_index"].as_i64();
            let right = pair[1]["source_index"].as_i64();
            if pair[0]["visual_row"] == pair[1]["visual_row"]
                && [left, right]
                    .iter()
                    .all(|&id| source_value(&roles, id).is_none_or(|r| r == "body"))
            {
                if let (Some(l), Some(r)) = (
                    source_value(&by_source, left),
                    source_value(&by_source, right),
                ) {
                    changed |= spatial::tight(
                        &spacing_char(l, tight, origins),
                        &spacing_char(r, tight, origins),
                    );
                }
            }
        }
        cells.insert(
            (cell["row"].as_i64().unwrap(), cell["col"].as_i64().unwrap()),
            (cell, roles),
        );
    }
    if !changed {
        return original(result);
    }
    table_native_candidate::render_with(result, |cell| {
        let key = (cell["row"].as_i64().unwrap(), cell["col"].as_i64().unwrap());
        let (prepared, roles) = cells.get(&key).ok_or_else(invalid)?;
        work.charge(
            chars.len() + tight.as_object().unwrap().len() + origins.as_object().unwrap().len(),
        )?;
        render_cell_parts_with_work(
            CellRender {
                cell,
                glyphs: &prepared["glyphs"],
                median_height: &result["text"]["median_glyph_height"],
                roles,
                chars: &by_source,
                tight,
                origins,
                preserve: Some(&Value::Bool(preserve)),
                possible: None,
            },
            work,
        )
    })
}
pub fn recover(v: &Value) -> Result<Value, Error> {
    recover_storage(std::borrow::Cow::Borrowed(v))
}
/// Production owns its assembled request; avoid cloning full page geometry again.
pub(crate) fn recover_owned(v: Value) -> Result<Value, Error> {
    recover_storage(std::borrow::Cow::Owned(v))
}
fn recover_storage(mut v: std::borrow::Cow<'_, Value>) -> Result<Value, Error> {
    validate_tree(&v)?;
    let mut work = Work(0);
    let result = match table_native_engine::recover_with_work(&v, &mut work) {
        Ok(result) => result,
        Err(error) => {
            return Ok(
                json!({"recovery_error":{"code":error.code.as_str(),"message":error.message}}),
            )
        }
    };
    if result.is_null() {
        return Ok(Value::Null);
    }
    let request = v.to_mut();
    request["result"] = result;
    let content = render_with_work(request, &mut work)?;
    let result = request["result"].take();
    Ok(json!({"content":content,"result":result}))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn corpus() -> Value {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-table-materialization.json");
        serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap()
    }
    #[test]
    fn old_complete_table_materialization() {
        let f = corpus();
        for c in f["renders"].as_array().unwrap() {
            assert_eq!(
                render(&c["request"]).unwrap_or_else(|e| panic!("{}: {e}", c["label"])),
                c["expected"].as_str().unwrap(),
                "{}",
                c["label"]
            );
        }
        for c in f["recoveries"].as_array().unwrap() {
            assert_eq!(
                recover(&c["request"]).unwrap_or_else(|e| panic!("{}: {e}", c["label"])),
                c["expected"],
                "{}",
                c["label"]
            );
        }
        for c in f["real"].as_array().unwrap() {
            assert_eq!(recover(&c["request"]).unwrap(), c["expected"]);
        }
    }
    #[test]
    fn materialization_does_not_reset_work_or_wrap_render_errors() {
        let f = corpus();
        let v = &f["renders"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["request"]["preserve_font_styles"] == true)
            .unwrap()["request"];
        assert_eq!(
            render_with_work(v, &mut Work(9_999_999)).unwrap_err().code,
            rapiddoc_core::ErrorCode::ResourceLimit
        );
        let v = json!({"table_input":{}});
        assert_eq!(
            recover(&v).unwrap()["recovery_error"]["code"],
            "invalid_input"
        );
    }
}
