//! 旧规则表核心、表题续行和注释记录；表注自动收集是上游阶段。
use super::table_members::{Lookup, Members};
use super::path_graphics::{array, bbox, integer, num, union, Box4};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
use std::collections::BTreeSet;
#[derive(Clone)]
pub(super) struct Fragment {
    pub(super) raw: Value,
    pub(super) local: Box4,
    pub(super) b: Box4,
    pub(super) id: i64,
}
#[derive(Clone)]
pub(super) struct Row {
    pub(super) fragments: Vec<Fragment>,
    pub(super) b: Box4,
    pub(super) center: f64,
    pub(super) visual: Value,
}
pub(super) fn row(v: &Value, total: &mut usize) -> Result<Row, Error> {
    let mut fragments = Vec::new();
    for f in array(&v["fragments"], 20000)? {
        *total += 1;
        if *total > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        fragments.push(Fragment {
            raw: f.clone(),
            local: bbox(&f["local_bbox"])?,
            b: bbox(&f["bbox"])?,
            id: integer(f, "line_index", 0)?,
        });
    }
    Ok(Row {
        fragments,
        b: bbox(&v["bbox"])?,
        center: num(&v["center_y"])?,
        visual: v.get("visual_row_id").cloned().unwrap_or(Value::Null),
    })
}
pub(super) fn clipped(
    row: &Row,
    corridor: &Box4,
    margin: f64,
    work: &mut Work,
) -> Result<Option<Row>, Error> {
    work.charge(row.fragments.len() + 1)?;
    let mut fs: Vec<_> = row
        .fragments
        .iter()
        .filter(|f| {
            let x = (f.local.n[0] + f.local.n[2]) / 2.;
            corridor.n[0] - margin <= x && x <= corridor.n[2] + margin
        })
        .cloned()
        .collect();
    if fs.is_empty() {
        return Ok(None);
    }
    work.charge(fs.len() * 16 + 1)?;
    fs.sort_by(|a, b| a.local.n[0].partial_cmp(&b.local.n[0]).unwrap());
    let center = super::table_native_candidate::sum(
        &fs.iter()
            .map(|f| (f.local.n[1] + f.local.n[3]) / 2.)
            .collect::<Vec<_>>(),
    ) / fs.len() as f64;
    let mut b = fs[0].local.clone();
    for f in &fs[1..] {
        b = union(b, &f.local);
    }
    Ok(Some(Row {
        fragments: fs,
        b,
        center,
        visual: row.visual.clone(),
    }))
}
pub(super) fn annotation(
    kind: &str,
    rows: &[Row],
    excluded: &dyn Lookup,
    local: Option<&Box4>,
    work: &mut Work,
) -> Result<Option<Value>, Error> {
    annotation_refs(
        kind,
        &rows.iter().collect::<Vec<_>>(),
        excluded,
        local,
        work,
    )
}
pub(super) fn annotation_refs(
    kind: &str,
    rows: &[&Row],
    excluded: &dyn Lookup,
    local: Option<&Box4>,
    work: &mut Work,
) -> Result<Option<Value>, Error> {
    let mut boxes: Vec<(i64, Box4)> = Vec::new();
    let mut positions = std::collections::HashMap::<i64, usize>::new();
    for r in rows {
        for f in &r.fragments {
            work.charge(boxes.len() + 16)?;
            if excluded.contains(&f.id)
                || local.is_some_and(|b| {
                    f.local.n[3] > b.n[1] && super::graphic_captions::axis(&f.local, b, 0) >= 0.05
                })
            {
                continue;
            }
            if let Some(&index) = positions.get(&f.id) {
                boxes[index].1 = union(boxes[index].1.clone(), &f.b);
            } else {
                positions.insert(f.id, boxes.len());
                boxes.push((f.id, f.b.clone()));
            }
        }
    }
    let Some((_, first)) = boxes.first() else {
        return Ok(None);
    };
    let mut b = first.clone();
    for (_, v) in &boxes[1..] {
        b = union(b, v);
    }
    let ids: BTreeSet<_> = boxes.iter().map(|(id, _)| *id).collect();
    let line_boxes = boxes
        .into_iter()
        .map(|(id, b)| Value::Array(vec![Value::from(id), Value::from(b.raw)]))
        .collect::<Vec<_>>();
    Ok(Some(
        value_object!({"kind":kind,"bbox":b.raw,"line_indices":json!(ids),"line_bboxes":Value::Array(line_boxes)}),
    ))
}
pub(super) fn output(r: &Row) -> Value {
    json!({"fragments":r.fragments.iter().map(|f|f.raw.clone()).collect::<Vec<_>>(),"bbox":r.b.raw,"center_y":r.center,"visual_row_id":r.visual})
}
fn indices(v: Option<&Value>, count: usize) -> Result<Vec<usize>, Error> {
    let Some(v) = v else {
        return Ok(Vec::new());
    };
    array(v, 4096)?
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|i| *i < count)
                .ok_or_else(invalid)
        })
        .collect()
}
pub fn prepare(v: &Value) -> Result<Value, Error> {
    prepare_with_work(v, &mut Work(0))
}
pub(crate) fn prepare_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    prepare_with_core(v, None, work)
}
pub(super) fn prepare_with_core(
    v: &Value,
    core_override: Option<&[Value]>,
    work: &mut Work,
) -> Result<Value, Error> {
    prepare_with_rows(v, core_override, None, None, work)
}
pub(super) fn prepare_with_rows(
    v: &Value,
    core_override: Option<&[Value]>,
    prepared_rows: Option<&[Row]>,
    prepared_core: Option<&[&Row]>,
    work: &mut Work,
) -> Result<Value, Error> {
    let prepared = prepare_typed(v, core_override, prepared_rows, prepared_core, work)?;
    let out = json!({"rule_bbox":prepared.rule.raw,"core_local_bbox":prepared.core.raw,"core_line_indices":prepared.ids,"caption_rows":prepared.caption_rows.iter().map(output).collect::<Vec<_>>(),"annotations":prepared.annotations});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(out)
}
pub(super) struct Prepared {
    pub(super) rule: Box4,
    pub(super) core: Box4,
    pub(super) ids: Members,
    pub(super) caption_rows: Vec<Row>,
    pub(super) annotations: Vec<Value>,
}
pub(super) fn prepare_typed(
    v: &Value,
    core_override: Option<&[Value]>,
    prepared_rows: Option<&[Row]>,
    prepared_core: Option<&[&Row]>,
    work: &mut Work,
) -> Result<Prepared, Error> {
    prepare_typed_members(v, core_override, prepared_rows, prepared_core, None, work)
}
pub(super) fn prepare_typed_members(
    v: &Value,
    core_override: Option<&[Value]>,
    prepared_rows: Option<&[Row]>,
    prepared_core: Option<&[&Row]>,
    prepared_members: Option<Members>,
    work: &mut Work,
) -> Result<Prepared, Error> {
    validate_tree(v)?;
    let em = num(&v["median_height"])?;
    if em <= 0. {
        return Err(invalid());
    }
    let mut total = 0;
    let owned_rows;
    let rows = if let Some(rows) = prepared_rows {
        total += rows.iter().map(|r| r.fragments.len()).sum::<usize>();
        rows
    } else {
        owned_rows = array(&v["rows"], 4096)?
            .iter()
            .map(|r| row(r, &mut total))
            .collect::<Result<Vec<_>, _>>()?;
        &owned_rows
    };
    let rules = array(&v["rule_bboxes"], 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    if rules.is_empty() {
        return Err(invalid());
    }
    let owned_core;
    let core_rows: Vec<&Row> = if let Some(core) = prepared_core {
        core.to_vec()
    } else if let Some(core) = core_override {
        if core.len() > rapiddoc_core::resources::limit(4096) {
            return Err(limit());
        }
        owned_core = core
            .iter()
            .map(|r| row(r, &mut total))
            .collect::<Result<Vec<_>, _>>()?;
        owned_core.iter().collect()
    } else {
        indices(v.get("core_row_indices"), rows.len())?
            .into_iter()
            .map(|i| &rows[i])
            .collect()
    };
    if core_rows.is_empty() {
        return Err(invalid());
    }
    let foot = indices(v.get("footnote_row_indices"), rows.len())?;
    work.charge(total + rules.len() + rows.len() + 1)?;
    let mut rule = rules[0].clone();
    for b in &rules[1..] {
        rule = union(rule, b);
    }
    let mut body = core_rows[0].b.clone();
    let mut source_ids = Vec::new();
    for row in core_rows {
        work.charge(row.fragments.len() * 16 + 1)?;
        body = union(body, &row.b);
        if prepared_members.is_none() {
            source_ids.extend(row.fragments.iter().map(|f| f.id));
        }
    }
    // Bulk construction preserves the same ordered unique IDs without repeated
    // tree insertion for each overlapping row interval.
    let ids = prepared_members
        .unwrap_or_else(|| Members::from(source_ids.into_iter().collect::<BTreeSet<_>>()));
    let core_box = union(rule.clone(), &body);
    let caption_id = v
        .get("caption_source_index")
        .filter(|v| !v.is_null())
        .map(|v| v.as_i64().ok_or_else(invalid))
        .transpose()?;
    let mut caption_rows = Vec::new();
    if let Some(id) = caption_id {
        let mut start = None;
        for (i, r) in rows.iter().enumerate() {
            work.charge(r.fragments.len() + 1)?;
            if r.fragments.iter().any(|f| f.id == id) {
                start = Some(i);
                break;
            }
        }
        if let Some(start) = start {
            let mut prior: Option<Box4> = None;
            for (i, r) in rows.iter().enumerate().skip(start) {
                if let Some(r) = clipped(r, &rule, 2. * em, work)? {
                    if i > start && r.center >= rule.n[1] {
                        break;
                    }
                    if prior
                        .as_ref()
                        .is_some_and(|b| (r.b.n[1] - b.n[3]).max(0.) > 2. * em)
                    {
                        break;
                    }
                    prior = Some(r.b.clone());
                    caption_rows.push(r);
                }
            }
            if caption_rows
                .last()
                .is_none_or(|r| rule.n[1] - r.b.n[3] > 2. * em)
            {
                caption_rows.clear();
            }
        }
    }
    let mut foot_rows = Vec::new();
    let mut foot_fragments = 0;
    for i in foot {
        foot_fragments += rows[i].fragments.len();
        if foot_fragments > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        work.charge(rows[i].fragments.len() + 1)?;
        foot_rows.push(rows[i].clone());
    }
    let mut annotations = Vec::new();
    if let Some(a) = annotation("caption", &caption_rows, &ids, Some(&core_box), work)? {
        annotations.push(a);
    }
    if let Some(a) = annotation("footnote", &foot_rows, &ids, None, work)? {
        annotations.push(a);
    }
    Ok(Prepared {
        rule,
        core: core_box,
        ids,
        caption_rows,
        annotations,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_rule_annotation_preparation() {
        let f: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-rule-annotation-preparation.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                prepare(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
    #[test]
    fn shared_budget() {
        let v = json!({"rows":[{"fragments":[],"bbox":[0,0,10,10],"center_y":5}],"rule_bboxes":[[0,0,10,10]],"core_row_indices":[0],"median_height":10});
        assert!(prepare_with_work(&v, &mut Work(10_000_000)).is_err());
        assert!(prepare(&v).is_ok());
    }
}
