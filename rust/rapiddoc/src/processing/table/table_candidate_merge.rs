//! 旧已物化候选合并；注释行框用有序[id,bbox]对跨JSON边界。
use super::table_members::{Lookup, Members};
use super::path_graphics::{array, bbox, num, overlap_smaller, union, Box4};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
use std::collections::BTreeSet;
#[derive(Clone)]
struct Annotation {
    raw: Value,
    kind: String,
    b: Box4,
    ids: BTreeSet<i64>,
    boxes: Vec<(i64, Box4)>,
}
pub(super) struct Candidate {
    raw: Value,
    b: Box4,
    local: Box4,
    core: Option<Box4>,
    angle: f64,
    score: f64,
    ids: Members,
    annotations: Vec<Annotation>,
}
pub(super) fn from_parts(
    b: Box4,
    local: Box4,
    core: Box4,
    angle: i64,
    score: f64,
    ids: Members,
    annotations: Vec<&Value>,
    work: &mut Work,
) -> Result<Candidate, Error> {
    let value = json!({"bbox":b.raw,"local_bbox":local.raw,"core_bbox":core.raw,"angle":angle,"score":score,"line_indices":[],"annotations":[],"inferred_grid":[],"preserve_inline_font_styles":false,"inferred_grid_authoritative":false});
    let mut candidate = parse(&value, &mut 0, work)?;
    candidate.ids = ids;
    candidate.annotations = parse_annotations(annotations, &mut 0, work)?;
    work.charge(candidate.annotations.len())?;
    Ok(candidate)
}
#[cfg(test)]
pub(super) fn into_value(c: Candidate) -> Value {
    output(c)
}
fn ids(v: &Value, total: &mut usize) -> Result<BTreeSet<i64>, Error> {
    let a = array(v, 20000)?;
    *total += a.len();
    if *total > crate::legacy_page::budget::current().candidate_members {
        return Err(limit());
    }
    a.iter().map(|v| v.as_i64().ok_or_else(invalid)).collect()
}
fn parse_annotations<'a>(
    values: impl IntoIterator<Item = &'a Value>,
    total: &mut usize,
    work: &mut Work,
) -> Result<Vec<Annotation>, Error> {
    let mut annotations = Vec::new();
    for a in values {
        let kind = a["kind"].as_str().ok_or_else(invalid)?;
        if !matches!(kind, "caption" | "footnote") {
            return Err(invalid());
        }
        let ai = ids(&a["line_indices"], total)?;
        let mut boxes = Vec::new();
        let mut seen = BTreeSet::new();
        for pair in array(&a["line_bboxes"], 20000)? {
            let pair = array(pair, 2)?;
            if pair.len() != 2 {
                return Err(invalid());
            }
            let id = pair[0].as_i64().ok_or_else(invalid)?;
            if !seen.insert(id) {
                return Err(invalid());
            }
            boxes.push((id, bbox(&pair[1])?));
        }
        *total += boxes.len();
        if *total > crate::legacy_page::budget::current().candidate_members {
            return Err(limit());
        }
        work.charge(ai.len() * 16 + boxes.len() * 16 + 1)?;
        annotations.push(Annotation {
            raw: a.clone(),
            kind: kind.into(),
            b: bbox(&a["bbox"])?,
            ids: ai,
            boxes,
        });
    }
    Ok(annotations)
}
fn parse(v: &Value, total: &mut usize, work: &mut Work) -> Result<Candidate, Error> {
    v.as_object().ok_or_else(invalid)?;
    let annotations = parse_annotations(array(&v["annotations"], 2000)?, total, work)?;
    let members = ids(&v["line_indices"], total)?;
    work.charge(members.len() * 16 + annotations.len() + 1)?;
    Ok(Candidate {
        raw: v.clone(),
        b: bbox(&v["bbox"])?,
        local: bbox(&v["local_bbox"])?,
        core: v
            .get("core_bbox")
            .filter(|v| !v.is_null())
            .map(bbox)
            .transpose()?,
        angle: num(&v["angle"])?,
        score: num(&v["score"])?,
        ids: members.into(),
        annotations,
    })
}
fn annotations(
    target: &mut Candidate,
    incoming: Vec<Annotation>,
    work: &mut Work,
) -> Result<(), Error> {
    for a in incoming {
        work.charge(target.annotations.len() + 1)?;
        if let Some(existing) = target.annotations.iter_mut().find(|b| b.kind == a.kind) {
            existing.b = union(existing.b.clone(), &a.b);
            work.charge(a.ids.len() * 16 + 1)?;
            existing.ids.extend(a.ids);
            for (id, b) in a.boxes {
                work.charge(existing.boxes.len() + 1)?;
                if let Some((_, prior)) = existing.boxes.iter_mut().find(|(i, _)| *i == id) {
                    *prior = union(prior.clone(), &b);
                } else {
                    existing.boxes.push((id, b));
                }
            }
        } else {
            target.annotations.push(a);
        }
    }
    let mut retained = Vec::new();
    for mut a in std::mem::take(&mut target.annotations) {
        work.charge(a.ids.len() * 16 + a.boxes.len() * 16 + 1)?;
        a.ids.retain(|id| !target.ids.contains(id));
        a.boxes.retain(|(id, _)| a.ids.contains(id));
        if a.ids.is_empty() {
            continue;
        }
        if let Some((_, b)) = a.boxes.first() {
            let mut bounds = b.clone();
            for (_, b) in &a.boxes[1..] {
                bounds = union(bounds, b);
            }
            a.b = bounds;
        }
        retained.push(a);
    }
    target.annotations = retained;
    Ok(())
}
fn output(mut c: Candidate) -> Value {
    c.raw["bbox"] = json!(c.b.raw);
    c.raw["local_bbox"] = json!(c.local.raw);
    c.raw["core_bbox"] = c.core.map_or(Value::Null, |b| json!(b.raw));
    c.raw["line_indices"] = json!(c.ids);
    c.raw["annotations"] = json!(c
        .annotations
        .into_iter()
        .map(|mut a| {
            a.raw["bbox"] = json!(a.b.raw);
            a.raw["line_indices"] = json!(a.ids);
            a.raw["line_bboxes"] = json!(a
                .boxes
                .into_iter()
                .map(|(id, b)| json!([id, b.raw]))
                .collect::<Vec<_>>());
            a.raw
        })
        .collect::<Vec<_>>());
    c.raw
}
fn enclosed_annotations(
    target: &mut Candidate,
    incoming: Vec<Annotation>,
    work: &mut Work,
) -> Result<(), Error> {
    let Some(core) = &target.core else {
        return Ok(());
    };
    let bounds = &target.b.n;
    let mut retained = Vec::new();
    for mut annotation in incoming {
        work.charge(annotation.boxes.len().saturating_mul(16) + 1)?;
        annotation.boxes.retain(|(id, b)| {
            let b = &b.n;
            let (x, y) = ((b[0] + b[2]) / 2., (b[1] + b[3]) / 2.);
            annotation.ids.contains(id)
                && !target.ids.contains(id)
                && bounds[0] <= b[0]
                && b[0] <= b[2]
                && b[2] <= bounds[2]
                && bounds[1] <= b[1]
                && b[1] <= b[3]
                && b[3] <= bounds[3]
                && !(core.n[0] <= x && x <= core.n[2] && core.n[1] <= y && y <= core.n[3])
        });
        let Some((_, first)) = annotation.boxes.first() else {
            continue;
        };
        annotation.b = annotation
            .boxes
            .iter()
            .skip(1)
            .fold(first.clone(), |b, (_, next)| union(b, next));
        annotation.ids = annotation.boxes.iter().map(|(id, _)| *id).collect();
        retained.push(annotation);
    }
    if !retained.is_empty() {
        annotations(target, retained, work)?;
    }
    Ok(())
}
pub fn merge(v: &Value) -> Result<Value, Error> {
    merge_with_work(v, &mut Work(0))
}
pub(crate) fn merge_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    validate_tree(v)?;
    let mut total = 0;
    let mut candidates = array(&v["candidates"], 5000)?
        .iter()
        .map(|v| parse(v, &mut total, work))
        .collect::<Result<Vec<_>, _>>()?;
    work.charge(candidates.len() * 16 + 1)?;
    // Bounded scores are exactly representable as double integers; retain winner's original JSON value.
    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut merged: Vec<Candidate> = Vec::new();
    for c in candidates {
        merge_candidate(&mut merged, c, work)?;
    }
    work.charge(merged.len() * 16 + 1)?;
    merged.sort_by(|a, b| {
        a.b.n[1]
            .partial_cmp(&b.b.n[1])
            .unwrap()
            .then(a.b.n[0].partial_cmp(&b.b.n[0]).unwrap())
    });
    let out = json!({"candidates":merged.into_iter().map(output).collect::<Vec<_>>()});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    #[test]
    fn enclosed_weak_annotations_preserve_complete_grid() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/upstream_068/caption_grid_merge.json.gz");
        let fixture: Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            std::fs::File::open(path).unwrap(),
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let actual = merge(&case["request"]).unwrap();
            if let Some(diff) =
                super::super::native_document::tests::diff(&actual, &case["expected"], "result")
            {
                panic!("{}: {diff}", case["label"]);
            }
        }
    }
    use super::*;
    #[test]
    fn original_pdf_candidate_capacity() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let f: Value = serde_json::from_slice(
            &std::fs::read(
                root.join("build/migration/native-pdf-candidate-capacity-canonical-members.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert!(merge(&c["request"]).is_err());
            let _scope = crate::legacy_page::budget::document(9);
            let got = merge(&c["request"]).unwrap();
            if let Some(d) = super::super::native_document::tests::diff(&got, &c["expected"], "") {
                panic!("{}: {d}", c["label"])
            }
        }
    }
    #[test]
    fn streaming_merge_matches_frozen_annotation_arbitration() {
        let _scope = crate::legacy_page::budget::host_page();
        let f: Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!(
                "../../../../../tests/fixtures/upstream_068/caption_grid_merge.json.gz"
            )[..],
        ))
        .unwrap();
        for case in f["cases"].as_array().unwrap() {
            let mut rows = case["request"]["candidates"].as_array().unwrap().clone();
            rows.sort_by(|a, b| {
                b["score"]
                    .as_f64()
                    .unwrap()
                    .partial_cmp(&a["score"].as_f64().unwrap())
                    .unwrap()
            });
            let mut merger = OrderedMerger::new();
            for row in rows {
                merger.add(&row, &mut Work(0)).unwrap();
            }
            assert_eq!(
                merger.finish(&mut Work(0)).unwrap(),
                case["expected"],
                "{}",
                case["label"]
            );
        }
    }
    #[test]
    fn old_materialized_merge() {
        let f: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-table-candidate-merge.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                merge(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
    #[test]
    fn shared_budget() {
        let v = json!({"candidates":[]});
        assert!(merge_with_work(&v, &mut Work(10_000_000)).is_err());
        assert!(merge(&v).is_ok());
    }
}

/// Consume independently materialized candidates in stable descending score order.
/// Shares the reference arbitration; only candidate lifetime and allocation change.
pub(super) struct OrderedMerger {
    merged: Vec<Candidate>,
    total: usize,
}
impl OrderedMerger {
    pub(super) fn new() -> Self {
        Self {
            merged: Vec::new(),
            total: 0,
        }
    }
    pub(super) fn add(&mut self, value: &Value, work: &mut Work) -> Result<(), Error> {
        let c = parse(value, &mut self.total, work)?;
        merge_candidate(&mut self.merged, c, work)
    }
    pub(super) fn add_candidate(&mut self, c: Candidate, work: &mut Work) -> Result<(), Error> {
        merge_candidate(&mut self.merged, c, work)
    }
    pub(super) fn finish(mut self, work: &mut Work) -> Result<Value, Error> {
        work.charge(self.merged.len() * 16 + 1)?;
        self.merged.sort_by(|a, b| {
            a.b.n[1]
                .partial_cmp(&b.b.n[1])
                .unwrap()
                .then(a.b.n[0].partial_cmp(&b.b.n[0]).unwrap())
        });
        Ok(json!({"candidates":self.merged.into_iter().map(output).collect::<Vec<_>>()}))
    }
}

fn merge_candidate(
    merged: &mut Vec<Candidate>,
    c: Candidate,
    work: &mut Work,
) -> Result<(), Error> {
    work.charge(merged.len() + 1)?;
    let Some(target) = merged
        .iter_mut()
        .find(|t| t.angle == c.angle && overlap_smaller(&c.b, &t.b) >= 0.2)
    else {
        merged.push(c);
        return Ok(());
    };
    if target.score >= 100. && c.score < 100. {
        enclosed_annotations(target, c.annotations, work)?;
        return Ok(());
    }
    target.b = union(target.b.clone(), &c.b);
    target.local = union(target.local.clone(), &c.local);
    target.core = match (target.core.take(), c.core) {
        (None, b) => b,
        (Some(a), None) => Some(a),
        (Some(a), Some(b)) => Some(union(a, &b)),
    };
    work.charge(c.ids.len() * 16 + 1)?;
    target.ids.union_with(&c.ids);
    annotations(target, c.annotations, work)?;

    Ok(())
}
