//! 旧跨正文草稿排除；只消费草稿几何/成员投影，不调用物化。
use super::table_members::{Lookup, Members};
use super::path_graphics::{array, bbox, integer, num, union, Box4};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
use std::collections::BTreeSet;
fn optional<'a>(v: &'a Value, k: &str, max: usize) -> Result<&'a [Value], Error> {
    v.get(k).map_or(Ok(&[]), |v| array(v, max))
}
fn flag(v: &Value, k: &str) -> Result<bool, Error> {
    v.get(k)
        .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))
}
struct Line {
    b: Box4,
    id: i64,
    eligible: bool,
}
struct Rule {
    b: Box4,
    h: bool,
    v: bool,
}
pub fn exclude(v: &Value) -> Result<Value, Error> {
    exclude_with_work(v, &mut Work(0))
}
pub(crate) fn exclude_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    validate_tree(v)?;
    let candidates: Vec<_> = array(&v["candidates"], 5000)?.iter().collect();
    exclude_inputs(&v["source"], &candidates, false, None, work)
}
pub(super) fn exclude_indices(
    source: &Value,
    candidates: &[&Value],
    work: &mut Work,
) -> Result<Value, Error> {
    exclude_inputs(source, candidates, true, None, work)
}
pub(super) fn exclude_lazy_members(
    source: &Value,
    candidates: &[&Value],
    members: &mut dyn FnMut(usize) -> Members,
    work: &mut Work,
) -> Result<Value, Error> {
    exclude_inputs(source, candidates, true, Some(members), work)
}
fn exclude_inputs(
    source: &Value,
    candidates: &[&Value],
    indices_only: bool,
    mut member_provider: Option<&mut dyn FnMut(usize) -> Members>,
    work: &mut Work,
) -> Result<Value, Error> {
    source.as_object().ok_or_else(invalid)?;
    let mut grids = Vec::new();
    for c in candidates {
        match c["kind"].as_str() {
            Some("table") => {
                if c.get("angle").map_or(Ok(0.), num)? == 0. {
                    grids.push(bbox(
                        c.get("core_bbox")
                            .filter(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
                            .unwrap_or(&c["bbox"]),
                    )?);
                }
            }
            Some("draft") => {}
            _ => return Err(invalid()),
        }
    }
    let line_input = optional(source, "lines", 4096)?;
    let rule_input = optional(source, "drawing_lines", 5000)?;
    work.charge(candidates.len() + 1)?;
    if grids.is_empty() {
        return Ok(if indices_only {
            json!({"rejected_indices":[]})
        } else {
            json!({"candidates":candidates,"rejected_indices":[]})
        });
    }
    let mut lines = Vec::new();
    for l in line_input {
        let text = l["text"].as_str().ok_or_else(invalid)?;
        if text.chars().count() > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        work.charge(text.len() + 1)?;
        let semantic = l
            .get("semantic_type")
            .filter(|v| !v.is_null())
            .map_or(Ok(""), |v| v.as_str().ok_or_else(invalid))?;
        let weight = l
            .get("dominant_font_weight")
            .filter(|v| !v.is_null())
            .map_or(Ok(400.), num)?;
        lines.push(Line {
            b: bbox(&l["bbox"])?,
            id: integer(l, "source_index", 0)?,
            eligible: l.get("angle").map_or(Ok(0.), num)? == 0.
                && !flag(l, "caption_start")?
                && !flag(l, "formula_candidate_only")?
                && semantic.is_empty()
                && (if weight == 0. { 400. } else { weight }) < 600.
                && super::graphic_captions::words(text) >= 6,
        });
    }
    let mut rules = Vec::new();
    for r in rule_input {
        let o = r["orientation"].as_str().ok_or_else(invalid)?;
        rules.push(Rule {
            b: bbox(&r["bbox"])?,
            h: o == "horizontal",
            v: o == "vertical",
        });
    }
    let mut rejected = Vec::new();
    let mut retained = Vec::new();
    let (mut boundary_total, mut member_total) = (0, 0);
    for (index, c) in candidates.iter().enumerate() {
        if c["kind"] == "table" {
            if !indices_only {
                retained.push(*c);
            }
            continue;
        }
        let angle = c.get("angle").map_or(Ok(0.), num)?;
        if angle != 0. || c.get("caption").is_some_and(|v| !v.is_null()) {
            if !indices_only {
                retained.push(*c);
            }
            continue;
        }
        let em = num(&c["median_height"])?;
        let boundaries = optional(c, "boundaries", 20000)?;
        boundary_total += boundaries.len();
        if boundary_total > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        if em <= 0. || boundaries.is_empty() {
            if !indices_only {
                retained.push(*c);
            }
            continue;
        }
        let boundaries = boundaries.iter().map(bbox).collect::<Result<Vec<_>, _>>()?;
        work.charge(boundaries.len() + 1)?;
        let first = boundaries.iter().map(|b| b.n[1]).reduce(f64::min).unwrap();
        let last = boundaries.iter().map(|b| b.n[3]).reduce(f64::max).unwrap();
        let mut bound = boundaries[0].clone();
        for b in &boundaries[1..] {
            bound = union(bound, b);
        }
        let mut members = if member_provider.is_none() {
            let raw = optional(c, "member_source_indices", 20000)?;
            member_total += raw.len();
            if member_total > rapiddoc_core::resources::limit(20000) {
                return Err(limit());
            }
            work.charge(raw.len() * 16 + 1)?;
            let members = raw
                .iter()
                .map(|v| v.as_i64().ok_or_else(invalid))
                .collect::<Result<BTreeSet<_>, _>>()?;
            Some(Members::from(members))
        } else {
            None
        };
        let mut conflict = false;
        for core in &grids {
            work.charge(1)?;
            let width = core.n[2] - core.n[0];
            let height = core.n[3] - core.n[1];
            if width < 8. * em
                || height < 4. * em
                || core.n[1] - first < 5. * em
                || !(core.n[1] - em <= last && last <= core.n[3] + em)
                || super::graphic_captions::axis(&bound, core, 0) < 0.9
            {
                continue;
            }
            work.charge(rules.len() * 2 + 1)?;
            let mut xs: Vec<_> = rules
                .iter()
                .filter(|r| {
                    r.v && core.n[0] - 0.5 * em <= (r.b.n[0] + r.b.n[2]) / 2.
                        && (r.b.n[0] + r.b.n[2]) / 2. <= core.n[2] + 0.5 * em
                        && r.b.n[1] <= core.n[1] + 0.5 * em
                        && r.b.n[3] >= core.n[3] - 0.5 * em
                })
                .map(|r| (r.b.n[0] + r.b.n[2]) / 2.)
                .collect();
            work.charge(xs.len() * 16 + 1)?;
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut tracks = Vec::new();
            for x in xs {
                if tracks.last().is_none_or(|p| x - p > 0.3 * em) {
                    tracks.push(x);
                }
            }
            let hs: Vec<_> = rules
                .iter()
                .filter(|r| r.h && super::graphic_captions::axis(&r.b, core, 0) >= 0.9)
                .collect();
            work.charge(hs.len() * 2 + 1)?;
            if tracks.len() < 3
                || ![core.n[1], core.n[3]].into_iter().all(|y| {
                    hs.iter()
                        .any(|r| ((r.b.n[1] + r.b.n[3]) / 2. - y).abs() <= 0.5 * em)
                })
            {
                continue;
            }
            work.charge(lines.len() * 16 + 1)?;
            let mut ys: Vec<_> = lines
                .iter()
                .filter(|l| {
                    let y = (l.b.n[1] + l.b.n[3]) / 2.;
                    l.eligible
                        && first + 0.5 * em < y
                        && y < core.n[1] - em
                        && l.b.n[2] - l.b.n[0] >= 0.65 * width
                        && super::graphic_captions::axis(&l.b, core, 0) >= 0.9
                        && members
                            .get_or_insert_with(|| member_provider.as_deref_mut().unwrap()(index))
                            .contains(&l.id)
                })
                .map(|l| (l.b.n[1] + l.b.n[3]) / 2.)
                .collect();
            work.charge(ys.len() * 16 + 1)?;
            ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut rows = Vec::new();
            for y in ys {
                if rows.last().is_none_or(|p| y - p > 0.6 * em) {
                    rows.push(y);
                }
            }
            if rows.len() >= 3 {
                conflict = true;
                break;
            }
        }
        if conflict {
            rejected.push(index);
        } else {
            if !indices_only {
                retained.push(*c);
            }
        }
    }
    let out = json!({"candidates":retained,"rejected_indices":rejected});
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
    use super::*;
    #[test]
    fn lazy_members_preserve_real_prose_rejection_and_skip_irrelevant_queries() {
        let _scope = crate::legacy_page::budget::host_page();
        let source = json!({"lines":(1..=3).map(|i| json!({"source_index":i,
            "text":"one two three four five six","bbox":[0,i*10,100,i*10+3]})).collect::<Vec<_>>(),
            "drawing_lines":[
                {"orientation":"vertical","bbox":[0,60,0,120]},
                {"orientation":"vertical","bbox":[50,60,50,120]},
                {"orientation":"vertical","bbox":[100,60,100,120]},
                {"orientation":"horizontal","bbox":[0,60,100,60]},
                {"orientation":"horizontal","bbox":[0,120,100,120]}]});
        let draft = json!({"kind":"draft","angle":0,"caption":null,"median_height":5,
            "boundaries":[[0,0,100,0],[0,120,100,120]],"member_source_indices":[1,2,3]});
        let mut two = draft.clone();
        two["member_source_indices"] = json!([1, 2]);
        let mut rotated = draft.clone();
        rotated["angle"] = json!(90);
        let mut lower = draft.clone();
        lower["boundaries"] = json!([[0, 70, 100, 70], [0, 120, 100, 120]]);
        let candidates = vec![
            draft,
            two,
            rotated,
            lower,
            json!({"kind":"table","bbox":[0,60,100,120],"core_bbox":[0,60,100,120]}),
        ];
        let refs = candidates.iter().collect::<Vec<_>>();
        let expected = exclude_indices(&source, &refs, &mut Work(0)).unwrap();
        assert_eq!(expected["rejected_indices"], json!([0]));
        let metadata = candidates
            .iter()
            .cloned()
            .map(|mut c| {
                c.as_object_mut().unwrap().remove("member_source_indices");
                c
            })
            .collect::<Vec<_>>();
        let mut calls = Vec::new();
        let actual = exclude_lazy_members(
            &source,
            &metadata.iter().collect::<Vec<_>>(),
            &mut |index| {
                calls.push(index);
                Members::from(
                    candidates[index]["member_source_indices"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_i64().unwrap())
                        .collect::<BTreeSet<_>>(),
                )
            },
            &mut Work(0),
        )
        .unwrap();
        assert_eq!(actual, expected);
        assert_eq!(calls, vec![0, 1]);
    }
    #[test]
    fn old_prose_conflicts() {
        let f: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-prose-rule-conflicts.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                exclude(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
    #[test]
    fn shared_budget() {
        let v = json!({"source":{},"candidates":[]});
        assert!(exclude_with_work(&v, &mut Work(10_000_000)).is_err());
        assert!(exclude(&v).is_ok());
    }
}
