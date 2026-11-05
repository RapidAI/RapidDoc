//! 旧has_native_closed_grid；上游候选显式输入，横纵边界和文字占用均由实际源证明。
use super::path_graphics::{array, bbox, integer, num, Box4};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
use std::collections::BTreeSet;
fn optional<'a>(v: &'a Value, k: &str, max: usize) -> Result<&'a [Value], Error> {
    v.get(k).map_or(Ok(&[]), |v| array(v, max))
}
fn core(v: &Value) -> Result<Box4, Error> {
    let b = v
        .get("core_bbox")
        .filter(|v| !v.is_null() && v.as_array().is_none_or(|a| !a.is_empty()))
        .unwrap_or(&v["bbox"]);
    bbox(b)
}
struct Line {
    b: Box4,
    id: i64,
    height: f64,
}
struct Rule {
    b: Box4,
    horizontal: bool,
    vertical: bool,
}
pub fn inspect(v: &Value) -> Result<Value, Error> {
    inspect_with_work(v, &mut Work(0))
}
pub(crate) fn inspect_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    validate_tree(v)?;
    let source = &v["source"];
    source.as_object().ok_or_else(invalid)?;
    let candidates = array(&v["candidates"], 5000)?;
    let allow = v
        .get("allow_two_by_two")
        .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))?;
    let forms = optional(source, "form_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let mut lines = Vec::new();
    for l in optional(source, "lines", 4096)? {
        lines.push(Line {
            b: bbox(&l["bbox"])?,
            id: integer(l, "source_index", 0)?,
            height: l.get("effective_height").map_or(Ok(0.), num)?,
        });
    }
    let mut rules = Vec::new();
    for r in optional(source, "drawing_lines", 5000)? {
        let orientation = r["orientation"].as_str().ok_or_else(invalid)?;
        rules.push(Rule {
            b: bbox(&r["bbox"])?,
            horizontal: orientation == "horizontal",
            vertical: orientation == "vertical",
        });
    }
    work.charge(forms.len() + lines.len() + rules.len() + 1)?;
    let mut flags = Vec::new();
    let mut id_count = 0;
    for c in candidates {
        c.as_object().ok_or_else(invalid)?;
        work.charge(1)?;
        if c.get("angle").map_or(Ok(0.), num)? != 0. {
            flags.push(false);
            continue;
        }
        let b = core(c)?;
        work.charge(forms.len() + 1)?;
        if forms
            .iter()
            .any(|f| super::filled_grid::closed_grid_form_excludes(&b, f))
        {
            flags.push(false);
            continue;
        }
        let raw_ids = optional(c, "line_indices", 20000)?;
        id_count += raw_ids.len();
        if id_count > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        work.charge(raw_ids.len() * 16 + lines.len() * 16 + 1)?;
        let ids = raw_ids
            .iter()
            .map(|v| v.as_i64().ok_or_else(invalid))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let members: Vec<_> = lines.iter().filter(|l| ids.contains(&l.id)).collect();
        if members.len() < if allow { 4 } else { 6 }
            || b.n[2] - b.n[0] <= 0.
            || b.n[3] - b.n[1] <= 0.
        {
            flags.push(false);
            continue;
        }
        let mut heights: Vec<_> = members
            .iter()
            .filter(|l| l.height > 0.)
            .map(|l| l.height)
            .collect();
        if heights.is_empty() {
            return Err(invalid());
        }
        work.charge(heights.len() * 16 + 1)?;
        heights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = heights.len();
        let em = if n % 2 == 1 {
            heights[n / 2]
        } else {
            (heights[n / 2 - 1] + heights[n / 2]) / 2.
        };
        let tolerance = 1f64.max(0.2 * em);
        work.charge(rules.len() * 2 + 1)?;
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for r in &rules {
            if r.horizontal
                && (r.b.n[0] - b.n[0]).abs() <= tolerance
                && (r.b.n[2] - b.n[2]).abs() <= tolerance
                && b.n[1] - tolerance <= r.b.n[1]
                && r.b.n[1] <= b.n[3] + tolerance
            {
                let x = (r.b.n[1] + r.b.n[3]) / 2.;
                ys.push(format!("{x:.2}").parse::<f64>().unwrap());
            }
            if r.vertical
                && (r.b.n[1] - b.n[1]).abs() <= tolerance
                && (r.b.n[3] - b.n[3]).abs() <= tolerance
                && b.n[0] - tolerance <= r.b.n[0]
                && r.b.n[0] <= b.n[2] + tolerance
            {
                let x = (r.b.n[0] + r.b.n[2]) / 2.;
                xs.push(format!("{x:.2}").parse::<f64>().unwrap());
            }
        }
        work.charge((xs.len() + ys.len()) * 16 + 1)?;
        for axis in [&mut xs, &mut ys] {
            axis.sort_by(|a, b| a.partial_cmp(b).unwrap());
            axis.dedup_by(|a, b| *a == *b);
        }
        if xs.len() < 3
            || ys.len() < if allow { 3 } else { 4 }
            || (xs[0] - b.n[0]).min(ys[0] - b.n[1]) < -tolerance
        {
            flags.push(false);
            continue;
        }
        if [
            (xs[0] - b.n[0]).abs(),
            (xs[xs.len() - 1] - b.n[2]).abs(),
            (ys[0] - b.n[1]).abs(),
            (ys[ys.len() - 1] - b.n[3]).abs(),
        ]
        .into_iter()
        .reduce(f64::max)
        .unwrap()
            > tolerance
        {
            flags.push(false);
            continue;
        }
        work.charge(
            members
                .len()
                .checked_mul(xs.len() + ys.len() + 16)
                .ok_or_else(limit)?,
        )?;
        let mut columns = BTreeSet::new();
        let mut rows = BTreeSet::new();
        for l in members {
            let cx = (l.b.n[0] + l.b.n[2]) / 2.;
            let cy = (l.b.n[1] + l.b.n[3]) / 2.;
            columns.insert(xs.iter().filter(|x| **x < cx).count());
            rows.insert(ys.iter().filter(|y| **y < cy).count());
        }
        flags.push(columns.len() >= 2 && rows.len() >= if allow { 2 } else { 3 });
    }
    Ok(json!({"closed_grid_flags":flags}))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_closed_grids() {
        let f: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-closed-grids.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                inspect(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
        for c in f["errors"].as_array().unwrap() {
            assert!(inspect(&c["request"]).is_err());
        }
        for r in f["rounding"].as_array().unwrap() {
            let x = r["value"].as_f64().unwrap();
            assert_eq!(
                format!("{x:.2}").parse::<f64>().unwrap(),
                r["expected"].as_f64().unwrap()
            );
        }
    }
    #[test]
    fn shared_budget() {
        let v = json!({"source":{},"candidates":[]});
        assert!(inspect_with_work(&v, &mut Work(10_000_000)).is_err());
        assert!(inspect(&v).is_ok());
    }
}
