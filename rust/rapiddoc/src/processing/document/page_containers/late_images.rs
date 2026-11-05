//! 旧几何校准后的并排图标题拆行与图内文字补认领。
use super::*;
use crate::legacy_model::{
    native_lines,
    path_graphics::{area, integer, overlap_first, overlap_smaller},
    table_native_candidate,
    table_native_multiline::spatial,
    table_text_styles::preparation,
};
use rapiddoc_native::{python_decimal, python_printable, text_boundary::python_space};
use std::collections::BTreeSet;
use std::borrow::Cow;

fn input_array(v: &mut Cow<'_, Value>, key: &str, max: usize) -> Result<Vec<Value>, Error> {
    array(&v[key], max)?;
    Ok(match v {
        Cow::Borrowed(v) => v[key].as_array().unwrap().clone(),
        Cow::Owned(v) => std::mem::take(v[key].as_array_mut().unwrap()),
    })
}
fn center(b: &Box4, axis: usize) -> f64 {
    (b.n[axis] + b.n[axis + 2]) * 0.5
}
fn intersects(a: &Box4, b: &Box4) -> bool {
    a.n[0] < b.n[2] && a.n[2] > b.n[0] && a.n[1] < b.n[3] && a.n[3] > b.n[1]
}
fn glyph(c: &Value, sz: [f64; 2]) -> Result<Option<Box4>, Error> {
    let raw = spatial::raw_string(c.get("char"), true);
    if raw.is_empty() || !raw.chars().all(python_printable) || raw.chars().all(python_space) {
        return Ok(None);
    }
    coerce(&c["bbox"])?.map_or(Ok(None), |b| clip(&b, sz))
}
fn rules(
    drawings: &[Value],
    images: &[Box4],
    tables: &[Box4],
    sz: [f64; 2],
    em: f64,
    w: &mut Work,
) -> Result<Vec<(Box4, Box4)>, Error> {
    let mut long = Vec::new();
    for line in drawings {
        let b = bbox(&line["bbox"])?;
        w.charge(tables.len() + 1)?;
        if line["orientation"] == "horizontal"
            && b.n[2] - b.n[0] >= (8. * em).max(0.18 * sz[0])
            && !tables.iter().any(|t| {
                center(&b, 0) >= t.n[0]
                    && center(&b, 0) <= t.n[2]
                    && center(&b, 1) >= t.n[1]
                    && center(&b, 1) <= t.n[3]
            })
        {
            long.push(b)
        }
    }
    let mut ordered = images.to_vec();
    ordered.sort_by(|a, b| a.n[0].total_cmp(&b.n[0]).then(a.n[1].total_cmp(&b.n[1])));
    let mut result: Vec<(Box4, Box4)> = Vec::new();
    for (i, left) in ordered.iter().enumerate() {
        for right in &ordered[i + 1..] {
            w.charge(tables.len() * 2 + long.len() * 2 + 1)?;
            if left.n[2] >= right.n[0]
                || (left.n[3].min(right.n[3]) - left.n[1].max(right.n[1])).max(0.)
                    < 0.7
                        * (left.n[3] - left.n[1])
                            .max(0.1)
                            .min((right.n[3] - right.n[1]).max(0.1))
                || tables
                    .iter()
                    .any(|t| overlap_smaller(left, t) >= 0.5 || overlap_smaller(right, t) >= 0.5)
            {
                continue;
            }
            let near = |r: &Box4, b: &Box4| {
                let width = (r.n[2] - r.n[0]).min(b.n[2] - b.n[0]);
                width > 0.
                    && (r.n[2].min(b.n[2]) - r.n[0].max(b.n[0])).max(0.) / width >= 0.8
                    && b.n[1] - r.n[3] >= -0.25 * em
                    && b.n[1] - r.n[3] <= 3. * em
            };
            for l in long.iter().filter(|r| near(r, left)) {
                for r in long.iter().filter(|r| near(r, right)) {
                    w.charge(result.len() + 1)?;
                    let gap = r.n[0] - l.n[2];
                    if l.n[2] < r.n[0]
                        && (center(l, 1) - center(r, 1)).abs() <= 0.5 * em
                        && gap >= 0.5 * em
                        && gap <= 5. * em
                        && !result.iter().any(|(a, b)| a.n == l.n && b.n == r.n)
                    {
                        result.push((l.clone(), r.clone()))
                    }
                }
            }
        }
    }
    Ok(result)
}
fn boundary(
    members: &[&Value],
    l: &Box4,
    r: &Box4,
    tables: &[Box4],
    sz: [f64; 2],
    em: f64,
    w: &mut Work,
) -> Result<Option<f64>, Error> {
    if members.is_empty()
        || members.iter().any(|m| {
            integer(m, "angle", 0).unwrap_or(-1) != 0
                || m["semantic_type"] != members[0]["semantic_type"]
        })
    {
        return Ok(None);
    }
    let bs = members
        .iter()
        .map(|m| bbox(&m["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let row = combined(&bs)?;
    w.charge(tables.len() + 1)?;
    let gap = l.n[1].min(r.n[1]) - row.n[3];
    if tables.iter().any(|b| intersects(&row, b)) || gap < -0.2 * em || gap > 1.5 * em {
        return Ok(None);
    }
    let mut gs = Vec::new();
    for m in members {
        for c in optional(m, "chars", 20000)? {
            w.charge(1)?;
            if let Some(b) = glyph(c, sz)? {
                gs.push(b)
            }
        }
    }
    let x = (l.n[2] + r.n[0]) * 0.5;
    if gs
        .iter()
        .any(|b| center(b, 0) < l.n[0] - em || center(b, 0) > r.n[2] + em)
    {
        return Ok(None);
    }
    let left = gs.iter().filter(|b| center(b, 0) < x).collect::<Vec<_>>();
    let right = gs.iter().filter(|b| center(b, 0) > x).collect::<Vec<_>>();
    if left.len() < 3 || right.len() < 3 {
        return Ok(None);
    }
    let lb = combined(left)?;
    let rb = combined(right)?;
    if (lb.n[2] - lb.n[0]).min(rb.n[2] - rb.n[0]) < 4. * em
        || rb.n[0] - lb.n[2] < 0.75 * em
        || !(lb.n[2] <= l.n[2] && l.n[2] <= rb.n[0] && lb.n[2] <= r.n[0] && r.n[0] <= rb.n[0])
    {
        return Ok(None);
    }
    Ok(Some(x))
}
fn run_text(cs: &[Value]) -> String {
    let raw = cs
        .iter()
        .map(|c| spatial::raw_string(c.get("char"), true))
        .collect::<String>();
    let clean = crate::legacy_model::code_containers::text::sanitize(&raw).replace('\n', "");
    let mut out = String::new();
    let mut space = false;
    for c in clean.chars() {
        if c == ' ' {
            if !space {
                out.push(c)
            }
            space = true
        } else {
            out.push(c);
            space = false
        }
    }
    out.trim_matches(python_space).to_owned()
}
pub(super) fn split_with_sources(v: &Value, w: &mut Work) -> Result<(Value, BTreeSet<i64>), Error> {
    split_impl(Cow::Borrowed(v), w)
}
pub(super) fn split_owned(v: Value, w: &mut Work) -> Result<(Value, BTreeSet<i64>), Error> {
    split_impl(Cow::Owned(v), w)
}
fn split_impl(mut v: Cow<'_, Value>, w: &mut Work) -> Result<(Value, BTreeSet<i64>), Error> {
    let lines = input_array(&mut v, "lines", 4096)?;
    for line in &lines {
        bbox(&line["bbox"])?;
        integer(line, "source_index", 0)?;
        integer(line, "angle", 0)?;
        integer(line, "run_index", 0)?;
        w.charge(line["text"].as_str().ok_or_else(invalid)?.len() + 1)?;
    }
    let images = array(&v["image_bboxes"], 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let sz = size(&v["page_size"])?;
    let mut horizontal = Vec::new();
    let mut heights = Vec::new();
    for l in &lines {
        let h = l.get("effective_height").map_or(Ok(0.), num)?;
        if integer(l, "angle", 0)? == 0 && h > 0. {
            horizontal.push(l);
            heights.push(h)
        }
    }
    if horizontal.is_empty() || images.len() < 2 {
        return Ok((Value::Array(lines), BTreeSet::new()));
    }
    let em = median(heights);
    let tables = array(&v["table_bboxes"], 4000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let pairs = rules(
        array(&v["drawing_lines"], 5000)?,
        &images,
        &tables,
        sz,
        em,
        w,
    )?;
    if pairs.is_empty() {
        return Ok((Value::Array(lines), BTreeSet::new()));
    }
    let mut groups: Vec<(i64, Vec<&Value>)> = Vec::new();
    for l in horizontal {
        if let Some(row) = l.get("visual_row_id").filter(|r| !r.is_null()) {
            let row = row.as_i64().ok_or_else(invalid)?;
            if let Some((_, members)) = groups.iter_mut().find(|(r, _)| *r == row) {
                members.push(l)
            } else {
                groups.push((row, vec![l]))
            }
        }
    }
    let mut by_row = Vec::new();
    for (row, members) in groups {
        let mut xs: Vec<f64> = Vec::new();
        for (l, r) in &pairs {
            if let Some(x) = boundary(&members, l, r, &tables, sz, em, w)? {
                if !xs.iter().any(|p| (x - p).abs() <= 0.5 * em) {
                    xs.push(x)
                }
            }
        }
        if !xs.is_empty() {
            by_row.push((row, members, xs))
        }
    }
    if by_row.is_empty() {
        return Ok((Value::Array(lines), BTreeSet::new()));
    }
    let mut next = lines
        .iter()
        .map(|l| integer(l, "source_index", 0))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(-1)
        .checked_add(1)
        .ok_or_else(limit)?
        .max(
            v.get("source_index_start")
                .filter(|v| !v.is_null())
                .map_or(Ok(0), |v| v.as_i64().ok_or_else(invalid))?,
        );
    let mut consumed = BTreeSet::new();
    let mut rebuilt_all = Vec::new();
    for (row, mut members, mut xs) in by_row {
        members.sort_by(|a, b| {
            a["bbox"][0]
                .as_f64()
                .unwrap()
                .total_cmp(&b["bbox"][0].as_f64().unwrap())
                .then(
                    integer(a, "run_index", 0)
                        .unwrap_or(0)
                        .cmp(&integer(b, "run_index", 0).unwrap_or(0)),
                )
                .then(
                    integer(a, "source_index", 0)
                        .unwrap_or(0)
                        .cmp(&integer(b, "source_index", 0).unwrap_or(0)),
                )
        });
        let mut cs = Vec::new();
        for m in &members {
            cs.extend_from_slice(optional(m, "chars", 20000)?)
        }
        if cs.len() > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        xs.sort_by(f64::total_cmp);
        let mut cuts = BTreeSet::new();
        for x in xs {
            for (i, c) in cs.iter().enumerate() {
                w.charge(1)?;
                if glyph(c, sz)?.is_some_and(|b| center(&b, 0) > x) {
                    cuts.insert(i);
                    break;
                }
            }
        }
        if cuts.is_empty() {
            continue;
        }
        let mut starts = vec![0];
        starts.extend(cuts);
        starts.push(cs.len());
        let mut rebuilt = Vec::new();
        for (run, range) in starts.windows(2).enumerate() {
            let chars = &cs[range[0]..range[1]];
            let text = run_text(chars);
            let mut bs = Vec::new();
            for c in chars {
                w.charge(1)?;
                if let Some(b) = glyph(c, sz)? {
                    bs.push(b)
                }
            }
            if text.is_empty() || bs.is_empty() {
                continue;
            }
            let template = &members[run.min(members.len() - 1)];
            let mut l = native_lines::defaults();
            for (k, v) in template.as_object().ok_or_else(invalid)? {
                l[k] = v.clone()
            }
            if l["source_bbox"].is_null() {
                l["source_bbox"] = template["bbox"].clone()
            }
            l["text"] = json!(text);
            l["bbox"] = json!(combined(&bs)?.raw);
            l["chars"] = json!(chars);
            l["visual_row_id"] = json!(row);
            l["run_index"] = json!(run);
            l["split_from_row"] = json!(true);
            l["preserve_split_boundary"] = json!(true);
            l["source_index"] = if run < members.len() {
                members[run]["source_index"].clone()
            } else {
                let id = next;
                next = next.checked_add(1).ok_or_else(limit)?;
                json!(id)
            };
            native_lines::markers(&mut l);
            w.charge(chars.len() * 5)?;
            let metrics = preparation::metrics(&l, sz)?;
            for (k, v) in metrics.as_object().unwrap() {
                l[k] = v.clone()
            }
            rebuilt.push(l);
        }
        if rebuilt.len() < 2 {
            continue;
        }
        for m in &members {
            consumed.insert(integer(m, "source_index", 0)?);
        }
        rebuilt_all.extend(rebuilt);
    }
    let mut out = lines
        .into_iter()
        .filter(|l| !consumed.contains(&integer(l, "source_index", 0).unwrap_or(0)))
        .collect::<Vec<_>>();
    out.extend(rebuilt_all);
    out.sort_by(|a, b| {
        integer(a, "angle", 0)
            .unwrap_or(0)
            .cmp(&integer(b, "angle", 0).unwrap_or(0))
            .then(
                a["bbox"][1]
                    .as_f64()
                    .unwrap()
                    .total_cmp(&b["bbox"][1].as_f64().unwrap()),
            )
            .then(
                a["bbox"][0]
                    .as_f64()
                    .unwrap()
                    .total_cmp(&b["bbox"][0].as_f64().unwrap()),
            )
            .then(
                integer(a, "source_index", 0)
                    .unwrap_or(0)
                    .cmp(&integer(b, "source_index", 0).unwrap_or(0)),
            )
    });
    Ok((Value::Array(out), consumed))
}
pub(super) fn reclaim_with_lines(v: &Value, w: &mut Work) -> Result<(Value, Vec<Value>), Error> {
    reclaim_source(&v["source"], v, w)
}
pub(super) fn reclaim_source(
    source: &Value,
    v: &Value,
    w: &mut Work,
) -> Result<(Value, Vec<Value>), Error> {
    reclaim_impl(source, Cow::Borrowed(v), false, w)
}
pub(super) fn reclaim_source_owned(
    source: &Value,
    v: Value,
    w: &mut Work,
) -> Result<(Value, Vec<Value>), Error> {
    reclaim_impl(source, Cow::Owned(v), true, w)
}
fn reclaim_impl(
    source: &Value,
    mut v: Cow<'_, Value>,
    compact: bool,
    w: &mut Work,
) -> Result<(Value, Vec<Value>), Error> {
    let sz = size(&source["page_size"])?;
    let maps = super::forms::maps(source, "form_member_sources", w)?;
    let mut lines = input_array(&mut v, "remaining_lines", 4096)?;
    let mut blocks = input_array(&mut v, "image_containers", 4000)?;
    let boxes = blocks
        .iter()
        .map(|b| bbox(&b["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let rasters = optional(source, "image_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let cores = array(&v["strong_graphic_bboxes"], 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let mut grouped = vec![Vec::new(); blocks.len()];
    let mut claimed = BTreeSet::new();
    for (i, l) in lines.iter().enumerate() {
        if !l["semantic_type"].is_null()
            || super::roles::caption(l["text"].as_str().ok_or_else(invalid)?)
        {
            continue;
        }
        let ink = bbox(
            l.get("ink_bbox")
                .filter(|b| !b.is_null())
                .unwrap_or(&l["bbox"]),
        )?;
        let mut owner = None;
        for (j, b) in boxes.iter().enumerate() {
            w.charge(rasters.len() + cores.len() + 1)?;
            if overlap_first(&ink, b) < 0.98
                || area(b) >= 0.6 * sz[0] * sz[1]
                    && rasters.iter().any(|r| overlap_first(b, r) >= 0.95)
            {
                continue;
            }
            if !super::captions::allow(l, b, &maps, w)? {
                let coverage = cores
                    .iter()
                    .filter(|c| overlap_first(c, b) >= 0.95)
                    .map(area)
                    .collect::<Vec<_>>();
                if table_native_candidate::sum(&coverage) < 0.25 * area(b) {
                    continue;
                }
            }
            if owner.is_none_or(|k| area(b) < area(&boxes[k])) {
                owner = Some(j)
            }
        }
        if let Some(j) = owner {
            grouped[j].push(i);
            claimed.insert(integer(l, "source_index", 0)?);
        }
    }
    for (j, members) in grouped.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        let mut numeric = 0;
        let mut horizontal = true;
        for &i in members {
            let text = lines[i]["text"]
                .as_str()
                .ok_or_else(invalid)?
                .trim_matches(python_space);
            if !text.is_empty()
                && text
                    .chars()
                    .all(|c| python_decimal(c).is_some() || ".,%+/−-".contains(c))
            {
                numeric += 1
            }
            horizontal &= integer(&lines[i], "angle", 0)? == 0;
        }
        let content = if numeric >= 6 && numeric as f64 >= 0.45 * members.len() as f64 && horizontal
        {
            let items = members
                .iter()
                .map(|&i| {
                    Ok((
                        lines[i]["text"].as_str().ok_or_else(invalid)?.to_owned(),
                        bbox(&lines[i]["bbox"])?.n,
                    ))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            crate::legacy_model::code_containers::text::sanitize(&spatial::project_items(
                &items, w,
            )?)
        } else {
            super::media::content(members, &mut lines, sz, w)?
        };
        let original = blocks[j]["content"].as_str().ok_or_else(invalid)?;
        blocks[j]["content"] = json!([original, content.as_str()]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n"));
    }
    let updates = compact.then(|| {
        lines
            .iter()
            .map(|l| {
                value_object!({
        "source_index":l["source_index"],"median_glyph_width":l["median_glyph_width"]})
            })
            .collect::<Vec<_>>()
    });
    let remaining = if compact {
        std::mem::take(&mut lines)
            .into_iter()
            .filter(|l| !claimed.contains(&integer(l, "source_index", 0).unwrap_or(0)))
            .collect()
    } else {
        lines
            .iter()
            .filter(|l| !claimed.contains(&integer(l, "source_index", 0).unwrap_or(0)))
            .cloned()
            .collect()
    };
    let mut out = json!({"reclaimed_line_indices":claimed});
    out["remaining_lines"] = Value::Array(remaining);
    out["image_containers"] = Value::Array(blocks);
    Ok((out, updates.unwrap_or(lines)))
}
