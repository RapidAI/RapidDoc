//! Ordered continuation of `_build_formula_like_blocks`; no independent replay shortcuts.
use super::*;
fn ids(block: &Value) -> Result<BTreeSet<i64>, Error> {
    optional(block, "_formula_members", 4096)?
        .iter()
        .map(|v| v.as_i64().ok_or_else(invalid))
        .collect()
}
fn unclaimed(p: &Page, claimed: &BTreeSet<i64>) -> Result<Vec<usize>, Error> {
    let mut out = Vec::new();
    for (i, l) in p.lines.iter().enumerate() {
        if !claimed.contains(&integer(l, "source_index", 0)?) {
            out.push(i)
        }
    }
    Ok(out)
}
fn distance(a: &Box4, b: &Box4) -> f64 {
    let x = (a.n[0] - b.n[2]).max(b.n[0] - a.n[2]).max(0.);
    let y = (a.n[1] - b.n[3]).max(b.n[1] - a.n[3]).max(0.);
    x.hypot(y)
}
fn bounds(block: &Value) -> Result<Box4, Error> {
    bbox(block.get("_tight_output_bbox").unwrap_or(&block["bbox"]))
}
fn anchored(
    p: &mut Page,
    indices: &[usize],
    w: &mut Work,
) -> Result<(Vec<Value>, BTreeSet<i64>), Error> {
    let mut blocks = Vec::new();
    let mut claimed = BTreeSet::new();
    let words = crate::cached_regex!("[A-Za-z]{4,}");
    for &i in indices {
        let marker = ink(p, i, 0)?;
        let Some((prefix, _)) = split_number(p.t(&marker)?) else {
            continue;
        };
        if prefix
            .chars()
            .any(|c| !python_space(c) && !".…·•—-".contains(c))
            || integer(p.l(&marker), "angle", 0)? != 0
        {
            continue;
        }
        let em = num(&p.l(&marker)["effective_height"])?.max(1.);
        let mb = bbox(&p.l(&marker)["bbox"])?;
        let mut members = Vec::new();
        for &j in indices {
            w.charge(1)?;
            let l = &p.lines[j];
            let b = bbox(&l["bbox"])?;
            if j != i
                && integer(l, "angle", 0)? == 0
                && !claimed.contains(&integer(l, "source_index", 0)?)
                && l["semantic_type"].is_null()
                && (cy(&b) - cy(&mb)).abs() <= 0.65 * em
                && b.n[2] < mb.n[0]
                && mb.n[0] - b.n[2] <= 16. * em
            {
                members.push(Row { index: j, b })
            }
        }
        if members.is_empty() {
            continue;
        }
        let mut ordered = members.clone();
        ordered.sort_by(|a, b| a.b.n[0].total_cmp(&b.b.n[0]));
        let text = ordered
            .iter()
            .map(|r| p.t(r))
            .collect::<Result<Vec<_>, _>>()?
            .join(" ");
        let b = p.bounds(&members)?;
        if !text.contains('=')
            || !text.chars().any(|c| "+−*/×÷-".contains(c))
            || text.chars().any(|c| ('\u{3400}'..='\u{9fff}').contains(&c))
            || words.is_match(&text)
            || b.n[2] - b.n[0] < 0.08 * p.size[0]
            || b.n[2] - b.n[0] > 0.6 * p.size[0]
            || (cx(&b) - p.size[0] * 0.5).abs() > 0.2 * p.size[0]
            || b.n[3] - b.n[1] > 2. * em
            || p.tables.iter().any(|t| intersects(&b, t))
        {
            continue;
        }
        if let Some(block) = materialize::block(p, &members, 0, p.id(&members[0])?, w)? {
            blocks.push(block);
            for r in members {
                claimed.insert(p.id(&r)?);
            }
        }
    }
    Ok((blocks, claimed))
}
pub(super) fn finish(
    p: &mut Page,
    drawings: &[Value],
    mut blocks: Vec<Value>,
    mut claimed: BTreeSet<i64>,
    paragraphs: Vec<Value>,
    w: &mut Work,
) -> Result<(Vec<Value>, Vec<Value>), Error> {
    let (more, indices) = recovery::mixed(p, &unclaimed(p, &claimed)?, w)?;
    blocks.extend(more);
    claimed.extend(indices);
    let flags = recovery::flags(p, w)?;
    let (more, indices) = recovery::detached(p, &unclaimed(p, &claimed)?, &[], &flags, w)?;
    blocks.extend(more);
    claimed.extend(indices);
    let (whole, _) = recovery::detached(
        p,
        &(0..p.lines.len()).collect::<Vec<_>>(),
        drawings,
        &flags,
        w,
    )?;
    for b in whole {
        let members = ids(&b)?;
        let intersecting = blocks
            .iter()
            .filter(|b| ids(b).is_ok_and(|s| !members.is_disjoint(&s)))
            .cloned()
            .collect::<Vec<_>>();
        let mut previous = BTreeSet::new();
        for b in &intersecting {
            previous.extend(ids(b)?)
        }
        if intersecting.len() == 1 && members.is_subset(&previous) {
            continue;
        }
        if intersecting
            .iter()
            .any(|b| !ids(b).unwrap().is_subset(&members))
        {
            continue;
        }
        blocks.retain(|b| !intersecting.iter().any(|old| eq(b, old)));
        blocks.push(b);
        claimed.extend(members)
    }
    let mut retained = Vec::new();
    for b in blocks {
        let members = ids(&b)?;
        let bb = bounds(&b)?;
        let mut hosts = false;
        for (i, l) in p.lines.iter().enumerate() {
            w.charge(1)?;
            let id = integer(l, "source_index", 0)?;
            if members.contains(&id)
                || claimed.contains(&id)
                || l["ink_bbox"].is_null()
                || !sentence(l["text"].as_str().ok_or_else(invalid)?)
            {
                continue;
            }
            let raw = Row {
                index: i,
                b: bbox(&l["bbox"])?,
            };
            if axis(&bb, &bbox(&l["ink_bbox"])?, 1) >= 0.25
                && (bb.n[0] - raw.b.n[2]).max(raw.b.n[0] - bb.n[2]).max(0.) <= 3. * p.h(&raw)?
            {
                hosts = true;
                break;
            }
        }
        if !members.is_empty()
            && hosts
            && !b["content"]
                .as_str()
                .ok_or_else(invalid)?
                .contains("\\tag{")
            && b["_spatial_display_band"] != true
        {
            claimed.retain(|id| !members.contains(id));
            for l in &mut p.lines {
                if members.contains(&integer(l, "source_index", 0)?) {
                    l["paragraph_formula_context"] = json!(true);
                    l["formula_candidate_only"] = json!(false);
                    l["compact_formula_cluster"] = json!(false)
                }
            }
        } else {
            retained.push(b)
        }
    }
    blocks = retained;
    let (more, indices) = anchored(p, &unclaimed(p, &claimed)?, w)?;
    blocks.extend(more);
    claimed.extend(indices);
    let em = if p.lines.is_empty() {
        10.
    } else {
        median(
            (0..p.lines.len())
                .map(|i| {
                    let r = Row {
                        index: i,
                        b: bbox(&p.lines[i]["bbox"])?,
                    };
                    p.h(&r)
                })
                .collect::<Result<Vec<_>, Error>>()?,
        )
    };
    for block in &mut blocks {
        let core = bounds(block)?;
        for (i, l) in p.lines.iter().enumerate() {
            w.charge(p.lines.len() + 1)?;
            let id = integer(l, "source_index", 0)?;
            if claimed.contains(&id) || l["ink_bbox"].is_null() {
                continue;
            }
            let text = l["text"]
                .as_str()
                .ok_or_else(invalid)?
                .trim_matches(python_space);
            let ib = bbox(&l["ink_bbox"])?;
            let n = text.chars().count();
            if n > 20
                || sentence(text)
                || !(n <= 2 || math(text))
                || cx(&ib) < core.n[0]
                || cx(&ib) > core.n[2]
                || distance(&core, &ib) > 0.7 * em
            {
                continue;
            }
            let mut veto = false;
            for (j, other) in p.lines.iter().enumerate() {
                if integer(other, "source_index", 0)? != id
                    && sentence(other["text"].as_str().ok_or_else(invalid)?)
                {
                    let ob = ink(p, j, 0)?.b;
                    if axis(&ib, &ob, 1) >= 0.4 && distance(&ib, &ob) <= em {
                        veto = true;
                        break;
                    }
                }
            }
            if veto {
                continue;
            }
            block["bbox"] =
                json!(combined([&bbox(&block["bbox"])?, &bbox(&p.lines[i]["bbox"])?])?.raw);
            block["_tight_output_bbox"] = json!(combined([&core, &ib])?.raw);
            block["content"] = json!(format!(
                "{}\n{}",
                block["content"].as_str().ok_or_else(invalid)?,
                text
            ));
            claimed.insert(id);
        }
    }
    let bands = bands::build(p, drawings, w)?;
    let mut recovered = BTreeSet::new();
    for band in bands {
        let bb = bounds(&band)?;
        let mut overlaps = Vec::new();
        for (i, b) in blocks.iter().enumerate() {
            if super::super::path_graphics::overlap_smaller(&bb, &bounds(b)?) >= 0.5 {
                overlaps.push(i)
            }
        }
        let mut contained = false;
        for &i in &overlaps {
            let old = bounds(&blocks[i])?;
            if (0..4).all(|k| {
                if k < 2 {
                    old.n[k] <= bb.n[k] + 1.
                } else {
                    old.n[k] >= bb.n[k] - 1.
                }
            }) {
                contained = true;
                break;
            }
        }
        if contained {
            continue;
        }
        let mut members = ids(&band)?;
        for &i in &overlaps {
            members.extend(ids(&blocks[i])?)
        }
        let mut rows = Vec::new();
        for (i, l) in p.lines.iter().enumerate() {
            if members.contains(&integer(l, "source_index", 0)?) {
                rows.push(ink(p, i, 0)?)
            }
        }
        if rows.is_empty() {
            continue;
        }
        let anchor = optional(&band, "_formula_members", 4096)?
            .last()
            .ok_or_else(invalid)?
            .as_i64()
            .ok_or_else(invalid)?;
        if let Some(b) = materialize::block(p, &rows, 0, anchor, w)? {
            blocks = blocks
                .into_iter()
                .enumerate()
                .filter(|(i, _)| !overlaps.contains(i))
                .map(|(_, b)| b)
                .collect();
            blocks.push(b);
            claimed.extend(&members);
            recovered.extend(members)
        }
    }
    for b in &mut blocks {
        b.as_object_mut()
            .ok_or_else(invalid)?
            .remove("_formula_members");
    }
    let mut remaining = Vec::new();
    for l in &p.lines {
        if !claimed.contains(&integer(l, "source_index", 0)?)
            && (l["formula_candidate_only"] != true || l["paragraph_formula_context"] == true)
        {
            remaining.push(l.clone())
        }
    }
    for l in paragraphs {
        if !recovered.contains(&integer(&l, "source_index", 0)?) {
            remaining.push(l)
        }
    }
    Ok((blocks, remaining))
}
