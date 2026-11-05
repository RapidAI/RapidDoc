//! All original inline-math fragment and paragraph-formula-context block recoveries.
use super::block_common as c;
use super::*;
fn raw(b: &Value) -> Result<Box4, Error> {
    bbox(&b["bbox"])
}
fn height(b: &Value, positive: bool) -> Result<f64, Error> {
    let hs = c::heights(b, positive)?;
    if hs.is_empty() {
        let b = raw(b)?;
        Ok(b.n[3] - b.n[1])
    } else {
        median(hs)
    }
}
fn page_width(p: [f64; 2], angle: i64) -> f64 {
    if angle == 90 || angle == 270 {
        p[1]
    } else {
        p[0]
    }
}
fn gap(a: &Box4, b: &Box4, k: usize) -> f64 {
    (a.n[k].max(b.n[k]) - a.n[k + 2].min(b.n[k + 2])).max(0.)
}
fn recovery(bs: &[Value], ids: &[usize]) -> Result<Value, Error> {
    let boxes = ids
        .iter()
        .map(|&i| raw(&bs[i]))
        .collect::<Result<Vec<_>, _>>()?;
    let max = boxes.iter().map(|b| b.n[2] - b.n[0]).fold(0f64, f64::max);
    let mut regions = boxes
        .iter()
        .filter(|b| max <= 0. || b.n[2] - b.n[0] <= 0.25 * max)
        .map(|b| json!(b.raw))
        .collect::<Vec<_>>();
    if regions.is_empty() {
        regions = boxes.iter().map(|b| json!(b.raw)).collect()
    }
    let mut b = c::merge(bs, ids, false)?;
    b["_recovered_inline_math_fragments"] = json!(true);
    let mut inline = optional(&b, "_inline_math_regions", 20000)?.to_vec();
    inline.extend(regions);
    b["_inline_math_regions"] = json!(inline);
    Ok(b)
}
pub(super) fn fragments(bs: Vec<Value>, p: [f64; 2], w: &mut Work) -> Result<Vec<Value>, Error> {
    let (mut used, mut replacements) = (BTreeSet::new(), BTreeMap::new());
    w.charge(bs.len() * bs.len())?;
    for (i, a) in bs.iter().enumerate() {
        let rs = c::rows(a)?;
        let angle = c::angle(a)?;
        let pw = page_width(p, angle);
        if used.contains(&i)
            || a["type"] != "text"
            || (a["_single_run_row_id"].is_null() && rs.len() > 2)
            || !a["bbox"].is_array()
            || !a["_line_heights"].is_array()
            || a["_line_heights"].as_array().unwrap().is_empty()
        {
            continue;
        }
        let ab = raw(a)?;
        if ab.n[2] - ab.n[0] < 0.35 * pw {
            continue;
        }
        let h = median(c::heights(a, true)?)?;
        let mut ids = vec![i];
        for (j, b) in bs.iter().enumerate() {
            if i == j
                || used.contains(&j)
                || b["type"] != "text"
                || c::angle(b)? != angle
                || !b["bbox"].is_array()
                || c::rows(b)?.len() != 1
            {
                continue;
            }
            let bb = raw(b)?;
            if bb.n[2] - bb.n[0] > 0.25 * (ab.n[2] - ab.n[0]) || axis(&ab, &bb, 0) <= 0. {
                continue;
            }
            let u = combined([&ab, &bb])?;
            if gap(&ab, &bb, 1) <= 0.75 * h && u.n[3] - u.n[1] <= 3.5 * h {
                ids.push(j)
            }
        }
        if ids.len() < 3 {
            continue;
        }
        replacements.insert(*ids.iter().min().unwrap(), recovery(&bs, &ids)?);
        used.extend(ids)
    }
    let bs = c::replace(bs, replacements, used);
    let bs = hostless(bs, p, w)?;
    let bs = residual(bs, p, w)?;
    continuations(bs, p, w)
}
fn residual(bs: Vec<Value>, p: [f64; 2], w: &mut Work) -> Result<Vec<Value>, Error> {
    let (mut used, mut replacements) = (BTreeSet::new(), BTreeMap::new());
    w.charge(bs.len() * bs.len())?;
    for (i, a) in bs.iter().enumerate() {
        let angle = c::angle(a)?;
        let pw = page_width(p, angle);
        if a["type"] != "text" || !a["bbox"].is_array() || c::rows(a)?.len() != 1 {
            continue;
        }
        let ab = raw(a)?;
        if ab.n[2] - ab.n[0] > 0.05 * pw {
            continue;
        }
        let ah = height(a, true)?;
        let mut hosts = vec![];
        for (j, b) in bs.iter().enumerate() {
            if i == j
                || used.contains(&j)
                || b["type"] != "text"
                || c::angle(b)? != angle
                || !b["bbox"].is_array()
            {
                continue;
            }
            let bb = raw(b)?;
            if bb.n[2] - bb.n[0] < 0.3 * pw || ab.n[0] < bb.n[0] || ab.n[2] > bb.n[2] {
                continue;
            }
            let bh = height(b, true)?;
            let overlap = (ab.n[3].min(bb.n[3]) - ab.n[1].max(bb.n[1])).max(0.);
            let h = ah.max(bh).max(0.1);
            let u = combined([&ab, &bb])?;
            if (overlap < 0.35 * ah.min(bh) && gap(&ab, &bb, 1) > 0.5 * h)
                || u.n[3] - u.n[1] > 3. * h
            {
                continue;
            }
            hosts.push(j)
        }
        if hosts.len() == 1 {
            let j = hosts[0];
            replacements.insert(i.min(j), recovery(&bs, &[i, j])?);
            used.insert(i);
            used.insert(j);
        }
    }
    Ok(c::replace(bs, replacements, used))
}
fn hostless(bs: Vec<Value>, p: [f64; 2], w: &mut Work) -> Result<Vec<Value>, Error> {
    let angles = bs
        .iter()
        .filter(|b| b["type"] == "text")
        .map(c::angle)
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut groups = vec![];
    for angle in angles {
        let pw = page_width(p, angle);
        let mut ids = vec![];
        for (i, b) in bs.iter().enumerate() {
            if b["type"] == "text"
                && c::angle(b)? == angle
                && b["bbox"].is_array()
                && c::rows(b)?.len() == 1
            {
                let b = raw(b)?;
                if b.n[2] - b.n[0] <= 0.25 * pw {
                    ids.push(i)
                }
            }
        }
        let mut pending = super::super::tiled_tables::pending::Pending::from_indices(&ids);
        while let Some(seed) = pending.pop() {
            let mut history = vec![seed];
            let mut changed = true;
            while changed {
                changed = false;
                let candidates = pending.ordered();
                w.charge(candidates.len() * history.len())?;
                for i in candidates {
                    let a = raw(&bs[i])?;
                    let ah = height(&bs[i], false)?;
                    let members = super::super::table_native_vector::python_set::ordered_signed(
                        &history.iter().map(|&n| n as i64).collect::<Vec<_>>(),
                    );
                    let mut connected = false;
                    for j in members {
                        let j = j as usize;
                        let b = raw(&bs[j])?;
                        let h = ah.max(height(&bs[j], false)?);
                        if gap(&a, &b, 1) <= 0.75 * h && gap(&a, &b, 0) <= 1.5 * h {
                            connected = true;
                            break;
                        }
                    }
                    if connected {
                        history.push(i);
                        pending.remove(i);
                        changed = true
                    }
                }
            }
            if history.len() < 4 {
                continue;
            }
            let mut hs = vec![];
            let mut fonts = vec![];
            let mut boxes = vec![];
            for &i in &history {
                hs.extend(c::heights(&bs[i], true)?);
                boxes.push(raw(&bs[i])?);
                for f in optional(&bs[i], "_font_signatures", 4096)? {
                    if !fonts.iter().any(|v| eq(v, f)) {
                        fonts.push(f.clone())
                    }
                }
            }
            if hs.is_empty() {
                continue;
            }
            let h = median(hs)?;
            let u = combined(boxes.iter())?;
            if fonts.len() < 2
                || !(0.25 * pw..=0.5 * pw).contains(&(u.n[2] - u.n[0]))
                || u.n[3] - u.n[1] > 3.5 * h
                || boxes.iter().filter(|b| b.n[2] - b.n[0] >= 3.5 * h).count() < 2
            {
                continue;
            }
            history.sort_unstable();
            groups.push(history);
        }
    }
    let (mut used, mut replacements) = (BTreeSet::new(), BTreeMap::new());
    for group in groups {
        if group.iter().any(|i| used.contains(i)) {
            continue;
        }
        replacements.insert(*group.iter().min().unwrap(), recovery(&bs, &group)?);
        used.extend(group)
    }
    Ok(c::replace(bs, replacements, used))
}
fn continuations(bs: Vec<Value>, p: [f64; 2], w: &mut Work) -> Result<Vec<Value>, Error> {
    if bs
        .iter()
        .filter(|b| b["_recovered_inline_math_fragments"] == true)
        .count()
        < 2
    {
        return Ok(bs);
    }
    let mut ids = (0..bs.len()).collect::<Vec<_>>();
    ids.sort_by(|&i, &j| {
        c::angle(&bs[i])
            .unwrap()
            .cmp(&c::angle(&bs[j]).unwrap())
            .then(
                c::key(&bs[i])
                    .unwrap_or((f64::INFINITY, f64::INFINITY))
                    .partial_cmp(&c::key(&bs[j]).unwrap_or((f64::INFINITY, f64::INFINITY)))
                    .unwrap(),
            )
    });
    let mut lanes: Vec<Vec<usize>> = vec![];
    for i in ids {
        let b = &bs[i];
        if b["type"] != "text" || !b["_lane_is_span"].is_boolean() || c::rows(b)?.is_empty() {
            continue;
        }
        let hs = c::heights(b, true)?;
        if hs.is_empty() {
            continue;
        }
        let mut found = false;
        for group in &mut lanes {
            w.charge(1)?;
            let a = &bs[group[0]];
            if c::angle(a)? != c::angle(b)? {
                continue;
            }
            let mut heights = c::heights(a, true)?;
            heights.extend_from_slice(&hs);
            let h = median(heights)?;
            if c::role(a, b, h) {
                group.push(i);
                found = true;
                break;
            }
        }
        if !found {
            lanes.push(vec![i])
        }
    }
    let mut chains = vec![];
    for mut lane in lanes {
        lane.sort_by(|&i, &j| {
            c::key(&bs[i])
                .unwrap()
                .partial_cmp(&c::key(&bs[j]).unwrap())
                .unwrap()
        });
        let mut chain = vec![lane[0]];
        for &i in &lane[1..] {
            let j = *chain.last().unwrap();
            let (a, b) = (&bs[j], &bs[i]);
            let ar = c::rows(a)?;
            let br = c::rows(b)?;
            let ab = combined(ar.iter())?;
            let bb = combined(br.iter())?;
            let mut hs = c::heights(a, true)?;
            hs.extend(c::heights(b, true)?);
            let h = median(hs)?;
            let lw = c::compatible(a, b, page_width(p, c::angle(a)?), h);
            if c::role(a, b, h)
                && ab.n[2] - ab.n[0] >= 0.8 * lw
                && bb.n[2] - bb.n[0] >= 0.8 * lw
                && (ab.n[0] - bb.n[0]).abs() <= 0.75 * h
                && (-0.75 * h..=0.75 * h).contains(&(bb.n[1] - ab.n[3]))
                && !c::conflict(a, b)
                && !c::skips(&bs, j, i, h)?
            {
                chain.push(i)
            } else {
                chains.push(chain);
                chain = vec![i]
            }
        }
        chains.push(chain)
    }
    let (mut used, mut replacements) = (BTreeSet::new(), BTreeMap::new());
    for chain in chains {
        if chain.len() < 3
            || chain
                .iter()
                .filter(|&&i| bs[i]["_recovered_inline_math_fragments"] == true)
                .count()
                < 2
        {
            continue;
        }
        let mut hs = vec![];
        let mut boxes = vec![];
        for &i in &chain {
            hs.extend(c::heights(&bs[i], true)?);
            boxes.extend(c::rows(&bs[i])?)
        }
        let h = median(hs)?;
        let u = combined(boxes.iter())?;
        if u.n[3] - u.n[1] > 24. * h {
            continue;
        }
        let mut b = c::merge(&bs, &chain, false)?;
        b["_recovered_inline_math_fragments"] = json!(true);
        replacements.insert(*chain.iter().min().unwrap(), b);
        used.extend(chain)
    }
    Ok(c::replace(bs, replacements, used))
}
fn local_union(b: &Value) -> Result<Option<Box4>, Error> {
    let rows = c::rows(b)?
        .into_iter()
        .filter(|b| b.n[2] > b.n[0] && b.n[3] > b.n[1])
        .collect::<Vec<_>>();
    if rows.is_empty() {
        Ok(None)
    } else {
        Ok(Some(combined(rows.iter())?))
    }
}
pub(super) fn formula_context(
    bs: Vec<Value>,
    p: [f64; 2],
    w: &mut Work,
) -> Result<Vec<Value>, Error> {
    let seeds = bs
        .iter()
        .enumerate()
        .filter(|(_, b)| b["type"] == "text" && b["_paragraph_formula_context"] == true)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    let (mut used, mut replacements) = (BTreeSet::new(), BTreeMap::new());
    for seed in seeds {
        if used.contains(&seed) {
            continue;
        }
        let mut history = vec![seed];
        loop {
            let mut changed = false;
            w.charge(bs.len() * history.len())?;
            for (i, a) in bs.iter().enumerate() {
                if history.contains(&i) || used.contains(&i) || a["type"] != "text" {
                    continue;
                }
                let members = super::super::table_native_vector::python_set::ordered_signed(
                    &history.iter().map(|&n| n as i64).collect::<Vec<_>>(),
                );
                for j in members {
                    let b = &bs[j as usize];
                    if c::angle(a)? != c::angle(b)? {
                        continue;
                    }
                    let mut hs = c::heights(a, true)?;
                    hs.extend(c::heights(b, true)?);
                    if hs.is_empty() {
                        continue;
                    }
                    let h = median(hs)?;
                    if !c::role(a, b, h) {
                        continue;
                    }
                    let (Some(ab), Some(bb)) = (local_union(a)?, local_union(b)?) else {
                        continue;
                    };
                    if gap(&ab, &bb, 1) > 1.5 * h {
                        continue;
                    }
                    let terminal = |b: &Value| {
                        regex(r#"[.!?。！？][\]\)}）】》”’'"]*$"#).is_match(
                            b["content"]
                                .as_str()
                                .unwrap_or("")
                                .trim_end_matches(python_space),
                        )
                    };
                    if ab.n[1] >= bb.n[3] && (a["_hard_break_before"] == true || terminal(b)) {
                        continue;
                    }
                    if bb.n[1] >= ab.n[3] && (b["_hard_break_before"] == true || terminal(a)) {
                        continue;
                    }
                    history.push(i);
                    changed = true;
                    break;
                }
                if changed {
                    break;
                }
            }
            if !changed {
                break;
            }
        }
        if history.len() < 2 {
            continue;
        }
        history.sort_unstable();
        let mut b = c::merge(&bs, &history, false)?;
        let rows = c::rows(&b)?;
        let max = rows.iter().map(|b| b.n[2] - b.n[0]).fold(0f64, f64::max);
        let body = rows
            .iter()
            .filter(|b| b.n[2] - b.n[0] >= 0.75 * max)
            .collect::<Vec<_>>();
        if body.len() >= 2 {
            let u = combined(rows.iter())?;
            let left = body.iter().map(|b| b.n[0]).fold(f64::INFINITY, f64::min);
            let right = body
                .iter()
                .map(|b| b.n[2])
                .fold(f64::NEG_INFINITY, f64::max);
            let local_box = bbox(&json!([left, u.n[1], right, u.n[3]]))?;
            let a = c::angle(&b)?;
            let lp = if a == 90 || a == 270 { [p[1], p[0]] } else { p };
            b["bbox"] = json!(local(&local_box, lp, (360 - a) % 360).raw)
        }
        replacements.insert(history[0], b);
        used.extend(history)
    }
    Ok(c::replace(bs, replacements, used))
}
