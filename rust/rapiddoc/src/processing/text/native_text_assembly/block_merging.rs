//! Original block-level continuation passes, in assembly order.
use super::block_common as c;
use super::*;
fn first(b: &Value) -> Result<Box4, Error> {
    c::rows(b)?.first().cloned().ok_or_else(invalid)
}
fn last(b: &Value) -> Result<Box4, Error> {
    c::rows(b)?.last().cloned().ok_or_else(invalid)
}
fn height(b: &Value, minimum: bool) -> Result<f64, Error> {
    let hs = c::heights(b, true)?;
    if !hs.is_empty() {
        return median(hs);
    }
    let b = bbox(&b["bbox"])?;
    Ok(if minimum {
        (b.n[3] - b.n[1]).max(0.1)
    } else {
        b.n[3] - b.n[1]
    })
}
fn text(b: &Value) -> &str {
    b["content"].as_str().unwrap_or("")
}
fn list(t: &str) -> bool {
    regex(r"(?i)^\s*(?:[（(]\s*(?:\d+|[ivxlcdm]+)\s*[）)]|[①-⑳]|[•●▪])").is_match(&fold(t))
}
fn figure(t: &str) -> bool {
    regex(r"(?i)^(?:图\s*[0-9０-９一二三四五六七八九十]|fig(?:ure)?\.?\s*[0-9])").is_match(&fold(t))
}
fn url(t: &str) -> bool {
    regex(r"(?i)^\s*(?:https?://|www\.)").is_match(&fold(t))
}
fn label(t: &str) -> bool {
    regex(r"^\s*[A-Za-z㐀-鿿]{1,12}\s*[:：]").is_match(t)
}
pub(super) fn prefix(bs: Vec<Value>, p: [f64; 2]) -> Result<Vec<Value>, Error> {
    let (mut used, mut replacements) = (BTreeSet::new(), BTreeMap::new());
    for (i, b) in bs.iter().enumerate() {
        let rs = c::rows(b)?;
        let t = text(b).trim_matches(python_space);
        let bullet = ["•", "●", "▪"].contains(&t) && b["_protected_hard_break_before"] == true;
        if used.contains(&i)
            || b["type"] != "text"
            || rs.len() != 1
            || !(bullet || regex(r"^(?:[（(]\s*\d{1,3}|\d{1,2}\s*[:：]\s*\d{2})$").is_match(t))
        {
            continue;
        }
        let hs = c::heights(b, true)?;
        let h = if hs.is_empty() {
            (rs[0].n[3] - rs[0].n[1]).max(0.1)
        } else {
            median(hs)?
        };
        let angle = c::angle(b)?;
        let pw = if angle == 90 || angle == 270 {
            p[1]
        } else {
            p[0]
        };
        let mut matches = vec![];
        for (j, host) in bs.iter().enumerate() {
            let hr = c::rows(host)?;
            if i == j
                || used.contains(&j)
                || host["type"] != "text"
                || c::angle(host)? != angle
                || hr.is_empty()
            {
                continue;
            }
            let gap = hr[0].n[0] - rs[0].n[2];
            let w = hr[0].n[2] - hr[0].n[0];
            if gap < 0.
                || gap > (if bullet { 3.5 } else { 1.25 }) * h
                || w < (if bullet { 2. * h } else { 0.15 * pw })
                || axis(&rs[0], &hr[0], 1) < 0.5
            {
                continue;
            }
            matches.push((gap, j))
        }
        if let Some((_, j)) = matches.into_iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
            let mut b = c::merge(&bs, &[i, j], true)?;
            b["type"] = json!("text");
            replacements.insert(i.min(j), b);
            used.insert(i);
            used.insert(j);
        }
    }
    Ok(c::replace(bs, replacements, used))
}
fn boundary(a: &Value, b: &Value, h: f64) -> Result<bool, Error> {
    let ar = c::rows(a)?;
    let br = c::rows(b)?;
    if ar.len().max(br.len()) < 3 || ar.len() + br.len() > 6 || !c::role(a, b, h) {
        return Ok(false);
    }
    let (upper, lower) = if num(&a["bbox"][1])? <= num(&b["bbox"][1])? {
        (ar, br)
    } else {
        (br, ar)
    };
    let a = upper
        .iter()
        .max_by(|a, b| {
            cy(a)
                .total_cmp(&cy(b))
                .then(a.n[0].total_cmp(&b.n[0]))
                .then(std::cmp::Ordering::Greater)
        })
        .ok_or_else(invalid)?;
    let b = lower
        .iter()
        .min_by(|a, b| cy(a).total_cmp(&cy(b)).then(a.n[0].total_cmp(&b.n[0])))
        .ok_or_else(invalid)?;
    let min = (a.n[3] - a.n[1]).min(b.n[3] - b.n[1]).max(0.1);
    let overlap = (a.n[3].min(b.n[3]) - a.n[1].max(b.n[1])).max(0.);
    let gap = b.n[0] - a.n[2];
    Ok(overlap / min >= 0.7 && (-0.2 * h..=0.75 * h).contains(&gap))
}
pub(super) fn overlapping(bs: Vec<Value>, p: [f64; 2]) -> Result<Vec<Value>, Error> {
    let (mut used, mut replacements) = (BTreeSet::new(), BTreeMap::new());
    for (i, a) in bs.iter().enumerate() {
        let angle = c::angle(a)?;
        let pw = if angle == 90 || angle == 270 {
            p[1]
        } else {
            p[0]
        };
        let ar = c::rows(a)?;
        if used.contains(&i) || a["type"] != "text" || !(1..=5).contains(&ar.len()) {
            continue;
        }
        let ab = bbox(&a["bbox"])?;
        if ab.n[2] - ab.n[0] < 0.3 * pw {
            continue;
        }
        let ah = height(a, false)?;
        for j in i + 1..bs.len() {
            let b = &bs[j];
            let br = c::rows(b)?;
            if used.contains(&j)
                || b["type"] != "text"
                || c::angle(b)? != angle
                || !(1..=5).contains(&br.len())
            {
                continue;
            }
            let bb = bbox(&b["bbox"])?;
            if bb.n[2] - bb.n[0] < 0.3 * pw {
                continue;
            }
            let later = if c::key(a)? >= c::key(b)? { a } else { b };
            if later["_hard_break_before"] == true {
                continue;
            }
            let h = ah.max(height(b, false)?).max(0.1);
            let overlap = (ab.n[3].min(bb.n[3]) - ab.n[1].max(bb.n[1])).max(0.);
            let min = (ab.n[3] - ab.n[1]).min(bb.n[3] - bb.n[1]).max(0.1);
            let compact = ar.len() <= 2
                && br.len() <= 2
                && ar.len().min(br.len()) == 1
                && (ab.n[0] - bb.n[0]).abs() <= h
                && overlap / min >= 0.5
                && axis(&ab, &bb, 0) >= 0.75;
            let bodies = combined([&ab, &bb])?;
            let boundary = boundary(a, b, h)? && bodies.n[3] - bodies.n[1] <= 6. * h;
            if c::conflict(a, b)
                || !(compact || boundary)
                || (!boundary && bodies.n[3] - bodies.n[1] > 4. * h)
            {
                continue;
            }
            replacements.insert(i.min(j), c::merge(&bs, &[i, j], false)?);
            used.insert(i);
            used.insert(j);
            break;
        }
    }
    Ok(c::replace(bs, replacements, used))
}
fn root(parents: &mut [usize], mut i: usize) -> usize {
    while parents[i] != i {
        parents[i] = parents[parents[i]];
        i = parents[i]
    }
    i
}
fn union(p: &mut [usize], i: usize, j: usize) {
    let i = root(p, i);
    let j = root(p, j);
    if i != j {
        p[j] = i
    }
}
pub(super) fn spatial(bs: Vec<Value>, p: [f64; 2], w: &mut Work) -> Result<Vec<Value>, Error> {
    let mut parents = (0..bs.len()).collect::<Vec<_>>();
    let angles = bs
        .iter()
        .filter(|b| b["type"] == "text" && !c::rows(b).unwrap().is_empty())
        .map(c::angle)
        .collect::<Result<BTreeSet<_>, _>>()?;
    for angle in angles {
        let ids = bs
            .iter()
            .enumerate()
            .filter(|(_, b)| {
                b["type"] == "text"
                    && c::angle(b).ok() == Some(angle)
                    && !c::rows(b).unwrap().is_empty()
            })
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        if ids.len() < 2 {
            continue;
        }
        w.charge(ids.len() * ids.len() * 3)?;
        let pw = if angle == 90 || angle == 270 {
            p[1]
        } else {
            p[0]
        };
        let hs = ids
            .iter()
            .map(|&i| c::heights(&bs[i], true))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let h = if hs.is_empty() { 1. } else { median(hs)? };
        let mut starts = BTreeSet::new();
        for &i in &ids {
            let r = c::rows(&bs[i])?;
            if r.len() >= 2 {
                let reference = c::reference(&bs[i], pw);
                let aw = r[0].n[2] - r[0].n[0];
                let bw = r[1].n[2] - r[1].n[0];
                if aw <= 0.35 * reference && bw >= (0.25 * reference).max(1.8 * aw) {
                    starts.insert(i);
                }
            }
        }
        // Short opener matches use original tuple minimum (gap, body index).
        let mut pairs = vec![];
        for &i in &ids {
            let ar = c::rows(&bs[i])?;
            if ar.len() != 1 {
                continue;
            }
            let a = &ar[0];
            let mut matches = vec![];
            for &j in &ids {
                if i == j || bs[j]["_hard_break_before"] == true || !c::role(&bs[i], &bs[j], h) {
                    continue;
                }
                let b = first(&bs[j])?;
                let gap = b.n[1] - a.n[3];
                let reference = c::compatible(&bs[i], &bs[j], pw, h);
                if !(0.0..=0.75 * h).contains(&gap)
                    || (b.n[0] - a.n[0]).abs() > 0.75 * h
                    || a.n[2] - a.n[0] > 0.35 * reference
                    || b.n[2] - b.n[0] < 0.6 * reference
                    || c::skips(&bs, i, j, h)?
                {
                    continue;
                }
                matches.push((gap, j))
            }
            if let Some((_, j)) = matches.into_iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
                pairs.push((i, j))
            }
        }
        for (i, j) in pairs {
            starts.insert(i);
            union(&mut parents, i, j)
        }
        let mut sorted = ids.clone();
        sorted.sort_by(|&i, &j| {
            c::key(&bs[i])
                .unwrap()
                .partial_cmp(&c::key(&bs[j]).unwrap())
                .unwrap()
        });
        for i in sorted {
            let roots = starts
                .iter()
                .map(|&i| root(&mut parents, i))
                .collect::<BTreeSet<_>>();
            if !roots.contains(&root(&mut parents, i)) {
                continue;
            }
            let a = last(&bs[i])?;
            let max = 0.75 * h;
            let mut matches = vec![];
            for &j in &ids {
                if i == j
                    || starts.contains(&j)
                    || bs[j]["_hard_break_before"] == true
                    || !c::role(&bs[i], &bs[j], max / 0.75)
                {
                    continue;
                }
                let b = first(&bs[j])?;
                let gap = b.n[1] - a.n[3];
                let dx = (b.n[0] - a.n[0]).abs();
                if !(-0.25 * max..=max).contains(&gap)
                    || (axis(&a, &b, 0) < 0.5 && dx > 3. * max)
                    || c::skips(&bs, i, j, max / 0.75)?
                {
                    continue;
                }
                matches.push((gap.max(0.), dx, j))
            }
            if let Some((_, _, j)) = matches.into_iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
                union(&mut parents, i, j)
            }
        }
        for &i in &ids {
            let raw = bbox(&bs[i]["bbox"])?;
            let parallel = ids.iter().any(|&j| {
                if i == j {
                    return false;
                }
                let b = bbox(&bs[j]["bbox"]).unwrap();
                (raw.n[3].min(b.n[3]) - raw.n[1].max(b.n[1])).max(0.) > 0.
                    && (b.n[0] >= raw.n[2] || raw.n[0] >= b.n[2])
            });
            if !parallel {
                continue;
            }
            let a = last(&bs[i])?;
            let mut matches = vec![];
            for &j in &ids {
                if i == j
                    || starts.contains(&j)
                    || bs[j]["_hard_break_before"] == true
                    || !c::role(&bs[i], &bs[j], h)
                {
                    continue;
                }
                let br = c::rows(&bs[j])?;
                let b = &br[0];
                let last = br.last().unwrap();
                let gap = b.n[1] - a.n[3];
                let bw = b.n[2] - b.n[0];
                if !(0.75 * h < gap && gap <= 1.5 * h)
                    || br.len() > 2
                    || (b.n[0] - a.n[0]).abs() > 0.75 * h
                    || bw > 0.85 * (a.n[2] - a.n[0])
                    || last.n[2] - last.n[0] > bw + 0.5 * h
                    || c::skips(&bs, i, j, h)?
                {
                    continue;
                }
                matches.push((gap, j))
            }
            if let Some((_, j)) = matches.into_iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
                union(&mut parents, i, j)
            }
        }
    }
    let mut groups: Vec<(usize, Vec<usize>)> = vec![];
    for i in 0..bs.len() {
        let r = root(&mut parents, i);
        if let Some((_, g)) = groups.iter_mut().find(|(k, _)| *k == r) {
            g.push(i)
        } else {
            groups.push((r, vec![i]))
        }
    }
    groups
        .into_iter()
        .map(|(_, ids)| {
            if ids.len() == 1 {
                Ok(bs[ids[0]].clone())
            } else {
                c::merge(&bs, &ids, false)
            }
        })
        .collect()
}
pub(super) fn list_intro(mut bs: Vec<Value>) -> Result<Vec<Value>, Error> {
    // Preserve the shallow snapshot's object identity while the live list changes.
    let mut objects = (0..bs.len()).collect::<Vec<_>>();
    let snapshot = bs.clone();
    let mut serial = bs.len();
    for (original, boundary) in snapshot.iter().enumerate() {
        if boundary["type"] != "text" || !regex(r"^\s*[（(]1[）)]\s").is_match(text(boundary)) {
            continue;
        }
        let rs = c::rows(boundary)?;
        if rs.is_empty() || boundary["_protected_hard_break_before"] == true {
            continue;
        }
        let hs = c::heights(boundary, false)?;
        let h = if hs.is_empty() {
            rs[0].n[3] - rs[0].n[1]
        } else {
            median(hs)?
        };
        let mut intros = vec![];
        for (i, b) in bs.iter().enumerate() {
            if objects[i] == original || b["type"] != "text" || c::rows(b)?.len() != 1 {
                continue;
            }
            let bb = bbox(&b["bbox"])?;
            if (0.0..=0.5 * h).contains(&(rs[0].n[1] - bb.n[3]))
                && (rs[0].n[0] - bb.n[0]).abs() < 0.3 * h
                && eq(&b["_font_signatures"], &boundary["_font_signatures"])
            {
                intros.push(i)
            }
        }
        if intros.len() == 1 {
            let intro = bs[intros[0]].clone();
            let i = bs.iter().position(|b| eq(b, &intro)).ok_or_else(invalid)?;
            let merged = c::merge(&[intro, boundary.clone()], &[0, 1], false)?;
            bs[i] = merged;
            objects[i] = serial;
            serial += 1;
            // Python list.remove compares values, even when the snapshot object
            // has already been replaced; preserve its original ValueError.
            let pos = bs
                .iter()
                .position(|b| eq(b, boundary))
                .ok_or_else(invalid)?;
            bs.remove(pos);
            objects.remove(pos);
        }
    }
    let ids = bs
        .iter()
        .enumerate()
        .filter(|(_, b)| b["type"] == "text" && !c::rows(b).unwrap().is_empty())
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    let (mut used, mut replacements) = (BTreeSet::new(), BTreeMap::new());
    for &boundary in &ids {
        if bs[boundary]["_hard_break_before"] != true || !list(text(&bs[boundary])) {
            continue;
        }
        let preceding = ids
            .iter()
            .copied()
            .filter(|i| {
                !used.contains(i) && c::key(&bs[*i]).unwrap() < c::key(&bs[boundary]).unwrap()
            })
            .collect::<Vec<_>>();
        let Some(&immediate) = preceding.iter().max_by(|&&i, &&j| {
            c::key(&bs[i])
                .unwrap()
                .partial_cmp(&c::key(&bs[j]).unwrap())
                .unwrap()
                .then(std::cmp::Ordering::Greater)
        }) else {
            continue;
        };
        let r = last(&bs[immediate])?;
        let hs = c::heights(&bs[immediate], true)?;
        let h = if hs.is_empty() {
            r.n[3] - r.n[1]
        } else {
            median(hs)?
        };
        let Some((l, rgt)) = c::interval(&bs[immediate], false) else {
            continue;
        };
        if r.n[2] - r.n[0] > 0.35 * (rgt - l)
            || !text(&bs[immediate])
                .trim_end_matches(python_space)
                .ends_with([':', '：'])
            || !c::role(&bs[immediate], &bs[boundary], h)
        {
            continue;
        }
        let mut group = vec![immediate];
        let mut cursor = immediate;
        while group.len() < 3 {
            let previous = preceding
                .iter()
                .copied()
                .filter(|i| {
                    !group.contains(i)
                        && c::key(&bs[*i]).unwrap() < c::key(&bs[cursor]).unwrap()
                        && c::role(&bs[*i], &bs[cursor], h)
                })
                .max_by(|&i, &j| {
                    c::key(&bs[i])
                        .unwrap()
                        .partial_cmp(&c::key(&bs[j]).unwrap())
                        .unwrap()
                        .then(std::cmp::Ordering::Greater)
                });
            let Some(previous) = previous else { break };
            let mut hs = c::heights(&bs[previous], true)?;
            hs.extend(c::heights(&bs[cursor], true)?);
            let lh = if hs.is_empty() { h } else { median(hs)? };
            let gap = first(&bs[cursor])?.n[1] - last(&bs[previous])?.n[3];
            if bs[previous]["_hard_break_before"] == true
                || !c::role(&bs[previous], &bs[cursor], lh)
                || !(-lh..=lh).contains(&gap)
            {
                break;
            }
            group.push(previous);
            cursor = previous
        }
        if group.len() >= 2 {
            replacements.insert(*group.iter().min().unwrap(), c::merge(&bs, &group, false)?);
            used.extend(group)
        }
    }
    Ok(c::replace(bs, replacements, used))
}
pub(super) fn unterminated(mut bs: Vec<Value>, w: &mut Work) -> Result<Vec<Value>, Error> {
    loop {
        let mut ids = bs
            .iter()
            .enumerate()
            .filter(|(_, b)| b["type"] == "text" && !c::rows(b).unwrap().is_empty())
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        ids.sort_by(|&i, &j| {
            c::key(&bs[i])
                .unwrap()
                .partial_cmp(&c::key(&bs[j]).unwrap())
                .unwrap()
        });
        w.charge(ids.len() * ids.len())?;
        let mut pairs = vec![];
        for (pos, &i) in ids.iter().enumerate() {
            let a = last(&bs[i])?;
            if let Some(j) = ids[pos + 1..]
                .iter()
                .copied()
                .find(|&j| axis(&a, &first(&bs[j]).unwrap(), 0) >= 0.5)
            {
                pairs.push((i, j))
            }
        }
        let mut found = None;
        for (i, j) in pairs {
            let (a, b) = (&bs[i], &bs[j]);
            let ar = c::rows(a)?;
            let br = c::rows(b)?;
            let first = ar.last().unwrap();
            let second = &br[0];
            let mut hs = c::heights(a, true)?;
            hs.extend(c::heights(b, true)?);
            let h = if hs.is_empty() {
                median(vec![first.n[3] - first.n[1], second.n[3] - second.n[1]])?
            } else {
                median(hs)?
            };
            let ai = c::interval(a, true);
            let bi = c::interval(b, true);
            let bnormal = c::interval(b, false);
            let row_h = median(vec![
                (first.n[3] - first.n[1]).max(0.1),
                (second.n[3] - second.n[1]).max(0.1),
            ])?;
            let span_h = h.max(row_h);
            let span = a["_lane_is_span"] == true
                && b["_lane_is_span"] == true
                && c::angle(a)? == c::angle(b)?
                && ai.zip(bi).is_some_and(|(a, b)| {
                    (a.0 - b.0).abs() <= 0.75 * span_h && (a.1 - b.1).abs() <= 0.75 * span_h
                });
            let at = text(a);
            let bt = text(b);
            let numbered = br.len() == 1
                && b["_hard_break_before"] != true
                && list(bt)
                && at.trim_end_matches(python_space).ends_with([':', '：']);
            let url = url(bt);
            let short =
                ar.len() >= 2 && br.len() == 1 && (second.n[0] - first.n[0]).abs() <= 0.75 * h;
            let narrow = numbered || url || short;
            let wide = !url
                && label(bt)
                && (if span { bi } else { bnormal })
                    .is_some_and(|(l, r)| second.n[2] - second.n[0] >= 0.5 * (r - l));
            if b["_protected_hard_break_before"] == true
                || (b["_hard_break_before"] == true && (span || !narrow))
                || b["_leading_emphasis_start"] == true
                || (wide && (span || !short))
                || !a["_hanging_indent_group"].is_null()
                || !b["_hanging_indent_group"].is_null()
                || figure(at)
                || (!span
                    && !numbered
                    && !url
                    && regex(r#"[.!?。！？:：;；][\]\)}）】》”’'"]*$"#)
                        .is_match(at.trim_end_matches(python_space)))
            {
                continue;
            }
            let actual = a["_lane_is_span"] == false
                && b["_lane_is_span"] == false
                && ai
                    .zip(bi)
                    .is_some_and(|(a, b)| b.0 <= a.0 - 3. * h && (b.1 - a.1).abs() <= 0.75 * h)
                && ar.len() >= 2
                && bt
                    .trim_start_matches(python_space)
                    .chars()
                    .next()
                    .is_some_and(char::is_lowercase)
                && (0.0..=0.5 * h).contains(&(second.n[1] - first.n[3]))
                && (second.n[0] - first.n[0]).abs() <= 0.25 * h
                && ((second.n[2] - first.n[2]).abs() <= 0.25 * h
                    || br.len() == 1
                        && second.n[2] - second.n[0] >= 0.5 * (first.n[2] - first.n[0]))
                && first.n[2] - first.n[0]
                    >= 0.85
                        * (ar.iter().map(|b| b.n[2]).fold(f64::NEG_INFINITY, f64::max)
                            - ar.iter().map(|b| b.n[0]).fold(f64::INFINITY, f64::min))
                && a["_font_signatures"]
                    .as_array()
                    .is_some_and(|a| !a.is_empty())
                && eq(&a["_font_signatures"], &b["_font_signatures"]);
            let interval = if span {
                ai
            } else if actual {
                Some((
                    ar.iter().map(|b| b.n[0]).fold(f64::INFINITY, f64::min),
                    ar.iter().map(|b| b.n[2]).fold(f64::NEG_INFINITY, f64::max),
                ))
            } else {
                if !c::role(a, b, h) {
                    continue;
                }
                c::interval(a, false)
            };
            let Some((l, r)) = interval else { continue };
            let width = r - l;
            let first_fill = if span {
                0.8
            } else if numbered {
                0.5
            } else {
                0.65
            };
            let second_fill = if span { 0.8 } else { 0.65 };
            let ch = if span { span_h } else { h };
            let first_w = if narrow {
                ar.iter()
                    .map(|b| b.n[2] - b.n[0])
                    .fold(f64::NEG_INFINITY, f64::max)
            } else {
                first.n[2] - first.n[0]
            };
            if first_w < first_fill * width
                || ((span || !narrow) && second.n[2] - second.n[0] < second_fill * width)
                || !(-ch..=1.5 * ch).contains(&(second.n[1] - first.n[3]))
                || c::conflict(a, b)
                || c::skips(&bs, i, j, ch)?
            {
                continue;
            }
            found = Some((i, j));
            break;
        }
        let Some((i, j)) = found else { return Ok(bs) };
        bs[i] = c::merge(&bs, &[i, j], false)?;
        bs.remove(j);
    }
}
