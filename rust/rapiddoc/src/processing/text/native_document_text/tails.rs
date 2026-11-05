//! Original footnote-projected tails and isolated first-page publication footers.
use super::*;
fn source(l: &Value) -> i64 {
    integer(l, "source_index", 0).unwrap()
}
fn order(a: &(usize, Box4), b: &(usize, Box4), lines: &[Value]) -> std::cmp::Ordering {
    a.1.n[1]
        .total_cmp(&b.1.n[1])
        .then(a.1.n[0].total_cmp(&b.1.n[0]))
        .then(source(&lines[a.0]).cmp(&source(&lines[b.0])))
}
pub(super) fn classify(pages: &mut [Value], w: &mut Work) -> Result<(), Error> {
    for p in pages.iter_mut() {
        trailing(p, w)?
    }
    if pages.len() >= 2 {
        isolated(&mut pages[0], w)?
    }
    Ok(())
}
fn anchors(lines: &[Value], group: &Value) -> Result<Vec<usize>, Error> {
    let mut by = BTreeMap::new();
    for (i, l) in lines.iter().enumerate() {
        by.insert(integer(l, "source_index", 0)?, i);
    }
    let mut out = Vec::new();
    for id in array(group, 4096)? {
        let id = id.as_i64().ok_or_else(invalid)?;
        if let Some(&i) = by.get(&id) {
            if lines[i]["semantic_type"] == "page_footnote" {
                out.push(i)
            }
        }
    }
    Ok(out)
}
fn trailing(p: &mut Value, w: &mut Work) -> Result<(), Error> {
    let s = sz(p)?;
    let mut lines = ls(p)?.to_vec();
    let mut groups = Vec::new();
    for g in optional(p, "page_footnote_groups", 4096)? {
        let ids = anchors(&lines, g)?;
        let angles = ids
            .iter()
            .map(|&i| integer(&lines[i], "angle", 0))
            .collect::<Result<BTreeSet<_>, _>>()?;
        if angles.len() != 1 {
            continue;
        }
        let a = *angles.first().unwrap();
        let bottom = ids
            .iter()
            .map(|&i| Ok(bs(&lines[i], s, a, false)?.n[3]))
            .collect::<Result<Vec<_>, Error>>()?
            .into_iter()
            .reduce(f64::max)
            .unwrap();
        groups.push((bottom, g.clone()));
    }
    groups.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (_, g) in groups {
        for i in members(p, &lines, &g, w)? {
            lines[i]["semantic_type"] = json!("footer")
        }
    }
    p["remaining_lines"] = json!(lines);
    Ok(())
}
fn members(p: &Value, lines: &[Value], g: &Value, w: &mut Work) -> Result<Vec<usize>, Error> {
    let s = sz(p)?;
    let ids = anchors(lines, g)?;
    let angles = ids
        .iter()
        .map(|&i| integer(&lines[i], "angle", 0))
        .collect::<Result<BTreeSet<_>, _>>()?;
    if angles.len() != 1 {
        return Ok(vec![]);
    }
    let a = *angles.first().unwrap();
    let lp = local_size(s, a);
    if lp[0] <= 0. || lp[1] <= 0. {
        return Ok(vec![]);
    }
    let anchors = ids
        .iter()
        .map(|&i| Ok((i, bs(&lines[i], s, a, false)?)))
        .collect::<Result<Vec<_>, Error>>()?;
    let ab = combined(anchors.iter().map(|(_, b)| b))?;
    if ab.n[3] < 0.75 * lp[1] {
        return Ok(vec![]);
    }
    let mut unresolved = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        w.charge(1)?;
        if l["semantic_type"].is_null() && integer(l, "angle", 0)? == a {
            unresolved.push((i, bs(l, s, a, false)?));
        }
    }
    let body = unresolved
        .iter()
        .filter(|(_, b)| {
            b.n[3] <= ab.n[1]
                && b.n[2] - b.n[0] >= 0.3 * lp[0]
                && cy(b) >= 0.1 * lp[1]
                && cy(b) <= 0.94 * lp[1]
                && axis(&ab, b, 0) >= 0.2
        })
        .collect::<Vec<_>>();
    if body.len() < 3 {
        return Ok(vec![]);
    }
    let em = median(
        body.iter()
            .map(|(i, b)| height(&lines[*i], b))
            .collect::<Result<Vec<_>, _>>()?,
    );
    if em <= 0.
        || unresolved
            .iter()
            .any(|(_, b)| b.n[1] <= ab.n[3] && ab.n[3] < b.n[3])
    {
        return Ok(vec![]);
    }
    let mut trailing = unresolved
        .into_iter()
        .filter(|(_, b)| b.n[1] > ab.n[3])
        .collect::<Vec<_>>();
    trailing.sort_by(|a, b| order(a, b, lines));
    if !(1..=3).contains(&trailing.len()) {
        return Ok(vec![]);
    }
    let first = &trailing[0];
    let cb = combined(trailing.iter().map(|(_, b)| b))?;
    let lefts = trailing.iter().map(|(_, b)| b.n[0]).collect::<Vec<_>>();
    if first.1.n[1] < 0.82 * lp[1]
        || cb.n[0] < ab.n[0] - 0.5 * em
        || cb.n[2] > ab.n[2] + 0.5 * em
        || (cb.n[0] - ab.n[0]).abs() > em
        || lefts.iter().copied().reduce(f64::max).unwrap()
            - lefts.iter().copied().reduce(f64::min).unwrap()
            > 0.75 * em
        || cb.n[2] - cb.n[0] > 0.6 * lp[0]
        || cb.n[3] - cb.n[1] > 4. * em
    {
        return Ok(vec![]);
    }
    for (i, b) in &trailing {
        if height(&lines[*i], b)? > 0.95 * em {
            return Ok(vec![]);
        }
    }
    let projecting = anchors
        .iter()
        .filter(|(_, b)| axis(b, &cb, 0) >= 0.2)
        .collect::<Vec<_>>();
    let Some(last) = projecting.into_iter().max_by(|a, b| order(a, b, lines)) else {
        return Ok(vec![]);
    };
    let continuation = (1.25 * em).max(0.01 * lp[1]);
    let gap = row_gap(&lines[last.0], &last.1, &first.1)?;
    if gap <= continuation || gap > 3. * em {
        return Ok(vec![]);
    }
    for pair in trailing.windows(2) {
        if row_gap(&lines[pair[0].0], &pair[0].1, &pair[1].1)? > continuation {
            return Ok(vec![]);
        }
    }
    let original = trailing
        .iter()
        .map(|(i, _)| bbox(&lines[*i]["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let cb = combined(original.iter())?;
    for block in optional(p, "fixed_blocks", 4000)? {
        if let Some(b) = coerce(&block["bbox"])?
            .map(|b| clip(&b, s))
            .transpose()?
            .flatten()
        {
            if intersects(&cb, &b) {
                return Ok(vec![]);
            }
        }
    }
    Ok(trailing.into_iter().map(|(i, _)| i).collect())
}
fn copyright(t: &str) -> bool {
    use rapiddoc_native::text_boundary::python_alnum;
    let original = t.chars().collect::<Vec<_>>();
    let cs = original
        .iter()
        .map(|&c| match c {
            'İ' | 'ı' => 'i',
            'ſ' => 's',
            'K' => 'k',
            _ => c.to_ascii_lowercase(),
        })
        .collect::<Vec<_>>();
    for i in 0..cs.len() {
        let end = if cs[i] == '©' {
            i + 1
        } else if cs[i..].starts_with(&['c', 'o', 'p', 'y', 'r', 'i', 'g', 'h', 't']) {
            i + 9
        } else {
            continue;
        };
        for j in end..cs.len() {
            if cs[j] == '\n' {
                break;
            }
            if cs[j..].starts_with(&['p', 'u', 'b', 'l', 'i', 's', 'h'])
                && (j == 0 || !(original[j - 1] == '_' || python_alnum(original[j - 1])))
            {
                return true;
            }
        }
    }
    false
}
fn isolated(p: &mut Value, w: &mut Work) -> Result<(), Error> {
    let s = sz(p)?;
    if s[0] <= 0. || s[1] <= 0. {
        return Ok(());
    }
    let mut lines = ls(p)?.to_vec();
    let mut body = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        if l["semantic_type"].is_null() && integer(l, "angle", 0)? == 0 {
            let b = bbox(&l["bbox"])?;
            if b.n[2] - b.n[0] >= 0.3 * s[0] && cy(&b) >= 0.1 * s[1] && cy(&b) <= 0.94 * s[1] {
                body.push((i, b))
            }
        }
    }
    if body.len() < 4 {
        return Ok(());
    }
    let em = median(
        body.iter()
            .map(|(i, b)| height(&lines[*i], b))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let mut copyrights = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        w.charge(1)?;
        if l["semantic_type"].is_null()
            && integer(l, "angle", 0)? == 0
            && bbox(&l["bbox"])?.n[1] >= 0.92 * s[1]
            && copyright(text(l)?)
        {
            copyrights.push(i)
        }
    }
    if copyrights.len() == 1 {
        let anchor = bbox(&lines[copyrights[0]]["bbox"])?;
        let mut members = Vec::new();
        for (i, l) in lines.iter().enumerate() {
            if l["semantic_type"].is_null() && integer(l, "angle", 0)? == 0 {
                let b = bbox(&l["bbox"])?;
                if (cy(&b) - cy(&anchor)).abs() <= 0.3 * em {
                    members.push((i, b))
                }
            }
        }
        let b = combined(members.iter().map(|(_, b)| b))?;
        let above = body
            .iter()
            .filter(|(_, bb)| bb.n[3] < b.n[1])
            .map(|(_, bb)| bb.n[3])
            .reduce(f64::max);
        let small = members
            .iter()
            .map(|(i, b)| Ok(height(&lines[*i], b)? <= em))
            .collect::<Result<Vec<_>, Error>>()?
            .into_iter()
            .all(|b| b);
        if b.n[2] - b.n[0] <= 0.6 * s[0] && above.is_some_and(|y| b.n[1] - y >= 0.5 * em) && small {
            for (i, _) in members {
                lines[i]["semantic_type"] = json!("footer")
            }
        }
    }
    let bottom = body.iter().map(|(_, b)| b.n[3]).reduce(f64::max).unwrap();
    if bottom >= 0.7 * s[1] {
        let containers = optional(p, "fixed_blocks", 4000)?
            .iter()
            .map(|b| coerce(&b["bbox"]))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        let mut candidates = Vec::new();
        for (i, l) in lines.iter().enumerate() {
            w.charge(containers.len() + 1)?;
            if l["semantic_type"].is_null() && integer(l, "angle", 0)? == 0 {
                let b = bbox(&l["bbox"])?;
                if b.n[1] >= 0.94 * s[1]
                    && b.n[2] - b.n[0] <= 0.6 * s[0]
                    && height(l, &b)? <= 0.95 * em
                    && b.n[1] - bottom >= 1.5 * em
                    && !containers.iter().any(|c| intersects(&b, c))
                {
                    candidates.push((i, b))
                }
            }
        }
        if !candidates.is_empty() {
            let b = combined(candidates.iter().map(|(_, b)| b))?;
            let min = candidates
                .iter()
                .map(|(_, b)| cy(b))
                .reduce(f64::min)
                .unwrap();
            let max = candidates
                .iter()
                .map(|(_, b)| cy(b))
                .reduce(f64::max)
                .unwrap();
            if b.n[2] - b.n[0] <= 0.6 * s[0]
                && (cx(&b) - 0.5 * s[0]).abs() <= 0.08 * s[0]
                && max - min <= 0.4 * em
            {
                for (i, _) in candidates {
                    lines[i]["semantic_type"] = json!("footer")
                }
            }
        }
    }
    p["remaining_lines"] = json!(lines);
    Ok(())
}
