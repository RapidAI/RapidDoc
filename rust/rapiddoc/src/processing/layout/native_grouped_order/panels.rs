//! Original numbered figure rows, parallel headings and numbered-step member groups.
use super::*;
pub(super) fn numbered_regions(
    regions: Vec<Vec<usize>>,
    bs: &[Value],
    w: &mut Work,
) -> Result<Vec<Vec<usize>>, Error> {
    let mut panels = vec![];
    let re = regex(r"(?i)^\s*(?:图|Figure\s*|Fig\.?\s*)(\d+)(?:\s|[:：.])");
    for (i, g) in regions.iter().enumerate() {
        let caps = g
            .iter()
            .copied()
            .filter(|&i| bs[i]["type"] == "caption")
            .collect::<Vec<_>>();
        let bodies = g
            .iter()
            .copied()
            .filter(|&i| bs[i]["type"] == "image")
            .collect::<Vec<_>>();
        if caps.len() != 1 || bodies.len() != 1 {
            continue;
        }
        let caption = caps[0];
        let Some(c) = re.captures(text(&bs[caption])) else {
            continue;
        };
        if bb(&bs[caption])?.n[1] > bb(&bs[bodies[0]])?.n[1] {
            continue;
        }
        let mut number = 0u64;
        for ch in c[1].chars() {
            let d = rapiddoc_native::python_decimal(ch).ok_or_else(invalid)?;
            number = number
                .checked_mul(10)
                .and_then(|n| n.checked_add(u64::from(d)))
                .ok_or_else(limit)?;
        }
        panels.push((i, caption, number))
    }
    if panels.len() < 4 {
        return Ok(regions);
    }
    let em = median(
        panels
            .iter()
            .map(|(_, i, _)| {
                let b = bb(&bs[*i]).unwrap();
                b.n[3] - b.n[1]
            })
            .collect(),
    );
    if em <= 0. {
        return Ok(regions);
    }
    panels.sort_by(|a, b| {
        let a = bb(&bs[a.1]).unwrap();
        let b = bb(&bs[b.1]).unwrap();
        a.n[1].total_cmp(&b.n[1]).then(a.n[0].total_cmp(&b.n[0]))
    });
    let mut rows: Vec<Vec<(usize, usize, u64)>> = vec![];
    for panel in &panels {
        if rows.last().is_none_or(|r| {
            (bb(&bs[panel.1]).unwrap().n[1] - bb(&bs[r[0].1]).unwrap().n[1]).abs() > 0.75 * em
        }) {
            rows.push(vec![])
        }
        rows.last_mut().unwrap().push(*panel)
    }
    if rows.len() < 2
        || rows
            .iter()
            .any(|r| r.len() != rows[0].len() || !(2..=3).contains(&r.len()))
    {
        return Ok(regions);
    }
    for row in &mut rows {
        row.sort_by(|a, b| bb(&bs[a.1]).unwrap().n[0].total_cmp(&bb(&bs[b.1]).unwrap().n[0]))
    }
    let ordered = rows.iter().flatten().collect::<Vec<_>>();
    if ordered
        .iter()
        .enumerate()
        .any(|(i, p)| p.2 != ordered[0].2 + i as u64)
    {
        return Ok(regions);
    }
    for row in rows.iter().skip(1) {
        for (k, p) in row.iter().enumerate() {
            if (bb(&bs[p.1])?.n[0] - bb(&bs[rows[0][k].1])?.n[0]).abs() > 0.75 * em {
                return Ok(regions);
            }
        }
    }
    let members = panels
        .iter()
        .flat_map(|(i, _, _)| regions[*i].iter().copied())
        .collect::<Vec<_>>();
    let bounds = union(bs, &members)?;
    for (i, b) in bs.iter().enumerate() {
        w.charge(1)?;
        if !members.contains(&i)
            && [
                "text",
                "paragraph_title",
                "doc_title",
                "table",
                "equation",
                "code",
            ]
            .iter()
            .any(|t| b["type"] == *t)
            && axis(&bb(b)?, &bounds, 0) >= 0.1
            && axis(&bb(b)?, &bounds, 1) >= 0.1
        {
            return Ok(regions);
        }
    }
    let selected = panels.iter().map(|(i, _, _)| *i).collect::<BTreeSet<_>>();
    let mut out = regions
        .iter()
        .enumerate()
        .filter(|(i, _)| !selected.contains(i))
        .map(|(_, g)| g.clone())
        .collect::<Vec<_>>();
    for row in rows {
        out.push(
            row.iter()
                .flat_map(|(i, _, _)| regions[*i].iter().copied())
                .collect(),
        )
    }
    Ok(out)
}
pub(super) fn parallel(
    bs: &[Value],
    excluded: &BTreeSet<usize>,
    w: &mut Work,
) -> Result<Vec<Vec<usize>>, Error> {
    let mut titles = bs
        .iter()
        .enumerate()
        .filter(|(i, b)| {
            !excluded.contains(i)
                && b["type"] == "paragraph_title"
                && b.get("angle").map_or(true, |v| eq(v, &json!(0)))
        })
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    titles.sort_by(|&i, &j| bb(&bs[i]).unwrap().n[1].total_cmp(&bb(&bs[j]).unwrap().n[1]));
    let mut bands: Vec<Vec<usize>> = vec![];
    for i in titles {
        let b = bb(&bs[i])?;
        if let Some(band) = bands.last() {
            if (b.n[1] - bb(&bs[band[0]])?.n[1]).abs() <= 0.5 * (b.n[3] - b.n[1]) {
                bands.last_mut().unwrap().push(i);
                continue;
            }
        }
        bands.push(vec![i])
    }
    let mut out = vec![];
    for mut band in bands {
        if band.len() < 3 {
            continue;
        }
        band.sort_by(|&i, &j| bb(&bs[i]).unwrap().n[0].total_cmp(&bb(&bs[j]).unwrap().n[0]));
        let em = median(
            band.iter()
                .map(|&i| {
                    let b = bb(&bs[i]).unwrap();
                    b.n[3] - b.n[1]
                })
                .collect(),
        );
        if em <= 0.
            || !truth(&bs[band[0]]["_font_signatures"])
            || band.iter().any(|&i| {
                let b = bb(&bs[i]).unwrap();
                !(0.9 * em..=1.1 * em).contains(&(b.n[3] - b.n[1]))
                    || !eq(&bs[i]["_font_signatures"], &bs[band[0]]["_font_signatures"])
            })
            || band
                .windows(2)
                .any(|p| bb(&bs[p[1]]).unwrap().n[0] - bb(&bs[p[0]]).unwrap().n[2] < 2. * em)
        {
            continue;
        }
        let mut groups = vec![];
        for (k, &i) in band.iter().enumerate() {
            let t = bb(&bs[i])?;
            let left = t.n[0];
            let right = if let Some(&j) = band.get(k + 1) {
                bb(&bs[j])?.n[0] - em
            } else {
                f64::INFINITY
            };
            let mut members = vec![];
            let mut preceding = vec![];
            for (j, b) in bs.iter().enumerate() {
                w.charge(1)?;
                if excluded.contains(&j)
                    || i == j
                    || !b.get("angle").map_or(true, |v| eq(v, &json!(0)))
                {
                    continue;
                }
                let bbb = bb(b)?;
                if (bbb.n[0] - left).abs() > 0.35 * em || bbb.n[2] > right {
                    continue;
                }
                if ["text", "image"].iter().any(|t| b["type"] == *t) && bbb.n[1] >= t.n[3] {
                    members.push(j)
                }
                if b["type"] == "paragraph_title" && (0.0..=2. * em).contains(&(t.n[1] - bbb.n[3]))
                {
                    preceding.push(j)
                }
            }
            members.sort_by(|&i, &j| bb(&bs[i]).unwrap().n[1].total_cmp(&bb(&bs[j]).unwrap().n[1]));
            if members.is_empty()
                || !(0.5 * em..=3. * em).contains(&(bb(&bs[members[0]])?.n[1] - t.n[3]))
            {
                break;
            }
            let mut bottom = bb(&bs[members[0]])?.n[3];
            let mut accepted = vec![members[0]];
            for &j in members.iter().skip(1) {
                let b = bb(&bs[j])?;
                if b.n[1] - bottom > 3. * em {
                    break;
                }
                accepted.push(j);
                bottom = b.n[3]
            }
            preceding.push(i);
            preceding.extend(accepted);
            groups.push(preceding)
        }
        if groups.len() != band.len() {
            continue;
        }
        let selected = groups.iter().flatten().copied().collect::<BTreeSet<_>>();
        let b = union(bs, &selected.iter().copied().collect::<Vec<_>>())?;
        let mut overlap = false;
        for (i, other) in bs.iter().enumerate() {
            if selected.contains(&i)
                || ["header", "footer", "page_number"]
                    .iter()
                    .any(|t| other["type"] == *t)
            {
                continue;
            }
            let o = bb(other)?;
            if b.n[2].min(o.n[2]) > b.n[0].max(o.n[0]) && b.n[3].min(o.n[3]) > b.n[1].max(o.n[1]) {
                overlap = true;
                break;
            }
        }
        if !overlap {
            out.extend(groups)
        }
    }
    Ok(out)
}
pub(super) fn steps(
    bs: &[Value],
    excluded: &BTreeSet<usize>,
    w: &mut Work,
) -> Result<Vec<Vec<usize>>, Error> {
    let available = bs
        .iter()
        .enumerate()
        .filter(|(i, b)| {
            !excluded.contains(i)
                && b["type"] == "text"
                && b.get("angle").map_or(true, |v| eq(v, &json!(0)))
                && b["content"].is_string()
        })
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    let mut children = available
        .iter()
        .copied()
        .filter(|&i| regex(r"^[a-z][.)]\s").is_match(text(&bs[i]).trim_matches(python_space)))
        .collect::<Vec<_>>();
    children.sort_by(|&i, &j| {
        let a = bb(&bs[i]).unwrap();
        let b = bb(&bs[j]).unwrap();
        a.n[1].total_cmp(&b.n[1]).then(a.n[0].total_cmp(&b.n[0]))
    });
    let mut out = vec![];
    for &i in &children {
        if !text(&bs[i])
            .trim_start_matches(python_space)
            .starts_with('a')
        {
            continue;
        }
        let first = bb(&bs[i])?;
        let em = first.n[3] - first.n[1];
        let mut run = vec![i];
        for &j in &children {
            w.charge(1)?;
            let prev = *run.last().unwrap();
            let pb = bb(&bs[prev])?;
            let cb = bb(&bs[j])?;
            if cb.n[1] <= pb.n[1] {
                continue;
            }
            let c = text(&bs[j])
                .trim_start_matches(python_space)
                .chars()
                .next()
                .ok_or_else(invalid)? as u32;
            let pc = text(&bs[prev])
                .trim_start_matches(python_space)
                .chars()
                .next()
                .ok_or_else(invalid)? as u32;
            if c == pc + 1
                && (cb.n[0] - first.n[0]).abs() <= 0.3 * em
                && (-0.15 * em..=em).contains(&(cb.n[1] - pb.n[3]))
            {
                run.push(j)
            } else {
                break;
            }
        }
        if run.len() < 3 {
            continue;
        }
        let parents = available
            .iter()
            .copied()
            .filter(|&j| {
                let b = bb(&bs[j]).unwrap();
                regex(r"^\d+[.)]\s").is_match(text(&bs[j]).trim_matches(python_space))
                    && (0.5 * em..=4. * em).contains(&(first.n[0] - b.n[0]))
                    && (-0.2 * em..=1.5 * em).contains(&(first.n[1] - b.n[3]))
            })
            .collect::<Vec<_>>();
        if parents.len() != 1 {
            continue;
        }
        let parent = parents[0];
        let t = text(&bs[parent]).trim_matches(python_space);
        let mut number = 0u64;
        for ch in t
            .chars()
            .take_while(|&c| rapiddoc_native::python_decimal(c).is_some())
        {
            number = number
                .checked_mul(10)
                .and_then(|n| {
                    n.checked_add(u64::from(rapiddoc_native::python_decimal(ch).unwrap()))
                })
                .ok_or_else(limit)?;
        }
        let mut following = 0;
        for &j in &available {
            let b = bb(&bs[j])?;
            if regex(&format!(r"^{}[.)]\s", number + 1))
                .is_match(text(&bs[j]).trim_matches(python_space))
                && (b.n[0] - bb(&bs[parent])?.n[0]).abs() <= 0.3 * em
                && (-0.2 * em..=2. * em).contains(&(b.n[1] - bb(&bs[*run.last().unwrap()])?.n[3]))
            {
                following += 1
            }
        }
        if following == 1 {
            let mut g = vec![parent];
            g.extend(run);
            out.push(g)
        }
    }
    Ok(out)
}
