//! Original repeated-number/right-edge anchors and source-stable spatial growth.
use super::evidence as e;
use super::*;
fn dedup(mut anchors: Vec<Anchor>, em: f64) -> Vec<Anchor> {
    anchors.sort_by(|a, b| {
        cy(&a.row.b)
            .total_cmp(&cy(&b.row.b))
            .then(b.row.b.n[2].total_cmp(&a.row.b.n[2]))
    });
    let mut out = Vec::<Anchor>::new();
    for a in anchors {
        if let Some(last) = out.last_mut() {
            if (cy(&a.row.b) - cy(&last.row.b)).abs() <= 1.5f64.max(0.35 * em) {
                if a.row.b.n[2] > last.row.b.n[2]
                    || a.row.b.n[2] == last.row.b.n[2] && a.row.b.n[0] < last.row.b.n[0]
                {
                    *last = a
                }
                continue;
            }
        }
        out.push(a)
    }
    out
}
fn row_prose(p: &Page, lane: &Lane, a: &Row, fragments: &[Row], em: f64) -> Result<bool, Error> {
    if fragments.len() < 3 || p.l(a)["visual_row_id"].is_null() {
        return Ok(false);
    }
    let sources = fragments
        .iter()
        .map(|r| p.id(r))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let geometry = lane
        .rows
        .iter()
        .filter(|r| sources.contains(&p.id(r).unwrap()))
        .cloned()
        .collect::<Vec<_>>();
    if geometry.len() < 3 {
        return Ok(false);
    }
    let width = (lane.right - lane.left).max(0.1);
    let b = p.bounds(&geometry)?;
    if b.n[2] - b.n[0] < 0.75 * width {
        return Ok(false);
    }
    let center = median(geometry.iter().map(|r| cy(&r.b)).collect());
    for r in &geometry {
        let h = p.h(r)?;
        if (cy(&r.b) - center).abs() > 0.25 * em || h < 0.7 * em || h > 1.3 * em {
            return Ok(false);
        }
    }
    let mut following = None;
    for r in &lane.rows {
        if !sources.contains(&p.id(r)?)
            && cy(&r.b) > center + 0.5 * em
            && following.is_none_or(|old: &Row| {
                cy(&r.b) < cy(&old.b) || cy(&r.b) == cy(&old.b) && r.b.n[0] < old.b.n[0]
            })
        {
            following = Some(r)
        }
    }
    let Some(r) = following else { return Ok(false) };
    let h = p.h(r)?;
    Ok(r.b.n[2] - r.b.n[0] >= 0.6 * width
        && (r.b.n[0] - lane.left).abs() <= 0.75 * em
        && h >= 0.75 * em
        && h <= 1.3 * em
        && r.b.n[1] - b.n[3] <= 1.25 * em)
}
pub(super) fn find(
    p: &Page,
    lane: &Lane,
    em: f64,
    font: &Value,
    w: &mut Work,
) -> Result<Vec<Anchor>, Error> {
    let width = (lane.right - lane.left).max(0.1);
    let interval = e::interval(p, lane, em)?;
    let markers = lane
        .rows
        .iter()
        .filter(|r| {
            split_number(p.t(r).unwrap()).is_some_and(|(a, _)| a.is_empty())
                && (lane.right - r.b.n[2]).abs() <= 3f64.max(0.02 * width)
        })
        .collect::<Vec<_>>();
    let mut anchors = Vec::new();
    for &r in &markers {
        w.charge(markers.len() + lane.rows.len() + 1)?;
        if !markers.iter().any(|other| {
            p.id(other).unwrap() != p.id(r).unwrap() && (cy(&other.b) - cy(&r.b)).abs() <= 6. * em
        }) {
            continue;
        }
        let mut peers = Vec::new();
        for other in &lane.rows {
            if p.id(other)? != p.id(r)? && cx(&other.b) < r.b.n[0] && seed(p, r, other, true)? {
                peers.push(other)
            }
        }
        if peers.len() < 2
            || !peers.iter().any(|r| {
                p.l(r)["formula_candidate_only"] == true
                    || p.l(r)["compact_formula_cluster"] == true
                    || p.coverage(r).unwrap() < 0.75
            })
        {
            continue;
        }
        anchors.push(Anchor {
            row: r.clone(),
            above: interval.is_none_or(|(top, _)| cy(&r.b) < top),
            below: interval.is_some_and(|(_, bottom)| cy(&r.b) > bottom),
            repeated: true,
        })
    }
    let Some((top, bottom)) = interval else {
        return Ok(dedup(anchors, em));
    };
    let repeated = anchors
        .iter()
        .map(|a| p.id(&a.row))
        .collect::<Result<BTreeSet<_>, _>>()?;
    for r in &lane.rows {
        if repeated.contains(&p.id(r)?) {
            continue;
        }
        let h = p.h(r)?;
        let rw = r.b.n[2] - r.b.n[0];
        let short = rw <= (4. * h).max(0.12 * width);
        let suffix = split_number(p.t(r)?).is_some();
        let wide = suffix
            && rw <= 0.75 * width
            && !font.is_null()
            && (!eq(&p.l(r)["font_signature"], font) || p.coverage(r)? < 0.75);
        if !short && !wide {
            continue;
        }
        let fragments = lane
            .rows
            .iter()
            .filter(|other| {
                !p.l(r)["visual_row_id"].is_null()
                    && eq(&p.l(other)["visual_row_id"], &p.l(r)["visual_row_id"])
                    && (p.l(r)["split_from_row"] == true || p.l(other)["split_from_row"] == true)
            })
            .cloned()
            .collect::<Vec<_>>();
        w.charge(fragments.len() + lane.rows.len() + 1)?;
        if fragments.len() >= 3
            && !fragments
                .iter()
                .filter(|other| p.id(other).unwrap() != p.id(r).unwrap())
                .any(|other| {
                    p.coverage(other).unwrap() < 0.75
                        || !font.is_null() && !eq(&p.l(other)["font_signature"], font)
                })
        {
            continue;
        }
        if row_prose(p, lane, r, &fragments, em)? {
            continue;
        }
        if (lane.right - r.b.n[2]).abs() > 3f64.max(0.02 * width) {
            continue;
        }
        let y = cy(&r.b);
        let below = bottom < y && y <= bottom + 6. * em;
        let above = top - 6. * em <= y && y < top;
        if !(top <= y && y <= bottom) && !below && !above {
            continue;
        }
        let mut peers = Vec::new();
        for other in &lane.rows {
            if p.id(other)? == p.id(r)? {
                continue;
            }
            let tagged = e::wide(p, r, other, width)?;
            if (other.b.n[2] - other.b.n[0] <= 0.75 * width || tagged)
                && cx(&other.b) < r.b.n[0]
                && seed(p, r, other, below || above || tagged)?
            {
                peers.push(other)
            }
        }
        if short && !suffix {
            let ordinary = crate::cached_regex!(r"^[A-Za-z]{3,}[.,]?$");
            let ordinary = ordinary.is_match(p.t(r)?.trim_matches(python_space));
            let mut lexical = true;
            for other in &peers {
                let t = p.t(other)?.trim_matches(python_space);
                let t = t.strip_suffix(['.', ',']).unwrap_or(t);
                let numeric = !t.is_empty()
                    && t.chars()
                        .all(|c| rapiddoc_native::python_decimal(c).is_some());
                if !(numeric && eq(&p.l(other)["font_signature"], &p.l(r)["font_signature"])
                    || e::display_prose(p.t(other)?) && !math(p.t(other)?))
                {
                    lexical = false;
                    break;
                }
            }
            if ordinary && p.coverage(r)? >= 0.75 && !peers.is_empty() && lexical {
                continue;
            }
            if peers.iter().any(|other| axis(&r.b, &other.b, 0) >= 0.5) {
                continue;
            }
            if !peers
                .iter()
                .any(|other| r.b.n[0] - other.b.n[2] >= 0.5f64.max(0.1 * h))
            {
                continue;
            }
        }
        if !peers.is_empty() {
            anchors.push(Anchor {
                row: r.clone(),
                below,
                above,
                repeated: false,
            })
        }
    }
    Ok(dedup(anchors, em))
}
pub(super) fn grow(
    p: &Page,
    lane: &Lane,
    anchor: &Anchor,
    top: f64,
    bottom: f64,
    claimed: &BTreeSet<i64>,
    font: &Value,
    em: f64,
    w: &mut Work,
) -> Result<Vec<Row>, Error> {
    let width = (lane.right - lane.left).max(0.1);
    let mut candidates = Vec::new();
    for r in &lane.rows {
        w.charge(lane.rows.len() + 1)?;
        if claimed.contains(&p.id(r)?)
            || !(r.b.n[2] - r.b.n[0] <= 0.8 * width || e::wide(p, &anchor.row, r, width)?)
            || cy(&r.b) < top
            || cy(&r.b) > bottom
            || e::body(p, r, lane, font, em)?
            || e::title(p, r, lane, font, em)?
            || e::prefix(
                p,
                r,
                lane,
                &anchor.row,
                font,
                em,
                if anchor.repeated { 0.5 } else { 0.75 },
            )?
        {
            continue;
        }
        candidates.push(r.clone())
    }
    let mut seeds = Vec::new();
    for r in &candidates {
        if p.id(r)? != p.id(&anchor.row)?
            && cx(&r.b) < anchor.row.b.n[0]
            && seed(
                p,
                &anchor.row,
                r,
                anchor.below || anchor.above || e::wide(p, &anchor.row, r, width)?,
            )?
            && !crosses(p, &anchor.row, r)?
        {
            seeds.push(r.clone())
        }
    }
    if seeds.is_empty() {
        return Ok(vec![]);
    }
    let mut members = vec![anchor.row.clone()];
    members.extend(seeds);
    closure(p, &mut members, &candidates, w)?;
    Ok(members)
}
fn closure(
    p: &Page,
    members: &mut Vec<Row>,
    candidates: &[Row],
    w: &mut Work,
) -> Result<(), Error> {
    let mut ids = members
        .iter()
        .map(|r| p.id(r))
        .collect::<Result<BTreeSet<_>, _>>()?;
    loop {
        let mut changed = false;
        for r in candidates {
            if ids.contains(&p.id(r)?) {
                continue;
            }
            let mut hit = false;
            for member in members.iter() {
                w.charge(1)?;
                if connected(p, member, r)? {
                    hit = true;
                    break;
                }
            }
            if hit {
                members.push(r.clone());
                ids.insert(p.id(r)?);
                changed = true
            }
        }
        if !changed {
            break;
        }
    }
    Ok(())
}
pub(super) fn expand_single(
    p: &Page,
    core: &Row,
    lane: &Lane,
    claimed: &BTreeSet<i64>,
    font: &Value,
    em: f64,
    w: &mut Work,
) -> Result<Vec<Row>, Error> {
    let width = (lane.right - lane.left).max(0.1);
    let mut candidates = Vec::new();
    for r in &lane.rows {
        w.charge(lane.rows.len() + 1)?;
        if p.id(r)? == p.id(core)?
            || claimed.contains(&p.id(r)?)
            || r.b.n[2] - r.b.n[0] > 0.35 * width
            || e::body(p, r, lane, font, em)?
            || e::title(p, r, lane, font, em)?
        {
            continue;
        }
        let narrow = r.b.n[2] - r.b.n[0] <= 1.5 * em
            && core.b.n[0] - em <= cx(&r.b)
            && cx(&r.b) <= core.b.n[2] + em;
        if (math(p.t(r)?)
            || p.l(r)["compact_formula_cluster"] == true
            || p.l(r)["formula_candidate_only"] == true
            || narrow)
            && connected(p, core, r)?
        {
            candidates.push(r.clone())
        }
    }
    let mut members = vec![core.clone()];
    members.extend(candidates);
    let mut all = Vec::new();
    for r in &lane.rows {
        if claimed.contains(&p.id(r)?)
            || r.b.n[2] - r.b.n[0] > 0.35 * width
            || e::body(p, r, lane, font, em)?
            || e::title(p, r, lane, font, em)?
        {
            continue;
        }
        if math(p.t(r)?)
            || p.l(r)["compact_formula_cluster"] == true
            || p.l(r)["formula_candidate_only"] == true
            || r.b.n[2] - r.b.n[0] <= 1.5 * em
        {
            all.push(r.clone())
        }
    }
    closure(p, &mut members, &all, w)?;
    Ok(members)
}
