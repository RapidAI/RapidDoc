//! Original document-title anchor scoring and same-/cross-lane expansion.
use super::*;
fn compatible(a: &Value, b: &Value) -> Result<bool, Error> {
    Ok(fonts_ok(a, b)? || coverage(a)?.min(coverage(b)?) < 0.85 && weights_ok(a, b)?)
}
pub(super) fn select(
    s: &mut State,
    lanes: &mut [Lane],
    p: [f64; 2],
    body: &Value,
    w: &mut Work,
) -> Result<Option<f64>, Error> {
    let mut profiles = Vec::new();
    for lane in lanes.iter_mut() {
        profiles.push(s.profile(lane, w)?)
    }
    let column = lanes
        .iter()
        .zip(&profiles)
        .filter(|(l, _)| !l.span)
        .map(|(_, p)| p.h)
        .collect::<Vec<_>>();
    let document_h = if !body.is_null() {
        num(&body["body_height"])?
    } else if !column.is_empty() {
        median(column)
    } else {
        profiles.iter().map(|p| p.h).reduce(f64::min).unwrap_or(1.)
    };
    let mut best = None;
    for (k, (lane, profile)) in lanes.iter().zip(&profiles).enumerate() {
        let width = (lane.right - lane.left).max(0.1);
        let available = lane
            .ids
            .iter()
            .copied()
            .filter(|&i| s.unresolved(i))
            .collect::<Vec<_>>();
        for (row, &i) in available.iter().enumerate() {
            w.charge(available.len() + 1)?;
            let b = &s.boxes[i];
            let h = s.h(i)?;
            let refh = profile.h.min(1.25 * document_h);
            let ratio = h / refh.max(0.1);
            let dratio = h / document_h.max(0.1);
            let wratio = (b.n[2] - b.n[0]) / width;
            let centered = (cx(b) - (lane.left + lane.right) / 2.).abs() <= 0.15 * width;
            let page_centered = (cx(b) - 0.5 * p[0]).abs() <= 0.15 * p[0];
            let pwr = (b.n[2] - b.n[0]) / p[0].max(0.1);
            let span = lane.span && centered && wratio >= 0.65 && dratio >= 1.3;
            let isolated = page_centered
                && pwr >= 0.5
                && dratio >= 1.12
                && cy(b) < 0.25 * p[1]
                && !available.iter().any(|&j| {
                    eq(&s.lines[j]["font_signature"], &s.lines[i]["font_signature"])
                        && s.boxes[j].n[1] > b.n[1]
                })
                && available
                    .iter()
                    .all(|&j| j == i || (s.boxes[j].n[1] - b.n[1]).abs() > 1.5 * h);
            if cy(b) > 0.45 * p[1]
                || ratio < 1.4 && !span && !isolated
                || wratio < 0.2
                || !centered && ratio < 1.7
                || !page_centered && pwr < 0.45 && ratio < 1.8
            {
                continue;
            }
            let preference = (0.45 - cy(b) / p[1]).max(0.);
            let top_weight = if body["has_style_scale_repairs"] == true {
                4.
            } else {
                1.
            };
            let score = ratio
                + if centered { 0.75 } else { 0. }
                + if page_centered { 1.25 } else { 0. }
                + if pwr >= 0.55 { 0.75 } else { 0. }
                + top_weight * preference
                - 0.02 * row as f64;
            if best.is_none_or(|(old, _, _)| score > old) {
                best = Some((score, k, i))
            }
        }
    }
    let Some((_, k, anchor)) = best else {
        return Ok(None);
    };
    let ab = s.boxes[anchor].clone();
    let ah = s.h(anchor)?;
    s.mark(anchor, "doc_title");
    let mut selected = vec![anchor];
    let ordered = lanes[k].ids.clone();
    let anchor_pos = ordered.iter().position(|&i| i == anchor).unwrap();
    for direction in [-1isize, 1] {
        let mut pos = anchor_pos as isize + direction;
        let mut previous = ab.clone();
        while pos >= 0 && (pos as usize) < ordered.len() {
            w.charge(1)?;
            let i = ordered[pos as usize];
            if !s.unresolved(i) {
                break;
            }
            let b = s.boxes[i].clone();
            let h = s.h(i)?;
            if b.n[1] > ab.n[3]
                && eq(
                    &s.lines[i]["font_signature"],
                    &s.lines[anchor]["font_signature"],
                )
                && h < 0.9 * ah
            {
                break;
            }
            if !(0.8..=1.25).contains(&(h / ah)) || !compatible(&s.lines[anchor], &s.lines[i])? {
                break;
            }
            let gap = (b.n[1] - previous.n[3]).max(previous.n[1] - b.n[3]).max(0.);
            let width = (lanes[k].right - lanes[k].left).max(0.1);
            let centered = (cx(&b) - cx(&ab)).abs() <= 0.18 * width;
            let aligned = (b.n[0] - ab.n[0]).abs() <= 0.75 * ah;
            if gap > 1.1 * ah || !centered && !aligned {
                break;
            }
            s.mark(i, "doc_title");
            selected.push(i);
            previous = b;
            pos += direction
        }
    }
    Ok(selected.iter().map(|&i| s.boxes[i].n[3]).reduce(f64::max))
}
pub(super) fn expand(
    s: &mut State,
    ids: &[usize],
    p: [f64; 2],
    bottom: Option<f64>,
    w: &mut Work,
) -> Result<Option<f64>, Error> {
    let mut titles = ids
        .iter()
        .copied()
        .filter(|&i| s.lines[i]["semantic_type"] == "doc_title")
        .collect::<Vec<_>>();
    if titles.is_empty() {
        return Ok(bottom);
    }
    let mut selected = titles.iter().copied().collect::<BTreeSet<_>>();
    loop {
        let b = s.bounds(&titles)?;
        let h = median(
            titles
                .iter()
                .map(|&i| s.h(i))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let mut candidates = ids
            .iter()
            .copied()
            .filter(|&i| {
                !selected.contains(&i) && s.unresolved(i) && cy(&s.boxes[i]) <= 0.45 * p[1]
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|&i, &j| {
            let distance = |i: usize| {
                (s.boxes[i].n[1] - b.n[3])
                    .abs()
                    .min((b.n[1] - s.boxes[i].n[3]).abs())
            };
            distance(i).total_cmp(&distance(j))
        });
        let mut chosen = None;
        for i in candidates {
            w.charge(titles.len() + 1)?;
            let c = &s.boxes[i];
            let ch = s.h(i)?;
            if c.n[1] > b.n[3]
                && titles
                    .iter()
                    .any(|&j| eq(&s.lines[i]["font_signature"], &s.lines[j]["font_signature"]))
                && ch < 0.9 * h
            {
                continue;
            }
            if !(0.8..=1.25).contains(&(ch / h.max(0.1))) {
                continue;
            }
            let mut fonts = false;
            for &j in &titles {
                if compatible(&s.lines[j], &s.lines[i])? {
                    fonts = true;
                    break;
                }
            }
            if !fonts {
                continue;
            }
            let gap = (c.n[1] - b.n[3]).max(b.n[1] - c.n[3]).max(0.);
            let centered = (cx(c) - cx(&b)).abs() <= 0.18 * p[0];
            let aligned = (c.n[0] - b.n[0]).abs() <= 0.75 * h;
            let tw = (b.n[2] - b.n[0]).max(0.1);
            let cw = c.n[2] - c.n[0];
            if gap > 1.1 * h || !(aligned && cw >= 0.25 * tw || centered && cw >= 0.45 * tw) {
                continue;
            }
            chosen = Some(i);
            break;
        }
        let Some(i) = chosen else { break };
        s.mark(i, "doc_title");
        titles.push(i);
        selected.insert(i);
    }
    Ok(titles.iter().map(|&i| s.boxes[i].n[3]).reduce(f64::max))
}
pub(super) fn additional(
    s: &mut State,
    ids: &[usize],
    p: [f64; 2],
    bottom: Option<f64>,
    body: &Value,
    w: &mut Work,
) -> Result<Option<f64>, Error> {
    let Some(bottom) = bottom else {
        return Ok(None);
    };
    if body.is_null() || body["has_style_scale_repairs"] != true {
        return Ok(Some(bottom));
    }
    let h = num(&body["body_height"])?.max(0.1);
    let mut ordered = ids.to_vec();
    s.order(&mut ordered);
    let mut bottoms = ordered
        .iter()
        .copied()
        .filter(|&i| s.lines[i]["semantic_type"] == "doc_title")
        .map(|i| s.boxes[i].n[3])
        .collect::<Vec<_>>();
    for (k, &i) in ordered.iter().enumerate() {
        w.charge(ordered.len() + 1)?;
        if !s.unresolved(i) {
            continue;
        }
        let b = &s.boxes[i];
        let y = cy(b);
        let width = (b.n[2] - b.n[0]) / p[0].max(0.1);
        if y < bottom + 4. * h
            || y > 0.58 * p[1]
            || !(0.4..=0.8).contains(&width)
            || (cx(b) - 0.5 * p[0]).abs() > 0.08 * p[0]
            || regular(&s.lines[i], body)?
        {
            continue;
        }
        let Some(j) = ordered[k + 1..]
            .iter()
            .copied()
            .find(|&j| s.unresolved(j) && cy(&s.boxes[j]) > y)
        else {
            continue;
        };
        let next = &s.boxes[j];
        let nw = (next.n[2] - next.n[0]) / p[0].max(0.1);
        let centered = (cx(next) - 0.5 * p[0]).abs() <= 0.12 * p[0];
        if !centered || nw > 0.45 || (next.n[1] - b.n[3]).max(0.) > 2. * h {
            continue;
        }
        bottoms.push(b.n[3]);
        s.mark(i, "doc_title")
    }
    Ok(Some(bottoms.into_iter().reduce(f64::max).unwrap_or(bottom)))
}
pub(super) fn fallback(
    s: &State,
    lanes: &mut [Lane],
    body: &Value,
    w: &mut Work,
) -> Result<bool, Error> {
    let mut heights = Vec::new();
    for l in lanes.iter() {
        for &i in &l.ids {
            if s.lines[i]["semantic_type"] == "doc_title" {
                heights.push(s.h(i)?)
            }
        }
    }
    let mut columns = Vec::new();
    for l in lanes {
        if !l.span {
            columns.push(s.profile(l, w)?.h)
        }
    }
    if heights.is_empty() || columns.is_empty() {
        return Ok(false);
    }
    let document_h = if !body.is_null() {
        num(&body["body_height"])?
    } else {
        median(columns)
    };
    Ok(heights.iter().any(|h| {
        let r = h / document_h.max(0.1);
        r >= 1.3 && r < 1.4
    }))
}
