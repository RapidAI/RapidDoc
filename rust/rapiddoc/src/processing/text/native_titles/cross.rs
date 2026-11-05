//! Original cross-lane centered/emphasized section-title passes.
use super::*;
fn candidates(s: &State, i: usize, lanes: &[Lane], p: [f64; 2]) -> Vec<usize> {
    lanes
        .iter()
        .enumerate()
        .filter(|(_, l)| {
            !l.span
                && l.ids.len() >= 5
                && l.right - l.left >= 0.2 * p[0]
                && l.left <= cx(&s.boxes[i])
                && cx(&s.boxes[i]) <= l.right
        })
        .map(|(k, _)| k)
        .collect()
}
fn choose(candidates: Vec<usize>, distance: impl Fn(usize) -> f64) -> Option<usize> {
    let mut best = None;
    for k in candidates {
        let d = distance(k);
        if best.is_none_or(|(_, old)| d < old) {
            best = Some((k, d))
        }
    }
    best.map(|(k, _)| k)
}
pub(super) fn centered(
    s: &mut State,
    ids: &[usize],
    lanes: &mut [Lane],
    p: [f64; 2],
    containers: &[Box4],
    page: i64,
    bottom: Option<f64>,
    w: &mut Work,
) -> Result<(), Error> {
    for &i in ids {
        if !s.unresolved(i) {
            continue;
        }
        let b = &s.boxes[i];
        if cy(b) < 0.07 * p[1] || cy(b) > 0.93 * p[1] {
            continue;
        }
        let Some(k) = choose(candidates(s, i, lanes, p), |k| {
            (cx(b) - 0.5 * (lanes[k].left + lanes[k].right)).abs()
        }) else {
            continue;
        };
        let profile = s.profile(&mut lanes[k], w)?;
        let lane = &lanes[k];
        let width = (lane.right - lane.left).max(0.1);
        let lw = b.n[2] - b.n[0];
        let h = s.h(i)?;
        if lw < 0.08 * width
            || lw > 0.7 * width
            || (cx(b) - 0.5 * (lane.left + lane.right)).abs() > 0.05 * width
            || !(0.75..=1.3).contains(&(h / profile.h.max(0.1)))
        {
            continue;
        }
        let l = &s.lines[i];
        let regular = !profile.font.is_null()
            && !l["font_signature"].is_null()
            && coverage(l)? >= 0.75
            && share_family(&l["font_signature"], &profile.font)?;
        let emphasized =
            matches!((profile.weight,weight(l)?),(Some(a),Some(b))if b>=(a+100.).max(1.15*a));
        if page == 0 && h < 0.9 * profile.h && regular && !emphasized {
            continue;
        }
        if bottom.is_some_and(|y| b.n[1] <= y + profile.h)
            || inside(b, containers)
            || near(b, containers, profile.h)
        {
            continue;
        }
        let mut above = Vec::new();
        let mut below = Vec::new();
        for &j in &lane.ids {
            w.charge(1)?;
            let c = &s.boxes[j];
            if j == i
                || !s.unresolved(j)
                || c.n[2] - c.n[0] < 0.5 * width
                || !(0.75..=1.3).contains(&(s.h(j)? / profile.h.max(0.1)))
            {
                continue;
            }
            if cy(c) < cy(b) {
                above.push(j)
            } else if cy(c) > cy(b) {
                below.push(j)
            }
        }
        let mut previous = None;
        for j in above {
            if previous.is_none_or(|old| cy(&s.boxes[j]) > cy(&s.boxes[old])) {
                previous = Some(j)
            }
        }
        let Some(previous) = previous else { continue };
        below.sort_by(|&a, &b| cy(&s.boxes[a]).total_cmp(&cy(&s.boxes[b])));
        if below.len() < 3 {
            continue;
        }
        let mut continuous = true;
        for pair in below[..3].windows(2) {
            if gap(s, pair[0], pair[1])? > profile.gap + 0.75 * profile.h {
                continuous = false;
                break;
            }
        }
        if !continuous {
            continue;
        }
        let ga = b.n[1] - s.boxes[previous].n[3];
        let gb = s.boxes[below[0]].n[1] - b.n[3];
        if ga < 0.3 * profile.h
            || gb < -0.1 * profile.h
            || ga.max(gb) < profile.gap + 0.2 * profile.h
        {
            continue;
        }
        s.mark(i, "paragraph_title");
    }
    Ok(())
}
pub(super) fn emphasized(
    s: &mut State,
    ids: &[usize],
    lanes: &mut [Lane],
    p: [f64; 2],
    containers: &[Box4],
    bottom: Option<f64>,
    body: &Value,
    w: &mut Work,
) -> Result<(), Error> {
    for &i in ids {
        let l = &s.lines[i];
        if !s.unresolved(i) || l["font_signature"].is_null() {
            continue;
        }
        let f = array(&l["font_signature"], 2)?;
        if f.len() != 2 {
            return Err(invalid());
        }
        let flags = f[1]
            .as_i64()
            .or_else(|| f[1].as_bool().map(i64::from))
            .ok_or_else(invalid)?;
        if flags & 64 == 0 || regular(l, body)? {
            continue;
        }
        let b = &s.boxes[i];
        if cy(b) < 0.07 * p[1] || cy(b) > 0.93 * p[1] {
            continue;
        }
        let Some(k) = choose(candidates(s, i, lanes, p), |k| {
            (b.n[0] - lanes[k].left).abs()
        }) else {
            continue;
        };
        let profile = s.profile(&mut lanes[k], w)?;
        let lane = &lanes[k];
        if profile.font.is_null() || eq(&l["font_signature"], &profile.font) || coverage(l)? < 0.75
        {
            continue;
        }
        let width = (lane.right - lane.left).max(0.1);
        let h = s.h(i)?;
        if b.n[2] - b.n[0] > 0.9 * width
            || (b.n[0] - lane.left).abs() > 2. * profile.h
            || !(0.8..=1.35).contains(&(h / profile.h.max(0.1)))
            || bottom.is_some_and(|y| b.n[1] <= y + 2. * profile.h)
            || inside(b, containers)
            || near(b, containers, profile.h)
        {
            continue;
        }
        let mut previous = None;
        let mut following = None;
        for &j in &lane.ids {
            w.charge(1)?;
            let c = &s.boxes[j];
            if j == i
                || !s.unresolved(j)
                || !eq(&s.lines[j]["font_signature"], &profile.font)
                || c.n[2] - c.n[0] < 0.45 * width
            {
                continue;
            }
            if cy(c) < cy(b) && previous.is_none_or(|old| cy(c) > cy(&s.boxes[old])) {
                previous = Some(j)
            } else if cy(c) > cy(b) && following.is_none_or(|old| cy(c) < cy(&s.boxes[old])) {
                following = Some(j)
            }
        }
        let (Some(previous), Some(following)) = (previous, following) else {
            continue;
        };
        let ga = b.n[1] - s.boxes[previous].n[3];
        let gb = s.boxes[following].n[1] - b.n[3];
        if ga < 0.35 * profile.h || gb > 3. * profile.h || gb < -0.1 * profile.h {
            continue;
        }
        s.mark(i, "paragraph_title");
    }
    Ok(())
}
