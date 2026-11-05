use super::*;
fn index(order: &[usize], bs: &[Value], id: usize) -> Result<usize, Error> {
    order
        .iter()
        .position(|&i| eq(&bs[i], &bs[id]))
        .ok_or_else(invalid)
}
fn marginal(b: &Value) -> bool {
    ["header", "footer", "page_number", "page_footnote"]
        .iter()
        .any(|t| b["type"] == *t)
}
fn zero(b: &Value) -> bool {
    eq(b.get("angle").unwrap_or(&Value::from(0)), &json!(0))
}
fn title(b: &Value) -> bool {
    b["type"] == "doc_title" || b["type"] == "paragraph_title"
}
fn first_top(ids: Vec<usize>, bs: &[Value]) -> Result<Option<usize>, Error> {
    let mut first = None;
    for i in ids {
        if first.is_none() || bb(&bs[i])?.n[1] < bb(&bs[first.unwrap()])?.n[1] {
            first = Some(i)
        }
    }
    Ok(first)
}
pub(super) fn stabilize(
    order: &mut [usize],
    bs: &[Value],
    p: [f64; 2],
    w: &mut Work,
) -> Result<(), Error> {
    for _ in 0..order.len() {
        let mut changed = false;
        for k in 0..order.len().saturating_sub(1) {
            w.charge(1)?;
            let a = &bs[order[k]];
            let b = &bs[order[k + 1]];
            if truth(&a["_visual_annotation_region_member"])
                || truth(&b["_visual_annotation_region_member"])
            {
                continue;
            }
            let (Some(ai), Some(bi)) = (
                a["_lane_interval"].as_array(),
                b["_lane_interval"].as_array(),
            ) else {
                continue;
            };
            if ai.len() != 2
                || bi.len() != 2
                || !eq(&a["_lane_is_span"], &b["_lane_is_span"])
                || angle(a)? != angle(b)?
            {
                continue;
            }
            let aa = local(&bb(a)?, p, angle(a)?);
            let ba = local(&bb(b)?, p, angle(b)?);
            let hs = heights(a)?
                .into_iter()
                .chain(heights(b)?)
                .collect::<Vec<_>>();
            let tolerance = 0.75 * hs.into_iter().reduce(f64::min).unwrap_or(1.);
            let n = |v: &Value| super::super::table_native_text::number(v)?.ok_or_else(invalid);
            if (n(&ai[0])? - n(&bi[0])?).abs() <= tolerance
                && (n(&ai[1])? - n(&bi[1])?).abs() <= tolerance
                && cy(&aa) > cy(&ba) + 0.1 * tolerance
                && axis(&aa, &ba, 0) >= 0.35
                && aa.n[3].min(ba.n[3]) - aa.n[1].max(ba.n[1]) >= 0.
            {
                order.swap(k, k + 1);
                changed = true
            }
        }
        if !changed {
            break;
        }
    }
    Ok(())
}
pub(super) fn leading(
    mut order: Vec<usize>,
    bs: &[Value],
    p: [f64; 2],
    w: &mut Work,
) -> Result<Vec<usize>, Error> {
    let original = order.clone();
    for &i in &original {
        let b = &bs[i];
        let a = bb(b)?;
        let h = a.n[3] - a.n[1];
        if !title(b)
            || !zero(b)
            || a.n[1] > 0.3 * p[1]
            || a.n[2] - a.n[0] > 0.65 * p[0]
            || (cx(&a) - 0.5 * p[0]).abs() > 0.08 * p[0]
        {
            continue;
        }
        let mut veto = false;
        let mut lower = vec![];
        for &j in &original {
            w.charge(1)?;
            if j == i || marginal(&bs[j]) {
                continue;
            }
            let c = bb(&bs[j])?;
            if c.n[1] < a.n[1] - 0.25 * h
                || c.n[3].min(a.n[3]) - c.n[1].max(a.n[1]) > 0.2 * (c.n[3] - c.n[1]).min(h)
            {
                veto = true
            }
            if (bs[j]["type"] == "paragraph_title" || bs[j]["type"] == "text")
                && c.n[0] < 0.25 * p[0]
                && (0. ..=1.5 * h).contains(&(c.n[1] - a.n[3]))
            {
                lower.push(j)
            }
        }
        if veto {
            continue;
        }
        if let Some(j) = first_top(lower, bs)? {
            let at = index(&order, bs, i)?;
            if at > index(&order, bs, j)? {
                order.remove(at);
                let pos = index(&order, bs, j)?;
                order.insert(pos, i)
            }
        }
    }
    let original = order.clone();
    let ordinary = original
        .iter()
        .copied()
        .filter(|&i| !marginal(&bs[i]))
        .collect::<Vec<_>>();
    for &i in &ordinary {
        let b = &bs[i];
        if !(b["type"] == "text" || title(b)) || !zero(b) {
            continue;
        }
        let a = bb(b)?;
        let h = a.n[3] - a.n[1];
        if h <= 0. || optional(b, "_text_lines", 4096)?.len() > 1 {
            continue;
        }
        let mut earlier = false;
        let mut sections = vec![];
        for &j in &ordinary {
            w.charge(1)?;
            if j == i {
                continue;
            }
            let c = bb(&bs[j])?;
            earlier |= c.n[1] < a.n[1] - 0.25 * h;
            if title(&bs[j]) && zero(&bs[j]) && (0. ..=1.5 * h).contains(&(c.n[1] - a.n[3])) {
                sections.push(j)
            }
        }
        if earlier {
            continue;
        }
        let Some(s) = first_top(sections, bs)? else {
            continue;
        };
        let sb = bb(&bs[s])?;
        let mut bodies = vec![];
        for &j in &ordinary {
            if j != i
                && bs[j]["type"] == "text"
                && zero(&bs[j])
                && optional(&bs[j], "_text_lines", 4096)?.len() >= 2
            {
                let c = bb(&bs[j])?;
                if (0. ..=2. * h).contains(&(c.n[1] - sb.n[3])) && c.n[2] - c.n[0] >= 0.55 * p[0] {
                    bodies.push(j)
                }
            }
        }
        let Some(body) = first_top(bodies, bs)? else {
            continue;
        };
        let c = bb(&bs[body])?;
        let width = c.n[2] - c.n[0];
        let mut hs = heights(&bs[body])?;
        hs.sort_by(f64::total_cmp);
        let em = if hs.is_empty() { h } else { hs[hs.len() / 2] };
        if h < 0.65 * em
            || h > 1.6 * em
            || a.n[2] - a.n[0] < 0.15 * width
            || a.n[2] - a.n[0] > 0.65 * width
            || (a.n[0] + a.n[2] - c.n[0] - c.n[2]).abs() > 0.16 * width
            || sb.n[0] - c.n[0] < -0.5 * em
            || sb.n[0] - c.n[0] > 3. * em
            || sb.n[2] > c.n[2] + 0.5 * em
        {
            continue;
        }
        let mut veto = false;
        for &j in &ordinary {
            w.charge(1)?;
            if [i, s, body].contains(&j) {
                continue;
            }
            let d = bb(&bs[j])?;
            if d.n[1] < sb.n[3] && d.n[3] > a.n[1]
                || d.n[1] < c.n[3] && d.n[3] > c.n[1] && (d.n[2] <= c.n[0] || d.n[0] >= c.n[2])
            {
                veto = true;
                break;
            }
        }
        if veto {
            continue;
        }
        let first = index(&order, bs, s)?;
        let last = index(&order, bs, i)?;
        if first < last && order[first..last].iter().all(|&j| j == s || j == body) {
            order.remove(last);
            order.insert(first, i)
        }
    }
    Ok(order)
}
pub(super) fn equations(
    mut order: Vec<usize>,
    bs: &[Value],
    w: &mut Work,
) -> Result<Vec<usize>, Error> {
    let original = order.clone();
    for &i in &original {
        if bs[i]["type"] != "equation" {
            continue;
        }
        let a = bb(&bs[i])?;
        let em = (a.n[3] - a.n[1]).max(1.);
        for &j in &original {
            w.charge(1)?;
            let parts = bs[j]["content"]
                .as_str()
                .and_then(super::super::native_formula_components::split_number);
            if bs[j]["type"] != "text" {
                continue;
            }
            let Some((prefix, _)) = parts else { continue };
            if prefix
                .chars()
                .any(|c| !python_space(c) && !".…·•—-".contains(c))
            {
                continue;
            }
            let b = bb(&bs[j])?;
            if b.n[0] <= a.n[2] || b.n[0] - a.n[2] > 8. * em || (cy(&b) - cy(&a)).abs() > 0.75 * em
            {
                continue;
            }
            let at = index(&order, bs, j)?;
            if at < index(&order, bs, i)? {
                order.remove(at);
                let pos = index(&order, bs, i)? + 1;
                order.insert(pos, j)
            }
        }
    }
    Ok(order)
}
