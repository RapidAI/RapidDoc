//! Original lane title scores, all body/layout vetoes and local expansion.
use super::lane_evidence as e;
use super::*;
pub(super) fn classify(
    s: &mut State,
    lane: &Lane,
    p: &Profile,
    page_size: [f64; 2],
    containers: &[Box4],
    bottom: Option<f64>,
    preserve: bool,
    physical: &BTreeMap<i64, (Option<f64>, Option<f64>)>,
    suppressed: &BTreeSet<i64>,
    body: &Value,
    titles: &Value,
    page: i64,
    w: &mut Work,
) -> Result<(), Error> {
    let width = (lane.right - lane.left).max(0.1);
    let rows = &lane.ids;
    let mut selected = super::selected::Selected::new();
    let boundary = e::boundary(s, lane, p, page_size[1], bottom)?;
    for (k, &i) in rows.iter().enumerate() {
        w.charge(rows.len() + containers.len() + 1)?;
        if !s.unresolved(i) || suppressed.contains(&integer(&s.lines[i], "source_index", 0)?) {
            continue;
        }
        let l = &s.lines[i];
        let b = &s.boxes[i];
        if cy(b) < 0.07 * page_size[1] || cy(b) > 0.93 * page_size[1] {
            continue;
        }
        let front = boundary.is_some_and(|y| cy(b) <= y);
        if front && !preserve
            || bottom.is_some_and(|y| y <= 0.6 * page_size[1] && cy(b) >= 0.84 * page_size[1])
        {
            continue;
        }
        let h = s.h(i)?;
        let (proto, conflict) = super::prototype::matching(s, i, lane, body, titles)?;
        let inside = inside(b, containers);
        let near = near(b, containers, p.h);
        let display = l["caption_start"] != true
            && h >= 1.3 * p.h
            && !l["font_signature"].is_null()
            && !p.font.is_null()
            && !share_family(&l["font_signature"], &p.font)?
            && (cx(b) - (lane.left + lane.right) / 2.).abs() <= 0.08 * width
            && b.n[2] - b.n[0] <= 0.9 * width
            && containers.iter().any(|c| {
                c.n[1] - b.n[3] >= 0.5 * p.h
                    && c.n[1] - b.n[3] <= 3. * p.h
                    && axis(b, c, 0) >= 0.65
                    && (cx(b) - cx(c)).abs() <= 0.08 * (c.n[2] - c.n[0])
            });
        if inside || near && !proto && !display {
            continue;
        }
        let regular = regular(l, body)?;
        let body_h = if body.is_null() {
            None
        } else {
            Some(num(&body["body_height"])?)
        };
        let regular_body = regular && body_h.is_some_and(|bh| h >= 0.9 * bh);
        let mut reference = p.h;
        if p.count < 3 {
            if let Some(bh) = body_h {
                reference = bh
            }
        }
        if regular_body {
            reference = reference.max(body_h.unwrap())
        }
        if reference == 0. {
            return Err(invalid());
        }
        let hr = h / reference;
        let wr = (b.n[2] - b.n[0]) / width;
        let center = (cx(b) - (lane.left + lane.right) / 2.).abs() <= 0.12 * width;
        let left = (b.n[0] - lane.left).abs() <= 0.65 * p.h;
        if !l["median_glyph_width"].is_null()
            && b.n[2] - b.n[0] <= 1.25 * num(&l["median_glyph_width"])?
            && hr < 1.18
        {
            continue;
        }
        let different = !regular
            && !p.font.is_null()
            && !l["font_signature"].is_null()
            && !share_family(&l["font_signature"], &p.font)?;
        let style = different && coverage(l)? >= 0.75;
        let low = different && (0.5..0.75).contains(&coverage(l)?);
        let support = if l["font_signature"].is_null() {
            1.
        } else {
            p.support
                .iter()
                .find(|(f, _)| eq(f, &l["font_signature"]))
                .map_or(1., |(_, v)| *v)
        };
        let emphasized =
            matches!((p.weight,weight(l)?),(Some(a),Some(b))if b>=(a+100.).max(1.15*a));
        let ga = e::normalized_gap(s, rows, k, -1, p.h, physical)?;
        let gb = e::normalized_gap(s, rows, k, 1, p.h, physical)?;
        let regular_ratio = p.gap / p.h.max(0.1);
        let excess_a = (ga - regular_ratio).max(0.);
        let excess_b = (gb - regular_ratio).max(0.);
        let spacing = ga >= 0.35;
        if e::sibling(s, rows, i)? || e::mixed(s, rows, k, width, p)? {
            continue;
        }
        if k > 0
            && s.lines[rows[k - 1]]["semantic_type"] == "paragraph_title"
            && wr >= 0.85
            && ga < 1.
            && !fonts_ok(&s.lines[rows[k - 1]], l)?
        {
            continue;
        }
        if e::continues(s, rows, k, width, p)? {
            continue;
        }
        let following = e::following(s, rows, k, lane, p)?;
        let transition = low && left && wr <= 0.65 && ga >= 0.1 && gb >= 0.2 && following;
        let compact = hr < 0.9
            && center
            && wr <= 0.7
            && ga >= 0.35
            && gb >= 0.2
            && (page != 0 || style || emphasized || proto)
            && e::compact(s, rows, k, width)?;
        if body_h == Some(0.) {
            return Err(invalid());
        }
        if body_h.is_some_and(|bh| (0.9..=1.1).contains(&(h / bh)))
            && !proto
            && (!center || wr >= 0.75)
            && !emphasized
            && excess_a < 0.1
            && excess_b < 0.1
        {
            continue;
        }
        if !proto && conflict && body_h.is_some_and(|bh| h <= 1.18 * bh) && !emphasized {
            continue;
        }
        if !emphasized && e::sentence(s, rows, k, width, p)? {
            continue;
        }
        let promotion = proto
            && (wr <= 0.9 || (b.n[2] - b.n[0]) / page_size[0].max(0.1) <= 0.8)
            && (center || left)
            && (body_h.is_some_and(|bh| h >= 1.18 * bh)
                || near
                || following && excess_a >= 0.1
                || excess_a >= 0.35);
        if regular_body && wr >= 0.8 && !emphasized && hr < 1.18 {
            continue;
        }
        let inline = body_h.is_some_and(|bh| h <= 1.15 * bh) && e::inline(s, rows, k, width, p)?;
        if promotion && !inline {
            s.mark(i, "paragraph_title");
            selected.insert(k);
            continue;
        }
        if body_h.is_some_and(|bh| (0.9..=1.1).contains(&(h / bh)))
            && !proto
            && center
            && !following
            && !compact
        {
            continue;
        }
        if hr < 0.9 && !front && !following && !compact {
            continue;
        }
        if wr >= 0.9 && hr < 1.15 && !(style && ga >= 0.65) {
            continue;
        }
        let mut score = 0.;
        if style || transition {
            score += 2.;
            if transition || support <= 0.2 {
                score += 0.75
            }
        }
        if emphasized {
            score += 1.
        }
        if hr >= 1.18 {
            score += 2.
        } else if hr <= 0.9 && center {
            score += 0.75
        }
        if wr <= 0.78 {
            score += 0.75
        }
        if center && wr <= 0.85 {
            score += 1.25
        }
        if ga >= 0.65 {
            score += 1.25
        } else if ga >= 0.35 {
            score += 0.5
        }
        if gb >= 0.35 {
            score += 0.75
        }
        if left {
            score += 0.25
        }
        let precise = (cx(b) - (lane.left + lane.right) / 2.).abs() <= 0.05 * width;
        let fallback = (0.75..=1.15).contains(&hr)
            && precise
            && wr <= 0.7
            && !front
            && !(page == 0 && cy(b) <= 0.2 * page_size[1])
            && (0.35..=1.5).contains(&ga)
            && (0.2..=1.5).contains(&gb)
            && following
            && !e::field(s, rows, k, width, p)?;
        let strong = hr >= 1.18 || style || transition || emphasized || fallback || compact;
        if score >= 4. && (spacing || transition) && strong {
            if e::inline(s, rows, k, width, p)? {
                continue;
            }
            s.mark(i, "paragraph_title");
            selected.insert(k);
        }
    }
    unify(s, rows, &mut selected, w)?;
    expand(s, lane, &mut selected, p, w)?;
    if preserve {
        if let Some(y) = boundary {
            protect(s, lane, p, y, w)?
        }
    }
    Ok(())
}
fn unify(
    s: &mut State,
    rows: &[usize],
    selected: &mut super::selected::Selected,
    w: &mut Work,
) -> Result<(), Error> {
    let mut groups = Vec::<(Value, Vec<usize>)>::new();
    for (k, &i) in rows.iter().enumerate() {
        let row = &s.lines[i]["visual_row_id"];
        if row.is_null() {
            continue;
        }
        if let Some((_, members)) = groups.iter_mut().find(|(id, _)| eq(id, row)) {
            members.push(k)
        } else {
            groups.push((row.clone(), vec![k]))
        }
    }
    for (_, members) in groups {
        let titles = members
            .iter()
            .copied()
            .filter(|&k| s.lines[rows[k]]["semantic_type"] == "paragraph_title")
            .collect::<Vec<_>>();
        if titles.is_empty() {
            continue;
        }
        let i = rows[titles[0]];
        let h = s.h(i)?;
        let mut compatible = true;
        for &k in &members {
            w.charge(1)?;
            if !(0.75..=1.25).contains(&(s.h(rows[k])? / h))
                || !fonts_ok(&s.lines[i], &s.lines[rows[k]])?
            {
                compatible = false;
                break;
            }
        }
        if compatible {
            for k in members {
                s.mark(rows[k], "paragraph_title");
                selected.insert(k);
            }
        } else {
            for k in titles {
                s.lines[rows[k]]["semantic_type"] = Value::Null;
                selected.remove(&k);
            }
        }
    }
    Ok(())
}
fn expand(
    s: &mut State,
    lane: &Lane,
    selected: &mut super::selected::Selected,
    p: &Profile,
    w: &mut Work,
) -> Result<(), Error> {
    let rows = &lane.ids;
    let width = (lane.right - lane.left).max(0.1);
    let mut pending = selected.ordered();
    while let Some(k) = pending.pop() {
        let i = rows[k];
        let h = s.h(i)?;
        for next in [k.checked_sub(1), Some(k + 1)].into_iter().flatten() {
            w.charge(1)?;
            if next >= rows.len() || selected.contains(&next) {
                continue;
            }
            let j = rows[next];
            let l = &s.lines[j];
            if !s.unresolved(j) || l["title_suppressed"] == true || !l["paragraph_group"].is_null()
            {
                continue;
            }
            let ch = s.h(j)?;
            let (a, b) = (&s.boxes[i], &s.boxes[j]);
            if !(0.75..=1.25).contains(&(ch / h)) || b.n[2] - b.n[0] > 0.8 * width {
                continue;
            }
            if !s.lines[i]["font_signature"].is_null()
                && !l["font_signature"].is_null()
                && !eq(&s.lines[i]["font_signature"], &l["font_signature"])
            {
                continue;
            }
            if !fonts_ok(&s.lines[i], l)? {
                continue;
            }
            let gap = (b.n[1] - a.n[3]).max(a.n[1] - b.n[3]).max(0.);
            let centers = (cx(b) - cx(a)).abs() <= 0.15 * width;
            let edges = (b.n[0] - a.n[0]).abs() <= 0.75 * p.h;
            let indent = (b.n[0] - a.n[0]).abs() <= 2. * p.h;
            if gap > 0.65 * p.h || !(centers || edges || indent) {
                continue;
            }
            s.mark(j, "paragraph_title");
            selected.insert(next);
            pending.push(next)
        }
    }
    Ok(())
}
fn protect(
    s: &mut State,
    lane: &Lane,
    p: &Profile,
    boundary: f64,
    w: &mut Work,
) -> Result<(), Error> {
    let members = lane
        .ids
        .iter()
        .copied()
        .filter(|&i| {
            s.lines[i]["semantic_type"] == "paragraph_title" && cy(&s.boxes[i]) <= boundary
        })
        .collect::<Vec<_>>();
    let mut connected = BTreeSet::new();
    for (k, &i) in members.iter().enumerate() {
        for &j in &members[k + 1..] {
            w.charge(1)?;
            if connects(s, i, j, lane, p.gap)? {
                connected.insert(i);
                connected.insert(j);
            }
        }
    }
    for i in members {
        s.lines[i]["semantic_type"] = if connected.contains(&i) {
            json!("text")
        } else {
            Value::Null
        }
    }
    Ok(())
}
fn connects(s: &State, i: usize, j: usize, lane: &Lane, regular: f64) -> Result<bool, Error> {
    let (a, b) = (&s.lines[i], &s.lines[j]);
    if !eq(&a["semantic_type"], &b["semantic_type"]) {
        return Ok(false);
    }
    if b["semantic_type"] == "paragraph_title"
        && b["explicit_section_title"] == true
        && !b["title_band_id"].is_null()
        && !eq(&b["title_band_id"], &a["title_band_id"])
    {
        return Ok(false);
    }
    let (ah, bh) = (s.h(i)?, s.h(j)?);
    let pair = ah.max(bh);
    let shared = !a["title_band_id"].is_null() && eq(&a["title_band_id"], &b["title_band_id"]);
    let confirmed = shared && a["structural_title"] == true && b["structural_title"] == true;
    if ah.max(bh) / ah.min(bh) > 1.35 && !confirmed {
        return Ok(false);
    }
    let g = gap(s, i, j)?;
    if g < -0.25 * pair
        || g > (1.25 * pair).max(regular + 0.75 * pair)
        || a["semantic_type"] == "paragraph_title" && g > 0.5 * pair && !shared
    {
        return Ok(false);
    }
    let conflict = !a["font_signature"].is_null()
        && !b["font_signature"].is_null()
        && coverage(a)? >= 0.75
        && coverage(b)? >= 0.75
        && !eq(&a["font_signature"], &b["font_signature"]);
    let uncertain = a["semantic_type"] == "doc_title"
        && coverage(a)?.min(coverage(b)?) < 0.85
        && match (weight(a)?, weight(b)?) {
            (Some(a), Some(b)) => (a - b).abs() < 100.,
            _ => true,
        };
    if conflict && !uncertain && !confirmed {
        return Ok(false);
    }
    let width = (lane.right - lane.left).max(0.1);
    let (a, b) = (&s.boxes[i], &s.boxes[j]);
    Ok((cx(a) - cx(b)).abs() <= 0.15 * width
        || (a.n[0] - b.n[0]).abs() <= 0.75 * pair
        || axis(a, b, 0) >= 0.35)
}
