//! Original mixed-body and detached mathematical-component recovery.
use super::evidence as e;
use super::*;
pub(super) fn flags(p: &Page, w: &mut Work) -> Result<Vec<bool>, Error> {
    let mut flags = Vec::new();
    let re = crate::cached_regex!(r"[a-z]{3,4}");
    for l in &p.lines {
        let text = l["text"].as_str().ok_or_else(invalid)?;
        w.charge(optional(l, "chars", 20000)?.len() + text.len() + 1)?;
        let words = super::super::native_page_preparation::formula_math_words(l)?;
        let words = array(&words, 20000)?;
        let word = |c| c == '_' || rapiddoc_native::text_boundary::python_alnum(c);
        let text = re.replace_all(text, |caps: &regex::Captures<'_>| {
            let m = caps.get(0).unwrap();
            if text[..m.start()].chars().last().is_none_or(|c| !word(c))
                && text[m.end()..].chars().next().is_none_or(|c| !word(c))
                && words.iter().any(|w| w.as_str() == Some(m.as_str()))
            {
                "x".to_owned()
            } else {
                m.as_str().to_owned()
            }
        });
        flags.push(e::display_prose(&text));
    }
    Ok(flags)
}
pub(super) fn mixed(
    p: &mut Page,
    indices: &[usize],
    w: &mut Work,
) -> Result<(Vec<Value>, BTreeSet<i64>), Error> {
    let mut blocks = Vec::new();
    let mut claimed = BTreeSet::new();
    let angles = indices
        .iter()
        .map(|&i| integer(&p.lines[i], "angle", 0))
        .collect::<Result<BTreeSet<_>, _>>()?;
    for a in angles {
        let mut geometry = Vec::new();
        for &i in indices {
            let l = &p.lines[i];
            if integer(l, "angle", 0)? == a
                && l["semantic_type"].is_null()
                && l["paragraph_formula_context"] != true
            {
                geometry.push(Row {
                    index: i,
                    b: local(&bbox(&l["bbox"])?, p.size, a),
                })
            }
        }
        let width = if a == 90 || a == 270 {
            p.size[1]
        } else {
            p.size[0]
        };
        let body = geometry
            .iter()
            .filter(|r| r.b.n[2] - r.b.n[0] >= 0.5 * width && e::prose_prefix(p.t(r).unwrap()))
            .collect::<Vec<_>>();
        if body
            .iter()
            .filter(|r| p.coverage(r).unwrap() < 0.75)
            .count()
            < 2
        {
            continue;
        }
        let h = median(body.iter().map(|r| p.h(r)).collect::<Result<Vec<_>, _>>()?);
        let left = median(body.iter().map(|r| r.b.n[0]).collect());
        let right = median(body.iter().map(|r| r.b.n[2]).collect());
        let bw = right - left;
        let mut barriers = geometry
            .iter()
            .filter(|r| {
                e::prose_prefix(p.t(r).unwrap())
                    && (r.b.n[2] - r.b.n[0] >= 0.5 * bw
                        || p.t(r)
                            .unwrap()
                            .chars()
                            .filter(|c| ('\u{3400}'..='\u{9fff}').contains(c))
                            .count()
                            >= 2)
            })
            .cloned()
            .collect::<Vec<_>>();
        barriers.sort_by(|a, b| a.b.n[1].total_cmp(&b.b.n[1]));
        for pair in barriers.windows(2) {
            let (top, bottom) = (pair[0].b.n[3], pair[1].b.n[1]);
            if bottom - top < 0.6 * h || bottom - top > 8. * h {
                continue;
            }
            let mut members = Vec::new();
            for r in &geometry {
                w.charge(1)?;
                let id = p.id(r)?;
                if id != p.id(&pair[0])?
                    && id != p.id(&pair[1])?
                    && !claimed.contains(&id)
                    && r.b.n[1] >= top - 0.2 * h
                    && r.b.n[3] <= bottom + 0.2 * h
                    && r.b.n[0] >= left
                    && r.b.n[2] <= right
                {
                    members.push(r.clone())
                }
            }
            if members.is_empty() {
                continue;
            }
            let b = p.bounds(&members)?;
            let text = members
                .iter()
                .map(|r| p.t(r))
                .collect::<Result<Vec<_>, _>>()?
                .join(" ");
            if !text.chars().any(|c| "=∑∫√≤≥<>".contains(c))
                || text
                    .chars()
                    .filter(|c| ('\u{3400}'..='\u{9fff}').contains(c))
                    .count()
                    >= 2
                || regex_words(&text, 4, None).len() >= 2
                || b.n[2] - b.n[0] < 0.15 * bw
                || b.n[2] - b.n[0] > 0.8 * bw
                || (cx(&b) - 0.5 * (left + right)).abs() > 0.1 * bw
            {
                continue;
            }
            let mut collision = false;
            for r in &members {
                w.charge(p.tables.len() + 1)?;
                let b = bbox(&p.l(r)["bbox"])?;
                if p.tables
                    .iter()
                    .any(|t| super::super::path_graphics::overlap_smaller(&b, t) > 0.)
                {
                    collision = true;
                    break;
                }
            }
            if collision {
                continue;
            }
            if let Some(block) = materialize::block(p, &members, a, p.id(&members[0])?, w)? {
                blocks.push(block);
                for r in members {
                    claimed.insert(p.id(&r)?);
                }
            }
        }
    }
    Ok((blocks, claimed))
}
pub(super) fn detached(
    p: &mut Page,
    indices: &[usize],
    drawings: &[Value],
    prose_flags: &[bool],
    w: &mut Work,
) -> Result<(Vec<Value>, BTreeSet<i64>), Error> {
    let mut blocks = Vec::new();
    let mut claimed = BTreeSet::new();
    let angle_values = indices
        .iter()
        .map(|&i| integer(&p.lines[i], "angle", 0))
        .collect::<Result<Vec<_>, _>>()?;
    let angles = super::super::table_native_vector::python_set::ordered_signed(&angle_values);
    for a in angles {
        let geometry = indices
            .iter()
            .copied()
            .filter(|&i| integer(&p.lines[i], "angle", 0).ok() == Some(a))
            .map(|i| ink(p, i, a))
            .collect::<Result<Vec<_>, _>>()?;
        if geometry.is_empty() {
            continue;
        }
        let em = median(
            geometry
                .iter()
                .map(|r| p.h(r))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let prose = geometry
            .iter()
            .filter(|r| prose_flags[r.index])
            .cloned()
            .collect::<Vec<_>>();
        let prose_layout = if a == 0 {
            layout(p, &prose.iter().map(|r| r.index).collect::<Vec<_>>(), w)?
        } else {
            vec![]
        };
        let prose_ids = prose
            .iter()
            .map(|r| p.id(r))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let mut candidates = Vec::new();
        for r in &geometry {
            if prose_ids.contains(&p.id(r)?) || !p.l(r)["semantic_type"].is_null() {
                continue;
            }
            let raw = bbox(&p.l(r)["bbox"])?;
            w.charge(p.tables.len() + 1)?;
            if !p.tables.iter().any(|t| intersects(&raw, t)) {
                candidates.push(r.clone())
            }
        }
        let operators = candidates
            .iter()
            .map(|r| p.t(r).map(math))
            .collect::<Result<Vec<_>, _>>()?;
        let mut pending = (0..candidates.len()).collect::<BTreeSet<_>>();
        while let Some(seed) = pending.pop_first() {
            let mut component = vec![seed];
            let mut frontier = vec![seed];
            while let Some(current) = frontier.pop() {
                let b = &candidates[current].b;
                let mut additions = Vec::new();
                for &i in &pending {
                    w.charge(1)?;
                    let c = &candidates[i].b;
                    if b.n[2] - b.n[0] > 3. * em
                        && c.n[2] - c.n[0] > 3. * em
                        && operators[current]
                        && operators[i]
                        && (cy(b) - cy(c)).abs() > 0.75 * em
                    {
                        continue;
                    }
                    let xgap = (b.n[0] - c.n[2]).max(c.n[0] - b.n[2]).max(0.);
                    let ygap = (b.n[1] - c.n[3]).max(c.n[1] - b.n[3]).max(0.);
                    if ygap <= 0.85 * em && axis(b, c, 0) > 0.1
                        || xgap <= 2. * em && axis(b, c, 1) >= 0.2
                    {
                        additions.push(i)
                    }
                }
                component.extend(&additions);
                frontier.extend(additions);
                for &i in &component {
                    pending.remove(&i);
                }
            }
            let mut members = component
                .iter()
                .map(|&i| candidates[i].clone())
                .collect::<Vec<_>>();
            if members.iter().any(|r| prose_flags[r.index])
                || !component.iter().any(|&i| operators[i])
            {
                continue;
            }
            let b = p.bounds(&members)?;
            let calculus = members
                .iter()
                .any(|r| p.t(r).unwrap().contains(['∂', '∫', '∑']));
            let numbered = members
                .iter()
                .any(|r| standalone(p.t(r).unwrap()).is_some());
            let strong = members
                .iter()
                .any(|r| p.t(r).unwrap().chars().filter(|&c| sm(c)).count() >= 2);
            let mut typed = false;
            for r in &members {
                let raw = Row {
                    index: r.index,
                    b: bbox(&p.l(r)["bbox"])?,
                };
                typed |= r.b.n[3] - r.b.n[1] >= 1.35 * p.h(&raw)? && p.coverage(r)? < 0.75
            }
            if !numbered && b.n[3] - b.n[1] < 1.35 * em && !calculus && !strong {
                continue;
            }
            let prose_rows = prose
                .iter()
                .filter(|r| {
                    r.b.n[2] - r.b.n[0] >= 0.35 * p.size[0]
                        && (cy(&r.b) - cy(&b)).abs() <= 8. * em
                        && axis(&r.b, &b, 0) > 0.5
                })
                .collect::<Vec<_>>();
            if !numbered
                && !calculus
                && members.len() == 1
                && b.n[3] - b.n[1] < 1.35 * em
                && !prose_rows.is_empty()
                && b.n[0] <= median(prose_rows.iter().map(|r| r.b.n[0]).collect()) + 0.5 * em
            {
                continue;
            }
            if component.len() == 1
                && prose.iter().any(|r| {
                    axis(&b, &r.b, 0) >= 0.2
                        && axis(&b, &r.b, 1) >= 0.2
                        && (cy(&b) - cy(&r.b)).abs() < 0.7 * em
                })
            {
                continue;
            }
            let mut fraction_rule = false;
            for rule in drawings {
                let horizontal = if a == 90 || a == 270 {
                    rule["orientation"] == "vertical"
                } else {
                    rule["orientation"] == "horizontal"
                };
                if !horizontal {
                    continue;
                }
                let r = local(&bbox(&rule["bbox"])?, p.size, a);
                if r.n[2] - r.n[0] >= 0.7 * em
                    && b.n[0] <= r.n[0]
                    && r.n[0] < r.n[2]
                    && r.n[2] <= b.n[2] + 0.2 * em
                    && b.n[1] + 0.3 * em < r.n[1]
                    && r.n[1] < b.n[3] - 0.3 * em
                    && members.iter().any(|m| m.b.n[3] <= r.n[1] + 0.35 * em)
                    && members.iter().any(|m| m.b.n[1] >= r.n[1] - 0.35 * em)
                {
                    fraction_rule = true;
                    break;
                }
            }
            let mut stacked = false;
            if members.len() >= 3 && members.iter().any(|r| p.t(r).unwrap().contains('=')) {
                for above in &members {
                    for below in &members {
                        w.charge(1)?;
                        let (a, b) = (&above.b, &below.b);
                        if above.index != below.index
                            && a.n[3] - b.n[1] <= 0.35 * em
                            && cy(b) - cy(a) >= 0.6 * em
                            && cy(b) - cy(a) <= 2. * em
                            && axis(a, b, 0) >= 0.7
                            && (a.n[2] - a.n[0]).max(b.n[2] - b.n[0]) <= 3. * em
                        {
                            stacked = true
                        }
                    }
                }
            }
            if !members.iter().any(|r| !p.l(r)["ink_bbox"].is_null())
                && !(numbered || strong || typed || fraction_rule || stacked)
            {
                continue;
            }
            if b.n[3] - b.n[1] > 8. * em || b.n[2] - b.n[0] < 3. * em {
                continue;
            }
            let corridor = if a == 0 {
                super::super::page_containers::layout_corridor(&b, &prose_layout, p.size[0])
            } else {
                None
            };
            if prose.iter().any(|r| {
                axis(&b, &r.b, 1) >= 0.15
                    && (cy(&b) - cy(&r.b)).abs() <= 0.7 * em
                    && ((b.n[0] - r.b.n[2]).max(r.b.n[0] - b.n[2]).max(0.) <= 3. * em
                        || corridor
                            .is_some_and(|(left, right)| left <= cx(&r.b) && cx(&r.b) <= right))
            }) {
                continue;
            }
            if margin(
                &b,
                if a == 90 || a == 270 {
                    p.size[0]
                } else {
                    p.size[1]
                },
            ) {
                continue;
            }
            let ids = members
                .iter()
                .map(|r| p.id(r))
                .collect::<Result<BTreeSet<_>, _>>()?;
            let mut markers = Vec::new();
            for r in &candidates {
                w.charge(prose.len() + 1)?;
                if ids.contains(&p.id(r)?)
                    || standalone(p.t(r)?).is_none()
                    || r.b.n[0] <= b.n[2]
                    || (cy(&r.b) - cy(&b)).abs() > 0.6 * em
                    || prose_rows.is_empty()
                {
                    continue;
                }
                if (r.b.n[2] - median(prose_rows.iter().map(|r| r.b.n[2]).collect())).abs()
                    > 2. * em
                    || prose.iter().any(|pr| {
                        pr.b.n[0] >= b.n[2] && pr.b.n[2] <= r.b.n[0] && axis(&pr.b, &r.b, 1) > 0.1
                    })
                {
                    continue;
                }
                markers.push(r.clone())
            }
            if markers.len() == 1 {
                members.extend(markers.clone());
                for r in markers {
                    claimed.insert(p.id(&r)?);
                }
                pending.retain(|&i| !claimed.contains(&p.id(&candidates[i]).unwrap()));
            }
            let anchor = members
                .iter()
                .find(|r| standalone(p.t(r).unwrap()).is_some())
                .unwrap_or(&members[0]);
            if let Some(block) = materialize::block(p, &members, a, p.id(anchor)?, w)? {
                blocks.push(block);
                for r in &members {
                    claimed.insert(p.id(r)?);
                }
            }
        }
    }
    Ok((blocks, claimed))
}
