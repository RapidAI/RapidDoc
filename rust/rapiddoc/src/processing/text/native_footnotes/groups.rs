use super::*;
pub(super) fn marker(
    t: &str,
    optional_punctuation: bool,
    content: bool,
) -> Option<(String, u32, usize)> {
    let cs = t.chars().collect::<Vec<_>>();
    let mut i = 0;
    while cs.get(i).is_some_and(|c| python_space(*c)) {
        i += 1
    }
    let start = i;
    let mut number = 0;
    while let Some(n) = cs.get(i).and_then(|c| python_decimal(*c)) {
        number = number * 10 + n as u32;
        i += 1;
        if i - start > 3 {
            return None;
        }
    }
    if i == start {
        return None;
    }
    let digits = cs[start..i].iter().collect();
    let punctuation = cs.get(i).is_some_and(|c| matches!(c, '.' | ')'));
    if punctuation {
        i += 1
    } else if !optional_punctuation {
        return None;
    }
    let end = i;
    while cs.get(i).is_some_and(|c| python_space(*c)) {
        i += 1
    }
    if i == end || content && i == cs.len() {
        return None;
    }
    Some((digits, number, end - start))
}
fn references<'a>(
    ls: impl IntoIterator<Item = &'a Value>,
    w: &mut Work,
) -> Result<BTreeSet<String>, Error> {
    let mut values = BTreeSet::new();
    for l in ls {
        let mut chars = Vec::new();
        for c in optional(l, "chars", 20000)? {
            let t = super::super::table_native_multiline::spatial::raw_string(c.get("char"), false);
            w.charge(t.len() + 1)?;
            if !t.trim_matches(python_space).is_empty()
                && !c["origin"].is_null()
                && c["origin"].as_array().is_some_and(|a| !a.is_empty())
            {
                chars.push((c, t))
            }
        }
        if chars.len() < 4 {
            continue;
        }
        let boxes = chars
            .iter()
            .map(|(c, _)| bbox(&c["bbox"]))
            .collect::<Result<Vec<_>, _>>()?;
        let em = median(boxes.iter().map(|b| b.n[3] - b.n[1]).collect());
        let baseline = median(
            chars
                .iter()
                .map(|(c, _)| {
                    let a = array(&c["origin"], 2)?;
                    if a.len() != 2 {
                        return Err(invalid());
                    }
                    num(&a[1])
                })
                .collect::<Result<Vec<_>, _>>()?,
        );
        let mut digits = String::new();
        for ((c, t), b) in chars.iter().zip(boxes) {
            if !t.is_empty()
                && t.chars().all(python_digit)
                && b.n[3] - b.n[1] <= 0.82 * em
                && baseline - num(&c["origin"][1])? >= 0.15 * em
            {
                digits.push_str(t)
            } else {
                if (1..=3).contains(&digits.chars().count()) {
                    values.insert(digits.clone());
                }
                digits.clear();
            }
        }
        if (1..=3).contains(&digits.chars().count()) {
            values.insert(digits);
        }
    }
    Ok(values)
}
fn rows<'a>(
    raw: &[Row<'a>],
    w: &mut Work,
) -> Result<(Vec<Row<'a>>, BTreeMap<i64, BTreeSet<i64>>), Error> {
    let mut order = raw.to_vec();
    order.sort_by(|a, b| {
        cy(&a.b)
            .total_cmp(&cy(&b.b))
            .then(a.b.n[0].total_cmp(&b.b.n[0]))
    });
    let mut groups = Vec::<Vec<Row<'a>>>::new();
    for r in order {
        let mut joined = None;
        for (i, g) in groups.iter().enumerate().rev() {
            w.charge(g.len() + 1)?;
            let reference = combined(g.iter().map(|r| &r.b))?;
            let em = h(&r)?.min(median(g.iter().map(h).collect::<Result<Vec<_>, _>>()?));
            if cy(&r.b) - cy(&reference) > 2. * em {
                break;
            }
            if (cy(&r.b) - cy(&reference)).abs() <= 0.4 * em
                && axis(&r.b, &reference, 1) >= 0.6
                && (r.b.n[0] - reference.n[2])
                    .abs()
                    .min((reference.n[0] - r.b.n[2]).abs())
                    <= 0.75 * em
            {
                joined = Some(i);
                break;
            }
        }
        if let Some(i) = joined {
            groups[i].push(r)
        } else {
            groups.push(vec![r])
        }
    }
    let mut result = Vec::new();
    let mut members = BTreeMap::new();
    for g in groups {
        let first = g.iter().min_by_key(|r| id(&r.l).unwrap()).unwrap();
        members.insert(
            id(&first.l)?,
            g.iter()
                .map(|r| id(&r.l))
                .collect::<Result<BTreeSet<_>, _>>()?,
        );
        if g.len() == 1 {
            result.push(g[0].clone())
        } else {
            let em = median(g.iter().map(h).collect::<Result<Vec<_>, _>>()?);
            let mut l = first.l.clone();
            let boxes = g
                .iter()
                .map(|r| bbox(&r.l["bbox"]))
                .collect::<Result<Vec<_>, _>>()?;
            l.to_mut()["bbox"] = json!(combined(boxes.iter())?.raw);
            l.to_mut()["effective_height"] = json!(em);
            l.to_mut()["em_height"] = json!(em);
            result.push(Row {
                l,
                b: combined(g.iter().map(|r| &r.b))?,
            })
        }
    }
    Ok((result, members))
}
fn unruled(rows: &[Row], p: [f64; 2], w: &mut Work) -> Result<Vec<BTreeSet<i64>>, Error> {
    let ordered = sorted(rows.to_vec(), false);
    let mut refs = None;
    let mut consumed = BTreeSet::<i64>::new();
    let mut groups = Vec::new();
    for (i, first) in ordered.iter().enumerate() {
        w.charge(ordered.len() * 2 + 1)?;
        let Some((value, _, _)) = marker(text(&first.l)?, false, true) else {
            continue;
        };
        if consumed.contains(&id(&first.l)?)
            || first.b.n[1] < 0.7 * p[1]
            || weight(&first.l)? >= 600.
        {
            continue;
        }
        let em = h(first)?;
        let above = ordered[..i]
            .iter()
            .filter(|r| {
                cy(&r.b) <= cy(&first.b) - 0.5 * em
                    && r.b.n[2] - r.b.n[0] >= 0.25 * p[0]
                    && axis(&r.b, &first.b, 0) >= 0.3
            })
            .collect::<Vec<_>>();
        if above.len() < 3 {
            continue;
        }
        let body = median(above.iter().map(|r| h(r)).collect::<Result<Vec<_>, _>>()?);
        let mut nearest = above[0];
        for r in &above[1..] {
            if cy(&r.b) > cy(&nearest.b) {
                nearest = r
            }
        }
        let space = gap(nearest, first)?;
        let small = em <= 0.9 * body && space >= 0.75 * body;
        let left = median(above.iter().map(|r| r.b.n[0]).collect());
        let spanning = first.b.n[2] - first.b.n[0] >= 0.65 * p[0]
            && em <= 1.25 * body
            && space >= 2. * body
            && left - first.b.n[0] >= 0.5 * body;
        let mut referenced = false;
        if !small && spanning {
            if refs.is_none() {
                refs = Some(references(ordered.iter().map(|r| r.l.as_ref()), w)?)
            }
            referenced = refs.as_ref().unwrap().contains(&value)
        }
        if !small && !referenced {
            continue;
        }
        let mut group = BTreeSet::from([id(&first.l)?]);
        let mut previous = first;
        for r in &ordered[i + 1..] {
            if r.b.n[1] < first.b.n[1] || axis(&first.b, &r.b, 0) < 0.3 {
                continue;
            }
            let size = h(r)?;
            let space = gap(previous, r)?;
            if marker(text(&r.l)?, false, true).is_some()
                || size > 1.15 * em
                || space > 1.5 * em
                || r.b.n[0] < first.b.n[0] - 0.4 * em
                || r.b.n[0] > first.b.n[0] + 2. * em
            {
                break;
            }
            if r.b.n[1] >= 0.92 * p[1]
                && r.b.n[2] - r.b.n[0] <= 0.55 * (first.b.n[2] - first.b.n[0])
                && size <= 0.95 * em
                && space >= 0.25 * em
            {
                break;
            }
            group.insert(id(&r.l)?);
            previous = r;
        }
        consumed.extend(&group);
        groups.push(group);
    }
    Ok(groups)
}
fn chart(
    rows: &[Row],
    visuals: &[Box4],
    reference: &[&Value],
    p: [f64; 2],
    w: &mut Work,
) -> Result<Vec<BTreeSet<i64>>, Error> {
    let ordered = sorted(rows.to_vec(), false);
    let mut refs = None;
    let mut consumed = BTreeSet::<i64>::new();
    let mut groups = Vec::new();
    for (i, first) in ordered.iter().enumerate() {
        w.charge(visuals.len() + ordered.len() + 1)?;
        let Some((value, mut number, _)) = marker(text(&first.l)?, true, true) else {
            continue;
        };
        let em = h(first)?;
        if consumed.contains(&id(&first.l)?) || first.b.n[1] < 0.7 * p[1] {
            continue;
        }
        let charts = visuals.iter().any(|b| {
            b.n[3] - b.n[1] >= 8. * em
                && b.n[2] - b.n[0] >= 0.2 * p[0]
                && (0.8 * em..=12. * em).contains(&(first.b.n[1] - b.n[3]))
                && axis(&first.b, b, 0) >= 0.6
        });
        if !charts || weight(&first.l)? >= 600. {
            continue;
        }
        if refs.is_none() {
            refs = Some(references(reference.iter().copied(), w)?)
        }
        if !refs.as_ref().unwrap().contains(&value) {
            continue;
        }
        let mut group = BTreeSet::from([id(&first.l)?]);
        let mut previous = first;
        for r in &ordered[i + 1..] {
            if axis(&first.b, &r.b, 0) < 0.3 {
                continue;
            }
            if !(0.85 * em..=1.15 * em).contains(&h(r)?)
                || !eq(&r.l["font_signature"], &first.l["font_signature"])
                || (r.b.n[0] - first.b.n[0]).abs() > 0.4 * em
                || gap(previous, r)? > 1.5 * em
            {
                break;
            }
            if let Some((_, n, _)) = marker(text(&r.l)?, true, true) {
                if n != number + 1 {
                    break;
                }
                consumed.extend(&group);
                groups.push(group);
                group = BTreeSet::new();
                number += 1;
            }
            group.insert(id(&r.l)?);
            previous = r;
        }
        consumed.extend(&group);
        groups.push(group);
    }
    Ok(groups)
}
pub(super) fn page(
    ls: &mut [Value],
    sz: [f64; 2],
    visual: &[Box4],
    tables: &[Box4],
    draw: &[Value],
    reference: &[Value],
    w: &mut Work,
) -> Result<Vec<BTreeSet<i64>>, Error> {
    let Some((raw, a, p)) = geometry(ls, sz, w)? else {
        return Ok(vec![]);
    };
    if raw.is_empty() {
        return Ok(vec![]);
    }
    let ink = raw
        .iter()
        .map(|r| {
            Ok(local(
                if r.l["ink_bbox"].is_null() {
                    bbox(&r.l["bbox"])?
                } else {
                    bbox(&r.l["ink_bbox"])?
                },
                sz,
                a,
            ))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let (rows, members) = rows(&raw, w)?;
    if p[0] <= 0. || p[1] <= 0. {
        return Ok(vec![]);
    }
    let em = median(rows.iter().map(h).collect::<Result<Vec<_>, _>>()?);
    let bounds: Vec<Value> = rows
        .iter()
        .map(|r| Value::Array(r.b.raw.to_vec()))
        .collect();
    let geometry: Vec<_> = rows
        .iter()
        .zip(&bounds)
        .map(|(r, b)| (r.l.as_ref(), b))
        .collect();
    let layout = lanes::infer_layout_rows(
        &geometry,
        &json!({"local_page_width":p[0],"median_height":em,"recalculate_intervals":false}),
        w,
        true,
    )?;
    let mut lanes = lane::parse(&layout)?;
    let rs = rules(draw, sz, a)?;
    let tr = table_rules(&rs, tables, w)?;
    let mut groups = unruled(&rows, p, w)?;
    if !visual.is_empty() {
        let refs = if reference.is_empty() {
            ls.iter()
                .filter(|l| l["semantic_type"].is_null())
                .collect::<Vec<_>>()
        } else {
            reference.iter().collect()
        };
        groups.extend(chart(&rows, visual, &refs, p, w)?)
    }
    for rule in &rs {
        w.charge(ink.len() + rs.len() + tr.len() + visual.len() + 1)?;
        if !rule.horizontal || cy(&rule.b) < 0.55 * p[1] {
            continue;
        }
        let rw = rule.b.n[2] - rule.b.n[0];
        let neighbors = ink
            .iter()
            .filter(|b| {
                b.n[2] - b.n[0] >= 0.4 * rw
                    && b.n[2] - b.n[0] <= rw + em
                    && axis(b, &rule.b, 0) >= 0.8
                    && (cx(b) - cx(&rule.b)).abs() <= 0.15 * rw
            })
            .collect::<Vec<_>>();
        if rw < 0.25 * p[0]
            && neighbors
                .iter()
                .any(|b| (0. ..=em).contains(&(cy(&rule.b) - b.n[3])))
            && neighbors
                .iter()
                .any(|b| (0. ..=em).contains(&(b.n[1] - cy(&rule.b))))
        {
            continue;
        }
        if belongs(rule, &rs, &tr, p[0]) {
            continue;
        }
        let expanded = expanded(&rule.original, rule.width.max(0.5))?;
        if visual.iter().any(|b| overlap(&expanded, b)) {
            continue;
        }
        let count = lanes.len();
        if rw >= 0.7 * p[0] {
            lanes.push(lane::Lane {
                left: rule.b.n[0],
                right: rule.b.n[2],
                span: false,
                indices: rows
                    .iter()
                    .enumerate()
                    .filter(|(_, r)| {
                        r.b.n[0] >= rule.b.n[0] - 0.25 * em && r.b.n[2] <= rule.b.n[2] + 0.25 * em
                    })
                    .map(|(i, _)| i)
                    .collect(),
            })
        }
        let mut group = BTreeSet::new();
        for l in &lanes {
            let lower = rs
                .iter()
                .filter(|s| {
                    s.horizontal
                        && s.b.n[2] - s.b.n[0] >= (4. * em).max(0.04 * p[0])
                        && s.b.n[1] - rule.b.n[3] > 0.5 * em
                        && axis(&rule.b, &s.b, 0) >= 0.8
                })
                .map(|s| s.b.n[1])
                .reduce(f64::min);
            let width = lane::width_reference(l, &lanes[..count], em);
            group.extend(lane::collect(
                l,
                &rows,
                &rule.b,
                p,
                em,
                width,
                cy(&rule.b) >= 0.55 * p[1],
                lower,
                w,
            )?);
        }
        lanes.truncate(count);
        if !group.is_empty() {
            groups.push(group)
        }
    }
    let mut merged = Vec::<BTreeSet<i64>>::new();
    for mut group in groups {
        let mut i = 0;
        while i < merged.len() {
            w.charge(group.len() + merged[i].len() + 1)?;
            if !group.is_disjoint(&merged[i]) {
                group.extend(merged.remove(i));
                i = 0
            } else {
                i += 1
            }
        }
        merged.push(group)
    }
    merged.sort_by_key(|g| *g.first().unwrap());
    let by_id = rows
        .iter()
        .map(|r| Ok((id(&r.l)?, r)))
        .collect::<Result<BTreeMap<_, _>, Error>>()?;
    for group in &mut merged {
        let rs = group
            .iter()
            .filter_map(|i| by_id.get(i))
            .collect::<Vec<_>>();
        if rs.is_empty() {
            continue;
        }
        let top = rs.iter().map(|r| r.b.n[1]).reduce(f64::min).unwrap();
        let bottom = rs.iter().map(|r| r.b.n[3]).reduce(f64::max).unwrap();
        let left = rs.iter().map(|r| r.b.n[0]).reduce(f64::min).unwrap();
        for r in &rows {
            w.charge(1)?;
            if r.b.n[2] - r.b.n[0] <= 1.5 * em
                && (left - 2. * em..=left).contains(&r.b.n[0])
                && r.b.n[2] <= left + 0.5 * em
                && (top - em..=bottom + em).contains(&cy(&r.b))
            {
                group.insert(id(&r.l)?);
            }
        }
    }
    let result = merged
        .iter()
        .map(|g| {
            g.iter()
                .flat_map(|i| members[i].iter().copied())
                .collect::<BTreeSet<_>>()
        })
        .collect::<Vec<_>>();
    let all = result.iter().flatten().copied().collect::<BTreeSet<_>>();
    for l in ls {
        if l["semantic_type"].is_null() && all.contains(&id(l)?) {
            l["semantic_type"] = json!("page_footnote")
        }
    }
    Ok(result)
}
