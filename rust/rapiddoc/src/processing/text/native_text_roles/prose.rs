use super::*;
fn signature(a: &Value, b: &Value) -> bool {
    eq(&a["font_signature"], &b["font_signature"])
}
fn italic(l: &Value) -> Result<bool, Error> {
    let a = array(&l["font_signature"], 2)?;
    if a.len() != 2 {
        return Err(invalid());
    }
    Ok(a[1].as_i64().ok_or_else(invalid)? & (1 << 6) != 0)
}
pub(super) fn quotes(ls: &mut [Value], bs: &[Box4], w: &mut Work) -> Result<(), Error> {
    let mut hs = Vec::new();
    for l in ls.iter() {
        if integer(l, "angle", 0)? == 0 && em(l)? > 0. {
            hs.push(em(l)?)
        }
    }
    if hs.is_empty() {
        return Ok(());
    }
    let body = median(hs);
    let mut styled = Vec::new();
    for (i, l) in ls.iter().enumerate() {
        if integer(l, "angle", 0)? == 0
            && l["semantic_type"].is_null()
            && l["caption_start"] != true
            && !l["font_signature"].is_null()
            && (italic(l)? || weight(l, 400.)? >= 600.)
            && (0.8..=1.2).contains(&(em(l)? / body))
        {
            styled.push(i)
        }
    }
    let mut rows = Vec::<Vec<usize>>::new();
    for i in ordered(styled, bs) {
        w.charge(rows.len() + 1)?;
        let peer = rows.iter().rposition(|r| {
            signature(&ls[r[0]], &ls[i])
                && !ls[r[0]]["visual_row_id"].is_null()
                && eq(&ls[r[0]]["visual_row_id"], &ls[i]["visual_row_id"])
                && (bs[i].n[0] - bs[*r.last().unwrap()].n[2])
                    .abs()
                    .min((bs[*r.last().unwrap()].n[0] - bs[i].n[2]).abs())
                    <= 3. * body
        });
        if let Some(j) = peer {
            rows[j].push(i)
        } else {
            rows.push(vec![i])
        }
    }
    let mut pending = (0..rows.len()).collect::<Vec<_>>();
    let mut group = next_group(ls)?;
    let close = constant_regex!(&format!(r#"[”"]{SPACE}*[.!?]?{SPACE}*$"#));
    while !pending.is_empty() {
        let mut run = vec![pending.remove(0)];
        loop {
            let previous = bounds(&rows[*run.last().unwrap()], bs)?;
            let mut next = None;
            for (p, &j) in pending.iter().enumerate() {
                w.charge(rows[j].len() + 1)?;
                let b = bounds(&rows[j], bs)?;
                if signature(&ls[rows[j][0]], &ls[rows[run[0]][0]])
                    && (-0.15 * body..=0.8 * body).contains(&(b.n[1] - previous.n[3]))
                    && axis(&b, &previous, 0) >= 0.5
                    && next.is_none_or(|(_, top)| b.n[1] < top)
                {
                    next = Some((p, b.n[1]))
                }
            }
            let Some((p, _)) = next else { break };
            run.push(pending.remove(p));
        }
        let ids = run
            .iter()
            .flat_map(|&r| rows[r].iter().copied())
            .collect::<Vec<_>>();
        let t = ids
            .iter()
            .map(|&i| text(&ls[i]))
            .collect::<Result<Vec<_>, _>>()?
            .join(" ");
        let t = trim(&t);
        if run.len() < 3 || english_words(t) < 15 || t.contains(['=', '<', '>', '∑', '∫']) {
            continue;
        }
        let opening = t.starts_with(['“', '"', '‘']);
        if !(italic(&ls[ids[0]])? && opening || run.len() >= 6 && close.is_match(t)) {
            continue;
        }
        mark(ids.iter().copied(), ls, group);
        group = group.checked_add(1).ok_or_else(limit)?;
        let b = bounds(&ids, bs)?;
        let mut preceding = Vec::new();
        for (i, l) in ls.iter().enumerate() {
            w.charge(1)?;
            if l["semantic_type"].is_null()
                && l["caption_start"] != true
                && integer(l, "angle", 0)? == 0
                && (0.8..=1.2).contains(&(em(l)? / body))
                && (0. ..=4. * body).contains(&(b.n[1] - bs[i].n[3]))
                && axis(&bs[i], &b, 0) >= 0.5
            {
                preceding.push(i)
            }
        }
        preceding.sort_by(|&a, &b| bs[a].n[1].total_cmp(&bs[b].n[1]));
        if let Some(&last) = preceding.last() {
            if text(&ls[last])?
                .trim_end_matches(python_space)
                .ends_with(':')
            {
                let mut intro = vec![last];
                for &i in preceding[..preceding.len() - 1].iter().rev() {
                    if !signature(&ls[i], &ls[intro[0]])
                        || !(-0.1 * body..=0.8 * body).contains(&(bs[intro[0]].n[1] - bs[i].n[3]))
                    {
                        break;
                    }
                    intro.insert(0, i)
                }
                let t = intro
                    .iter()
                    .map(|&i| text(&ls[i]))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(" ");
                if english_words(&t) >= 6 {
                    mark(intro, ls, group);
                    group = group.checked_add(1).ok_or_else(limit)?;
                }
            }
        }
    }
    Ok(())
}
pub(super) fn inline(
    ls: &mut [Value],
    bs: &[Box4],
    _sz: [f64; 2],
    regions: &[Value],
    w: &mut Work,
) -> Result<(), Error> {
    if regions.len() < 2 {
        return Ok(());
    }
    let ids = ordered(
        (0..ls.len())
            .filter(|&i| {
                ls[i]["angle"] == 0
                    && ls[i]["semantic_type"].is_null()
                    && ls[i]["caption_start"] != true
                    && ls[i]["paragraph_group"].is_null()
                    && !ls[i]["font_signature"].is_null()
            })
            .collect(),
        bs,
    );
    let mut rows = Vec::<Vec<usize>>::new();
    for i in ids {
        w.charge(rows.len() + 1)?;
        if let Some(j) = rows.iter().position(|r| {
            !ls[r[0]]["visual_row_id"].is_null()
                && eq(&ls[r[0]]["visual_row_id"], &ls[i]["visual_row_id"])
                && signature(&ls[r[0]], &ls[i])
        }) {
            rows[j].push(i)
        } else {
            rows.push(vec![i])
        }
    }
    let mut sources = BTreeMap::new();
    let mut boxes = Vec::new();
    let mut hits = Vec::new();
    for (r, ids) in rows.iter().enumerate() {
        boxes.push(bounds(ids, bs)?);
        hits.push(BTreeSet::new());
        for &i in ids {
            sources.insert(integer(&ls[i], "source_index", 0)?, r);
        }
    }
    for (index, region) in regions.iter().enumerate() {
        let b = bbox(&region["bbox"])?;
        for s in array(&region["source_indices"], 4096)? {
            w.charge(1)?;
            if let Some(&r) = sources.get(&s.as_i64().ok_or_else(invalid)?) {
                boxes[r] = combined([&boxes[r], &b])?;
                hits[r].insert(index);
            }
        }
    }
    let mut pending = (0..rows.len()).collect::<Vec<_>>();
    pending.sort_by(|&a, &b| boxes[a].n[1].total_cmp(&boxes[b].n[1]));
    let mut group = next_group(ls)?;
    while !pending.is_empty() {
        let mut run = vec![pending.remove(0)];
        let h = median(
            rows[run[0]]
                .iter()
                .map(|&i| height(&ls[i], &bs[i]))
                .collect::<Result<Vec<_>, _>>()?,
        );
        while !pending.is_empty() {
            let r = pending[0];
            let before = &boxes[*run.last().unwrap()];
            let b = &boxes[r];
            w.charge(rows[r].len() + 1)?;
            let mut same = true;
            for &i in &rows[r] {
                if !(0.85..=1.15).contains(&(height(&ls[i], &bs[i])? / h)) {
                    same = false
                }
            }
            if !signature(&ls[rows[r][0]], &ls[rows[run[0]][0]])
                || !same
                || !(-0.15 * h..=0.8 * h).contains(&(b.n[1] - before.n[3]))
                || axis(before, b, 0) < 0.5
            {
                break;
            }
            run.push(pending.remove(0));
        }
        let ids = run
            .iter()
            .flat_map(|&r| rows[r].iter().copied())
            .collect::<Vec<_>>();
        let t = ids
            .iter()
            .map(|&i| text(&ls[i]))
            .collect::<Result<Vec<_>, _>>()?
            .join(" ");
        let hit = run
            .iter()
            .flat_map(|&r| hits[r].iter().copied())
            .collect::<BTreeSet<_>>();
        if run.len() < 3 || hit.len() < 2 || english_words(&t) < 20 {
            continue;
        }
        mark(ids, ls, group);
        group = group.checked_add(1).ok_or_else(limit)?;
    }
    Ok(())
}
