use super::*;
fn sig(a: &Value, b: &Value) -> bool {
    eq(&a["font_signature"], &b["font_signature"])
}
fn reference_at<'a>(r: &'a Value, ls: &'a [Value]) -> &'a Value {
    r.get("_role_alias")
        .and_then(Value::as_u64)
        .and_then(|i| ls.get(i as usize))
        .unwrap_or(r)
}
fn uppercase(s: &str) -> bool {
    use rapiddoc_native::text_boundary::{python_lower, python_upper};
    static D: OnceLock<Value> = OnceLock::new();
    let title = D.get_or_init(|| serde_json::from_str(include_str!("unicode.json")).unwrap());
    let mut upper = false;
    for c in s.chars() {
        if python_upper(c) {
            upper = true
        } else if python_lower(c)
            || title["titlecase"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v.as_u64() == Some(c as u64))
        {
            return false;
        }
    }
    upper
}
pub(super) fn classify(
    ls: &mut [Value],
    bs: &[Box4],
    sz: [f64; 2],
    visual: &[Value],
    tables: &[Value],
    page: i64,
    reference: &[Cow<'_, Value>],
    w: &mut Work,
) -> Result<(), Error> {
    let available = (0..ls.len())
        .filter(|&i| {
            ls[i]["angle"] == 0
                && (ls[i]["semantic_type"].is_null() || ls[i]["semantic_type"] == "paragraph_title")
                && ls[i]["caption_start"] != true
                && ls[i]["title_suppressed"] != true
                && !ls[i]["font_signature"].is_null()
        })
        .collect::<Vec<_>>();
    let mut regular = Vec::new();
    let mut hs = Vec::new();
    for l in reference {
        w.charge(text(l)?.len() + 1)?;
        if integer(l, "angle", 0)? == 0
            && em(l)? > 0.
            && (english_words(text(l)?) >= 4 || text(l)?.chars().filter(|&c| cjk(c)).count() >= 10)
        {
            regular.push(l.as_ref());
            hs.push(em(l)?)
        }
    }
    if regular.len() < 3 {
        return Ok(());
    }
    let body = median(hs);
    // The reference list aliases source lines; synchronize mutations for original
    // objects only, as indicated by the caller's reference source identities.
    let mut short = Vec::new();
    for &i in &available {
        let t = text(&ls[i])?;
        let n = t
            .chars()
            .filter(|c| c.is_ascii_alphabetic() || cjk(*c))
            .count();
        let t = trim(t);
        if (3..=70).contains(&n)
            && words(t) <= 12
            && !t.contains(['=', '<', '>', '∑', '∫'])
            && !t.ends_with(['.', '!', '。', '！', ';', '；'])
        {
            short.push(i)
        }
    }
    let mut light = Vec::new();
    for &i in &short {
        w.charge(regular.len() + 1)?;
        if (0.85..=1.25).contains(&(em(&ls[i])? / body))
            && !ls[i]["dominant_font_weight"].is_null()
            && num(&ls[i]["dominant_font_weight"])? <= 350.
            && !regular
                .iter()
                .any(|r| sig(r, &ls[i]) && words(r["text"].as_str().unwrap()) >= 12)
        {
            light.push(i)
        }
    }
    let visuals = visual.iter().map(bbox).collect::<Result<Vec<_>, _>>()?;
    let tables = tables.iter().map(bbox).collect::<Result<Vec<_>, _>>()?;
    for &i in &light {
        w.charge(light.len() + regular.len() + visuals.len() + ls.len() + 1)?;
        let h = em(&ls[i])?;
        let mut peers = 0;
        for &j in &light {
            if sig(&ls[j], &ls[i]) && (0.9..=1.1).contains(&(em(&ls[j])? / h)) {
                peers += 1
            }
        }
        if peers < 2 {
            continue;
        }
        let mut followers = 0;
        for entry in &regular {
            let r = reference_at(entry, ls);
            let b = bbox(&r["bbox"])?;
            if !sig(&r, &ls[i])
                && (b.n[0] - bs[i].n[0]).abs() <= 0.6 * h
                && (0. ..=3. * h).contains(&(b.n[1] - bs[i].n[3]))
                && b.n[2] - b.n[0] >= bs[i].n[2] - bs[i].n[0]
            {
                followers += 1
            }
        }
        let above = visuals.iter().any(|b| {
            (0. ..=2. * h).contains(&(b.n[1] - bs[i].n[3]))
                && axis(&bs[i], b, 0) >= 0.7
                && ((cx(&bs[i]) - sz[0] / 2.).abs() <= h || (cx(&bs[i]) - cx(b)).abs() <= h)
        });
        if followers >= 2 || above {
            band(&[i], ls)?
        }
    }
    for &i in &short {
        w.charge(tables.len() + ls.len() + 1)?;
        if !ls[i]["semantic_type"].is_null() || !bold(&ls[i])? || em(&ls[i])? < 1.25 * body {
            continue;
        }
        if tables.iter().any(|b| {
            (0.3 * body..=2. * body).contains(&(b.n[1] - bs[i].n[3]))
                && (0.2..=0.7).contains(&((bs[i].n[2] - bs[i].n[0]) / (b.n[2] - b.n[0]).max(0.1)))
                && (cx(&bs[i]) - cx(b)).abs() <= 0.05 * (b.n[2] - b.n[0])
        }) {
            band(&[i], ls)?
        }
    }
    multiline(ls, bs, &short, body, page, w)?;
    let mut labels = Vec::new();
    for &i in &short {
        if !ls[i]["semantic_type"].is_null()
            || !text(&ls[i])?
                .trim_end_matches(python_space)
                .ends_with([':', '：'])
            || !(3..=8).contains(&words(text(&ls[i])?))
        {
            continue;
        }
        let h = em(&ls[i])?;
        let mut followers = 0;
        for entry in &regular {
            let r = reference_at(entry, ls);
            w.charge(1)?;
            let b = bbox(&r["bbox"])?;
            if r["semantic_type"].is_null()
                && (0.8..=1.2).contains(&(em(&r)? / h))
                && (0. ..=2. * h).contains(&(b.n[0] - bs[i].n[0]))
                && (0. ..=3. * h).contains(&(b.n[1] - bs[i].n[3]))
                && weight(&ls[i], 400.)? - weight(&r, 400.)? >= 100.
            {
                followers += 1
            }
        }
        if followers >= 2 {
            labels.push(i)
        }
    }
    for &i in &labels {
        w.charge(labels.len() + ls.len() + 1)?;
        if labels.iter().filter(|&&j| sig(&ls[j], &ls[i])).count() >= 3 {
            band(&[i], ls)?
        }
    }
    Ok(())
}
fn multiline(
    ls: &mut [Value],
    bs: &[Box4],
    short: &[usize],
    body: f64,
    page: i64,
    w: &mut Work,
) -> Result<(), Error> {
    let mut large = Vec::new();
    for &i in short {
        if em(&ls[i])? >= 1.4 * body
            && (weight(&ls[i], 400.)? >= 600.
                || page > 0 && text(&ls[i])?.chars().filter(|&c| cjk(c)).count() >= 3)
        {
            large.push(i)
        }
    }
    let mut rows = Vec::<Vec<usize>>::new();
    for i in ordered(large, bs) {
        let mut peer = None;
        for (r, ids) in rows.iter().enumerate() {
            w.charge(ids.len() + 1)?;
            let right = ids
                .iter()
                .map(|&j| bs[j].n[2])
                .fold(f64::NEG_INFINITY, f64::max);
            let dx = bs[i].n[0] - right;
            let all = ids
                .iter()
                .chain(std::iter::once(&i))
                .all(|&j| text(&ls[j]).unwrap().chars().any(cjk));
            let gap = if all {
                6. * em(&ls[i])?
            } else {
                0.5 * em(&ls[i])?
            };
            if sig(&ls[ids[0]], &ls[i])
                && axis(&bs[ids[0]], &bs[i], 1) >= 0.75
                && dx >= -0.1 * body
                && dx <= gap
            {
                peer = Some(r)
            }
        }
        if let Some(r) = peer {
            rows[r].push(i)
        } else {
            rows.push(vec![i])
        }
    }
    let mut pending = (0..rows.len()).collect::<Vec<_>>();
    let mut bands = Vec::new();
    while !pending.is_empty() {
        let mut run = vec![pending.remove(0)];
        loop {
            let b = bounds(&rows[*run.last().unwrap()], bs)?;
            let h = em(&ls[rows[run[0]][0]])?;
            let mut next = None;
            for (p, &r) in pending.iter().enumerate() {
                w.charge(rows[r].len() + 1)?;
                let c = bounds(&rows[r], bs)?;
                if sig(&ls[rows[r][0]], &ls[rows[run[0]][0]])
                    && (0.9..=1.1).contains(&(em(&ls[rows[r][0]])? / h))
                    && (-0.2 * h..=0.6 * h).contains(&(c.n[1] - b.n[3]))
                    && cy(&c) > cy(&b) + 0.5 * h
                    && ((c.n[0] - b.n[0]).abs() <= 0.4 * h || (cx(&c) - cx(&b)).abs() <= 0.4 * h)
                    && next.is_none_or(|(_, top)| c.n[1] < top)
                {
                    next = Some((p, c.n[1]))
                }
            }
            let Some((p, _)) = next else { break };
            run.push(pending.remove(p));
        }
        if run.len() >= 2 {
            bands.push(
                run.iter()
                    .flat_map(|&r| rows[r].iter().copied())
                    .collect::<Vec<_>>(),
            )
        }
    }
    for ids in &bands {
        w.charge(ls.len() + bands.iter().map(Vec::len).sum::<usize>() + 1)?;
        let chinese = page > 0
            && text(&ls[ids[0]])?.contains(['：', ':'])
            && ids
                .iter()
                .all(|&i| text(&ls[i]).unwrap().chars().filter(|&c| cjk(c)).count() >= 3);
        let repeated = ids.iter().all(|&i| {
            uppercase(text(&ls[i]).unwrap())
                && text(&ls[i])
                    .unwrap()
                    .chars()
                    .any(|c| c.is_ascii_uppercase())
        }) && bands
            .iter()
            .filter(|other| {
                sig(&ls[other[0]], &ls[ids[0]])
                    && (0.9..=1.1)
                        .contains(&(em(&ls[other[0]]).unwrap() / em(&ls[ids[0]]).unwrap()))
                    && other.iter().all(|&i| uppercase(text(&ls[i]).unwrap()))
            })
            .count()
            >= 2;
        if chinese || repeated {
            band(ids, ls)?
        }
    }
    Ok(())
}
