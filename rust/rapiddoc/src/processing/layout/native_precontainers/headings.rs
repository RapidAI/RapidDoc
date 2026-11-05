use super::*;
fn trimmed(l: &Value) -> Result<&str, Error> {
    Ok(text(l)?.trim_matches(python_space))
}
fn available(ls: &[Value]) -> Result<Vec<usize>, Error> {
    ls.iter()
        .enumerate()
        .filter_map(|(i, l)| match integer(l, "angle", 0) {
            Ok(0) if l["semantic_type"].is_null() || l["semantic_type"] == "paragraph_title" => {
                Some(Ok(i))
            }
            Ok(_) => None,
            Err(e) => Some(Err(e)),
        })
        .collect()
}
fn number(l: &Value, k: &str, default: f64) -> Result<f64, Error> {
    if l[k].is_null() {
        Ok(default)
    } else {
        num(&l[k])
    }
}
fn weight(l: &Value, default: f64) -> Result<f64, Error> {
    let x = number(l, "dominant_font_weight", default)?;
    Ok(if x == 0. { default } else { x })
}
pub(super) fn classify(source: &Value, containers: &[Value], w: &mut Work) -> Result<Value, Error> {
    classify_mode(source, containers, w, true)
}
pub(super) fn classify_mode(
    source: &Value,
    containers: &[Value],
    w: &mut Work,
    appendix_only: bool,
) -> Result<Value, Error> {
    let mut out = source.clone();
    let mut ls = array(&source["lines"], 4096)?.to_vec();
    let sz = size(&source["page_size"])?;
    let bs = ls
        .iter()
        .map(|l| bbox(&l["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let cs = containers.iter().map(bbox).collect::<Result<Vec<_>, _>>()?;
    let ids = available(&ls)?;
    let gate = constant_regex!(&format!(r"^[A-Z](?:{SPACE}|$)"));
    let prefix = constant_regex!(&format!(
        r"^(?:[0-9]+(?:\.[0-9]+)*\.?|[A-Z])(?:{SPACE}|$|[-–—]{SPACE}*[A-Z])"
    ));
    let marker = constant_regex!(r"^(?:[0-9]+(?:\.[0-9]+)*\.?|[A-Z])$");
    let letters = constant_regex!(r"[A-Za-z\u{3400}-\u{9fff}]");
    let caps = constant_regex!(&format!(r"^[A-Z](?:{SPACE}+[A-Z])*$"));
    let year = constant_regex!(&format!(r"^[0-9]{{4,}}(?:{SPACE}|$)"));
    let start = constant_regex!(&format!(r"^(?:[0-9]+(?:\.[0-9]+)*\.?|[A-Z])(?:{SPACE}|$)"));
    for &i in &ids {
        w.charge(ids.len() + cs.len() + text(&ls[i])?.len() + 1)?;
        if !ls[i]["title_band_id"].is_null() && ls[i]["structural_title"] == true {
            continue;
        }
        let t = trimmed(&ls[i])?;
        if appendix_only && !gate.is_match(t) || !prefix.is_match(&mapped(t)) {
            continue;
        }
        let h = height(&ls[i], &bs[i])?;
        let mut members = vec![i];
        if marker.is_match(&mapped(t)) {
            let mut companion = None;
            for &j in &ids {
                if j != i
                    && bs[j].n[0] >= bs[i].n[2]
                    && bs[j].n[0] - bs[i].n[2] <= 2. * h
                    && axis(&bs[j], &bs[i], 1) >= 0.6
                {
                    if companion.is_none_or(|k: usize| bs[j].n[0] < bs[k].n[0]) {
                        companion = Some(j)
                    }
                }
            }
            let Some(j) = companion else { continue };
            members.push(j);
        }
        let body = *members.last().unwrap();
        let label = members
            .iter()
            .map(|&j| trimmed(&ls[j]))
            .collect::<Result<Vec<_>, _>>()?
            .join(" ");
        let bounds = combined(members.iter().map(|&j| &bs[j]))?;
        if !bold(&ls[body])?
            || number(&ls[body], "font_coverage", 0.)? < 0.75
            || words(&label) > 15
            || !letters.is_match(text(&ls[body])?)
            || caps.is_match(trimmed(&ls[body])?)
            || year.is_match(&mapped(&label))
            || label.ends_with([
                '.', '!', '?', '。', '！', '？', ':', '：', ';', '；', ',', '，',
            ])
            || bounds.n[2] - bounds.n[0] > 0.8 * sz[0]
            || cs
                .iter()
                .any(|b| axis(&bounds, b, 0) > 0.5 && axis(&bounds, b, 1) > 0.5)
        {
            continue;
        }
        if words(&label) >= 5 {
            let mut next = None;
            for &j in &ids {
                w.charge(members.len() + 1)?;
                if members
                    .iter()
                    .any(|&k| crate::legacy_page::eq(&ls[j], &ls[k]))
                {
                    continue;
                }
                let dx = bs[j].n[0] - bounds.n[0];
                let dy = bs[j].n[1] - bounds.n[3];
                if ((h < dx && dx <= 3. * h)
                    || (dx.abs() <= 0.25 * h && !start.is_match(&mapped(trimmed(&ls[j])?))))
                    && dy >= -0.25 * h
                    && dy <= 0.75 * h
                    && cy(&bs[j]) > cy(&bounds) + 0.5 * h
                    && bs[j].n[2] <= bounds.n[2] + h
                    && bold(&ls[j])?
                    && crate::legacy_page::eq(&ls[body]["font_signature"], &ls[j]["font_signature"])
                {
                    let ratio = height(&ls[j], &bs[j])? / h;
                    if (0.9..=1.1).contains(&ratio)
                        && next.is_none_or(|k: usize| bs[j].n[1] < bs[k].n[1])
                    {
                        next = Some(j)
                    }
                }
            }
            if let Some(j) = next {
                members.push(j)
            }
        }
        let id = integer(&ls[i], "source_index", 0)?;
        for j in members {
            promote(&mut ls[j], id)
        }
    }
    if !appendix_only {
        super::super::native_titles::final_roman_headings(&mut ls, &bs, &cs, sz, w)?;
    }
    extreme(&mut ls, &bs, &cs, sz, w)?;
    out["lines"] = json!(ls);
    Ok(out)
}
fn extreme(
    ls: &mut [Value],
    bs: &[Box4],
    cs: &[Box4],
    sz: [f64; 2],
    w: &mut Work,
) -> Result<(), Error> {
    let mut repaired = Vec::new();
    for (i, l) in ls.iter().enumerate() {
        if l["style_scale_repaired"] == true && !l["source_bbox"].is_null() {
            let b = bbox(&l["source_bbox"])?;
            let h = height(l, &bs[i])?;
            if b.n[3] - b.n[1] >= 4. * h {
                repaired.push(h)
            }
        }
    }
    if repaired.len() < 6 {
        return Ok(());
    }
    let em = median(repaired);
    let ids = available(ls)?;
    for &i in &ids {
        w.charge(ids.len() * 2 + cs.len() + text(&ls[i])?.len() + 1)?;
        let t = trimmed(&ls[i])?;
        let h = height(&ls[i], &bs[i])?;
        let cx = (bs[i].n[0] + bs[i].n[2]) * 0.5;
        if weight(&ls[i], 0.)? < 600.
            || number(&ls[i], "font_coverage", 0.)? < 0.4
            || h < 0.85 * em
            || h > 1.5 * em
            || !(2..=20).contains(&words(t))
            || t.chars()
                .next()
                .is_some_and(|c| python_decimal(c).is_some() || ['•', '●', '▪'].contains(&c))
            || t.ends_with(['.', '!', '?', '。', '！', '？'])
            || ls[i]["caption_start"] == true
            || ls[i]["formula_candidate_only"] == true
            || !optional(&ls[i], "inline_math_regions", 4000)?.is_empty()
            || bs[i].n[2] - bs[i].n[0] > 0.85 * sz[0]
            || cs.iter().any(|b| {
                cx >= b.n[0] && cx <= b.n[2] && cy(&bs[i]) >= b.n[1] && cy(&bs[i]) <= b.n[3]
            })
        {
            continue;
        }
        if ids
            .iter()
            .any(|&j| j != i && axis(&bs[j], &bs[i], 1) >= 0.6)
        {
            continue;
        }
        let mut followers = Vec::new();
        let mut previous = None;
        for &j in &ids {
            if j == i {
                continue;
            }
            let dx = bs[j].n[0] - bs[i].n[0];
            let dy = bs[j].n[1] - bs[i].n[3];
            if weight(&ls[j], 400.)? < 600.
                && dx >= -0.25 * em
                && dx <= 2.5 * em
                && dy >= 0.2 * em
                && dy <= 8. * em
            {
                followers.push(j)
            }
            if bs[j].n[3] <= bs[i].n[1] && axis(&bs[j], &bs[i], 0) >= 0.2 {
                previous = Some(previous.map_or(bs[j].n[3], |p: f64| p.max(bs[j].n[3])))
            }
        }
        if followers.len() < 2 {
            continue;
        }
        let following = *followers
            .iter()
            .min_by(|&&a, &&b| bs[a].n[1].total_cmp(&bs[b].n[1]))
            .unwrap();
        let above = previous.map_or(em, |p| bs[i].n[1] - p);
        let below = bs[following].n[1] - bs[i].n[3];
        if above < 0.3 * em || above > 3. * em || below > 2.5 * em {
            continue;
        }
        let id = integer(&ls[i], "source_index", 0)?;
        promote(&mut ls[i], id);
    }
    Ok(())
}
