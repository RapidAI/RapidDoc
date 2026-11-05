use super::*;
fn number(s: &str) -> bool {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^[0-9]+(?:\.[0-9]+)+$").unwrap())
        .is_match(&mapped(trim(s)))
}
pub(super) fn bilingual(ls: &mut [Value], bs: &[Box4], w: &mut Work) -> Result<(), Error> {
    let ids = ordered((0..ls.len()).filter(|&i| ls[i]["angle"] == 0).collect(), bs);
    let next = constant_regex!(&format!(
        r"^[0-9]+(?:\.[0-9]+)*{SPACE}+[^{}]",
        &SPACE[1..SPACE.len() - 1]
    ));
    static C: OnceLock<Regex> = OnceLock::new();
    let chinese = C.get_or_init(|| Regex::new(r"[\u{3400}-\u{9fff}]{3,}").unwrap());
    for &i in &ids {
        w.charge(ids.len() * 4 + ls.len() + 1)?;
        if !ls[i]["semantic_type"].is_null() || !number(text(&ls[i])?) {
            continue;
        }
        let h = em(&ls[i])?.max(1.);
        let followers = ids
            .iter()
            .copied()
            .filter(|&j| {
                j != i
                    && ls[j]["semantic_type"].is_null()
                    && (0. ..=2.5 * h).contains(&(bs[j].n[1] - bs[i].n[1]))
                    && (0.3 * h..=20. * h).contains(&(bs[j].n[0] - bs[i].n[0]))
            })
            .collect::<Vec<_>>();
        let Some(&first) = followers
            .iter()
            .min_by(|&&a, &&b| bs[a].n[1].total_cmp(&bs[b].n[1]))
        else {
            continue;
        };
        let mut title = followers
            .into_iter()
            .filter(|&j| (cy(&bs[j]) - cy(&bs[first])).abs() <= 0.35 * h)
            .collect::<Vec<_>>();
        title.sort_by(|&a, &b| bs[a].n[0].total_cmp(&bs[b].n[0]));
        let t = title
            .iter()
            .map(|&j| text(&ls[j]))
            .collect::<Result<Vec<_>, _>>()?
            .join(" ");
        if !t.chars().any(cjk)
            || !t
                .as_bytes()
                .windows(3)
                .any(|b| b.iter().all(u8::is_ascii_alphabetic))
            || t.contains(['。', '；', '：', ';', ':'])
            || t.chars().count() > 140
        {
            continue;
        }
        let bottom = title
            .iter()
            .map(|&j| bs[j].n[3])
            .fold(f64::NEG_INFINITY, f64::max);
        let mut definition = false;
        for &j in &ids {
            if j != i
                && !title.iter().any(|&k| eq(&ls[k], &ls[j]))
                && (0. ..=2.5 * h).contains(&(bs[j].n[1] - bottom))
                && bs[j].n[0] >= bs[i].n[0] - 0.5 * h
                && chinese.is_match(text(&ls[j])?)
            {
                definition = true;
                break;
            }
        }
        if !definition {
            continue;
        }
        let mut members = vec![i];
        members.extend(title);
        band(&members, ls)?;
        let mut preceding = None;
        for &j in &ids {
            if bs[j].n[1] < bs[i].n[1]
                && ls[j]["semantic_type"].is_null()
                && bs[j].n[0] >= bs[i].n[0] - h
                && preceding.is_none_or(|k: usize| bs[j].n[1] > bs[k].n[1])
            {
                preceding = Some(j)
            }
        }
        if let Some(j) = preceding {
            ls[j]["paragraph_terminal"] = json!(true)
        }
        let mut end = f64::INFINITY;
        for &j in &ids {
            if bs[j].n[1] > bs[i].n[1]
                && (number(text(&ls[j])?) || next.is_match(&mapped(trim(text(&ls[j])?))))
            {
                end = end.min(bs[j].n[1])
            }
        }
        for &j in &ids {
            if bs[j].n[1] >= bottom && bs[j].n[1] < end && ls[j]["semantic_type"].is_null() {
                ls[j]["title_suppressed"] = json!(true)
            }
        }
    }
    Ok(())
}
pub(super) fn appendix(
    ls: &mut [Value],
    bs: &[Box4],
    sz: [f64; 2],
    w: &mut Work,
) -> Result<(), Error> {
    let marker = constant_regex!(&format!(r"^附{SPACE}*录{SPACE}*([A-Z])$"));
    let section = constant_regex!(&format!(
        r"^[A-Z]\.[0-9]+{SPACE}+[^{}]",
        &SPACE[1..SPACE.len() - 1]
    ));
    for i in 0..ls.len() {
        w.charge(ls.len() * 2 + 1)?;
        if !marker.is_match(trim(text(&ls[i])?))
            || !ls[i]["semantic_type"].is_null()
            || integer(&ls[i], "angle", 0)? != 0
        {
            continue;
        }
        let h = em(&ls[i])?.max(1.);
        let center = cx(&bs[i]);
        let members = (0..ls.len())
            .filter(|&j| {
                ls[j]["angle"] == 0
                    && ls[j]["semantic_type"].is_null()
                    && bs[j].n[1] >= bs[i].n[1]
                    && bs[j].n[1] <= bs[i].n[3] + 3. * h
                    && (cx(&bs[j]) - center).abs() <= h
                    && bs[j].n[2] - bs[j].n[0] < 0.5 * sz[0]
            })
            .collect::<Vec<_>>();
        let mut sections = false;
        for j in 0..ls.len() {
            if section.is_match(&mapped(text(&ls[j])?)) {
                let bottom = members
                    .iter()
                    .map(|&k| bs[k].n[3])
                    .reduce(f64::max)
                    .ok_or_else(invalid)?;
                if (0. ..=4. * h).contains(&(bs[j].n[1] - bottom)) {
                    sections = true
                }
            }
        }
        if members.len() >= 2 && sections {
            band(&members, ls)?
        }
    }
    Ok(())
}
