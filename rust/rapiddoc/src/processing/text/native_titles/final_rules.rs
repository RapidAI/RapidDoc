//! Existing Python final-page title/front-matter refinements in original semantics.
use super::super::{native_layout, native_page_preparation as prep, native_precontainers::bold};
use super::*;
use regex::Regex;
use unicode_normalization::UnicodeNormalization;
const SPACE: &str = r"[\u{9}-\u{d}\u{1c}-\u{20}\u{85}\u{a0}\u{1680}\u{2000}-\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}]";
fn text(l: &Value) -> Result<&str, Error> {
    l["text"].as_str().ok_or_else(invalid)
}
fn trim(t: &str) -> &str {
    t.trim_matches(python_space)
}
fn regex_fold(t: &str) -> String {
    t.chars()
        .map(|c| match c {
            'İ' | 'ı' => 'i',
            'ſ' => 's',
            'K' => 'k',
            _ => c,
        })
        .collect()
}
fn prefix_word(re: &Regex, t: &str) -> bool {
    let t = regex_fold(t);
    re.find(&t).is_some_and(|m| {
        t[m.end()..]
            .chars()
            .next()
            .is_none_or(|c| c != '_' && !rapiddoc_native::text_boundary::python_alnum(c))
    })
}
fn label_left(l: &Value) -> Result<f64, Error> {
    if !l["native_title_label_left"].is_null() {
        return num(&l["native_title_label_left"]);
    }
    let re = constant_regex!(&format!(r"^(?:[A-Za-z]|\d{{1,2}})[.)]{SPACE}+"));
    if re.is_match(text(l)?) && !optional(l, "chars", 20000)?.is_empty() {
        let mut ended = false;
        for c in optional(l, "chars", 20000)? {
            let s = c
                .get("char")
                .map_or(Ok(""), |v| v.as_str().ok_or_else(invalid))?;
            if ended && !trim(s).is_empty() {
                return num(&c["bbox"][0]);
            }
            if s == "." || s == ")" {
                ended = true
            }
        }
    }
    num(&l["bbox"][0])
}
fn words(t: &str) -> usize {
    t.split(python_space).filter(|s| !s.is_empty()).count()
}
fn em(l: &Value) -> Result<f64, Error> {
    num(&l["effective_height"])
}
fn wt(l: &Value, default: f64) -> Result<f64, Error> {
    Ok(weight(l)?.filter(|&n| n != 0.).unwrap_or(default))
}
fn cjk(t: &str) -> usize {
    t.chars()
        .filter(|c| ('\u{3400}'..='\u{9fff}').contains(c))
        .count()
}
fn mapped(t: &str) -> String {
    t.chars()
        .map(|c| rapiddoc_native::python_decimal(c).map_or(c, |d| (b'0' + d) as char))
        .collect()
}
fn available(l: &Value) -> bool {
    l["semantic_type"].is_null() || l["semantic_type"] == "paragraph_title"
}
fn group(ls: &[Value]) -> Result<i64, Error> {
    ls.iter()
        .filter(|l| !l["paragraph_group"].is_null())
        .map(|l| integer(l, "paragraph_group", 0))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(-1)
        .checked_add(1)
        .ok_or_else(limit)
}
fn band(ls: &mut [Value], members: &[usize]) -> Result<(), Error> {
    let id = members
        .iter()
        .map(|&i| integer(&ls[i], "source_index", 0))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .min()
        .ok_or_else(invalid)?;
    let group = group(ls)?;
    for &i in members {
        ls[i]["semantic_type"] = json!("paragraph_title");
        ls[i]["structural_title"] = json!(true);
        ls[i]["explicit_section_title"] = json!(true);
        ls[i]["title_band_id"] = json!(id);
        ls[i]["paragraph_group"] = json!(group);
        ls[i]["title_suppressed"] = json!(false)
    }
    Ok(())
}
fn promote(ls: &mut [Value], members: &[usize], id: i64) {
    for &i in members {
        ls[i]["semantic_type"] = json!("paragraph_title");
        ls[i]["structural_title"] = json!(true);
        ls[i]["explicit_section_title"] = json!(true);
        ls[i]["title_band_id"] = json!(id);
        ls[i]["title_suppressed"] = json!(false)
    }
}
fn boxes(ls: &[Value]) -> Result<Vec<Box4>, Error> {
    ls.iter().map(|l| bbox(&l["bbox"])).collect()
}
fn ordered(ls: &[Value], bs: &[Box4]) -> Vec<usize> {
    let mut ids = (0..ls.len()).collect::<Vec<_>>();
    ids.sort_by(|&i, &j| {
        bs[i].n[1]
            .total_cmp(&bs[j].n[1])
            .then(bs[i].n[0].total_cmp(&bs[j].n[0]))
    });
    ids
}
pub(super) fn roman(
    ls: &mut [Value],
    bs: &[Box4],
    cs: &[Box4],
    p: [f64; 2],
    w: &mut Work,
) -> Result<(), Error> {
    let ids = ls
        .iter()
        .enumerate()
        .filter(|(_, l)| l["angle"] == 0 && (available(l) || l["semantic_type"] == "doc_title"))
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    let canonical = |i: usize| -> Result<f64, Error> {
        let e = ls[i].get("em_height").map_or(Ok(0.), num)?;
        let h = if e > 0. { e } else { em(&ls[i])? };
        Ok((if h != 0. { h } else { bs[i].n[3] - bs[i].n[1] }).max(0.1))
    };
    let scales = ids
        .iter()
        .copied()
        .filter(|&i| words(text(&ls[i]).unwrap()) >= 4)
        .map(canonical)
        .collect::<Result<Vec<_>, _>>()?;
    if scales.len() < 3 {
        return Ok(());
    }
    let bh = median(scales);
    let marker = crate::cached_regex!(r"\A[IVXLCDM]{1,8}\.\z");
    let roman = crate::cached_regex!(r"\A[IVXLCDM]+\.?\z");
    let letters = crate::cached_regex!(r"[A-Za-z]{2,}|[\u{3400}-\u{9fff}]{2,}");
    for &i in &ids {
        let scale = |l: &Value, b: &Box4| -> Result<f64, Error> {
            let e = l.get("em_height").map_or(Ok(0.), num)?;
            let h = if e > 0. { e } else { em(l)? };
            Ok((if h != 0. { h } else { b.n[3] - b.n[1] }).max(0.1))
        };
        let h = scale(&ls[i], &bs[i])?;
        if !marker.is_match(trim(text(&ls[i])?))
            || h < 1.8 * bh
            || wt(&ls[i], 0.)? < 600.
            || coverage(&ls[i])? < 0.75
            || bs[i].n[2] - bs[i].n[0] > 5. * h
            || inside(&bs[i], cs)
        {
            continue;
        }
        let mut candidates = Vec::new();
        for &j in &ids {
            w.charge(1)?;
            let c = &bs[j];
            let b = &bs[i];
            let sh = scale(&ls[j], c)?;
            if i != j
                && ls[j]["caption_start"] != true
                && (c.n[0] - b.n[0]).abs() <= 0.25 * h
                && (-0.25 * h..=0.5 * h).contains(&(c.n[1] - b.n[3]))
                && cy(c) > cy(b) + 0.5 * h
                && (0.6 * h..=1.4 * h).contains(&sh)
                && sh >= 1.8 * bh
                && wt(&ls[j], 0.)? >= 600.
                && coverage(&ls[j])? >= 0.75
                && letters.is_match(text(&ls[j])?)
                && !roman.is_match(trim(text(&ls[j])?))
                && c.n[2] - c.n[0] <= 0.6 * p[0]
                && !inside(c, cs)
            {
                candidates.push(j)
            }
        }
        let Some(j) = candidates
            .into_iter()
            .min_by(|&a, &b| cy(&bs[a]).total_cmp(&cy(&bs[b])))
        else {
            continue;
        };
        let mut members = vec![i, j];
        for _ in 0..2 {
            let before = *members.last().unwrap();
            let h = scale(&ls[j], &bs[j])?;
            let mut continuations = Vec::new();
            for &k in &ids {
                w.charge(members.len() + 1)?;
                if members.iter().any(|&m| eq(&ls[k], &ls[m])) || ls[k]["caption_start"] == true {
                    continue;
                }
                let c = &bs[k];
                if (c.n[0] - bs[j].n[0]).abs() <= 0.25 * h
                    && (-0.2 * h..=0.4 * h).contains(&(c.n[1] - bs[before].n[3]))
                    && cy(c) > cy(&bs[before]) + 0.5 * h
                    && eq(&ls[k]["font_signature"], &ls[j]["font_signature"])
                    && (0.85..=1.15).contains(&(scale(&ls[k], c)? / h))
                    && !inside(c, cs)
                {
                    continuations.push(k)
                }
            }
            if let Some(k) = continuations
                .into_iter()
                .min_by(|&a, &b| bs[a].n[1].total_cmp(&bs[b].n[1]))
            {
                members.push(k)
            } else {
                break;
            }
        }
        let id = integer(&ls[i], "source_index", 0)?;
        promote(ls, &members, id)
    }
    Ok(())
}
fn repeated(ls: &mut [Value], bs: &[Box4], body: &Value, w: &mut Work) -> Result<(), Error> {
    let h = if body.is_null() {
        let hs = ls
            .iter()
            .enumerate()
            .filter(|(_, l)| words(text(l).unwrap()) >= 5)
            .map(|(i, l)| height(l, &bs[i]))
            .collect::<Result<Vec<_>, _>>()?;
        if hs.is_empty() {
            1.
        } else {
            median(hs)
        }
    } else {
        num(&body["body_height"])?
    };
    let re = constant_regex!(&format!(
        r"^{SPACE}*(?:[•●▪]{SPACE}+|(?P<number>\d{{1,2}})[.)]{SPACE}+)\S"
    ));
    let mut candidates = Vec::new();
    for (i, l) in ls.iter().enumerate() {
        let Some(m) = re.captures(text(l)?) else {
            continue;
        };
        if l["angle"] != 0
            || !available(l)
            || l["structural_title"] == true
            || l["explicit_section_title"] == true
            || l["caption_start"] == true
            || l["font_signature"].is_null()
            || coverage(l)? < 0.75
            || height(l, &bs[i])? > 1.15 * h
            || bold(l)?
        {
            continue;
        }
        let number = m
            .name("number")
            .map(|n| mapped(n.as_str()).parse::<i64>().unwrap());
        candidates.push((i, number))
    }
    for &(i, n) in &candidates {
        let h = height(&ls[i], &bs[i])?;
        let mut found = false;
        for &(j, m) in &candidates {
            w.charge(1)?;
            if i != j
                && (n.is_none() && m.is_none() || n.zip(m).is_some_and(|(a, b)| (a - b).abs() == 1))
                && eq(&ls[i]["font_signature"], &ls[j]["font_signature"])
                && (0.9 * h..=1.1 * h).contains(&height(&ls[j], &bs[j])?)
                && (bs[i].n[0] - bs[j].n[0]).abs() <= 0.5 * h
                && (0.75 * h..=8. * h).contains(&(cy(&bs[i]) - cy(&bs[j])).abs())
            {
                found = true;
                break;
            }
        }
        if found {
            ls[i]["semantic_type"] = Value::Null;
            ls[i]["title_suppressed"] = json!(true)
        }
    }
    Ok(())
}
fn suppress(ls: &mut [Value], bs: &[Box4], w: &mut Work) -> Result<(), Error> {
    for i in 0..ls.len() {
        let l = &ls[i];
        if !matches!(
            l["semantic_type"].as_str(),
            Some("paragraph_title" | "doc_title")
        ) || l["angle"] != 0
            || l["structural_title"] == true
            || l["explicit_section_title"] == true
            || l["caption_start"] == true
        {
            continue;
        }
        let h = em(l)?.max(1.);
        let t = text(l)?;
        if cjk(t) < 6
            || !t.chars().any(|c| "，,、；;".contains(c))
            || t.trim_end_matches(python_space)
                .ends_with(['。', '！', '？', ':', '：'])
        {
            continue;
        }
        let b = &bs[i];
        let mut followers = Vec::new();
        for (j, l) in ls.iter().enumerate() {
            w.charge(1)?;
            let c = &bs[j];
            if i != j
                && l["angle"] == 0
                && (l["semantic_type"].is_null() || l["semantic_type"] == "text")
                && (-0.25 * h..=0.85 * h).contains(&(c.n[1] - b.n[3]))
                && (-3. * h..=0.5 * h).contains(&(c.n[0] - b.n[0]))
                && cjk(text(l)?) > 0
            {
                followers.push(j)
            }
        }
        let Some(j) = followers
            .into_iter()
            .min_by(|&a, &b| bs[a].n[1].total_cmp(&bs[b].n[1]))
        else {
            continue;
        };
        if bs[j].n[2] - bs[j].n[0] >= 0.8 * (b.n[2] - b.n[0]) {
            let mut previous = false;
            for (k, l) in ls.iter().enumerate() {
                w.charge(1)?;
                let c = &bs[k];
                if k != i
                    && l["angle"] == 0
                    && l["semantic_type"].is_null()
                    && l["structural_title"] != true
                    && (0.0..=h).contains(&(b.n[1] - c.n[3]))
                    && (c.n[0] - b.n[0]).abs() <= 0.5 * h
                    && c.n[2] - c.n[0] >= 0.85 * (b.n[2] - b.n[0])
                    && text(l)?.chars().any(|c| "，,、；;".contains(c))
                    && !text(l)?
                        .trim_end_matches(python_space)
                        .ends_with(['。', '！', '？'])
                {
                    previous = true;
                    break;
                }
            }
            if !previous {
                continue;
            }
        }
        ls[i]["title_suppressed"] = json!(true);
        ls[i]["semantic_type"] = Value::Null;
        ls[i]["title_band_id"] = Value::Null;
        ls[i]["paragraph_group"] = Value::Null;
    }
    Ok(())
}
fn short_cjk(ls: &mut [Value], bs: &[Box4], w: &mut Work) -> Result<(), Error> {
    let rows = ordered(ls, bs);
    let date = constant_regex!(&format!(
        r"》{SPACE}*[—–-]+{SPACE}*\d{{4}}[-./]\d{{1,2}}[-./]\d{{1,2}}{SPACE}*$"
    ));
    for &i in &rows {
        let l = &ls[i];
        let t = trim(text(l)?);
        if !l["semantic_type"].is_null()
            || !l["paragraph_group"].is_null()
            || l["caption_start"] == true
            || l["formula_candidate_only"] == true
            || l["angle"] != 0
            || l["font_signature"].is_null()
            || !(4..=16).contains(&t.chars().count())
            || cjk(t) != t.chars().count()
        {
            continue;
        }
        let b = &bs[i];
        let mut following = Vec::new();
        for &j in &rows {
            w.charge(1)?;
            let l = &ls[j];
            if i != j
                && l["semantic_type"].is_null()
                && l["paragraph_group"].is_null()
                && l["caption_start"] != true
                && l["angle"] == 0
                && (bs[j].n[0] - b.n[0]).abs() <= 0.3 * em(&ls[i])?
                && bs[j].n[1] >= b.n[3]
            {
                following.push(j)
            }
        }
        if following.len() < 2 {
            continue;
        }
        let (a, c) = (following[0], following[1]);
        let h = em(&ls[a])?.max(em(&ls[c])?);
        let width = bs[a].n[2] - bs[a].n[0];
        if h <= 0.
            || width < 12. * h
            || b.n[2] - b.n[0] >= 0.5 * width
            || !(1.0..=1.65).contains(&(em(&ls[i])? / h))
            || em(&ls[a])?.min(em(&ls[c])?) < 0.9 * h
            || !fonts_ok(&ls[i], &ls[a])?
            || !fonts_ok(&ls[a], &ls[c])?
            || !(0.0..=1.1 * h).contains(&(bs[a].n[1] - b.n[3]))
            || !(-0.1 * h..=0.65 * h).contains(&(bs[c].n[1] - bs[a].n[3]))
            || cjk(text(&ls[a])?) < 12
            || (cjk(text(&ls[a])?) + cjk(text(&ls[c])?) < 18 && !date.is_match(text(&ls[c])?))
        {
            continue;
        }
        if rows.iter().any(|&j| {
            i != j
                && ls[j]["angle"] == 0
                && !matches!(
                    ls[j]["semantic_type"].as_str(),
                    Some("paragraph_title" | "doc_title")
                )
                && bs[j].n[3] <= b.n[1]
                && axis(&bs[j], b, 0) >= 0.5
                && b.n[1] - bs[j].n[3] >= 0.
                && b.n[1] - bs[j].n[3] < 1.2 * h
        }) {
            continue;
        }
        ls[i]["semantic_type"] = json!("paragraph_title")
    }
    Ok(())
}
fn runin(ls: &mut [Value], p: [f64; 2], w: &mut Work) -> Result<(), Error> {
    let angles = ls
        .iter()
        .map(|l| integer(l, "angle", 0))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut layouts = BTreeMap::new();
    for a in angles {
        layouts.insert(a, native_layout::build(ls, p, a, &[], w)?);
    }
    let prefix = crate::cached_regex!(r"\A[a-zA-Z][.)]\z");
    for i in 0..ls.len() {
        let was = ls[i]["semantic_type"] == "paragraph_title";
        if !was
            && !(ls[i]["semantic_type"].is_null()
                && wt(&ls[i], 0.)? >= 550.
                && words(text(&ls[i])?) <= 8)
        {
            continue;
        }
        let a = integer(&ls[i], "angle", 0)?;
        let b = local(
            &bbox(if ls[i]["ink_bbox"].is_null() {
                &ls[i]["bbox"]
            } else {
                &ls[i]["ink_bbox"]
            })?,
            p,
            a,
        );
        let h = height(&ls[i], &b)?;
        for j in 0..ls.len() {
            w.charge(1)?;
            if i == j
                || integer(&ls[j], "angle", 0)? != a
                || !ls[j]["semantic_type"].is_null()
                || ls[j]["formula_candidate_only"] == true
                || ls[j]["compact_formula_cluster"] == true
            {
                continue;
            }
            if !was
                && (ls[j]["dominant_font_weight"].is_null()
                    || num(&ls[i]["dominant_font_weight"])?
                        < num(&ls[j]["dominant_font_weight"])? + 100.)
            {
                continue;
            }
            let c = local(
                &bbox(if ls[j]["ink_bbox"].is_null() {
                    &ls[j]["bbox"]
                } else {
                    &ls[j]["ink_bbox"]
                })?,
                p,
                a,
            );
            let ch = height(&ls[j], &c)?;
            if h > 1.3 * ch || trim(text(&ls[j])?).chars().count() < 3 {
                continue;
            }
            if (cy(&b) - cy(&c)).abs() <= 0.45 * h.min(ch)
                && b.n[2].min(c.n[2]) - b.n[0].max(c.n[0]) <= 0.5 * h
                && !layouts.get(&a).ok_or_else(invalid)?.separated(&b, &c, w)?
                && prep::formula_same_baseline(&b, h, &c, ch, 5. * h.max(ch))
            {
                ls[i]["semantic_type"] = Value::Null;
                if b.n[0] < c.n[0] {
                    ls[i]["leading_emphasis_width"] = json!(b.n[2] - b.n[0]);
                    ls[i]["leading_typography_width"] = json!(b.n[2] - b.n[0])
                }
                ls[i]["title_suppressed"] = json!(true);
                ls[i]["structural_title"] = json!(false);
                ls[i]["explicit_section_title"] = json!(false);
                let raw = bbox(&ls[i]["bbox"])?;
                for k in 0..ls.len() {
                    w.charge(1)?;
                    let kb = bbox(&ls[k]["bbox"])?;
                    if ls[k]["semantic_type"] == "paragraph_title"
                        && !prefix.is_match(trim(text(&ls[i])?))
                        && integer(&ls[k], "angle", 0)? == a
                        && ls[k]["explicit_section_title"] != true
                        && (kb.n[0] - raw.n[0]).abs() < h
                        && (0.0..=h).contains(&(raw.n[1] - kb.n[3]))
                        && eq(&ls[k]["font_signature"], &ls[i]["font_signature"])
                    {
                        ls[k]["semantic_type"] = Value::Null;
                        ls[k]["title_suppressed"] = json!(true)
                    }
                }
                break;
            }
        }
    }
    Ok(())
}
fn panel_rows(seed: usize, ls: &[Value], bs: &[Box4], w: &mut Work) -> Result<Vec<usize>, Error> {
    let mut result = vec![seed];
    let h = em(&ls[seed])?;
    for i in ordered(ls, bs) {
        let before = *result.last().unwrap();
        w.charge(1)?;
        if i == seed || ls[i]["angle"] != 0 || bs[i].n[1] <= bs[before].n[1] {
            continue;
        }
        let mut label = result
            .iter()
            .map(|&i| text(&ls[i]))
            .collect::<Result<Vec<_>, _>>()?;
        label.push(text(&ls[i])?);
        if available(&ls[i])
            && eq(&ls[i]["font_signature"], &ls[seed]["font_signature"])
            && (bs[i].n[0] - bs[seed].n[0]).abs() <= 0.3 * h
            && (0.9 * h..=1.1 * h).contains(&em(&ls[i])?)
            && (-0.2 * h..=0.8 * h).contains(&(bs[i].n[1] - bs[before].n[3]))
            && words(&label.join(" ")) <= 22
            && !trim(text(&ls[i])?).ends_with(['.', '!', '?', '。', '！', '？', ';', '；'])
        {
            result.push(i);
            if result.len() == 3 {
                break;
            }
        }
    }
    Ok(result)
}
fn panels(
    ls: &mut [Value],
    bs: &[Box4],
    images: &[Box4],
    width: f64,
    w: &mut Work,
) -> Result<(), Error> {
    let seeds = ordered(ls, bs)
        .into_iter()
        .filter(|&i| {
            ls[i]["angle"] == 0
                && available(&ls[i])
                && ls[i]["title_suppressed"] != true
                && !ls[i]["font_signature"].is_null()
                && (2..=12).contains(&words(text(&ls[i]).unwrap()))
                && !trim(text(&ls[i]).unwrap())
                    .ends_with(['.', '!', '?', '。', '！', '？', ';', '；'])
        })
        .collect::<Vec<_>>();
    let mut used = BTreeSet::<usize>::new();
    let mut runs = Vec::new();
    for i in seeds {
        if used.contains(&i) {
            continue;
        }
        let rows = panel_rows(i, ls, bs, w)?;
        used.extend(&rows);
        let b = combined(rows.iter().map(|&i| &bs[i]))?;
        runs.push((i, rows, b))
    }
    for (k, (i, _, b)) in runs.iter().enumerate() {
        let h = em(&ls[*i])?;
        let mut larger = false;
        for (j, l) in ls.iter().enumerate() {
            w.charge(1)?;
            if j != *i
                && l["angle"] == 0
                && (bs[j].n[0] - b.n[0]).abs() <= 0.3 * h
                && (1.1 * h..=1.5 * h).contains(&em(l)?)
                && (-0.1 * h..=0.8 * h).contains(&(b.n[1] - bs[j].n[3]))
            {
                larger = true;
                break;
            }
        }
        if larger {
            continue;
        }
        let mut peers = runs
            .iter()
            .enumerate()
            .filter(|(_, r)| {
                eq(&ls[r.0]["font_signature"], &ls[*i]["font_signature"])
                    && (0.9 * h..=1.1 * h).contains(&em(&ls[r.0]).unwrap())
                    && (r.2.n[1] - b.n[1]).abs() <= 0.5 * h
            })
            .map(|(k, _)| k)
            .collect::<Vec<_>>();
        peers.sort_by(|&a, &b| runs[a].2.n[0].total_cmp(&runs[b].2.n[0]));
        if peers.len() >= 2
            && peers
                .windows(2)
                .all(|p| runs[p[1]].2.n[0] - runs[p[0]].2.n[2] >= 2. * h)
        {
            let mut verified = true;
            for (pos, &r) in peers.iter().enumerate() {
                let (_, members, b) = &runs[r];
                let right = peers.get(pos + 1).map_or(width, |&r| runs[r].2.n[0] - h);
                let mut follows = false;
                for (j, l) in ls.iter().enumerate() {
                    w.charge(1)?;
                    if !members.contains(&j)
                        && l["semantic_type"].is_null()
                        && (bs[j].n[0] - b.n[0]).abs() <= 0.4 * h
                        && (b.n[3]..=b.n[3] + 3. * h).contains(&bs[j].n[1])
                        && bs[j].n[2] < right
                        && em(l)? <= 0.9 * h
                        && words(text(l)?) >= 4
                    {
                        follows = true
                    }
                }
                let graphic = images.iter().any(|c| {
                    (c.n[0] - b.n[0]).abs() <= 1.5 * h
                        && (b.n[3]..=b.n[3] + 4. * h).contains(&c.n[1])
                        && c.n[2] <= right
                        && c.n[3] - c.n[1] >= 3. * h
                });
                if !follows && !graphic {
                    verified = false
                }
            }
            if verified {
                for &r in &peers {
                    promote(ls, &runs[r].1, integer(&ls[runs[r].0], "source_index", 0)?);
                }
            }
        }
        if wt(&ls[*i], 0.)? < 600. || b.n[2] - b.n[0] > 0.25 * width {
            continue;
        }
        let side = runs
            .iter()
            .enumerate()
            .filter(|(_, r)| {
                eq(&ls[r.0]["font_signature"], &ls[*i]["font_signature"])
                    && wt(&ls[r.0], 0.).unwrap() >= 600.
                    && (r.2.n[0] - b.n[0]).abs() <= 0.3 * h
                    && (0.9 * h..=1.1 * h).contains(&em(&ls[r.0]).unwrap())
            })
            .map(|(k, _)| k)
            .collect::<Vec<_>>();
        if side.len() < 2 {
            continue;
        }
        if side.iter().all(|&r| {
            let b = &runs[r].2;
            images.iter().any(|c| {
                c.n[0] <= b.n[0] + h
                    && c.n[2] >= b.n[2] - h
                    && ((0.0..=3. * h).contains(&(b.n[1] - c.n[3]))
                        || (0.0..=3. * h).contains(&(c.n[1] - b.n[3])))
                    && c.n[3] - c.n[1] >= 3. * h
            })
        }) {
            for r in side {
                promote(ls, &runs[r].1, integer(&ls[runs[r].0], "source_index", 0)?);
            }
        }
        let _ = k;
    }
    Ok(())
}
fn displaced(ls: &mut [Value], bs: &[Box4], images: &[Box4], w: &mut Work) -> Result<(), Error> {
    let mut rows = (0..ls.len()).collect::<Vec<_>>();
    rows.sort_by_key(|&i| integer(&ls[i], "source_index", 0).unwrap());
    let tail = constant_regex!(&format!(
        r"\A[a-z][A-Za-z-]*(?:{SPACE}+[a-z][A-Za-z-]*){{0,2}}[.!?]\z"
    ));
    for k in 0..rows.len().saturating_sub(2) {
        let (a, b, c) = (rows[k], rows[k + 1], rows[k + 2]);
        if [a, b, c].iter().any(|&i| {
            ls[i]["angle"] != 0
                || !ls[i]["semantic_type"].is_null()
                || !ls[i]["paragraph_group"].is_null()
                || ls[i]["caption_start"] == true
                || ls[i]["formula_candidate_only"] == true
        }) {
            continue;
        }
        let h = height(&ls[a], &bs[a])?;
        if ls[a]["font_signature"].is_null()
            || !eq(&ls[a]["font_signature"], &ls[b]["font_signature"])
            || !eq(&ls[a]["font_signature"], &ls[c]["font_signature"])
            || [b, c]
                .iter()
                .any(|&i| !(0.9..=1.1).contains(&(height(&ls[i], &bs[i]).unwrap() / h)))
            || words(text(&ls[a])?) < 8
            || words(text(&ls[b])?) < 8
            || ls[b]["paragraph_terminal"] == true
            || text(&ls[b])?
                .trim_end_matches(python_space)
                .ends_with(['.', '!', '?', ';', ':', '。', '！', '？', '；', '：'])
            || !tail.is_match(trim(text(&ls[c])?))
            || !(-0.1 * h..=0.6 * h).contains(&(bs[b].n[1] - bs[a].n[3]))
            || (bs[a].n[0] - bs[b].n[0]).abs() > 0.25 * h
            || (bs[a].n[0] - bs[c].n[0]).abs() > 0.25 * h
            || bs[c].n[2] - bs[c].n[0] > 5. * h
        {
            continue;
        }
        w.charge(images.len() + 1)?;
        let count = images
            .iter()
            .filter(|im| {
                im.n[3] - im.n[1] >= 8. * h
                    && (bs[b].n[3] - 0.2 * h..=bs[b].n[3] + 2. * h).contains(&im.n[1])
                    && (bs[c].n[1] - h..=bs[c].n[3] + h).contains(&im.n[3])
                    && axis(im, &bs[a], 0) >= 0.5
                    && axis(im, &bs[c], 0) <= 0.1
            })
            .count();
        if count != 1 {
            continue;
        }
        let g = group(ls)?;
        for i in [a, b, c] {
            ls[i]["paragraph_group"] = json!(g)
        }
        if k > 0 {
            let j = rows[k - 1];
            let l = &ls[j];
            if l["angle"] == 0
                && l["semantic_type"].is_null()
                && eq(&l["font_signature"], &ls[a]["font_signature"])
                && (2..=6).contains(&words(text(l)?))
                && !text(l)?
                    .trim_end_matches(python_space)
                    .ends_with(['.', '!', '?', '。', '！', '？', ':', '：', ';', '；'])
                && (bs[j].n[0] - bs[a].n[0]).abs() <= 0.25 * h
                && (h..=3. * h).contains(&(bs[a].n[1] - bs[j].n[3]))
            {
                band(ls, &[j])?
            }
        }
    }
    Ok(())
}
fn wrapped(ls: &mut Vec<Value>, p: [f64; 2], w: &mut Work) -> Result<(), Error> {
    let original = ls.len();
    let mut next = ls
        .iter()
        .map(|l| integer(l, "source_index", 0))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .unwrap_or(-1)
        + 1;
    for i in 0..original {
        if ls[i]["angle"] != 0
            || !ls[i]["semantic_type"].is_null()
            || ls[i]["caption_start"] == true
        {
            continue;
        }
        if ls[i]["wrapped_title_parts"].is_null() && !optional(&ls[i], "chars", 20000)?.is_empty() {
            prep::freeze_wrapped_title(&mut ls[i], p, w)?
        }
        if ls[i]["wrapped_title_parts"].is_null() {
            continue;
        }
        let parts = array(&ls[i]["wrapped_title_parts"], 2)?;
        if parts.len() != 2 {
            return Err(invalid());
        }
        let mut suffix = parts[0].clone();
        let body = parts[1].clone();
        let pb = bbox(&suffix["bbox"])?;
        let h = height(&ls[i], &bbox(&ls[i]["bbox"])?)?;
        let mut previous = Vec::new();
        for j in 0..original {
            w.charge(1)?;
            let l = &ls[j];
            let b = bbox(&l["bbox"])?;
            let left = label_left(l)?;
            if l["semantic_type"] == "paragraph_title"
                && l["angle"] == 0
                && wt(l, 0.)? >= 600.
                && l["paragraph_terminal"] != true
                && !trim(text(l)?).ends_with(['.', '!', '?', '。', '！', '？'])
                && (left - pb.n[0]).abs() <= 0.5 * h
                && (-0.1 * h..=0.65 * h).contains(&(pb.n[1] - b.n[3]))
                && !l["font_signature"].is_null()
                && eq(&l["font_signature"], &suffix["font_signature"])
            {
                previous.push(j)
            }
        }
        if previous.len() != 1 {
            continue;
        }
        let head = previous[0];
        let hb = bbox(&ls[head]["bbox"])?;
        suffix["source_index"] = json!(next);
        next += 1;
        let mut members = vec![head];
        let suffix_index = ls.len();
        ls.push(suffix);
        members.push(suffix_index);
        for j in 0..original {
            w.charge(1)?;
            let b = bbox(&ls[j]["bbox"])?;
            if j != head
                && ls[j]["semantic_type"] == "paragraph_title"
                && eq(&ls[j]["font_signature"], &ls[head]["font_signature"])
                && axis(&b, &hb, 1) >= 0.7
                && (0.0..=3. * h).contains(&(hb.n[0] - b.n[2]))
            {
                members.push(j)
            }
        }
        band(ls, &members)?;
        for k in [
            "text",
            "bbox",
            "source_bbox",
            "font_signature",
            "dominant_font_weight",
            "median_glyph_width",
        ] {
            ls[i][k] = body[k].clone()
        }
        ls[i]["ink_bbox"] = Value::Null;
        ls[i]["chars"] = json!([]);
        ls[i]["leading_emphasis_width"] = Value::Null;
        ls[i]["wrapped_title_parts"] = Value::Null;
        let g = group(ls)?;
        let mut paragraph = vec![i];
        let mut after = (0..original)
            .filter(|&j| {
                integer(&ls[j], "source_index", 0).unwrap()
                    > integer(&ls[i], "source_index", 0).unwrap()
            })
            .collect::<Vec<_>>();
        after.sort_by_key(|&j| integer(&ls[j], "source_index", 0).unwrap());
        for j in after {
            let before = *paragraph.last().unwrap();
            let b = bbox(&ls[j]["bbox"])?;
            let bb = bbox(&ls[before]["bbox"])?;
            let l = &ls[j];
            if !l["semantic_type"].is_null()
                || l["caption_start"] == true
                || l["angle"] != 0
                || !eq(&l["font_signature"], &ls[i]["font_signature"])
                || !(0.9..=1.1).contains(&(height(l, &b)? / h))
                || !(-0.1 * h..=0.65 * h).contains(&(b.n[1] - bb.n[3]))
                || (b.n[0] - pb.n[0]).abs() > 0.5 * h
            {
                break;
            }
            paragraph.push(j)
        }
        for j in paragraph {
            ls[j]["paragraph_group"] = json!(g)
        }
    }
    Ok(())
}
fn front(ls: &mut [Value], bs: &[Box4], p: [f64; 2], w: &mut Work) -> Result<(), Error> {
    let role = |t: &str| {
        let s = t
            .nfkc()
            .filter(|&c| !python_space(c) && c != ':' && c != '：')
            .flat_map(char::to_lowercase)
            .collect::<String>();
        ["abstract", "summary", "摘要", "内容摘要"].contains(&s.as_str())
    };
    let dates = constant_regex!(&format!(
        r"(?i)^{SPACE}*[（(]?(?:received|accepted|published)"
    ));
    let mut abstracts = ls
        .iter()
        .enumerate()
        .filter(|(_, l)| role(text(l).unwrap()))
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    if abstracts.is_empty() {
        abstracts = ls
            .iter()
            .enumerate()
            .filter(|(_, l)| prefix_word(&dates, text(l).unwrap()))
            .map(|(i, _)| i)
            .collect()
    }
    if abstracts.is_empty() {
        return Ok(());
    }
    let end = abstracts
        .iter()
        .map(|&i| bs[i].n[1])
        .reduce(f64::min)
        .unwrap();
    let candidates = (0..ls.len())
        .filter(|&i| bs[i].n[1] >= 0.12 * p[1] && bs[i].n[1] < end && ls[i]["angle"] == 0)
        .collect::<Vec<_>>();
    let marker = constant_regex!(&format!(r"^{SPACE}*(\d{{1,2}}){SPACE}+\D"));
    let mut markers = Vec::new();
    for &i in &candidates {
        if let Some(m) = marker.captures(text(&ls[i])?) {
            markers.push((i, mapped(&m[1]).parse::<i64>().map_err(|_| invalid())?))
        }
    }
    if markers.is_empty() {
        return Ok(());
    }
    let institution = crate::cached_regex!(
        r"(?i)university|institute|department|laboratory|centre|center|大学|学院|研究所|实验室|医院"
    );
    if !candidates.iter().any(|&i| {
        institution.is_match(&regex_fold(text(&ls[i]).unwrap()))
            || super::super::table_block::output::field(text(&ls[i]).unwrap()) == Some("contact")
    }) {
        return Ok(());
    }
    let y = markers
        .iter()
        .map(|(i, _)| bs[*i].n[1])
        .reduce(f64::min)
        .unwrap();
    let left = median(markers.iter().map(|(i, _)| bs[*i].n[0]).collect());
    let mut by_source = BTreeMap::new();
    for (i, n) in markers {
        by_source.insert(integer(&ls[i], "source_index", 0)?, n);
    }
    let only = crate::cached_regex!(r"\A\d{1,2}\z");
    for &i in &candidates {
        if only.is_match(trim(text(&ls[i])?)) && (bs[i].n[0] - left).abs() < em(&ls[i])? {
            by_source.insert(
                integer(&ls[i], "source_index", 0)?,
                mapped(trim(text(&ls[i])?))
                    .parse::<i64>()
                    .map_err(|_| invalid())?,
            );
        }
    }
    let mut ordered = candidates;
    ordered.sort_by(|&i, &j| {
        let b = if ls[i]["ink_bbox"].is_null() {
            bs[i].clone()
        } else {
            bbox(&ls[i]["ink_bbox"]).unwrap()
        };
        let c = if ls[j]["ink_bbox"].is_null() {
            bs[j].clone()
        } else {
            bbox(&ls[j]["ink_bbox"]).unwrap()
        };
        cy(&b)
            .total_cmp(&cy(&c))
            .then(bs[i].n[0].total_cmp(&bs[j].n[0]))
    });
    let stop = crate::cached_regex!(r"(?i)^(?:received|accepted|date|draft)");
    let mut active = None;
    let mut previous: Option<usize> = None;
    for i in ordered {
        w.charge(1)?;
        if bs[i].n[1] < y {
            continue;
        }
        if let Some(n) = by_source.get(&integer(&ls[i], "source_index", 0)?) {
            active = Some(-n)
        } else if let Some(j) = previous {
            if bs[i].n[1] - bs[j].n[3] > 1.5 * em(&ls[i])?.max(em(&ls[j])?) {
                active = None
            }
        }
        if prefix_word(&stop, text(&ls[i])?) {
            active = None
        }
        if let Some(g) = active {
            ls[i]["paragraph_group"] = json!(g);
            ls[i]["semantic_type"] = Value::Null;
            ls[i]["title_suppressed"] = json!(true)
        }
        previous = Some(i)
    }
    Ok(())
}
pub(super) fn run(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let mut ls = array(&v["lines"], 4096)?.to_vec();
    let bs = boxes(&ls)?;
    let p = size(&v["page_size"])?;
    let images = optional(v, "image_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    match v["kind"].as_str().ok_or_else(invalid)? {
        "front" => front(&mut ls, &bs, p, w)?,
        "suppress" => suppress(&mut ls, &bs, w)?,
        "runin" => runin(&mut ls, p, w)?,
        "repeated" => repeated(&mut ls, &bs, &v["document_body_profile"], w)?,
        "panels" => panels(&mut ls, &bs, &images, p[0], w)?,
        "displaced" => displaced(&mut ls, &bs, &images, w)?,
        "cjk" => short_cjk(&mut ls, &bs, w)?,
        "wrapped" => wrapped(&mut ls, p, w)?,
        "numbered" => {
            let source = super::super::native_precontainers::final_headings(
                &json!({"lines":ls,"page_size":p}),
                optional(v, "container_bboxes", 5000)?,
                w,
            )?;
            ls = array(&source["lines"], 4096)?.to_vec()
        }
        _ => return Err(invalid()),
    }
    let out = json!({"lines":ls});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        Err(limit())
    } else {
        Ok(out)
    }
}
