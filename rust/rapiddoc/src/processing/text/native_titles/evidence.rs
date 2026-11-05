use super::*;
use rapiddoc_native::{python_decimal, text_boundary::python_alnum};
use unicode_normalization::UnicodeNormalization;
fn fold(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'İ' | 'ı' => 'i',
            'ſ' => 's',
            'K' => 'k',
            _ => c.to_ascii_lowercase(),
        })
        .collect()
}
fn words(s: &str) -> String {
    s.chars()
        .map(|c| if python_space(c) { ' ' } else { c })
        .collect()
}
fn abstract_line(t: &str) -> bool {
    let t = fold(t);
    let t = t.trim_start_matches(python_space);
    let rest = t
        .strip_prefix("abstract")
        .or_else(|| t.strip_prefix("摘要"));
    rest.is_some_and(|t| t.trim_start_matches(python_space).starts_with([':', '：']))
}
fn keywords(t: &str) -> bool {
    let t = fold(t);
    let t = t.trim_start_matches(python_space);
    if let Some(t) = t.strip_prefix("关键词") {
        return t.trim_start_matches(python_space).starts_with([':', '：']);
    }
    let Some(t) = t.strip_prefix("key") else {
        return false;
    };
    let t = t.trim_start_matches(python_space);
    let Some(t) = t.strip_prefix("word") else {
        return false;
    };
    t.strip_prefix('s')
        .unwrap_or(t)
        .trim_start_matches(python_space)
        .starts_with([':', '：'])
}
pub(super) fn front(s: &mut State, p: [f64; 2], w: &mut Work) -> Result<(), Error> {
    let english = crate::cached_regex!(r"[A-Za-z]{3,}");
    let author = crate::cached_regex!(r"[A-Z]{2,} +[A-Z][a-z]+");
    let strings = s
        .lines
        .iter()
        .map(|l| l["text"].as_str().map(str::to_owned).ok_or_else(invalid))
        .collect::<Result<Vec<_>, _>>()?;
    if strings.iter().any(|t| abstract_line(t)) && strings.iter().any(|t| keywords(t)) {
        let mut candidates = Vec::new();
        for (i, l) in s.lines.iter().enumerate() {
            let b = &s.boxes[i];
            if integer(l, "angle", 0)? == 0
                && s.unresolved(i)
                && l["title_suppressed"] != true
                && b.n[1] >= 0.075 * p[1]
                && b.n[1] <= 0.18 * p[1]
                && (cx(b) - p[0] / 2.).abs() <= 0.04 * p[0]
                && english.is_match(&strings[i])
            {
                candidates.push(i)
            }
        }
        candidates.sort_by(|&i, &j| s.boxes[i].n[1].total_cmp(&s.boxes[j].n[1]));
        if (2..=4).contains(&candidates.len())
            && candidates
                .iter()
                .map(|&i| s.boxes[i].n[2] - s.boxes[i].n[0])
                .reduce(f64::max)
                .unwrap()
                >= 0.65 * p[0]
        {
            let last = *candidates.last().unwrap();
            let bottom = s.boxes[last].n[3];
            let em = s.h(last)?;
            let mut author_tops = Vec::new();
            for (i, b) in s.boxes.iter().enumerate() {
                w.charge(strings[i].len() + 1)?;
                if b.n[1] <= bottom || b.n[1] >= bottom + 6. * em {
                    continue;
                }
                let value = words(&strings[i]);
                if author.find_iter(&value).any(|m| {
                    value[..m.start()]
                        .chars()
                        .last()
                        .is_none_or(|c| !(python_alnum(c) || c == '_'))
                }) {
                    author_tops.push(b.n[1])
                }
            }
            if !author_tops.is_empty()
                && author_tops.into_iter().reduce(f64::min).unwrap() - bottom >= em
            {
                let band = candidates
                    .iter()
                    .map(|&i| integer(&s.lines[i], "source_index", 0))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .min()
                    .unwrap();
                for i in candidates {
                    s.mark(i, "doc_title");
                    s.lines[i]["title_band_id"] = json!(band);
                    s.lines[i]["structural_title"] = json!(true)
                }
            }
        }
    }
    let normalized = strings
        .iter()
        .map(|t| t.nfkc().collect::<String>())
        .collect::<Vec<_>>();
    let abstracts = normalized
        .iter()
        .enumerate()
        .filter(|(_, t)| abstract_line(t))
        .map(|(i, _)| s.boxes[i].n[1])
        .collect::<Vec<_>>();
    let institution = crate::cached_regex!(r"^ *[0-9]{1,2}[A-Z][a-z]");
    for i in 0..s.lines.len() {
        if !s.unresolved(i) {
            continue;
        }
        let mapped = normalized[i]
            .chars()
            .map(|c| {
                python_decimal(c).map_or(if python_space(c) { ' ' } else { c }, |d| {
                    (b'0' + d) as char
                })
            })
            .collect::<String>();
        if !institution.is_match(&mapped)
            || !abstracts
                .iter()
                .any(|&y| s.boxes[i].n[3] < y && y <= s.boxes[i].n[3] + 12. * s.h(i).unwrap())
        {
            continue;
        }
        let mut preceding = false;
        for j in 0..s.lines.len() {
            w.charge(1)?;
            if s.boxes[j].n[3] < s.boxes[i].n[1]
                && s.boxes[i].n[1] - s.boxes[j].n[3] <= 8. * s.h(j)?
                && s.boxes[j].n[2] - s.boxes[j].n[0] >= 0.6 * p[0]
                && normalized[j]
                    .chars()
                    .filter(|c| c.is_ascii_alphabetic())
                    .count()
                    >= 20
            {
                preceding = true;
                break;
            }
        }
        if preceding {
            s.mark(i, "text");
            s.lines[i]["title_suppressed"] = json!(true)
        }
    }
    Ok(())
}
pub(super) fn physical(s: &State, ids: &[usize], w: &mut Work) -> Result<Vec<Value>, Error> {
    let mut out = Vec::<(i64, Value)>::new();
    for &i in ids {
        let mut above = None;
        let mut below = None;
        for &j in ids {
            w.charge(1)?;
            if i == j
                || axis(&s.boxes[i], &s.boxes[j], 0) < 0.1
                || !s.lines[i]["visual_row_id"].is_null()
                    && eq(&s.lines[i]["visual_row_id"], &s.lines[j]["visual_row_id"])
            {
                continue;
            }
            let (y, other) = (cy(&s.boxes[i]), cy(&s.boxes[j]));
            if other < y {
                let g = gap(s, j, i)?.max(0.);
                above = Some(above.map_or(g, |old: f64| old.min(g)))
            } else if other > y {
                let g = gap(s, i, j)?.max(0.);
                below = Some(below.map_or(g, |old: f64| old.min(g)))
            }
        }
        let id = integer(&s.lines[i], "source_index", 0)?;
        let value = json!({"source_index":id,"above":above,"below":below});
        if let Some((_, v)) = out.iter_mut().find(|(key, _)| *key == id) {
            *v = value
        } else {
            out.push((id, value))
        }
    }
    Ok(out.into_iter().map(|(_, v)| v).collect())
}
pub(super) fn suppression(
    s: &State,
    ids: &[usize],
    lanes: &[Lane],
    containers: &[Box4],
    em: f64,
    w: &mut Work,
) -> Result<Vec<i64>, Error> {
    let mut candidates = Vec::new();
    for (lane_no, lane) in lanes.iter().enumerate() {
        if lane.span {
            continue;
        }
        let width = (lane.right - lane.left).max(0.1);
        for pair in lane.ids.windows(2) {
            w.charge(1)?;
            let (i, j) = (pair[0], pair[1]);
            let (a, b) = (&s.boxes[i], &s.boxes[j]);
            let g = gap(s, i, j)?;
            if !s.unresolved(i)
                || !s.unresolved(j)
                || a.n[2] - a.n[0] > 0.35 * width
                || b.n[2] - b.n[0] < (0.6 * width).max(1.8 * (a.n[2] - a.n[0]))
                || (b.n[0] - a.n[0]).abs() > 0.75 * em
                || g < -0.25 * em
                || g > 0.75 * em
            {
                continue;
            }
            candidates.push((lane_no, integer(&s.lines[i], "source_index", 0)?, cy(a)))
        }
    }
    candidates.sort_by(|a, b| a.2.total_cmp(&b.2));
    let mut bands = Vec::<Vec<(usize, i64, f64)>>::new();
    for c in candidates {
        let mut target = None;
        for (i, band) in bands.iter().enumerate() {
            w.charge(band.len() + 1)?;
            if (c.2 - median(band.iter().map(|v| v.2).collect())).abs() <= 0.5 * em {
                target = Some(i);
                break;
            }
        }
        if let Some(i) = target {
            bands[i].push(c)
        } else {
            bands.push(vec![c])
        }
    }
    let paired = bands
        .into_iter()
        .filter(|b| b.iter().map(|v| v.0).collect::<BTreeSet<_>>().len() >= 2)
        .collect::<Vec<_>>();
    let mut suppressed = BTreeSet::new();
    if paired.len() >= 2 {
        for b in paired {
            suppressed.extend(b.into_iter().map(|v| v.1))
        }
    }
    let mut rows = Vec::<(Value, Vec<usize>)>::new();
    for &i in ids {
        let id = &s.lines[i]["visual_row_id"];
        if id.is_null() {
            continue;
        }
        if let Some((_, members)) = rows.iter_mut().find(|(v, _)| eq(v, id)) {
            members.push(i)
        } else {
            rows.push((id.clone(), vec![i]))
        }
    }
    for (_, members) in rows {
        if members.len() < 2 {
            continue;
        }
        let b = s.bounds(&members)?;
        w.charge(containers.len() + 1)?;
        if containers.iter().any(|c| {
            axis(&b, c, 0) >= 0.35 && (b.n[1] - c.n[3]).max(c.n[1] - b.n[3]).max(0.) <= 1.5 * em
        }) {
            for i in members {
                suppressed.insert(integer(&s.lines[i], "source_index", 0)?);
            }
        }
    }
    Ok(suppressed.into_iter().collect())
}
