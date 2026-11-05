//! Exact local body-continuity and front-matter evidence used by lane titles.
use super::*;
pub(super) fn boundary(
    s: &State,
    l: &Lane,
    p: &Profile,
    height: f64,
    bottom: Option<f64>,
) -> Result<Option<f64>, Error> {
    let Some(bottom) = bottom else {
        return Ok(None);
    };
    let default = (bottom + 2.5 * p.h).max(0.39 * height);
    let width = (l.right - l.left).max(0.1);
    for pair in l.ids.windows(2) {
        let (i, j) = (pair[0], pair[1]);
        let (a, b) = (&s.boxes[i], &s.boxes[j]);
        if a.n[1] <= bottom {
            continue;
        }
        if (a.n[2] - a.n[0]) / width <= 0.35
            && (b.n[2] - b.n[0]) / width >= 0.75
            && ((a.n[0] - l.left).abs() <= 0.65 * p.h
                || (cx(a) - (l.left + l.right) / 2.).abs() <= 0.12 * width)
            && (0.75..=1.35).contains(&(s.h(j)? / p.h))
            && gap(s, i, j)?.max(0.) <= 1.5 * p.h
        {
            return Ok(Some(default.min(a.n[1] - 0.1)));
        }
    }
    Ok(Some(default))
}
pub(super) fn normalized_gap(
    s: &State,
    ids: &[usize],
    k: usize,
    d: isize,
    h: f64,
    physical: &BTreeMap<i64, (Option<f64>, Option<f64>)>,
) -> Result<f64, Error> {
    let j = k as isize + d;
    let lg = if j >= 0 && (j as usize) < ids.len() {
        if d < 0 {
            gap(s, ids[j as usize], ids[k])?
        } else {
            gap(s, ids[k], ids[j as usize])?
        }
    } else {
        0.5 * h
    };
    let pair = physical
        .get(&integer(&s.lines[ids[k]], "source_index", 0)?)
        .copied()
        .unwrap_or((None, None));
    Ok(lg
        .min(if d < 0 { pair.0 } else { pair.1 }.unwrap_or(lg))
        .max(0.)
        / h.max(0.1))
}
pub(super) fn sibling(s: &State, ids: &[usize], i: usize) -> Result<bool, Error> {
    if s.lines[i]["visual_row_id"].is_null() {
        return Ok(false);
    }
    for &j in ids {
        if i != j
            && eq(&s.lines[i]["visual_row_id"], &s.lines[j]["visual_row_id"])
            && !fonts_ok(&s.lines[i], &s.lines[j])?
        {
            return Ok(true);
        }
    }
    Ok(false)
}
pub(super) fn mixed(
    s: &State,
    ids: &[usize],
    k: usize,
    width: f64,
    p: &Profile,
) -> Result<bool, Error> {
    let i = ids[k];
    if coverage(&s.lines[i])? >= 0.95
        || s.boxes[i].n[2] - s.boxes[i].n[0] < 0.85 * width
        || !(0.85..=1.15).contains(&(s.h(i)? / p.h))
        || k + 1 >= ids.len()
    {
        return Ok(false);
    }
    let j = ids[k + 1];
    Ok(s.unresolved(j)
        && (0.8..=1.2).contains(&(s.h(j)? / p.h))
        && gap(s, i, j)? <= p.gap + 0.25 * p.h)
}
fn shared(s: &State, a: usize, b: usize) -> Result<bool, Error> {
    let before = &s.lines[a];
    let after = &s.lines[b];
    let len = after["text"].as_str().ok_or_else(invalid)?.chars().count();
    if len < 25 {
        return Ok(false);
    }
    let aa = before["native_cjk_style"].as_array();
    let bb = after["native_cjk_style"].as_array();
    if let (Some(aa), Some(bb)) = (aa, bb) {
        if !aa.is_empty() && !bb.is_empty() {
            if aa.len() < 4 || bb.len() < 4 {
                return Err(invalid());
            }
            let h = num(&aa[3])?;
            return Ok(eq(&json!(aa[..3]), &json!(bb[..3]))
                && h > 0.
                && (0.9..=1.1).contains(&(num(&bb[3])? / h)));
        }
    }
    let cjk = |c: &Value| {
        c["char"].as_str().is_some_and(|s| {
            s.chars().count() == 1 && s.chars().all(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
        })
    };
    let before = optional(before, "chars", 20000)?
        .iter()
        .filter(|c| cjk(c))
        .collect::<Vec<_>>();
    let after = optional(after, "chars", 20000)?
        .iter()
        .filter(|c| cjk(c))
        .collect::<Vec<_>>();
    if before.len() < 3 || after.is_empty() {
        return Ok(false);
    }
    let font = |c: &Value| json!([c["font"]["name"], c["font"]["flags"], c["font"]["weight"]]);
    let fonts = before.iter().map(|c| font(c)).collect::<Vec<_>>();
    let matches = after
        .into_iter()
        .filter(|c| fonts.iter().any(|f| eq(f, &font(c))))
        .collect::<Vec<_>>();
    if matches.is_empty() {
        return Ok(false);
    }
    let heights = |cs: &Vec<&Value>| {
        cs.iter()
            .map(|c| bbox(&c["bbox"]).map(|b| b.n[3] - b.n[1]))
            .collect::<Result<Vec<_>, _>>()
    };
    let h = median(heights(&before)?);
    Ok(h > 0. && (0.9..=1.1).contains(&(median(heights(&matches)?) / h)))
}
pub(super) fn continues(
    s: &State,
    ids: &[usize],
    k: usize,
    width: f64,
    p: &Profile,
) -> Result<bool, Error> {
    if k == 0 {
        return Ok(false);
    }
    let (i, previous) = (ids[k], ids[k - 1]);
    if !s.unresolved(previous) {
        return Ok(false);
    }
    let previous_row = if s.lines[previous]["visual_row_id"].is_null() {
        vec![previous]
    } else {
        ids[..k]
            .iter()
            .copied()
            .filter(|&j| {
                eq(
                    &s.lines[j]["visual_row_id"],
                    &s.lines[previous]["visual_row_id"],
                )
            })
            .collect::<Vec<_>>()
    };
    let style = previous_row
        .iter()
        .copied()
        .filter(|&j| {
            !s.lines[j]["font_signature"].is_null() && coverage(&s.lines[j]).unwrap() >= 0.5
        })
        .collect::<Vec<_>>();
    let shared = shared(s, previous, i)?;
    if !s.lines[i]["font_signature"].is_null() && !style.is_empty() && !shared {
        let mut match_style = false;
        for j in style {
            if share_family(&s.lines[j]["font_signature"], &s.lines[i]["font_signature"])?
                && weights_ok(&s.lines[j], &s.lines[i])?
            {
                match_style = true
            }
        }
        if !match_style {
            return Ok(false);
        }
    }
    let b = s.bounds(&previous_row)?;
    let ph = median(
        previous_row
            .iter()
            .map(|&j| s.h(j))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let h = s.h(i)?;
    let pair = ph.max(h);
    let g = if s.lines[previous]["restored_inline_cluster"] == true {
        (s.boxes[i].n[1] - b.n[3]).max(-0.25 * s.h(previous)?)
    } else {
        s.boxes[i].n[1] - (b.n[1] + s.h(previous)?)
    };
    Ok(((0.8..=1.25).contains(&(h / ph)) || shared)
        && b.n[2] - b.n[0] >= 0.75 * width
        && (s.boxes[i].n[0] - b.n[0]).abs() <= 2. * pair
        && g >= -0.25 * pair
        && g <= p.gap + pair)
}
pub(super) fn field(
    s: &State,
    ids: &[usize],
    k: usize,
    width: f64,
    p: &Profile,
) -> Result<bool, Error> {
    let i = ids[k];
    if !(0.85..=1.15).contains(&(s.h(i)? / p.h)) {
        return Ok(false);
    }
    for j in [k.checked_sub(1), Some(k + 1)]
        .into_iter()
        .flatten()
        .filter_map(|k| ids.get(k).copied())
    {
        if s.unresolved(j)
            && (0.85..=1.15).contains(&(s.h(j)? / p.h))
            && s.boxes[j].n[2] - s.boxes[j].n[0] >= 0.35 * width
            && (s.boxes[j].n[0] - s.boxes[i].n[0]).abs() <= 0.75 * p.h
            && gap(s, j, i)?.max(gap(s, i, j)?) <= 1.5 * p.h
        {
            return Ok(true);
        }
    }
    Ok(false)
}
pub(super) fn inline(
    s: &State,
    ids: &[usize],
    k: usize,
    width: f64,
    p: &Profile,
) -> Result<bool, Error> {
    let i = ids[k];
    if !(0.85..=1.15).contains(&(s.h(i)? / p.h))
        || s.boxes[i].n[2] - s.boxes[i].n[0] < 0.9 * width
        || k + 1 >= ids.len()
    {
        return Ok(false);
    }
    let j = ids[k + 1];
    let l = &s.lines[j];
    let body = p.font.is_null()
        || l["font_signature"].is_null()
        || coverage(l)? < 0.75
        || share_family(&l["font_signature"], &p.font)?;
    Ok(s.unresolved(j)
        && body
        && (0.75..=1.15).contains(&(s.h(j)? / p.h))
        && s.boxes[j].n[2] - s.boxes[j].n[0] >= 0.75 * width
        && gap(s, i, j)? <= p.gap + 0.25 * p.h)
}
pub(super) fn following(
    s: &State,
    ids: &[usize],
    k: usize,
    lane: &Lane,
    p: &Profile,
) -> Result<bool, Error> {
    if k + 1 >= ids.len() {
        return Ok(false);
    }
    let (i, j) = (ids[k], ids[k + 1]);
    let (a, b) = (&s.boxes[i], &s.boxes[j]);
    let width = (lane.right - lane.left).max(0.1);
    let related = axis(a, b, 0) >= 0.35 || (a.n[0] - b.n[0]).abs() <= 0.75 * p.h;
    let centered = (cx(a) - (lane.left + width / 2.)).abs() <= 0.12 * width;
    let compact = a.n[2] - a.n[0] <= 0.35 * width && (a.n[0] - lane.left).abs() <= 0.65 * p.h;
    let max = if centered {
        1.5
    } else if compact {
        1.25
    } else {
        0.75
    } * p.h;
    let g = gap(s, i, j)?;
    Ok(s.unresolved(j)
        && related
        && g >= -0.25 * p.h
        && g <= max
        && (0.9..=1.2).contains(&(s.h(j)? / p.h))
        && (b.n[2] - b.n[0]) / width >= 0.45)
}
pub(super) fn compact(s: &State, ids: &[usize], k: usize, width: f64) -> Result<bool, Error> {
    if k + 3 >= ids.len() {
        return Ok(false);
    }
    let h = s.h(ids[k])?;
    let mut previous = ids[k];
    for &i in &ids[k + 1..k + 4] {
        let g = gap(s, previous, i)?;
        if !s.unresolved(i)
            || !(0.75..=1.25).contains(&(s.h(i)? / h))
            || (s.boxes[i].n[2] - s.boxes[i].n[0]) / width < 0.45
            || g < -0.25 * h
            || g > 0.75 * h
        {
            return Ok(false);
        }
        previous = i
    }
    Ok(true)
}
pub(super) fn sentence(
    s: &State,
    ids: &[usize],
    k: usize,
    width: f64,
    p: &Profile,
) -> Result<bool, Error> {
    if k + 2 >= ids.len() {
        return Ok(false);
    }
    let first = &s.lines[ids[k]];
    if first["explicit_section_title"] == true || first["structural_title"] == true {
        return Ok(false);
    }
    let word = crate::cached_regex!(r"[A-Za-z]{2,}");
    use rapiddoc_native::text_boundary::python_alnum;
    for &i in &ids[k..k + 3] {
        let l = &s.lines[i];
        let b = &s.boxes[i];
        let text = l["text"].as_str().ok_or_else(invalid)?;
        let count = word
            .find_iter(text)
            .filter(|m| {
                let word = |c| python_alnum(c) || c == '_';
                text[..m.start()].chars().last().is_none_or(|c| !word(c))
                    && text[m.end()..].chars().next().is_none_or(|c| !word(c))
            })
            .count();
        if !s.unresolved(i)
            || !(0.85..=1.15).contains(&(s.h(i)? / p.h))
            || b.n[2] - b.n[0] < 0.8 * width
            || (b.n[0] - s.boxes[ids[k]].n[0]).abs() > 0.3 * p.h
            || !fonts_ok(first, l)?
            || count < 6
        {
            return Ok(false);
        }
        if !l["font_signature"].is_null() && super::super::native_precontainers::bold(l)? {
            return Ok(false);
        }
    }
    if !ids[k..k + 2].iter().any(|&i| {
        s.lines[i]["text"]
            .as_str()
            .unwrap()
            .contains([',', ';', '(', ')'])
    }) {
        return Ok(false);
    }
    let last = s.lines[ids[k + 2]]["text"]
        .as_str()
        .ok_or_else(invalid)?
        .trim_end_matches(python_space);
    let last = last.trim_end_matches(|c| "])’\"'".contains(c));
    if !last.ends_with(['.', '!', '?', ':', '：']) {
        return Ok(false);
    }
    for pair in ids[k..k + 3].windows(2) {
        let g = gap(s, pair[0], pair[1])?;
        if g < -0.1 * p.h || g > p.gap + 0.3 * p.h {
            return Ok(false);
        }
    }
    Ok(true)
}
