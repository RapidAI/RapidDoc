//! Original body/title barriers, numbered and isolated-line evidence.
use super::*;
use unicode_normalization::UnicodeNormalization;
pub(super) fn wide(p: &Page, a: &Row, r: &Row, width: f64) -> Result<bool, Error> {
    Ok(p.l(a)["style_scale_repaired"] == true
        && standalone(p.t(a)?).is_some_and(|s| number(&s))
        && r.b.n[2] - r.b.n[0] > 0.75 * width
        && r.b.n[2] - r.b.n[0] <= 0.95 * width
        && math(p.t(r)?))
}
pub(super) fn body(p: &Page, r: &Row, lane: &Lane, font: &Value, em: f64) -> Result<bool, Error> {
    let l = p.l(r);
    let width = (lane.right - lane.left).max(0.1);
    let h = p.h(r)?;
    if l["style_scale_repaired"] != true {
        return Ok(!font.is_null()
            && eq(&l["font_signature"], font)
            && p.coverage(r)? >= 0.75
            && r.b.n[2] - r.b.n[0] >= 0.3 * width
            && h >= 0.8 * em
            && h <= 1.25 * em);
    }
    let mut scales = Vec::new();
    for other in &lane.rows {
        if other.b.n[2] - other.b.n[0] >= 0.35 * width && !math(p.t(other)?) {
            scales.push(p.h(other)?)
        }
    }
    let scale = if scales.is_empty() {
        em
    } else {
        median(scales)
    };
    let left = (r.b.n[0] - lane.left).abs() <= 3f64.max(0.75 * scale);
    Ok(!math(p.t(r)?)
        && (r.b.n[2] - r.b.n[0] >= 0.3 * width || left && r.b.n[2] - r.b.n[0] >= 0.08 * width)
        && h >= 0.75 * scale
        && h <= 1.35 * scale
        && (!font.is_null() && eq(&l["font_signature"], font) && p.coverage(r)? >= 0.75 || left))
}
pub(super) fn title(p: &Page, r: &Row, lane: &Lane, font: &Value, em: f64) -> Result<bool, Error> {
    let l = p.l(r);
    let width = (lane.right - lane.left).max(0.1);
    let h = p.h(r)?;
    Ok(!font.is_null()
        && !l["font_signature"].is_null()
        && !eq(&l["font_signature"], font)
        && p.coverage(r)? >= 0.75
        && h >= 1.1 * em
        && h <= 1.6 * em
        && r.b.n[2] - r.b.n[0] >= 0.25 * width
        && (r.b.n[0] - lane.left).abs() <= em)
}
pub(super) fn prefix(
    p: &Page,
    r: &Row,
    lane: &Lane,
    anchor: &Row,
    font: &Value,
    em: f64,
    coverage: f64,
) -> Result<bool, Error> {
    let l = p.l(r);
    if l["formula_candidate_only"] == true {
        return Ok(false);
    }
    let h = p.h(r)?;
    let ah = p.h(anchor)?;
    if cy(&r.b) > cy(&anchor.b) - 0.2 * h.max(ah)
        || (r.b.n[0] - lane.left).abs() > 3f64.max(0.75 * em)
    {
        return Ok(false);
    }
    if l["style_scale_repaired"] != true && p.l(anchor)["style_scale_repaired"] != true {
        return Ok(!font.is_null()
            && eq(&l["font_signature"], font)
            && p.coverage(r)? >= coverage
            && h >= 0.8 * em
            && h <= 1.25 * em);
    }
    let width = (lane.right - lane.left).max(0.1);
    if r.b.n[2] - r.b.n[0] < 0.08 * width || math(p.t(r)?) {
        return Ok(false);
    }
    let mut scales = Vec::new();
    for other in &lane.rows {
        if other.b.n[2] - other.b.n[0] >= 0.35 * width
            && (other.b.n[0] - lane.left).abs() <= 3f64.max(0.75 * em)
            && !math(p.t(other)?)
        {
            scales.push(p.h(other)?)
        }
    }
    let scale = if scales.is_empty() {
        em
    } else {
        median(scales)
    };
    Ok(h >= 0.75 * scale
        && h <= 1.35 * scale
        && (font.is_null() || eq(&l["font_signature"], font) || p.coverage(r)? <= coverage))
}
pub(super) fn hanging(p: &Page, r: &Row, lane: &Lane, em: f64) -> Result<bool, Error> {
    if p.l(r)["font_signature"].is_null() {
        return Ok(false);
    }
    let Some((above, below)) = neighbors(p, r, &lane.rows, |_| true)? else {
        return Ok(false);
    };
    let width = (lane.right - lane.left).max(0.1);
    let before = cy(&r.b) - cy(&above.b);
    let after = cy(&below.b) - cy(&r.b);
    Ok(eq(&p.l(above)["font_signature"], &p.l(r)["font_signature"])
        && (above.b.n[0] - r.b.n[0]).abs() <= 0.5 * em
        && r.b.n[2] - r.b.n[0] <= 0.9 * (above.b.n[2] - above.b.n[0])
        && before >= 0.65 * em
        && before <= 1.6 * em
        && after >= 0.65 * em
        && after <= 1.6 * em
        && below.b.n[0] <= r.b.n[0] - 1.5 * em
        && below.b.n[2] - below.b.n[0] >= 0.75 * width)
}
fn neighbors<'a>(
    p: &Page,
    r: &Row,
    rows: &'a [Row],
    pred: impl Fn(&Row) -> bool,
) -> Result<Option<(&'a Row, &'a Row)>, Error> {
    let mut above = None;
    let mut below = None;
    for other in rows {
        if p.id(other)? == p.id(r)? || !pred(other) {
            continue;
        }
        if cy(&other.b) < cy(&r.b) && above.is_none_or(|a: &Row| cy(&other.b) > cy(&a.b)) {
            above = Some(other)
        } else if cy(&other.b) > cy(&r.b) && below.is_none_or(|a: &Row| cy(&other.b) < cy(&a.b)) {
            below = Some(other)
        }
    }
    Ok(above.zip(below))
}
pub(super) fn single_numbered(p: &Page, r: &Row, lane: &Lane, em: f64) -> Result<bool, Error> {
    if p.l(r)["style_scale_repaired"] != true {
        return Ok(false);
    }
    let Some((prefix, marker)) = split_number(p.t(r)?) else {
        return Ok(false);
    };
    if prefix.is_empty() || !number(&marker) || !math(&prefix) {
        return Ok(false);
    }
    let width = (lane.right - lane.left).max(0.1);
    let ratio = (r.b.n[2] - r.b.n[0]) / width;
    Ok((0.12..=0.98).contains(&ratio)
        && (cx(&r.b) - 0.5 * (lane.left + lane.right)).abs() <= 0.2 * width
        && r.b.n[3] - r.b.n[1] <= 4. * em
        && !hanging(p, r, lane, em)?)
}
pub(super) fn compact(p: &Page, r: &Row, lane: &Lane, em: f64) -> Result<bool, Error> {
    if p.l(r)["compact_formula_cluster"] != true {
        return Ok(false);
    }
    let width = (lane.right - lane.left).max(0.1);
    if r.b.n[2] - r.b.n[0] > 0.6 * width || r.b.n[3] - r.b.n[1] > 3. * em {
        return Ok(false);
    }
    let delta = (cx(&r.b) - 0.5 * (lane.left + lane.right)).abs() / width;
    let left = (r.b.n[0] - lane.left) / width;
    let right = (lane.right - r.b.n[2]) / width;
    if delta > 0.2 && !((0.03..=0.25).contains(&left) && right >= 0.35) {
        return Ok(false);
    }
    let Some((above, below)) = neighbors(p, r, &lane.rows, |other| {
        other.b.n[2] - other.b.n[0] >= 0.45 * width
            && p.h(other).is_ok_and(|h| h >= 0.8 * em && h <= 1.25 * em)
    })?
    else {
        return Ok(false);
    };
    Ok(cy(&r.b) - cy(&above.b) <= 8. * em && cy(&below.b) - cy(&r.b) <= 8. * em)
}
pub(super) fn nearby_number(p: &Page, r: &Row, lane: &Lane, em: f64) -> Result<bool, Error> {
    for other in &lane.rows {
        if p.id(other)? != p.id(r)?
            && standalone(p.t(other)?).is_some()
            && other.b.n[0] > cx(&r.b)
            && (cy(&other.b) - cy(&r.b)).abs() <= 2.5 * em
        {
            return Ok(true);
        }
    }
    Ok(false)
}
fn punctuated(p: &Page, r: &Row, lane: &Lane, em: f64) -> Result<bool, Error> {
    for other in &lane.rows {
        if p.id(other)? == p.id(r)? {
            continue;
        }
        let Some((prefix, _)) = split_number(p.t(other)?) else {
            continue;
        };
        let t = prefix.trim_matches(python_space);
        if t.is_empty()
            || t.chars().count() > 3
            || t.chars().any(rapiddoc_native::text_boundary::python_alnum)
        {
            continue;
        }
        let vg = (other.b.n[1].max(r.b.n[1]) - other.b.n[3].min(r.b.n[3])).max(0.);
        let hg = (other.b.n[0] - r.b.n[2]).max(0.);
        if other.b.n[0] >= r.b.n[2] - 0.5 * em && vg <= 0.75 * em && hg <= 4. * em {
            return Ok(true);
        }
    }
    Ok(false)
}
pub(super) fn unnumbered(
    p: &Page,
    r: &Row,
    lane: &Lane,
    em: f64,
    font: &Value,
) -> Result<bool, Error> {
    let l = p.l(r);
    if l["compact_formula_cluster"] == true
        || !math(p.t(r)?)
        || font.is_null()
        || l["font_signature"].is_null()
        || eq(&l["font_signature"], font)
        || p.coverage(r)? >= 0.75
    {
        return Ok(false);
    }
    let width = (lane.right - lane.left).max(0.1);
    let rw = r.b.n[2] - r.b.n[0];
    if rw < 0.15 * width
        || rw > 0.8 * width
        || (cx(&r.b) - 0.5 * (lane.left + lane.right)).abs() > 0.08 * width
        || r.b.n[3] - r.b.n[1] > 1.8 * em
        || hanging(p, r, lane, em)?
    {
        return Ok(false);
    }
    for other in &lane.rows {
        if p.id(other)? != p.id(r)?
            && standalone(p.t(other)?).is_some()
            && (cy(&other.b) - cy(&r.b)).abs() <= 4. * em
        {
            return Ok(false);
        }
    }
    if punctuated(p, r, lane, em)? {
        return Ok(false);
    }
    let Some((above, below)) = neighbors(p, r, &lane.rows, |other| {
        eq(&p.l(other)["font_signature"], font)
            && p.coverage(other).is_ok_and(|v| v >= 0.75)
            && other.b.n[2] - other.b.n[0] >= 0.45 * width
            && p.h(other).is_ok_and(|h| h >= 0.8 * em && h <= 1.25 * em)
    })?
    else {
        return Ok(false);
    };
    Ok(cy(&r.b) - cy(&above.b) <= 4. * em && cy(&below.b) - cy(&r.b) <= 4. * em)
}
pub(super) fn body_font(p: &Page, lane: &Lane, em: f64) -> Result<Value, Error> {
    let width = (lane.right - lane.left).max(0.1);
    let mut counts = Vec::<(Value, usize)>::new();
    for r in &lane.rows {
        let l = p.l(r);
        let h = p.h(r)?;
        if r.b.n[2] - r.b.n[0] < 0.35 * width
            || h < 0.8 * em
            || h > 1.25 * em
            || l["font_signature"].is_null()
            || p.coverage(r)? < 0.75
        {
            continue;
        }
        if let Some((_, n)) = counts.iter_mut().find(|(f, _)| eq(f, &l["font_signature"])) {
            *n += 1
        } else {
            counts.push((l["font_signature"].clone(), 1))
        }
    }
    counts.sort_by(|(a, n), (b, m)| {
        n.cmp(m)
            .then(a[0].as_str().unwrap().cmp(b[0].as_str().unwrap()))
            .then(
                a[1].as_f64()
                    .or_else(|| a[1].as_bool().map(|b| i32::from(b) as f64))
                    .unwrap()
                    .total_cmp(
                        &b[1]
                            .as_f64()
                            .or_else(|| b[1].as_bool().map(|b| i32::from(b) as f64))
                            .unwrap(),
                    ),
            )
    });
    Ok(counts.last().map_or(Value::Null, |(f, _)| f.clone()))
}
pub(super) fn interval(p: &Page, lane: &Lane, em: f64) -> Result<Option<(f64, f64)>, Error> {
    let width = (lane.right - lane.left).max(0.1);
    let mut rows = Vec::new();
    for r in &lane.rows {
        if r.b.n[2] - r.b.n[0] >= (4. * p.h(r)?).max(0.35 * width) {
            rows.push(r)
        }
    }
    rows.sort_by(|a, b| {
        a.b.n[1]
            .total_cmp(&b.b.n[1])
            .then(a.b.n[0].total_cmp(&b.b.n[0]))
    });
    if rows.len() < 3 {
        return Ok(None);
    }
    let dense = rows
        .iter()
        .enumerate()
        .filter(|(i, r)| {
            *i > 0 && r.b.n[1] - rows[i - 1].b.n[3] <= 1.5 * em
                || *i + 1 < rows.len() && rows[i + 1].b.n[1] - r.b.n[3] <= 1.5 * em
        })
        .map(|(_, r)| *r)
        .collect::<Vec<_>>();
    if dense.len() < 3 {
        Ok(None)
    } else {
        Ok(Some((
            dense.iter().map(|r| r.b.n[1]).reduce(f64::min).unwrap(),
            dense.iter().map(|r| r.b.n[3]).reduce(f64::max).unwrap(),
        )))
    }
}
pub(super) fn display_prose(t: &str) -> bool {
    if t.chars()
        .filter(|c| ('\u{3400}'..='\u{9fff}').contains(c))
        .count()
        >= 2
    {
        return true;
    }
    let words = crate::cached_regex!(r"[A-Za-z]{4,}-\d+");
    let word = |c| c == '_' || rapiddoc_native::text_boundary::python_alnum(c);
    for m in words.find_iter(t) {
        if t[..m.start()].chars().last().is_none_or(|c| !word(c))
            && t[m.end()..].chars().next().is_none_or(|c| !word(c))
            && m.as_str()
                .chars()
                .filter(|c| c.is_ascii_lowercase())
                .count()
                >= 3
        {
            return true;
        }
    }
    let residue = super::super::native_vector_formulas::formula_residue(t);
    let re = crate::cached_regex!(r"[A-Za-z]{3,}");
    for m in re.find_iter(&residue) {
        if residue[..m.start()].chars().last().is_some_and(word)
            || residue[m.end()..].chars().next().is_some_and(word)
        {
            continue;
        }
        let mut tail = residue[m.end()..].chars().peekable();
        while tail.peek().is_some_and(|&c| python_space(c)) {
            tail.next();
        }
        if tail.peek().is_some_and(|c| matches!(c, '(' | '[')) {
            continue;
        }
        let t = m.as_str();
        if [
            "lim", "sin", "cos", "tan", "log", "exp", "min", "max", "with", "and", "for",
        ]
        .contains(&t.to_ascii_lowercase().as_str())
        {
            continue;
        }
        let caps = crate::cached_regex!(r"^[a-z]{0,2}[A-Z]{2,}$");
        if !caps.is_match(t) {
            return true;
        }
    }
    false
}
pub(super) fn prose_prefix(t: &str) -> bool {
    let residue = super::super::native_vector_formulas::formula_residue(t);
    let re = crate::cached_regex!(r"[({\[（［【｛][^)}\]）］】｝]*[)}\]）］】｝]");
    let residue = re.replace_all(&residue, " ");
    residue
        .chars()
        .filter(|c| ('\u{3400}'..='\u{9fff}').contains(c))
        .count()
        >= 2
        || residue
            .split(python_space)
            .filter(|t| crate::cached_regex!(r"[A-Za-z]{2,}").is_match(t))
            .count()
            >= 2
}
pub(super) fn left_prose(
    p: &Page,
    members: &[Row],
    lane: &Lane,
    em: f64,
) -> Result<(bool, bool), Error> {
    let mut left = false;
    let mut fragmented = false;
    for r in members {
        let t = p.t(r)?.nfkc().collect::<String>();
        let t = t.trim_matches(python_space);
        let Some((i, _)) = t
            .char_indices()
            .find(|(_, c)| "=<>≤≥≠≈∝∈∑∫√±×÷−".contains(*c))
        else {
            continue;
        };
        if !prose_prefix(&t[..i]) {
            continue;
        }
        if (r.b.n[0] - lane.left).abs() <= 0.75 * em {
            left = true
        }
        for other in members {
            if other.index != r.index
                && (other.b.n[0] - lane.left).abs() <= 0.75 * em
                && r.b.n[0] - other.b.n[2] >= 0.
                && r.b.n[0] - other.b.n[2] <= 0.75 * em
                && axis(&r.b, &other.b, 1) >= 0.7
            {
                fragmented = true;
                break;
            }
        }
    }
    Ok((left || fragmented, fragmented))
}
