use super::*;
#[derive(PartialEq)]
pub(super) struct Lane {
    pub left: f64,
    pub right: f64,
    pub span: bool,
    pub indices: Vec<usize>,
}
pub(super) fn parse(v: &Value) -> Result<Vec<Lane>, Error> {
    array(v, 4096)?
        .iter()
        .map(|l| {
            Ok(Lane {
                left: num(&l["left"])?,
                right: num(&l["right"])?,
                span: l["is_span"].as_bool().ok_or_else(invalid)?,
                indices: array(&l["line_indices"], 4096)?
                    .iter()
                    .map(|i| i.as_u64().map(|i| i as usize).ok_or_else(invalid))
                    .collect::<Result<Vec<_>, _>>()?,
            })
        })
        .collect()
}
pub(super) fn width_reference(l: &Lane, lanes: &[Lane], em: f64) -> f64 {
    let width = (l.right - l.left).max(0.1);
    let mut stable = lanes
        .iter()
        .filter(|l| !l.span && l.indices.len() >= 3)
        .collect::<Vec<_>>();
    stable.sort_by(|a, b| a.left.total_cmp(&b.left));
    if let Some(i) = stable.iter().position(|s| *s == l) {
        if i + 1 < stable.len() {
            return width.max(stable[i + 1].left - l.left - 6f64.max(0.75 * em));
        }
    }
    width
}
fn lowercase(l: &Value) -> Result<Option<f64>, Error> {
    let mut hs = Vec::new();
    for c in optional(l, "chars", 20000)? {
        let t = super::super::table_native_multiline::spatial::raw_string(c.get("char"), false);
        if t.len() == 1 && t.as_bytes()[0].is_ascii_lowercase() {
            if let Some(b) = coerce(&c["tight_bbox"])? {
                if b.n[3] > b.n[1] {
                    hs.push(b.n[3] - b.n[1])
                }
            }
        }
    }
    Ok((hs.len() >= 6).then(|| median(hs)))
}
fn single(
    first: &Row,
    above: &[&Row],
    rule: &Box4,
    p: [f64; 2],
    em: f64,
    w: &mut Work,
) -> Result<bool, Error> {
    let raw = bbox(&first.l["bbox"])?;
    let Some((_, _, marker_len)) = groups::marker(text(&first.l)?, false, false) else {
        return Ok(false);
    };
    if rule.n[2] - rule.n[0] < 0.7 * p[0]
        || cy(rule) < 0.8 * p[1]
        || weight(&first.l)? >= 600.
        || (raw.n[0] - rule.n[0]).abs() > em
        || raw.n[2] - raw.n[0] > 0.7 * (rule.n[2] - rule.n[0])
        || above.len() < 3
        || above.iter().any(|r| raw.n[1] - r.b.n[3] < 0.75 * em)
    {
        return Ok(false);
    }
    let letters = optional(&first.l, "chars", 20000)?
        .iter()
        .filter(|c| {
            !super::super::table_native_multiline::spatial::raw_string(c.get("char"), false)
                .trim_matches(python_space)
                .is_empty()
        })
        .collect::<Vec<_>>();
    if letters.len() <= marker_len {
        return Ok(false);
    }
    let mut marker = Vec::new();
    for c in &letters[..marker_len] {
        let Some(b) = coerce(&c["tight_bbox"])? else {
            return Ok(false);
        };
        marker.push(b)
    }
    let Some(following) = coerce(&letters[marker_len]["tight_bbox"])? else {
        return Ok(false);
    };
    let b = combined(marker.iter())?;
    if b.n[2] - b.n[0] > 1.5 * em || following.n[0] - b.n[2] < 0.5 * em {
        return Ok(false);
    }
    let note = lowercase(&first.l)?;
    let mut body = Vec::new();
    for r in above {
        w.charge(optional(&r.l, "chars", 20000)?.len() + 1)?;
        if let Some(n) = lowercase(&r.l)? {
            body.push(n)
        }
    }
    Ok(note.is_some_and(|n| body.len() >= 3 && n <= 0.85 * median(body)))
}
pub(super) fn collect(
    l: &Lane,
    rows: &[Row],
    rule: &Box4,
    p: [f64; 2],
    page_em: f64,
    width_ref: f64,
    allow_column: bool,
    lower: Option<f64>,
    w: &mut Work,
) -> Result<BTreeSet<i64>, Error> {
    let empty = || Ok(BTreeSet::new());
    let mut items = l
        .indices
        .iter()
        .map(|&i| rows.get(i).ok_or_else(invalid))
        .collect::<Result<Vec<_>, _>>()?;
    items.retain(|r| r.l["semantic_type"].is_null());
    if items.is_empty() {
        return empty();
    }
    items.sort_by(|a, b| {
        a.b.n[1]
            .total_cmp(&b.b.n[1])
            .then(a.b.n[0].total_cmp(&b.b.n[0]))
            .then(id(&a.l).unwrap().cmp(&id(&b.l).unwrap()))
    });
    w.charge(items.len() * items.len() + items.len() * 8 + 1)?;
    for r in &items {
        if r.l["caption_start"] == true
            && r.b.n[1] >= rule.n[1]
            && r.b.n[1] <= rule.n[3] + 3. * h(r)?
        {
            return empty();
        }
    }
    let width = 0.1f64.max(l.right - l.left).max(width_ref);
    let em = median(items.iter().map(|r| h(r)).collect::<Result<Vec<_>, _>>()?);
    let rw = (rule.n[2] - rule.n[0]).max(0.);
    let short = rw < (4. * em).max(0.04 * p[0]);
    if short && rw < (2. * em).max(0.08 * width) {
        return empty();
    }
    let mut strict = (rule.n[0] - l.left).abs() <= (2. * em).max(0.04 * width);
    let wide = items
        .iter()
        .filter(|r| r.b.n[3] <= rule.n[1] && r.b.n[2] - r.b.n[0] >= 0.35 * width)
        .map(|r| h(r))
        .collect::<Result<Vec<_>, _>>()?;
    let marker_ref = em
        .max(page_em)
        .max(if wide.len() >= 3 { median(wide) } else { 0. });
    for (i, r) in items.iter().enumerate() {
        if !r.l["note_marker_value"].is_null()
            && (r.b.n[0] - rule.n[0]).abs() <= 0.5 * em
            && (0. ..=3. * em).contains(&(r.b.n[1] - rule.n[3]))
            && r.b.n[2] - r.b.n[0] <= 1.25 * em
        {
            for (j, b) in items.iter().enumerate() {
                if j != i
                    && b.b.n[0] >= r.b.n[2]
                    && b.b.n[0] - r.b.n[2] <= 2. * em
                    && b.b.n[2] - b.b.n[0] >= 4. * em
                    && axis(&r.b, &b.b, 1) >= 0.6
                    && h(b)? <= 0.9 * marker_ref
                {
                    strict = true;
                    break;
                }
            }
        }
    }
    let relaxed = rule.n[0] < l.left && l.left - rule.n[0] <= 2.25 * em;
    let centered = !l.span
        && cy(rule) >= 0.7 * p[1]
        && (0.35 * width..=0.7 * width).contains(&rw)
        && (cx(rule) - 0.5 * (l.left + l.right)).abs() <= 0.08 * width;
    if !strict && !relaxed && !centered {
        return empty();
    }
    let regular = cy(rule) >= (if short { 0.55 } else { 0.7 }) * p[1] && rw <= 0.65 * width;
    let column = allow_column
        && !l.span
        && (0.65 * width..=1.05 * width).contains(&rw)
        && (rule.n[2] - l.right).abs() <= (2. * em).max(0.05 * width);
    if !regular && !column && !centered {
        return empty();
    }
    let mut first = None;
    for (i, r) in items.iter().enumerate() {
        if rule.n[2].min(r.b.n[2]) <= rule.n[0].max(r.b.n[0]) || r.b.n[2] - r.b.n[0] < 2. * em {
            continue;
        }
        let dy = r.b.n[1] - rule.n[3];
        if dy < -0.5 * em {
            continue;
        }
        if lower.is_some_and(|y| r.b.n[1] >= y) {
            break;
        }
        if dy <= (3. * em).max(0.035 * p[1]) {
            first = Some(i)
        }
        break;
    }
    let Some(i) = first else { return empty() };
    let first = items[i];
    if regular
        && items[..i]
            .iter()
            .any(|r| r.b.n[1] < cy(rule) && cy(rule) < r.b.n[3] && axis(&r.b, rule, 0) >= 0.5)
    {
        return empty();
    }
    if short {
        let above = items[..i]
            .iter()
            .filter(|r| r.b.n[1] < rule.n[1])
            .copied()
            .collect::<Vec<_>>();
        let reference = page_em.max(if above.is_empty() {
            0.
        } else {
            median(above.iter().map(|r| h(r)).collect::<Result<Vec<_>, _>>()?)
        });
        let numbered = first.l["footnote_marker_start"] == true
            || items.iter().any(|r| {
                r.b.n[2] - r.b.n[0] <= 1.5 * em
                    && (r.b.n[0] - rule.n[0]).abs() <= 0.5 * em
                    && (0. ..=1.5 * em).contains(&(first.b.n[0] - r.b.n[2]))
                    && axis(&r.b, &first.b, 1) >= 0.6
            });
        let slot = (em..=2.25 * em).contains(&(first.b.n[0] - rule.n[0]));
        if !strict
            || !(numbered || slot)
            || reference <= 0.
            || h(first)? > 0.9 * reference
            || above.iter().any(|r| {
                r.b.n[3] > rule.n[1] - 0.25 * reference
                    || first.b.n[1] - r.b.n[3] < 0.75 * reference
            })
        {
            return empty();
        }
    }
    if column {
        let heights = items
            .iter()
            .filter(|r| r.b.n[3] <= rule.n[1] + 0.5 * em)
            .map(|r| h(r))
            .collect::<Result<Vec<_>, _>>()?;
        let count = heights.len();
        let reference = page_em.max(if heights.is_empty() {
            0.
        } else {
            median(heights)
        });
        let continuation = items[i + 1..]
            .iter()
            .filter(|r| r.b.n[1] >= first.b.n[1] + 0.5 * em && r.b.n[1] <= first.b.n[3] + 3. * em)
            .map(|r| &r.b)
            .collect::<Vec<_>>();
        let left = if continuation.is_empty() {
            first.b.n[0]
        } else {
            median(continuation.iter().map(|b| b.n[0]).collect())
        };
        let prefix = if first.l["source_bbox"].is_null() {
            first.b.clone()
        } else {
            bbox(&first.l["source_bbox"])?
        };
        let hanging = rw >= 0.7 * p[0]
            && cy(rule) >= 0.75 * p[1]
            && weight(&first.l)? < 600.
            && h(first)? <= 1.25 * reference
            && ((prefix.n[2] - prefix.n[0] <= 1.5 * em
                && (prefix.n[0] - rule.n[0]).abs() <= em
                && (0.5 * em..=2. * em).contains(&(left - prefix.n[2])))
                || items.iter().any(|r| {
                    r.b.n[2] - r.b.n[0] <= 1.5 * em
                        && (r.b.n[0] - rule.n[0]).abs() <= em
                        && (0. ..=2. * em).contains(&(first.b.n[0] - r.b.n[2]))
                        && axis(&r.b, &first.b, 1) >= 0.8
                }))
            && continuation
                .iter()
                .filter(|b| (b.n[0] - left).abs() <= 0.3 * em)
                .count()
                >= 2
            && items[..i]
                .iter()
                .filter(|r| r.b.n[3] <= rule.n[1])
                .all(|r| first.b.n[1] - r.b.n[3] >= 0.75 * reference);
        let above = items[..i]
            .iter()
            .filter(|r| r.b.n[3] <= rule.n[1])
            .copied()
            .collect::<Vec<_>>();
        let single = single(first, &above, rule, p, reference, w)?;
        if count < 3 || h(first)? > 0.95 * reference && !(hanging || single) {
            return empty();
        }
    }
    let mut members = vec![first];
    for r in &items[i + 1..] {
        if axis(&first.b, &r.b, 0) < 0.5 {
            continue;
        }
        if lower.is_some_and(|y| r.b.n[1] >= y)
            || gap(members.last().unwrap(), r)? > (1.25 * em).max(0.01 * p[1])
        {
            break;
        }
        members.push(r)
    }
    if relaxed && !strict {
        let reference = em.max(page_em);
        let overlap = (rule.n[2].min(first.b.n[2]) - rule.n[0].max(first.b.n[0])).max(0.);
        if members.len() < 2 || h(first)? > 0.9 * reference || overlap / rw.max(0.1) < 0.8 {
            return empty();
        }
    }
    if centered {
        let reference = em.max(page_em);
        if items[..i].iter().any(|r| {
            r.b.n[1] < rule.n[1]
                && (rule.n[2].min(r.b.n[2]) - rule.n[0].max(r.b.n[0])).max(0.)
                    >= 0.2 * rw.min((r.b.n[2] - r.b.n[0]).max(0.1))
                && r.b.n[3] > rule.n[1] - 0.75 * reference
        }) {
            return empty();
        }
        let mh = median(
            members
                .iter()
                .map(|r| h(r))
                .collect::<Result<Vec<_>, _>>()?,
        );
        if members.len() < 2 || mh > 0.9 * reference {
            return empty();
        }
    }
    members.iter().map(|r| id(&r.l)).collect()
}
