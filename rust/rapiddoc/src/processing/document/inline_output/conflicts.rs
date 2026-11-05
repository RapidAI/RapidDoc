//! Upstream d878cba: preserve match provenance until script conflicts are resolved.
use super::*;
#[derive(Clone)]
pub(super) struct Match {
    pub line: Line,
    pub source_range: Range,
    pub range: Range,
    // Alphabetical order: dehyphenated, exact, formula, fragment.
    kind: u8,
}
fn append(out: &mut Vec<Match>, line: &Line, source: &Range, start: usize, end: usize, kind: u8) {
    let mut range = source.clone();
    range.start = start;
    range.end = end;
    out.push(Match {
        line: line.clone(),
        source_range: source.clone(),
        range,
        kind,
    });
}
fn mapped(out: &mut Vec<Match>, line: &Line, mapping: &[Option<usize>], kind: u8) {
    for r in &line.ranges {
        let mut active: Option<(usize, usize, usize)> = None;
        for (source, index) in mapping.iter().enumerate().take(r.end).skip(r.start) {
            if active.is_some_and(|(_, _, last)| index.is_none_or(|i| i != last + 1)) {
                let (source_start, start, last) = active.take().unwrap();
                let mut sr = r.clone();
                sr.start = source_start;
                sr.end = source;
                append(out, line, &sr, start, last + 1, kind);
            }
            if let Some(index) = index {
                if let Some((_, _, last)) = &mut active {
                    *last = *index
                } else {
                    active = Some((source, *index, *index));
                }
            }
        }
        if let Some((source_start, start, last)) = active {
            let mut sr = r.clone();
            sr.start = source_start;
            append(out, line, &sr, start, last + 1, kind);
        }
    }
}
pub(super) fn collect(
    tokens: &[Token],
    lines: &[Line],
    work: &mut Work,
) -> Result<Vec<Match>, Error> {
    let hay: Vec<_> = tokens.iter().map(|t| t.value).collect();
    let mut out = Vec::new();
    let mut cursor = 0;
    for (index, line) in lines.iter().enumerate() {
        if let Some(start) = matching::find(&hay, &line.text, cursor, work)? {
            for r in &line.ranges {
                append(&mut out, line, r, start + r.start, start + r.end, 1);
            }
            cursor = start + line.text.len();
            continue;
        }
        if let Some((end, mapping)) = matching::formula(&line.text, tokens, cursor, work)? {
            mapped(&mut out, line, &mapping, 2);
            cursor = end;
            continue;
        }
        if let Some((end, mapping)) =
            matching::dehyphen(&hay, line, lines.get(index + 1), cursor, false, work)?
        {
            mapped(&mut out, line, &mapping, 0);
            cursor = end;
            continue;
        }
        for r in &line.ranges {
            if let Some(start) = matching::fallback(&hay, line, r, cursor, false, work)? {
                let end = start + r.end - r.start;
                append(&mut out, line, r, start, end, 3);
                cursor = end;
            }
        }
    }
    for (index, line) in lines.iter().enumerate() {
        let exact = matching::occurrences(&hay, &line.text, 0, work)?;
        if exact.len() == 1 {
            for r in &line.ranges {
                append(&mut out, line, r, exact[0] + r.start, exact[0] + r.end, 1);
            }
            continue;
        }
        if let Some((_, mapping)) = matching::formula(&line.text, tokens, 0, work)? {
            mapped(&mut out, line, &mapping, 2);
            continue;
        }
        if let Some((_, mapping)) =
            matching::dehyphen(&hay, line, lines.get(index + 1), 0, false, work)?
        {
            mapped(&mut out, line, &mapping, 0);
            continue;
        }
        for r in &line.ranges {
            if let Some(start) = matching::fallback(&hay, line, r, 0, false, work)? {
                append(&mut out, line, r, start, start + r.end - r.start, 3);
            }
        }
    }
    resolve(out, work)
}
fn resolve(matches: Vec<Match>, work: &mut Work) -> Result<Vec<Match>, Error> {
    let mut positions = BTreeMap::new();
    let mut candidates = Vec::<Match>::new();
    for m in matches {
        if m.range.start >= m.range.end {
            continue;
        }
        let key = (
            m.line.source,
            m.line.bbox.map(|v| if v == 0. { 0 } else { v.to_bits() }),
            m.line.text.clone(),
            m.source_range.start,
            m.source_range.end,
            m.source_range.styles,
            m.range.start,
            m.range.end,
        );
        if let Some(&index) = positions.get(&key) {
            let previous: &mut Match = &mut candidates[index];
            if previous.kind == 3 && m.kind != 3 {
                *previous = m;
            }
        } else {
            positions.insert(key, candidates.len());
            candidates.push(m);
        }
    }
    let mut events = BTreeMap::<usize, Vec<(usize, bool)>>::new();
    for (i, m) in candidates.iter().enumerate() {
        events.entry(m.range.start).or_default().push((i, true));
        events.entry(m.range.end).or_default().push((i, false));
    }
    let mut active = BTreeSet::new();
    let mut retained = BTreeMap::<usize, Vec<(usize, usize)>>::new();
    let mut previous = None;
    for (position, events) in events {
        if let Some(start) = previous.filter(|&start| start < position) {
            work.charge(active.len())?;
            let whole = active.iter().any(|&i: &usize| candidates[i].kind != 3);
            let strongest: Vec<_> = active
                .iter()
                .copied()
                .filter(|&i| !whole || candidates[i].kind != 3)
                .collect();
            let styles = strongest
                .iter()
                .fold(0, |mask, &i| mask | candidates[i].source_range.styles)
                & 0x30;
            if styles.count_ones() == 1 {
                for i in strongest {
                    let segments = retained.entry(i).or_default();
                    if let Some(last) = segments.last_mut().filter(|last| last.1 == start) {
                        last.1 = position
                    } else {
                        segments.push((start, position));
                    }
                }
            }
        }
        for (i, entering) in events {
            if entering {
                active.insert(i);
            } else {
                active.remove(&i);
            }
        }
        previous = Some(position);
    }
    let mut out = Vec::new();
    for (i, segments) in retained {
        for (start, end) in segments {
            let mut m = candidates[i].clone();
            let source = m.source_range.start + start - m.range.start;
            m.source_range.start = source;
            m.source_range.end = source + end - start;
            m.range.start = start;
            m.range.end = end;
            out.push(m);
        }
    }
    out.sort_by(|a, b| {
        a.range
            .start
            .cmp(&b.range.start)
            .then(a.range.end.cmp(&b.range.end))
            .then(a.line.source.cmp(&b.line.source))
            .then_with(|| {
                for i in 0..4 {
                    let order = a.line.bbox[i].total_cmp(&b.line.bbox[i]);
                    if order.is_ne() {
                        return order;
                    }
                }
                std::cmp::Ordering::Equal
            })
            .then(a.line.text.cmp(&b.line.text))
            .then(a.source_range.start.cmp(&b.source_range.start))
            .then(a.source_range.end.cmp(&b.source_range.end))
            .then_with(|| style_key(a.source_range.styles).cmp(&style_key(b.source_range.styles)))
            .then(a.kind.cmp(&b.kind))
    });
    Ok(out)
}
pub(super) fn resolved_mask(mut mask: u8, native: u8) -> u8 {
    if native != 0 {
        mask = (mask & !0x30) | native;
    }
    if mask & 0x30 == 0x30 {
        mask &= !0x30;
    }
    mask
}
pub(super) fn effective(
    content: &[char],
    styles: &[Range],
    owned: bool,
    work: &mut Work,
) -> Result<Vec<Range>, Error> {
    let (markers, scripts) = if owned {
        spans::script_markers(content).unwrap_or_default()
    } else {
        (Vec::new(), Vec::new())
    };
    let mut excluded = markers;
    let mut i = 0;
    while i + 1 < content.len() {
        if content[i..].starts_with(&['\\', '(']) {
            if let Some(end) = formula_end(content, i + 2, work)? {
                excluded.push((i, end + 2));
                i = end + 2;
                continue;
            }
        }
        i += 1;
    }
    let mut boundaries = BTreeSet::from([0, content.len()]);
    for r in styles.iter().chain(&scripts) {
        boundaries.extend([r.start, r.end]);
    }
    for &(a, b) in &excluded {
        boundaries.extend([a, b]);
    }
    let boundaries: Vec<_> = boundaries
        .into_iter()
        .filter(|&i| i <= content.len())
        .collect();
    let mut out = Vec::new();
    for pair in boundaries.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        if excluded.iter().any(|&(a, b)| a <= start && end <= b) {
            continue;
        }
        let native = scripts
            .iter()
            .filter(|r| r.start <= start && end <= r.end)
            .fold(0, |m, r| m | r.styles);
        let mask = styles
            .iter()
            .filter(|r| r.start <= start && end <= r.end)
            .fold(0, |m, r| m | r.styles);
        let mask = resolved_mask(mask, native) & 0x30;
        if mask != 0 {
            out.push(Range {
                start,
                end,
                styles: mask,
                source: 0,
                target: None,
            });
        }
    }
    Ok(merge_raw_styles(content, out))
}
