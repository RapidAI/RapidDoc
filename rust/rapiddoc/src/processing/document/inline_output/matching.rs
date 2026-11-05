use super::*;
use rapiddoc_native::text_boundary::python_lower;
pub(super) fn find(
    hay: &[char],
    needle: &[char],
    start: usize,
    work: &mut Work,
) -> Result<Option<usize>, Error> {
    if start > hay.len() {
        return Ok(None);
    }
    if needle.is_empty() {
        return Ok(Some(start));
    }
    if needle.len() > hay.len() - start {
        return Ok(None);
    }
    for i in start..=hay.len() - needle.len() {
        work.charge(needle.len())?;
        if hay[i..i + needle.len()] == *needle {
            return Ok(Some(i));
        }
    }
    Ok(None)
}
pub(super) fn occurrences(
    hay: &[char],
    needle: &[char],
    start: usize,
    work: &mut Work,
) -> Result<Vec<usize>, Error> {
    let mut out = Vec::new();
    if needle.is_empty() {
        return Ok(out);
    }
    let mut i = start;
    while let Some(p) = find(hay, needle, i, work)? {
        out.push(p);
        i = p + 1
    }
    Ok(out)
}
pub(super) fn fallback(
    hay: &[char],
    line: &Line,
    r: &Range,
    start: usize,
    link: bool,
    work: &mut Work,
) -> Result<Option<usize>, Error> {
    let target = &line.text[r.start..r.end];
    let positions = occurrences(hay, target, start, work)?;
    if positions.is_empty() {
        return Ok(None);
    }
    let geometric = r.styles & ((1 << 2) | (1 << 3)) != 0;
    if positions.len() == 1 && (link || geometric || target.len() >= 3) {
        return Ok(Some(positions[0]));
    }
    let left = &line.text[r.start.saturating_sub(12)..r.start];
    let right = &line.text[r.end..(r.end + 12).min(line.text.len())];
    let mut best = 0usize;
    let mut winners = Vec::new();
    for p in positions {
        work.charge(left.len() + right.len())?;
        let score = usize::from(!left.is_empty() && hay[p.saturating_sub(left.len())..p] == *left)
            + usize::from(
                !right.is_empty()
                    && hay.get(p + target.len()..p + target.len() + right.len()) == Some(right),
            );
        if winners.is_empty() || score > best {
            best = score;
            winners.clear()
        }
        if score == best {
            winners.push(p)
        }
    }
    let required = usize::from(!left.is_empty()) + usize::from(!right.is_empty());
    Ok(
        if winners.len() == 1
            && if !link && geometric {
                best > 0
            } else {
                required > 0 && best == required
            }
        {
            Some(winners[0])
        } else {
            None
        },
    )
}
pub(super) fn formula(
    line: &[char],
    tokens: &[Token],
    start: usize,
    work: &mut Work,
) -> Result<Option<(usize, Vec<Option<usize>>)>, Error> {
    if line.is_empty() || start >= tokens.len() || !tokens[start..].iter().any(|t| t.gap) {
        return Ok(None);
    }
    for first in start..tokens.len() {
        if !tokens[first].gap && tokens[first].value != line[0] {
            continue;
        }
        let mut states = vec![(0usize, Vec::new())];
        for (index, token) in tokens.iter().enumerate().skip(first) {
            let mut next = Vec::new();
            let mut seen = BTreeSet::new();
            for (source, mapping) in states {
                if source >= line.len() {
                    continue;
                }
                let candidates = if token.gap {
                    source + 1..line.len()
                } else if line[source] == token.value {
                    source..source + 1
                } else {
                    continue;
                };
                for matched in candidates {
                    work.charge(1)?;
                    if line[matched] != token.value {
                        continue;
                    }
                    let key = matched + 1;
                    if !seen.insert(key) {
                        continue;
                    }
                    work.charge(mapping.len() + matched - source + 1)?;
                    let mut m = mapping.clone();
                    m.extend(std::iter::repeat_n(None, matched - source));
                    m.push(Some(index));
                    next.push((key, m));
                }
            }
            if let Some((_, mapping)) = next.iter().find(|(s, _)| *s == line.len()) {
                return Ok(Some((index + 1, mapping.clone())));
            }
            if next.is_empty() {
                break;
            }
            states = next;
        }
    }
    Ok(None)
}
pub(super) fn dehyphen(
    hay: &[char],
    line: &Line,
    next: Option<&Line>,
    start: usize,
    link: bool,
    work: &mut Work,
) -> Result<Option<(usize, Vec<Option<usize>>)>, Error> {
    let Some(next) = next else { return Ok(None) };
    if line.source.checked_add(1) != Some(next.source)
        || line.text.is_empty()
        || next.text.is_empty()
        || !python_lower(next.text[0])
    {
        return Ok(None);
    }
    let trimmed: Vec<_> = line
        .text
        .iter()
        .copied()
        .rev()
        .skip_while(|c| python_space(*c))
        .collect();
    if trimmed.len() < 2
        || !"-\u{ad}\u{2010}\u{2011}\u{2043}".contains(trimmed[0])
        || !trimmed[1].is_ascii_alphabetic()
    {
        return Ok(None);
    }
    if link
        && !line.ranges.iter().any(|a| {
            a.start < a.end
                && a.end == line.text.len()
                && next
                    .ranges
                    .iter()
                    .any(|b| b.start == 0 && b.end > 0 && a.target == b.target)
        })
    {
        return Ok(None);
    }
    let prefix = &line.text[..line.text.len() - 1];
    let positions = occurrences(hay, prefix, start, work)?
        .into_iter()
        .filter(|p| hay.get(p + prefix.len()) == Some(&next.text[0]))
        .collect::<Vec<_>>();
    if positions.len() != 1 {
        return Ok(None);
    }
    let p = positions[0];
    let mut mapping: Vec<_> = (p..p + prefix.len()).map(Some).collect();
    mapping.push(None);
    Ok(Some((p + prefix.len(), mapping)))
}
fn map_range(r: &Range, mapping: &[Option<usize>]) -> Vec<Range> {
    let mut out = Vec::new();
    let mut active: Option<Range> = None;
    for index in mapping.iter().take(r.end).skip(r.start) {
        if active.as_ref().is_some_and(|a| Some(a.end) != *index) {
            out.push(active.take().unwrap())
        }
        if let Some(i) = index {
            if let Some(a) = &mut active {
                a.end = i + 1
            } else {
                let mut a = r.clone();
                a.start = *i;
                a.end = i + 1;
                active = Some(a)
            }
        }
    }
    if let Some(a) = active {
        out.push(a)
    }
    out
}
fn merge_links(ranges: &[Range], work: &mut Work) -> Result<Vec<Range>, Error> {
    let ranges: Vec<_> = ranges
        .iter()
        .filter(|r| r.start < r.end && r.target.as_ref().is_some_and(|s| !s.is_empty()))
        .collect();
    let points: BTreeSet<_> = ranges.iter().flat_map(|r| [r.start, r.end]).collect();
    let points: Vec<_> = points.into_iter().collect();
    work.charge(points.len().saturating_mul(ranges.len()))?;
    let mut out: Vec<Range> = Vec::new();
    for pair in points.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        let active: Vec<_> = ranges
            .iter()
            .filter(|r| r.start < end && r.end > start)
            .collect();
        let targets: BTreeSet<_> = active.iter().map(|r| &r.target).collect();
        if targets.len() != 1 {
            continue;
        }
        let target = (*targets.first().unwrap()).clone();
        let source = active.iter().map(|r| r.source).min().unwrap();
        if out
            .last()
            .is_some_and(|r| r.end == start && r.target == target && r.source == source)
        {
            out.last_mut().unwrap().end = end
        } else {
            out.push(Range {
                start,
                end,
                target,
                source,
                styles: 0,
            })
        }
    }
    Ok(out)
}
pub(super) fn match_ranges(
    tokens: &[Token],
    lines: &[Line],
    link: bool,
    work: &mut Work,
) -> Result<Vec<Range>, Error> {
    let hay: Vec<_> = tokens.iter().map(|t| t.value).collect();
    let mut output = Vec::new();
    let mut cursor = 0;
    for (i, l) in lines.iter().enumerate() {
        if let Some(p) = find(&hay, &l.text, cursor, work)? {
            for r in &l.ranges {
                let mut r = r.clone();
                r.start += p;
                r.end += p;
                output.push(r)
            }
            cursor = p + l.text.len();
            continue;
        }
        let mapping = if let Some(m) = formula(&l.text, tokens, cursor, work)? {
            Some(m)
        } else {
            dehyphen(&hay, l, lines.get(i + 1), cursor, link, work)?
        };
        if let Some((end, mapping)) = mapping {
            for r in &l.ranges {
                output.extend(map_range(r, &mapping))
            }
            cursor = end;
            continue;
        }
        for r in &l.ranges {
            if let Some(p) = fallback(&hay, l, r, cursor, link, work)? {
                let mut r = r.clone();
                let n = r.end - r.start;
                r.start = p;
                r.end = p + n;
                cursor = r.end;
                output.push(r)
            }
        }
    }
    if link {
        merge_links(&output, work)
    } else {
        Ok(merge_styles(&output))
    }
}

pub(super) fn match_script(
    tokens: &[Token],
    line: &Line,
    work: &mut Work,
) -> Result<Vec<Range>, Error> {
    let hay: Vec<_> = tokens.iter().map(|t| t.value).collect();
    let exact = occurrences(&hay, &line.text, 0, work)?;
    if exact.len() == 1 {
        return Ok(line
            .ranges
            .iter()
            .map(|r| {
                let mut r = r.clone();
                r.start += exact[0];
                r.end += exact[0];
                r
            })
            .collect());
    }
    if let Some((_, mapping)) = formula(&line.text, tokens, 0, work)? {
        return Ok(line
            .ranges
            .iter()
            .flat_map(|r| map_range(r, &mapping))
            .collect());
    }
    let mut out = Vec::new();
    for r in &line.ranges {
        if let Some(p) = fallback(&hay, line, r, 0, false, work)? {
            let mut r = r.clone();
            r.end = p + r.end - r.start;
            r.start = p;
            out.push(r)
        }
    }
    Ok(merge_styles(&out))
}
