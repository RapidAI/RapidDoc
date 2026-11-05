//! 旧aligned_numeric.py实际执行链：字形重分行、数值列、逻辑表体和候选；不读取PDF。
use super::table_native_sparse::local_rules;
use super::{
    table_native_candidate,
    table_native_text::{self, Glyph, Row, Text, Token},
    table_native_vector::{number, validate_text},
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{
    python_decimal,
    text_boundary::{python_digit, python_space},
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len() / 2;
    if values.len() % 2 == 1 {
        values[n]
    } else {
        (values[n - 1] + values[n]) / 2.
    }
}
fn union<'a>(a: impl Iterator<Item = &'a [f64; 4]>) -> Result<[f64; 4], Error> {
    let a: Vec<_> = a.collect();
    if a.is_empty() {
        return Err(invalid());
    }
    Ok([
        a.iter().map(|b| b[0]).fold(f64::INFINITY, f64::min),
        a.iter().map(|b| b[1]).fold(f64::INFINITY, f64::min),
        a.iter().map(|b| b[2]).fold(f64::NEG_INFINITY, f64::max),
        a.iter().map(|b| b[3]).fold(f64::NEG_INFINITY, f64::max),
    ])
}
pub fn numeric_token(s: &str) -> bool {
    let s = s.trim_matches(python_space);
    if ["-", "–", "—"].contains(&s) {
        return true;
    }
    let c: Vec<_> = s.chars().collect();
    let mut i = 0;
    if c.get(i).is_some_and(|c| ['+', '−', '-'].contains(c)) {
        i += 1
    }
    if c.get(i) == Some(&'(') {
        i += 1
    }
    let first = i;
    while c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
        i += 1
    }
    if i == first {
        return false;
    }
    while c.get(i).is_some_and(|c| ['.', ','].contains(c)) {
        i += 1;
        let first = i;
        while c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
            i += 1
        }
        if i == first {
            return false;
        }
    }
    if c.get(i) == Some(&')') {
        i += 1
    }
    while c.get(i).is_some_and(|c| python_space(*c)) {
        i += 1
    }
    if c.get(i).is_some_and(|c| ['%', '‰'].contains(c)) {
        i += 1
    }
    i == c.len()
}
fn numeric(t: &Token) -> bool {
    numeric_token(&t.text)
}
fn regroup_text(text: &Text, work: &mut Work) -> Result<Text, Error> {
    let mut ordered: Vec<_> = text.glyphs.iter().collect();
    ordered.sort_by(|a, b| {
        ((a.bbox[1] + a.bbox[3]) / 2.)
            .partial_cmp(&((b.bbox[1] + b.bbox[3]) / 2.))
            .unwrap()
            .then(a.bbox[0].partial_cmp(&b.bbox[0]).unwrap())
    });
    let mut groups: Vec<Vec<Glyph>> = Vec::new();
    let mut centers = Vec::new();
    let threshold = (0.3 * text.median_glyph_height).max(0.75);
    for g in ordered {
        let cy = (g.bbox[1] + g.bbox[3]) / 2.;
        if centers.last().is_none_or(|last| cy - last > threshold) {
            groups.push(vec![g.clone()]);
            centers.push(cy)
        } else {
            let group = groups.last_mut().unwrap();
            group.push(g.clone());
            work.charge(group.len())?;
            *centers.last_mut().unwrap() =
                median(group.iter().map(|g| (g.bbox[1] + g.bbox[3]) / 2.).collect());
        }
    }
    let (mut glyphs, mut rows) = (Vec::new(), Vec::new());
    for (index, mut group) in groups.into_iter().enumerate() {
        for g in &mut group {
            g.visual_row = index as i64
        }
        group.sort_by(|a, b| a.bbox[0].partial_cmp(&b.bbox[0]).unwrap());
        work.charge(group.len())?;
        rows.push(Row {
            row_index: index as i64,
            bbox: union(group.iter().map(|g| &g.bbox))?,
            tokens: table_native_text::tokens(
                &group,
                0.35 * text.median_glyph_width,
                text.median_glyph_height,
            ),
            glyph_ids: group.iter().map(|g| g.glyph_id).collect(),
        });
        glyphs.extend(group)
    }
    glyphs.sort_by_key(|g| g.glyph_id);
    Ok(Text {
        glyphs,
        rows,
        median_glyph_width: text.median_glyph_width,
        median_glyph_height: text.median_glyph_height,
    })
}
type NumericGroup = (Vec<usize>, Vec<f64>);
fn row_groups(text: &Text, work: &mut Work) -> Result<Vec<NumericGroup>, Error> {
    let em = text.median_glyph_height;
    let mut groups = Vec::new();
    let (mut active, mut anchors): (Vec<usize>, Vec<f64>) = (Vec::new(), Vec::new());
    for (index, row) in text.rows.iter().enumerate() {
        work.charge(row.tokens.iter().map(|t| t.text.len() + 1).sum())?;
        let values: Vec<_> = row.tokens.iter().filter(|t| numeric(t)).collect();
        if values.len() < 2 {
            continue;
        }
        let positions: Vec<_> = values.iter().map(|t| t.bbox[2]).collect();
        let mut prefix = row
            .tokens
            .iter()
            .any(|t| !numeric(t) && t.bbox[2] < values[0].bbox[0]);
        if !prefix && index > 0 {
            let prior = &text.rows[index - 1];
            if row.bbox[1] - prior.bbox[3] < em {
                prefix = prior
                    .tokens
                    .iter()
                    .any(|t| !numeric(t) && t.bbox[2] < values[0].bbox[0])
            }
        }
        if active.is_empty() && !prefix {
            continue;
        }
        if positions.windows(2).any(|p| p[1] - p[0] < 2. * em) {
            continue;
        }
        work.charge(positions.len() * anchors.len())?;
        let matches = !anchors.is_empty()
            && positions.iter().all(|x| {
                anchors
                    .iter()
                    .map(|a| (x - a).abs())
                    .fold(f64::INFINITY, f64::min)
                    <= 0.65 * em
            });
        let prefix_limit = values
            .iter()
            .map(|t| t.bbox[0])
            .fold(f64::INFINITY, f64::min)
            - 0.5 * em;
        let between = if let Some(last) = active.last() {
            &text.rows[last + 1..index]
        } else {
            &[]
        };
        work.charge(between.iter().map(|r| r.tokens.len()).sum())?;
        let barrier = between.iter().any(|r| {
            r.tokens
                .iter()
                .any(|t| t.bbox[0] < prefix_limit && t.bbox[2] > prefix_limit + em)
        });
        if active.last().is_some_and(|last| {
            !matches || barrier || row.bbox[1] - text.rows[*last].bbox[3] > 22. * em
        }) {
            groups.push((std::mem::take(&mut active), anchors.clone()))
        }
        if active.is_empty() || positions.len() > anchors.len() {
            anchors = positions
        }
        active.push(index);
    }
    if !active.is_empty() {
        groups.push((active, anchors))
    }
    Ok(groups)
}
fn nearest(values: &[f64], target: f64) -> usize {
    values
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (**a - target)
                .abs()
                .partial_cmp(&(**b - target).abs())
                .unwrap()
        })
        .unwrap()
        .0
}
fn column_tracks(
    text: &Text,
    indices: &[usize],
    header: &[usize],
    width: f64,
    work: &mut Work,
) -> Result<Option<Vec<f64>>, Error> {
    let em = text.median_glyph_height;
    let values: Vec<Vec<_>> = indices
        .iter()
        .map(|i| text.rows[*i].tokens.iter().filter(|t| numeric(t)).collect())
        .collect();
    let count = values.iter().map(Vec::len).max().unwrap_or(0);
    let Some(anchors) = values.iter().find(|v| v.len() == count) else {
        return Ok(None);
    };
    if anchors.len() < 2 {
        return Ok(None);
    }
    let rights: Vec<_> = anchors.iter().map(|t| t.bbox[2]).collect();
    let mut buckets = vec![Vec::new(); anchors.len()];
    for row in &values {
        for token in row {
            work.charge(anchors.len())?;
            let owner = nearest(&rights, token.bbox[2]);
            if (token.bbox[2] - rights[owner]).abs() > 0.65 * em {
                return Ok(None);
            }
            buckets[owner].push(token.bbox)
        }
    }
    if buckets.iter().any(|b| b.len() < 2) {
        return Ok(None);
    }
    let first = buckets[0]
        .iter()
        .map(|b| b[0])
        .fold(f64::INFINITY, f64::min);
    let mut prefix: Vec<_> = text.rows[indices[0]..]
        .iter()
        .flat_map(|r| &r.tokens)
        .filter(|t| t.bbox[2] < first && !numeric(t))
        .map(|t| t.bbox[2])
        .collect();
    if prefix.is_empty() && indices[0] > 0 {
        prefix = text.rows[indices[0].saturating_sub(2)..indices[0]]
            .iter()
            .flat_map(|r| &r.tokens)
            .filter(|t| t.bbox[2] < first && !numeric(t))
            .map(|t| t.bbox[2])
            .collect()
    }
    let Some(left) = prefix.into_iter().max_by(f64::total_cmp) else {
        return Ok(None);
    };
    if left >= first {
        return Ok(None);
    }
    let mut tracks = vec![0., (left + first) / 2.];
    for p in buckets.windows(2) {
        let low = p[0].iter().map(|b| b[2]).fold(f64::NEG_INFINITY, f64::max);
        let high = p[1].iter().map(|b| b[0]).fold(f64::INFINITY, f64::min);
        if high - low < 0.3 * em {
            return Ok(None);
        }
        tracks.push((low + high) / 2.)
    }
    let centers: Vec<_> = anchors
        .iter()
        .map(|t| (t.bbox[0] + t.bbox[2]) / 2.)
        .collect();
    let mut header_cols = vec![Vec::new(); anchors.len()];
    for i in header {
        let mut slots = BTreeMap::<usize, Vec<[f64; 4]>>::new();
        for token in &text.rows[*i].tokens {
            let c = (token.bbox[0] + token.bbox[2]) / 2.;
            if c < tracks[1] || c > rights[rights.len() - 1] + em {
                continue;
            }
            work.charge(centers.len())?;
            slots
                .entry(nearest(&centers, c))
                .or_default()
                .push(token.bbox)
        }
        if slots.len() == anchors.len() && slots.values().all(|v| v.len() == 1) {
            for (owner, items) in slots {
                header_cols[owner].extend(items)
            }
        }
    }
    for i in 0..anchors.len() - 1 {
        let low = buckets[i]
            .iter()
            .chain(&header_cols[i])
            .map(|b| b[2])
            .fold(f64::NEG_INFINITY, f64::max);
        let high = buckets[i + 1]
            .iter()
            .chain(&header_cols[i + 1])
            .map(|b| b[0])
            .fold(f64::INFINITY, f64::min);
        if high - low >= 0.3 * em {
            tracks[i + 2] = (low + high) / 2.
        }
    }
    let last = buckets
        .last()
        .unwrap()
        .iter()
        .map(|b| b[2])
        .fold(f64::NEG_INFINITY, f64::max);
    let mut extra: Vec<_> = header
        .iter()
        .flat_map(|i| &text.rows[*i].tokens)
        .filter(|t| t.bbox[0] > last + 0.5 * em)
        .collect();
    extra.sort_by(|a, b| a.bbox[0].partial_cmp(&b.bbox[0]).unwrap());
    let mut extra_centers = Vec::new();
    for token in extra {
        let c = (token.bbox[0] + token.bbox[2]) / 2.;
        if extra_centers.last().is_none_or(|last| c - last > 2. * em) {
            extra_centers.push(c)
        }
    }
    if !extra_centers.is_empty() {
        let candidates: Vec<_> = text.rows[header.len()..]
            .iter()
            .flat_map(|r| &r.tokens)
            .filter(|t| t.bbox[0] > last + 0.5 * em)
            .collect();
        if candidates.is_empty() {
            return Ok(None);
        }
        let first = candidates
            .iter()
            .map(|t| t.bbox[0])
            .fold(f64::INFINITY, f64::min);
        tracks.push((last + first) / 2.);
        for i in 0..extra_centers.len() - 1 {
            let mut b = [Vec::new(), Vec::new()];
            for token in &candidates {
                work.charge(extra_centers.len())?;
                let owner = nearest(&extra_centers, (token.bbox[0] + token.bbox[2]) / 2.);
                if owner == i || owner == i + 1 {
                    b[owner - i].push(token.bbox)
                }
            }
            if b.iter().any(Vec::is_empty) {
                return Ok(None);
            }
            let low = b[0].iter().map(|b| b[2]).fold(f64::NEG_INFINITY, f64::max);
            let high = b[1].iter().map(|b| b[0]).fold(f64::INFINITY, f64::min);
            if high - low < 0.3 * em {
                return Ok(None);
            }
            tracks.push((low + high) / 2.);
        }
    }
    tracks.push(width);
    Ok(Some(tracks))
}
fn section(s: &str) -> bool {
    let c: Vec<_> = s.chars().collect();
    let Some(first) = c.first() else { return false };
    let chinese = "一二三四五六七八九十";
    let n = if chinese.contains(*first) {
        c.iter().take_while(|c| chinese.contains(**c)).count()
    } else {
        c.iter()
            .take_while(|c| python_decimal(**c).is_some())
            .count()
    };
    n > 0 && c.get(n).is_some_and(|c| ['、', '.'].contains(c))
}
fn unit(s: &str) -> bool {
    let s = s
        .strip_suffix('\n')
        .unwrap_or(s)
        .rsplit('\n')
        .next()
        .unwrap_or("");
    let c: Vec<_> = s.chars().collect();
    c.len() >= 3
        && c.last().is_some_and(|c| ['）', ')'].contains(c))
        && c[..c.len() - 2].iter().any(|c| ['（', '('].contains(c))
}
fn time(s: &str) -> bool {
    let c: Vec<_> = s.chars().collect();
    (0..c.len()).any(|i| {
        if python_decimal(c[i]).is_none() {
            return false;
        }
        for n in [1, 2] {
            if c.get(i..i + n)
                .is_some_and(|a| a.iter().all(|c| python_decimal(*c).is_some()))
                && c.get(i + n).is_some_and(|c| ['：', ':'].contains(c))
                && c.get(i + n + 1..i + n + 3)
                    .is_some_and(|a| a.iter().all(|c| python_decimal(*c).is_some()))
            {
                return true;
            }
        }
        false
    })
}
fn body_rows(
    text: &Text,
    first: usize,
    tracks: &[f64],
    work: &mut Work,
) -> Result<Vec<Vec<usize>>, Error> {
    let em = text.median_glyph_height;
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut pending: Vec<usize> = Vec::new();
    for index in first..text.rows.len() {
        let row = &text.rows[index];
        work.charge(row.tokens.iter().map(|t| t.text.len() + 1).sum())?;
        let num = row
            .tokens
            .iter()
            .any(|t| numeric(t) && t.bbox[0] >= tracks[1]);
        let label = row.tokens.iter().any(|t| t.bbox[2] <= tracks[1]);
        let only = label && !num;
        if !pending.is_empty() && (num || !only) {
            let section = pending
                .iter()
                .flat_map(|i| &text.rows[*i].tokens)
                .any(|t| section(&t.text));
            let isunit = label
                && pending.len() == 1
                && text.rows[pending[0]].tokens.iter().any(|t| {
                    unit(&t.text)
                        && t.bbox[2] <= tracks[1]
                        && row
                            .tokens
                            .iter()
                            .filter(|t| t.bbox[2] <= tracks[1])
                            .map(|t| t.bbox[0])
                            .fold(f64::INFINITY, f64::min)
                            > t.bbox[0] + 0.5 * em
                });
            let gap = row.bbox[1] - text.rows[*pending.last().unwrap()].bbox[3];
            if gap > em || section && label && gap > 0.5 * em || isunit && gap > 0.25 * em {
                groups.push(std::mem::take(&mut pending))
            }
            pending.push(index);
            groups.push(std::mem::take(&mut pending));
        } else if num {
            groups.push(vec![index])
        } else if only {
            if !groups.is_empty() && pending.is_empty() {
                let previous = groups.last().unwrap();
                let bottom = previous
                    .iter()
                    .flat_map(|i| &text.rows[*i].tokens)
                    .filter(|t| numeric(t))
                    .map(|t| t.bbox[3])
                    .max_by(f64::total_cmp)
                    .unwrap_or(-1.);
                if row.bbox[1] < bottom - 0.1 * em {
                    let prior: Vec<_> = previous
                        .iter()
                        .flat_map(|i| &text.rows[*i].tokens)
                        .filter(|t| t.bbox[2] <= tracks[1])
                        .collect();
                    if !prior.is_empty()
                        && row.bbox[0]
                            > prior
                                .iter()
                                .map(|t| t.bbox[0])
                                .fold(f64::INFINITY, f64::min)
                                + 0.5 * em
                    {
                        groups.last_mut().unwrap().push(index);
                        continue;
                    }
                }
            }
            if !pending.is_empty() {
                let prior = pending
                    .iter()
                    .flat_map(|i| &text.rows[*i].tokens)
                    .any(|t| section(&t.text));
                let gap = row.bbox[1] - text.rows[*pending.last().unwrap()].bbox[3];
                if gap > em || prior && gap > 0.5 * em {
                    groups.push(std::mem::take(&mut pending))
                }
            }
            pending.push(index)
        } else if let Some(last) = groups.last_mut() {
            last.push(index)
        }
        work.charge(pending.iter().map(|i| text.rows[*i].tokens.len()).sum())?;
    }
    if !pending.is_empty() {
        groups.push(pending)
    }
    Ok(groups)
}
fn build(
    input: &Value,
    text: &Text,
    groups: &[NumericGroup],
    work: &mut Work,
) -> Result<Option<Value>, Error> {
    let b = input["table_bbox"]
        .as_array()
        .filter(|a| a.len() == 4)
        .ok_or_else(invalid)?;
    let b = [
        number(&b[0])?,
        number(&b[1])?,
        number(&b[2])?,
        number(&b[3])?,
    ];
    let a = input.get("angle").map_or(Ok(0.), number)?.trunc() as i64;
    let (w, h) = if [90, 270].contains(&a.rem_euclid(360)) {
        (b[3] - b[1], b[2] - b[0])
    } else {
        (b[2] - b[0], b[3] - b[1])
    };
    let rules = local_rules(input, w, h)?;
    if rules
        .iter()
        .filter(|r| r.orientation == "horizontal")
        .count()
        < 2
    {
        return Ok(None);
    }
    let nh = rules
        .iter()
        .filter(|r| r.orientation == "horizontal" && r.end - r.start >= 0.8 * w)
        .count();
    let nv = rules
        .iter()
        .filter(|r| r.orientation == "vertical" && r.end - r.start >= 0.6 * h)
        .count();
    if nh >= 4 || nh >= 3 && nv >= 3 {
        return Ok(None);
    }
    if groups.len() != 1 || groups[0].0.len() < 2 || groups[0].1.len() > 4 {
        return Ok(None);
    }
    let (indices, anchors) = (&groups[0].0, &groups[0].1);
    let first = indices[0];
    if first == 0 || first > 8 {
        return Ok(None);
    }
    let mut header: Vec<_> = (0..first).collect();
    if text
        .glyphs
        .iter()
        .filter(|g| g.visual_row < first as i64)
        .any(|g| {
            !g.text.is_empty()
                && g.text.chars().all(python_digit)
                && g.bbox[3] - g.bbox[1] < 0.65 * text.median_glyph_height
        })
    {
        return Ok(None);
    }
    let Some(tracks) = column_tracks(text, indices, &header, w, work)? else {
        return Ok(None);
    };
    if header.iter().any(|i| {
        text.rows[*i]
            .tokens
            .iter()
            .filter(|t| t.bbox[2] <= tracks[1])
            .count()
            >= 2
    }) && indices
        .iter()
        .filter(|i| {
            text.rows[**i]
                .tokens
                .iter()
                .filter(|t| t.bbox[2] <= tracks[1])
                .count()
                >= 2
        })
        .count()
        >= 3
    {
        return Ok(None);
    }
    let labels: String = text
        .glyphs
        .iter()
        .filter(|g| g.bbox[0] < tracks[1])
        .map(|g| g.text.as_str())
        .collect();
    if time(&labels) {
        return Ok(None);
    }
    let em = text.median_glyph_height;
    if !header.iter().any(|i| text.rows[*i].tokens.len() >= 2) {
        return Ok(None);
    }
    let end = header
        .iter()
        .filter(|i| text.rows[**i].tokens.iter().any(|t| t.bbox[0] >= tracks[1]))
        .map(|i| i + 1)
        .max()
        .unwrap_or(first);
    header = (0..end).collect();
    let body = body_rows(text, end, &tracks, work)?;
    if body.len() < 2 {
        return Ok(None);
    }
    let header_labels: Vec<_> = header
        .iter()
        .copied()
        .filter(|i| text.rows[*i].tokens.iter().all(|t| t.bbox[2] <= tracks[1]))
        .collect();
    header.retain(|i| !header_labels.contains(i));
    let mut head: Vec<_> = header.iter().map(|i| vec![*i]).collect();
    if header
        .windows(2)
        .any(|p| text.rows[p[0]].bbox[3] > text.rows[p[1]].bbox[1] + 0.4 * em)
    {
        head = vec![header.clone()]
    }
    let head_count = head.len();
    let logical: Vec<_> = head.iter().chain(&body).cloned().collect();
    let number_cols = anchors.len() + 1;
    if tracks.len() <= number_cols {
        return Err(invalid());
    }
    let mut boundaries = vec![0.];
    for p in logical.windows(2) {
        let a: Vec<_> = p[0]
            .iter()
            .flat_map(|i| {
                text.rows[*i]
                    .tokens
                    .iter()
                    .filter(|t| header.contains(i) || t.bbox[2] <= tracks[number_cols])
            })
            .collect();
        let b: Vec<_> = p[1]
            .iter()
            .flat_map(|i| {
                text.rows[*i]
                    .tokens
                    .iter()
                    .filter(|t| header.contains(i) || t.bbox[2] <= tracks[number_cols])
            })
            .collect();
        if a.is_empty() || b.is_empty() {
            return Err(invalid());
        }
        let low = a
            .iter()
            .map(|t| t.bbox[3])
            .fold(f64::NEG_INFINITY, f64::max);
        let high = b.iter().map(|t| t.bbox[1]).fold(f64::INFINITY, f64::min);
        if low > high + 0.4 * em {
            return Ok(None);
        }
        boundaries.push((low + high) / 2.)
    }
    boundaries.push(h);
    let cols = tracks.len() - 1;
    if !(4..=5).contains(&cols) {
        return Ok(None);
    }
    let mut spans = BTreeMap::new();
    if !header_labels
        .iter()
        .flat_map(|i| &text.rows[*i].tokens)
        .collect::<Vec<_>>()
        .is_empty()
    {
        spans.insert((0, 0), (head_count, 1));
    }
    for (row, members) in head.iter().enumerate() {
        for token in members.iter().flat_map(|i| &text.rows[*i].tokens) {
            work.charge(cols + rules.len())?;
            let crossed: Vec<_> = (0..cols)
                .filter(|c| token.bbox[0] < tracks[c + 1] && token.bbox[2] > tracks[*c])
                .collect();
            if crossed.len() == 2
                && ((token.bbox[0] + token.bbox[2]) / 2. - tracks[crossed[1]]).abs() <= 0.65 * em
            {
                spans.insert((row, crossed[0]), (1, 2));
            }
            let next = if row + 1 < logical.len() {
                text.rows[logical[row + 1][0]].bbox[1] + 0.25 * em
            } else {
                h
            };
            let matching = rules
                .iter()
                .filter(|r| {
                    r.orientation == "horizontal"
                        && token.bbox[3] - 0.2 * em <= r.coordinate
                        && r.coordinate <= next
                        && r.start <= token.bbox[0]
                        && r.end >= token.bbox[2]
                        && r.end - r.start < 0.85 * w
                        && ((r.start + r.end - token.bbox[0] - token.bbox[2]) / 2.).abs()
                            <= 0.75 * em
                })
                .min_by(|a, b| (a.end - a.start).partial_cmp(&(b.end - b.start)).unwrap());
            if let Some(r) = matching {
                let covered: Vec<_> = (1..cols)
                    .filter(|c| {
                        r.start - em <= (tracks[*c] + tracks[c + 1]) / 2.
                            && (tracks[*c] + tracks[c + 1]) / 2. <= r.end + em
                    })
                    .collect();
                if covered.len() > 1
                    && covered.len() < cols
                    && covered.windows(2).all(|p| p[1] == p[0] + 1)
                {
                    spans.insert((row, covered[0]), (1, covered.len()));
                }
            }
        }
    }
    for col in number_cols..cols {
        if body.iter().any(|rows| {
            rows.iter().flat_map(|i| &text.rows[*i].tokens).any(|t| {
                tracks[col] <= (t.bbox[0] + t.bbox[2]) / 2.
                    && (t.bbox[0] + t.bbox[2]) / 2. <= tracks[col + 1]
            })
        }) {
            spans.insert((head_count, col), (body.len(), 1));
        }
    }
    if logical.len() * cols > rapiddoc_core::resources::limit(2000) {
        return Err(limit());
    }
    let mut specs = Vec::new();
    let mut occupied = BTreeSet::new();
    for row in 0..logical.len() {
        for col in 0..cols {
            if occupied.contains(&(row, col)) {
                continue;
            }
            let (rs, cs) = spans.get(&(row, col)).copied().unwrap_or((1, 1));
            if row + rs >= boundaries.len() || col + cs >= tracks.len() {
                return Err(invalid());
            }
            for r in row..row + rs {
                for c in col..col + cs {
                    occupied.insert((r, c));
                }
            }
            specs.push(json!({"row":row,"col":col,"rowspan":rs,"colspan":cs,"bbox":[tracks[col],boundaries[row],tracks[col+cs],boundaries[row+rs]]}));
        }
    }
    work.charge(text.glyphs.len() * specs.len() * 3 + text.glyphs.len() * text.glyphs.len() * 2)?;
    let result = table_native_candidate::build(
        &json!({"source":"sparse_multiline","rows":logical.len(),"cols":cols,"specs":specs,"text":text,"structure_support":1.,"row_stability":1.,"column_stability":1.,"issues":["evidence=aligned_numeric_grid"],"use_grid_index":true}),
    )?;
    let candidate = result["candidate"].clone();
    if candidate.is_null() {
        return Ok(None);
    }
    let cells = candidate["cells"].as_array().ok_or_else(invalid)?;
    for i in header {
        for token in &text.rows[i].tokens {
            let sources: BTreeSet<_> = token.source_char_indices.iter().copied().collect();
            let mut owners = 0;
            for c in cells {
                let ids = c["source_char_indices"].as_array().ok_or_else(invalid)?;
                work.charge(ids.len())?;
                if ids
                    .iter()
                    .any(|v| v.as_i64().is_some_and(|n| sources.contains(&n)))
                {
                    owners += 1
                }
            }
            if owners > 1 {
                return Ok(None);
            }
        }
    }
    Ok(Some(candidate))
}
fn read_text(v: &Value) -> Result<Text, Error> {
    let text = validate_text(v)?;
    if text.median_glyph_height > 1e9
        || text.median_glyph_width.abs() > 1e9
        || text.glyphs.iter().any(|g| {
            g.spacing_bbox
                .is_some_and(|b| b.iter().any(|n| !n.is_finite() || n.abs() > 1e9))
        })
    {
        return Err(limit());
    }
    Ok(text)
}
pub fn regroup(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let text = read_text(&v["text"])?;
    Ok(json!(regroup_text(&text, &mut Work(0))?))
}
/// 页面候选编排的完整旧numeric_region_bounds；只复用组行，不在整页提前构建候选。
pub(in crate::processing) fn regions(
    input: &Value,
    original: &Text,
    work: &mut Work,
) -> Result<Vec<[f64; 4]>, Error> {
    let text = regroup_text(original, work)?;
    let groups = row_groups(&text, work)?;
    let w = number(&input["page_size"][0])?;
    let h = number(&input["page_size"][1])?;
    let rules = local_rules(input, w, h)?;
    let em = text.median_glyph_height;
    let mut out = Vec::new();
    for (rows, anchors) in groups {
        if rows.len() < 3 || anchors.len() < 3 {
            continue;
        }
        work.charge(
            text.rows.iter().map(|r| r.tokens.len() + 1).sum::<usize>() * 4 + rules.len() * 6 + 1,
        )?;
        let first = rows[0];
        let mut last = *rows.last().unwrap();
        let minimum = rows
            .iter()
            .flat_map(|i| text.rows[*i].tokens.iter())
            .filter(|t| numeric(t))
            .map(|t| t.bbox[0])
            .reduce(f64::min)
            .ok_or_else(invalid)?;
        let labels = text.rows[first..=last]
            .iter()
            .flat_map(|r| r.tokens.iter())
            .filter(|t| t.bbox[2] < minimum && !numeric(t))
            .collect::<Vec<_>>();
        if labels.is_empty() || anchors.len() < 2 {
            continue;
        }
        let mut left = labels.iter().map(|t| t.bbox[0]).reduce(f64::min).unwrap();
        let prefix = (labels.iter().map(|t| t.bbox[2]).reduce(f64::max).unwrap() + minimum) / 2.;
        let mut right = anchors.iter().copied().reduce(f64::max).unwrap();
        let mut top = first;
        while top > 0 {
            let prior = &text.rows[top - 1];
            let current = &text.rows[top];
            if current.bbox[1] - prior.bbox[3] > 5. * em
                || prior
                    .tokens
                    .iter()
                    .any(|t| t.bbox[0] < left - 3. * em || t.bbox[2] > right + em)
                || prior
                    .tokens
                    .iter()
                    .any(|t| t.bbox[0] < prefix - em && t.bbox[2] > prefix + em)
                || (prior.tokens.len() == 1
                    && prior.bbox[0] >= minimum
                    && prior.bbox[2] - prior.bbox[0] > 8. * em)
            {
                break;
            }
            top -= 1;
        }
        let mut header = &text.rows[top..first];
        for t in header
            .iter()
            .flat_map(|r| &r.tokens)
            .filter(|t| t.bbox[2] < minimum)
        {
            left = left.min(t.bbox[0]);
        }
        if let Some(multi) = header.iter().find(|r| r.tokens.len() >= 2) {
            let boundary = rules
                .iter()
                .filter(|r| {
                    r.orientation == "horizontal"
                        && r.start <= minimum
                        && r.end >= right - em
                        && 0. <= multi.bbox[1] - r.coordinate
                        && multi.bbox[1] - r.coordinate <= em
                })
                .map(|r| r.coordinate)
                .reduce(f64::max);
            if let Some(boundary) = boundary {
                top = header
                    .iter()
                    .find(|r| r.bbox[3] > boundary)
                    .ok_or_else(invalid)?
                    .row_index as usize;
                header = &text.rows[top..first];
            }
        }
        let numeric_width = right - minimum;
        let has_header = header
            .iter()
            .any(|r| r.tokens.len() >= 2 && r.tokens.iter().any(|t| !numeric(t)));
        let physical = rules
            .iter()
            .filter(|r| {
                r.orientation == "horizontal"
                    && r.end - r.start >= (3. * em).max(0.3 * numeric_width)
                    && r.start <= right
                    && r.end >= minimum
                    && text.rows[top].bbox[1] - em <= r.coordinate
                    && r.coordinate <= text.rows[last].bbox[3] + em
            })
            .count();
        if !has_header || physical < 2 {
            continue;
        }
        while last + 1 < text.rows.len() {
            let next = &text.rows[last + 1];
            if next.bbox[1] - text.rows[last].bbox[3] > 0.8 * em
                || next.tokens.iter().any(|t| t.bbox[2] >= minimum)
            {
                break;
            }
            last += 1;
        }
        for t in text.rows[top..=last].iter().flat_map(|r| &r.tokens) {
            right = right.max(t.bbox[2]);
        }
        let top_y = text.rows[top].bbox[1];
        let bottom_y = text.rows[last].bbox[3];
        let borders = rules
            .iter()
            .filter(|r| {
                r.orientation == "horizontal"
                    && r.start <= left + em
                    && r.end >= right - em
                    && top_y - em <= r.coordinate
                    && r.coordinate <= bottom_y + em
                    && r.end - r.start <= 1.15 * (right - left) + 2. * em
            })
            .collect::<Vec<_>>();
        for r in borders {
            left = left.min(r.start);
            right = right.max(r.end);
        }
        let near_top = rules
            .iter()
            .filter(|r| {
                r.orientation == "horizontal"
                    && r.start <= minimum
                    && r.end >= right - em
                    && 0. <= top_y - r.coordinate
                    && top_y - r.coordinate <= em
            })
            .map(|r| r.coordinate)
            .reduce(f64::min)
            .unwrap_or(top_y - 0.2 * em);
        let near_bottom = rules
            .iter()
            .filter(|r| {
                r.orientation == "horizontal"
                    && r.start <= minimum
                    && r.end >= right - em
                    && 0. <= r.coordinate - bottom_y
                    && r.coordinate - bottom_y <= em
            })
            .map(|r| r.coordinate)
            .reduce(f64::max)
            .unwrap_or(bottom_y + 0.2 * em);
        out.push([left - 0.2 * em, near_top, right + 0.2 * em, near_bottom]);
        if out.len() > rapiddoc_core::resources::limit(5000) {
            return Err(limit());
        }
    }
    Ok(out)
}
pub fn generate(v: &Value) -> Result<Value, Error> {
    generate_with_work(v, &mut Work(0))
}
/// 页面产生器只需区域确认，不生成旧检测器未消费的诊断HTML。
pub(in crate::processing) fn accepts(
    input: &Value,
    original: &Text,
    work: &mut Work,
) -> Result<bool, Error> {
    let text = regroup_text(original, work)?;
    let groups = row_groups(&text, work)?;
    Ok(build(input, &text, &groups, work)?.is_some())
}
pub(in crate::processing) fn generate_with_work(
    v: &Value,
    work: &mut Work,
) -> Result<Value, Error> {
    validate_tree(v)?;
    let original = read_text(&v["text"])?;
    let regroup = v
        .get("regroup")
        .map_or(Ok(true), |x| x.as_bool().ok_or_else(invalid))?;
    let text = if regroup {
        regroup_text(&original, work)?
    } else {
        original
    };
    if text
        .rows
        .iter()
        .enumerate()
        .any(|(i, r)| r.row_index != i as i64)
    {
        return Err(invalid());
    }
    let groups = row_groups(&text, work)?;
    let candidate = build(&v["table_input"], &text, &groups, work)?;
    let html = candidate
        .as_ref()
        .map(table_native_candidate::render)
        .transpose()?;
    Ok(json!({"text":text,"numeric_groups":groups,"candidate":candidate,"html":html}))
}
