//! 旧目录完整提取器；稳定行身份、Unicode 页码/标题、语义认领及 tight 输出。
use super::*;
use crate::legacy_model::path_graphics::{integer, overlap_smaller};
use rapiddoc_native::{
    python_decimal,
    text_boundary::{python_alnum, python_space},
};
use std::collections::{BTreeMap, BTreeSet};
fn text(l: &Value) -> Result<&str, Error> {
    l["text"].as_str().ok_or_else(invalid)
}
fn trim(s: &str) -> &str {
    s.trim_matches(python_space)
}
fn lower(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'İ' | 'ı' => 'i',
            'ſ' => 's',
            'K' => 'k',
            _ => c.to_ascii_lowercase(),
        })
        .collect()
}
fn heading(s: &str) -> bool {
    let s = lower(trim(s));
    let p: Vec<_> = s.split(python_space).filter(|s| !s.is_empty()).collect();
    p == ["contents"]
        || p == ["table", "of", "contents"]
        || s.chars().filter(|&c| !python_space(c)).collect::<String>() == "目录"
}
fn roman(c: char) -> bool {
    "ivxlcdmIVXLCDM".contains(c)
}
fn word(c: char) -> bool {
    python_alnum(c) || c == '_'
}
fn page(c: char) -> bool {
    c.is_ascii_digit() || ('０'..='９').contains(&c)
}
fn ends(s: &str) -> bool {
    let c: Vec<_> = s.trim_end_matches(python_space).chars().collect();
    let mut end = c.len();
    while end > 0 && (python_space(c[end - 1]) || ".。．、,，;；:：)）]】".contains(c[end - 1]))
    {
        end -= 1;
    }
    if end == 0 {
        return false;
    }
    let digit = page(c[end - 1]);
    if !digit && !roman(c[end - 1]) {
        return false;
    }
    let mut start = end;
    while start > 0
        && if digit {
            page(c[start - 1])
        } else {
            roman(c[start - 1])
        }
    {
        start -= 1;
    }
    start == 0 || !(c[start - 1].is_ascii_alphanumeric() || ('０'..='９').contains(&c[start - 1]))
}
fn leader(s: &str, minimum: usize) -> bool {
    let mut count = 0;
    for c in s.chars() {
        if ".．·".contains(c) {
            count += 1;
            if count >= minimum {
                return true;
            }
        } else if !python_space(c) {
            count = 0;
        }
    }
    false
}
fn only_page(s: &str, space: bool) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| python_decimal(c).is_some() || roman(c) || (space && python_space(c)))
}
fn digits(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| python_decimal(c).is_some())
}
fn numbered(s: &str) -> Option<(usize, &str)> {
    let mut end = 0;
    let mut n = 0usize;
    for (i, c) in s.char_indices() {
        if let Some(d) = python_decimal(c) {
            n = n.saturating_mul(10).saturating_add(d as usize);
            end = i + c.len_utf8();
        } else {
            break;
        }
    }
    if end == 0 || !s[end..].starts_with(['.', ')']) {
        return None;
    }
    let rest = s[end + 1..].trim_start_matches(python_space);
    let rest = rest.strip_suffix('\n').unwrap_or(rest);
    if rest.contains('\n') {
        return None;
    }
    if rest.chars().count() < 2 {
        return None;
    }
    Some((n, rest))
}
fn checked_numbered(s: &str) -> Result<Option<(usize, &str)>, Error> {
    let result = numbered(s);
    if result.is_some()
        && s.chars()
            .take_while(|&c| python_decimal(c).is_some())
            .count()
            > 4300
    {
        return Err(invalid());
    }
    Ok(result)
}
fn section(s: &str) -> bool {
    let s = lower(s);
    for prefix in ["part", "chapter", "section", "book", "volume", "vol."] {
        if let Some(rest) = s.strip_prefix(prefix) {
            let cut = rest.trim_start_matches(python_space);
            if cut.len() == rest.len() {
                continue;
            }
            let mut end = 0;
            let roman_branch = cut.chars().next().is_some_and(|c| "ivxlcdm".contains(c));
            for (i, c) in cut.char_indices() {
                if if roman_branch {
                    "ivxlcdm".contains(c)
                } else {
                    python_decimal(c).is_some()
                } {
                    end = i + c.len_utf8()
                } else {
                    break;
                }
            }
            if end > 0
                && cut[end..]
                    .chars()
                    .next()
                    .is_some_and(|c| ".：:".contains(c) || python_space(c))
            {
                return true;
            }
        }
    }
    false
}
fn starts_section(s: &str) -> bool {
    let s = lower(s);
    for p in ["part", "chapter", "section"] {
        if let Some(r) = s.strip_prefix(p) {
            if r.chars().next().is_none_or(|c| !word(c)) {
                return true;
            }
        }
    }
    let mut end = 0;
    for (i, c) in s.char_indices() {
        if python_decimal(c).is_some() {
            end = i + c.len_utf8()
        } else {
            break;
        }
    }
    end > 0 && s[end..].starts_with(['.', ')']) && s[end + 1..].chars().next().is_some_and(word)
}
fn numeric_token(s: &str) -> bool {
    let c: Vec<_> = s.chars().collect();
    let mut i = 0;
    if c.get(i).is_some_and(|c| "(（".contains(*c)) {
        i += 1
    }
    if c.get(i).is_some_and(|c| "+-".contains(*c)) {
        i += 1
    }
    if !c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
        return false;
    }
    i += 1;
    while c
        .get(i)
        .is_some_and(|c| python_decimal(*c).is_some() || *c == ',')
    {
        i += 1
    }
    if c.get(i) == Some(&'.') {
        i += 1;
        let start = i;
        while c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
            i += 1
        }
        if i == start {
            return false;
        }
    }
    if c.get(i).is_some_and(|c| ")）".contains(*c)) {
        i += 1
    }
    i == c.len()
}
pub(super) fn local(b: &Box4, sz: [f64; 2], angle: i64) -> Box4 {
    let [x, y, r, t] = b.n;
    let [w, h] = sz;
    if ![90, 180, 270].contains(&angle) {
        return b.clone();
    }
    let n = match angle {
        90 => [y, w - r, t, w - x],
        180 => [w - r, h - t, w - x, h - y],
        _ => [h - t, x, h - y, r],
    };
    Box4 {
        n,
        raw: n.map(|v| json!(v)),
    }
}
fn line_height(l: &Value, b: &Box4, w: &mut Work) -> Result<f64, Error> {
    w.charge(text(l)?.len() + 1)?;
    let mut h = height(l, b)?;
    if !leader(text(l)?, 3) {
        return Ok(h);
    }
    let mut hs = Vec::new();
    for ch in optional(l, "chars", 20000)? {
        w.charge(1)?;
        let s = ch["char"].as_str().unwrap_or("");
        if !s.is_empty() && s.chars().all(python_alnum) && !ch["bbox"].is_null() {
            let b = local(&bbox(&ch["bbox"])?, [1., 1.], integer(l, "angle", 0)?);
            if b.n[3] > b.n[1] {
                hs.push(b.n[3] - b.n[1]);
            }
        }
    }
    if !hs.is_empty() {
        h = h.max(median(hs));
    }
    Ok(h)
}
#[derive(Clone)]
struct Row {
    identity: usize,
    m: Vec<usize>,
    boxes: Vec<Box4>,
    b: Box4,
    content: String,
    page: bool,
}
fn row(
    m: Vec<usize>,
    boxes: Vec<Box4>,
    lines: &[Value],
    identity: usize,
    skip_empty: bool,
) -> Result<Row, Error> {
    let b = combined(&boxes)?;
    let mut pieces = Vec::new();
    for &i in &m {
        let t = trim(text(&lines[i])?);
        if !skip_empty || !t.is_empty() {
            pieces.push(t)
        }
    }
    let content = pieces.join(" ");
    let page = ends(&content);
    Ok(Row {
        identity,
        m,
        boxes,
        b,
        content,
        page,
    })
}
fn cy(b: &Box4) -> f64 {
    (b.n[1] + b.n[3]) / 2.
}
fn em(r: &Row, ls: &[Value], w: &mut Work) -> Result<f64, Error> {
    let mut hs = Vec::new();
    for (&i, b) in r.m.iter().zip(&r.boxes) {
        hs.push(line_height(&ls[i], b, w)?);
    }
    Ok(median(hs))
}
fn rows(ls: &[Value], sz: [f64; 2], a: i64, w: &mut Work) -> Result<Vec<Row>, Error> {
    let mut groups: Vec<(Vec<usize>, Vec<Box4>)> = Vec::new();
    let mut by_key = BTreeMap::new();
    for (i, l) in ls.iter().enumerate() {
        w.charge(1)?;
        if integer(l, "angle", 0)? != a || !l["semantic_type"].is_null() {
            continue;
        }
        let b = local(&bbox(&l["bbox"])?, sz, a);
        if b.n[3] - b.n[1] > 1.7 * line_height(l, &b, w)? {
            continue;
        }
        let key = if let Some(id) = l.get("visual_row_id").filter(|v| !v.is_null()) {
            (0, id.as_i64().ok_or_else(invalid)?)
        } else {
            (1, integer(l, "source_index", 0)?)
        };
        let next = groups.len();
        let group = *by_key.entry(key).or_insert(next);
        if group == next {
            groups.push((Vec::new(), Vec::new()));
        }
        groups[group].0.push(i);
        groups[group].1.push(b);
    }
    let mut rs = Vec::new();
    for (m, bs) in groups {
        w.charge(m.len() * 16 + 1)?;
        let mut pairs: Vec<_> = m.into_iter().zip(bs).collect();
        pairs.sort_by(|(i, b), (j, c)| {
            b.n[0]
                .partial_cmp(&c.n[0])
                .unwrap()
                .then(
                    integer(&ls[*i], "run_index", 0)
                        .unwrap_or(0)
                        .cmp(&integer(&ls[*j], "run_index", 0).unwrap_or(0)),
                )
                .then(
                    ls[*i]["source_index"]
                        .as_i64()
                        .cmp(&ls[*j]["source_index"].as_i64()),
                )
        });
        let (m, bs) = pairs.into_iter().unzip();
        let r = row(m, bs, ls, rs.len(), true)?;
        if !r.content.is_empty() {
            rs.push(r);
        }
    }
    w.charge(rs.len() * 16 + 1)?;
    rs.sort_by(|a, b| {
        a.b.n[1]
            .partial_cmp(&b.b.n[1])
            .unwrap()
            .then(a.b.n[0].partial_cmp(&b.b.n[0]).unwrap())
            .then(
                a.m.iter()
                    .map(|&i| integer(&ls[i], "source_index", 0).unwrap())
                    .min()
                    .cmp(
                        &b.m.iter()
                            .map(|&i| integer(&ls[i], "source_index", 0).unwrap())
                            .min(),
                    ),
            )
    });
    let mut out: Vec<Row> = Vec::new();
    for r in rs {
        if let Some(p) = out.last() {
            let h = (p.b.n[3] - p.b.n[1]).max(r.b.n[3] - r.b.n[1]);
            if (cy(&r.b) - cy(&p.b)).abs() <= 0.35 * h
                && (r.b.n[0] >= p.b.n[2] || p.b.n[0] >= r.b.n[2])
            {
                let mut pairs: Vec<_> =
                    p.m.iter()
                        .copied()
                        .chain(r.m)
                        .zip(p.boxes.iter().cloned().chain(r.boxes))
                        .collect();
                pairs.sort_by(|(_, a), (_, b)| a.n[0].partial_cmp(&b.n[0]).unwrap());
                let (m, bs) = pairs.into_iter().unzip();
                let id = p.identity;
                *out.last_mut().unwrap() = row(m, bs, ls, id, false)?;
                continue;
            }
        }
        out.push(r);
    }
    for (i, r) in out.iter_mut().enumerate() {
        r.identity = i;
    }
    Ok(out)
}
fn join(rs: &[Row], ls: &[Value], w: &mut Work) -> Result<Vec<Row>, Error> {
    let mut out: Vec<Row> = Vec::new();
    for r in rs {
        w.charge(1)?;
        if let Some(p) = out.last() {
            let h = em(p, ls, w)?;
            let last = *p.m.last().unwrap();
            let label = &ls[p.m[0]];
            if p.page
                && !r.page
                && p.m.len() >= 2
                && only_page(trim(text(&ls[last])?), false)
                && p.boxes.last().unwrap().n[0] > p.b.n[0] + 8. * h
                && !starts_section(&r.content)
                && !label["font_signature"].is_null()
                && r.m.iter().all(|&i| {
                    crate::legacy_page::eq(&ls[i]["font_signature"], &label["font_signature"])
                })
                && (0.9..=1.1).contains(&(em(r, ls, w)? / h))
                && (r.b.n[0] - p.b.n[0]).abs() <= 0.5 * h
                && (-0.25 * h..=0.75 * h).contains(&(r.b.n[1] - p.b.n[3]))
                && r.b.n[2] < p.boxes.last().unwrap().n[0] - h
            {
                let m = p.m[..p.m.len() - 1]
                    .iter()
                    .copied()
                    .chain(r.m.iter().copied())
                    .chain([last])
                    .collect();
                let bs = p.boxes[..p.boxes.len() - 1]
                    .iter()
                    .cloned()
                    .chain(r.boxes.iter().cloned())
                    .chain([p.boxes.last().unwrap().clone()])
                    .collect();
                let mut merged = row(m, bs, ls, p.identity, false)?;
                merged.page = true;
                *out.last_mut().unwrap() = merged;
                continue;
            }
        }
        out.push(r.clone());
    }
    Ok(out)
}
fn unpaged(rs: &[Row], ls: &[Value], w: &mut Work) -> Result<Vec<(Row, Vec<Row>)>, Error> {
    let mut out = Vec::new();
    for (p, head) in rs.iter().enumerate().take(rs.len().saturating_sub(1)) {
        if !heading(&head.content) {
            continue;
        }
        let first = &rs[p + 1];
        let h = em(first, ls, w)?;
        let font = &ls[first.m[0]]["font_signature"];
        if checked_numbered(&first.content)?.is_none_or(|(n, _)| n != 1)
            || h <= 0.
            || font.is_null()
            || !(0.0..=6. * h).contains(&(first.b.n[1] - head.b.n[3]))
            || !(1.1..=3.).contains(&(em(head, ls, w)? / h))
        {
            continue;
        }
        let mut candidate: Vec<Row> = Vec::new();
        for r in &rs[p + 1..] {
            w.charge(1)?;
            let rh = em(r, ls, w)?;
            if digits(&r.content) && rh < 0.5 * h {
                continue;
            }
            let Some((n, t)) = checked_numbered(&r.content)? else {
                break;
            };
            let ascii_words = t
                .split(|c: char| !c.is_ascii_alphabetic())
                .filter(|s| !s.is_empty())
                .count();
            if n != candidate.len() + 1
                || r.page
                || ascii_words > 14
                || t.chars()
                    .filter(|c| ('\u{4e00}'..='\u{9fff}').contains(c))
                    .count()
                    > 50
                || t.trim_end_matches(python_space)
                    .chars()
                    .last()
                    .is_some_and(|c| ".!?。！？".contains(c))
                || !(0.9..=1.1).contains(&(rh / h))
                || r.m
                    .iter()
                    .any(|&i| !crate::legacy_page::eq(&ls[i]["font_signature"], font))
                || (r.b.n[0] - first.b.n[0]).abs() > 0.5 * h
                || candidate
                    .last()
                    .is_some_and(|p| !(0.8 * h..=3. * h).contains(&(cy(&r.b) - cy(&p.b))))
            {
                break;
            }
            candidate.push(r.clone());
        }
        if candidate.len() >= 5 {
            out.push((head.clone(), candidate));
        }
    }
    Ok(out)
}
fn bands(rs: &[Row], ls: &[Value], w: &mut Work) -> Result<Vec<Vec<Row>>, Error> {
    if rs.is_empty() {
        return Ok(Vec::new());
    }
    let mut hs = Vec::new();
    for r in rs {
        let mut high = f64::NEG_INFINITY;
        for &i in &r.m {
            high = high.max(line_height(&ls[i], &r.b, w)?);
        }
        hs.push(high)
    }
    let max = 3.25 * median(hs).max(0.1);
    let mut out = vec![vec![rs[0].clone()]];
    for r in &rs[1..] {
        if cy(&r.b) - cy(&out.last().unwrap().last().unwrap().b) > max {
            out.push(Vec::new());
        }
        out.last_mut().unwrap().push(r.clone());
    }
    Ok(out)
}
fn trim_band(rs: &[Row]) -> Vec<Row> {
    let mut start = 0;
    let mut end = rs.len();
    while start < end && !rs[start].page {
        start += 1
    }
    while end > start && (!rs[end - 1].page || only_page(&rs[end - 1].content, true)) {
        end -= 1;
    }
    rs[start..end].to_vec()
}
fn find_heading<'a>(all: &'a [Row], rs: &[Row], width: f64) -> Option<&'a Row> {
    let first = rs.first()?;
    let pos = all.iter().position(|r| r.identity == first.identity)?;
    if pos == 0 {
        return None;
    }
    let h = &all[pos - 1];
    if h.page {
        return None;
    }
    let em = median(rs.iter().map(|r| (r.b.n[3] - r.b.n[1]).max(0.1)).collect());
    let centered = ((h.b.n[0] + h.b.n[2]) / 2. - 0.5 * width).abs() <= 0.15 * width;
    let pitch = cy(&rs[0].b) - cy(&h.b);
    (h.b.n[2] - h.b.n[0] <= 0.4 * width
        && (centered || heading(&h.content))
        && (1.25 * em..=if heading(&h.content) {
            12. * em
        } else {
            5. * em
        })
            .contains(&pitch))
    .then_some(h)
}
fn stable(rs: &[Row], width: f64, explicit: bool) -> bool {
    if rs.len() < 5 || width <= 0. {
        return false;
    }
    if !explicit {
        let numeric = rs
            .iter()
            .filter(|r| {
                r.content
                    .split(python_space)
                    .filter(|s| !s.is_empty() && numeric_token(s))
                    .count()
                    >= 3
            })
            .count();
        if numeric as f64 >= 3f64.max(0.5 * rs.len() as f64) {
            return false;
        }
    }
    let pages: Vec<_> = rs.iter().filter(|r| r.page).collect();
    if pages.len() < 4usize.max((0.7 * rs.len() as f64).ceil() as usize) {
        return false;
    }
    let h = median(rs.iter().map(|r| (r.b.n[3] - r.b.n[1]).max(0.1)).collect());
    let right = median(pages.iter().map(|r| r.b.n[2]).collect());
    let tolerance = (1.5 * h).max(0.02 * width);
    if right < (if explicit { 0.5 } else { 0.75 }) * width
        || pages
            .iter()
            .filter(|r| (r.b.n[2] - right).abs() <= tolerance)
            .count() as f64
            / (pages.len() as f64)
            < 0.7
    {
        return false;
    }
    let wide = rs
        .iter()
        .filter(|r| r.b.n[2] - r.b.n[0] >= (if explicit { 0.4 } else { 0.55 }) * width)
        .count() as f64
        / rs.len() as f64;
    let sidecars = rs
        .iter()
        .filter(|r| {
            r.m.len() >= 2
                && r.boxes.last().is_some_and(|b| {
                    b.n[0] >= 0.75 * width && b.n[2] - b.n[0] <= (0.12 * width).max(4. * h)
                })
        })
        .count();
    if wide < 0.7 && sidecars < 2 {
        return false;
    }
    if rs.iter().filter(|r| r.b.n[0] <= 0.3 * width).count() as f64 / (rs.len() as f64) < 0.7 {
        return false;
    }
    let pitches: Vec<_> = rs.windows(2).map(|r| cy(&r[1].b) - cy(&r[0].b)).collect();
    if pitches.is_empty() || pitches.iter().any(|&p| p <= 0.) {
        return false;
    }
    let pitch = median(pitches.clone());
    pitches
        .iter()
        .filter(|&&p| p >= 0.55 * pitch && p <= 1.75 * pitch)
        .count() as f64
        / pitches.len() as f64
        >= 0.7
}
fn leading(all: &[Row], rs: Vec<Row>, ls: &[Value], w: &mut Work) -> Result<Vec<Row>, Error> {
    let Some(first) = rs.first() else {
        return Ok(rs);
    };
    let pos = all
        .iter()
        .position(|r| r.identity == first.identity)
        .unwrap_or(0);
    if pos == 0 {
        return Ok(rs);
    }
    let p = &all[pos - 1];
    if p.page || !section(&p.content) {
        return Ok(rs);
    }
    let font = &ls[p.m[0]]["font_signature"];
    let h = em(p, ls, w)?;
    if font.is_null() || h <= 0. || !(0.25 * h..=3. * h).contains(&(first.b.n[1] - p.b.n[3])) {
        return Ok(rs);
    }
    let mut peers = 0;
    for r in &rs {
        if !r.page
            && section(&r.content)
            && crate::legacy_page::eq(&ls[r.m[0]]["font_signature"], font)
            && (0.9..=1.1).contains(&(em(r, ls, w)? / h))
            && (r.b.n[0] - p.b.n[0]).abs() <= 0.5 * h
        {
            peers += 1;
        }
    }
    if peers >= 2 {
        Ok(std::iter::once(p.clone()).chain(rs).collect())
    } else {
        Ok(rs)
    }
}
fn block(rs: &[Row], a: i64, ls: &[Value], sz: [f64; 2], w: &mut Work) -> Result<Value, Error> {
    let members: Vec<_> = rs.iter().flat_map(|r| r.m.iter().copied()).collect();
    let bs = members
        .iter()
        .map(|&i| bbox(&ls[i]["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut out = json!({"type":"index","bbox":combined(&bs)?.raw,"angle":a,"content":rs.iter().map(|r|r.content.as_str()).collect::<Vec<_>>().join("\n")});
    let mut tight = Vec::new();
    let mut changed = false;
    for (&i, b) in members.iter().zip(&bs) {
        w.charge(1)?;
        let c = if let Some(ink) = coerce(&ls[i]["ink_bbox"])? {
            clip(
                &bbox(&json!([
                    ink.n[0] - 1.,
                    ink.n[1] - 1.,
                    ink.n[2] + 1.,
                    ink.n[3] + 1.
                ]))?,
                sz,
            )?
        } else {
            None
        };
        changed |= c.is_some();
        tight.push(c.unwrap_or_else(|| b.clone()));
    }
    if changed {
        out["_tight_output_bbox"] = json!(combined(&tight)?.raw);
    }
    Ok(out)
}
fn claim(rs: &[Row], ls: &mut [Value], ids: &mut BTreeSet<usize>) {
    for r in rs {
        for &i in &r.m {
            ls[i]["semantic_type"] = json!("index");
            ids.insert(i);
        }
    }
}
pub(super) fn extract(v: &Value, w: &mut Work) -> Result<Value, Error> {
    extract_mode(v, w, false)
}
pub(super) fn extract_final_page(v: &Value, w: &mut Work) -> Result<Value, Error> {
    extract_mode(v, w, true)
}
fn extract_mode(v: &Value, w: &mut Work, original_identities: bool) -> Result<Value, Error> {
    extract_lines(
        array(&v["lines"], 4096)?.to_vec(),
        v,
        w,
        original_identities,
    )
}
pub(super) fn extract_owned(mut v: Value, w: &mut Work) -> Result<Value, Error> {
    array(&v["lines"], 4096)?;
    let Value::Array(lines) = v["lines"].take() else {
        return Err(invalid());
    };
    extract_lines(lines, &v, w, false)
}
fn extract_lines(
    mut ls: Vec<Value>,
    v: &Value,
    w: &mut Work,
    original_identities: bool,
) -> Result<Value, Error> {
    let sz = size(&v["page_size"])?;
    let containers = optional(v, "container_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let mut unique = BTreeSet::new();
    let mut angles = BTreeSet::new();
    let mut count = 0;
    for l in &ls {
        if !unique.insert(integer(l, "source_index", 0)?) && !original_identities {
            return Err(invalid());
        }
        let t = text(l)?;
        if t.chars().count() > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        w.charge(t.len() * 12 + 1)?;
        count += optional(l, "chars", 20000)?.len();
        if count > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        integer(l, "run_index", 0)?;
        if l["semantic_type"].is_null() {
            angles.insert(integer(l, "angle", 0)?);
        }
    }
    let flag = |k: &str| {
        v.get(k)
            .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))
    };
    let require = flag("require_heading")?;
    let explicit = flag("require_explicit_heading")?;
    let mut claimed = BTreeSet::new();
    let mut blocks = Vec::new();
    for a in angles {
        let rs = rows(&ls, sz, a, w)?;
        if rs.len() < 5 {
            continue;
        }
        let width = if [90, 270].contains(&a) { sz[1] } else { sz[0] };
        let cs: Vec<_> = containers.iter().map(|b| local(b, sz, a)).collect();
        let mut eligible = Vec::new();
        for r in rs {
            w.charge(cs.len() + 1)?;
            if !cs.iter().any(|b| overlap_smaller(&r.b, b) >= 0.35) {
                eligible.push(r);
            }
        }
        let leaders: Vec<_> = eligible
            .iter()
            .filter(|r| r.page && leader(&r.content, 4) && r.b.n[2] - r.b.n[0] >= 0.5 * width)
            .cloned()
            .collect();
        if leaders.len() >= 5 && leaders.len() as f64 >= 0.6 * eligible.len() as f64 {
            let right = median(leaders.iter().map(|r| r.b.n[2]).collect());
            let aligned: Vec<_> = leaders
                .iter()
                .filter(|r| (r.b.n[2] - right).abs() <= 0.02 * width)
                .cloned()
                .collect();
            if right >= 0.75 * width && aligned.len() as f64 >= 0.8 * leaders.len() as f64 {
                claim(&aligned, &mut ls, &mut claimed);
                blocks.push(block(&join(&aligned, &ls, w)?, a, &ls, sz, w)?);
                eligible.retain(|r| !r.m.iter().any(|i| claimed.contains(i)));
            }
        }
        for (head, candidate) in unpaged(&eligible, &ls, w)? {
            for i in head.m {
                ls[i]["semantic_type"] = json!("paragraph_title");
            }
            claim(&candidate, &mut ls, &mut claimed);
            blocks.push(block(&candidate, a, &ls, sz, w)?);
        }
        eligible.retain(|r| !r.m.iter().any(|i| claimed.contains(i)));
        for band in bands(&eligible, &ls, w)? {
            w.charge(eligible.len() + band.len() + 1)?;
            let candidate = trim_band(&band);
            let head = find_heading(&eligible, &candidate, width);
            let explicit_head = head.is_some_and(|h| heading(&h.content));
            if explicit && !explicit_head {
                continue;
            }
            if !stable(&candidate, width, explicit_head) || require && head.is_none() {
                continue;
            }
            let candidate = join(&leading(&eligible, candidate, &ls, w)?, &ls, w)?;
            if let Some(h) = head {
                for &i in &h.m {
                    ls[i]["semantic_type"] = json!("paragraph_title");
                }
            }
            claim(&candidate, &mut ls, &mut claimed);
            blocks.push(block(&candidate, a, &ls, sz, w)?);
        }
    }
    let remaining: Vec<_> = ls
        .iter()
        .enumerate()
        .filter(|(i, _)| !claimed.contains(i))
        .map(|(_, l)| l.clone())
        .collect();
    let ids: BTreeSet<_> = claimed
        .iter()
        .map(|&i| integer(&ls[i], "source_index", 0).unwrap())
        .collect();
    let mut out = json!({"claimed_line_indices":ids});
    out["lines"] = Value::Array(ls);
    out["index_blocks"] = Value::Array(blocks);
    out["remaining_lines"] = Value::Array(remaining);
    Ok(out)
}
