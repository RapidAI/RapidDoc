//! 无框重复数值列及有表题双色表头双列文字表，保留实际字形对齐证明。
use super::*;
use numeric_segments::{clustered, grid};
pub(super) fn token(s: &str) -> bool {
    let c: Vec<_> = s.trim_matches(python_space).chars().collect();
    let mut i = 0;
    if c.get(i).is_some_and(|c| "+−-".contains(*c)) {
        i += 1;
    }
    let start = i;
    while c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
        i += 1;
    }
    if i == start {
        return false;
    }
    while c.get(i).is_some_and(|c| ".,".contains(*c))
        && c.get(i + 1).is_some_and(|c| python_decimal(*c).is_some())
    {
        i += 1;
        while c.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
            i += 1;
        }
    }
    while c.get(i).is_some_and(|c| python_space(*c)) {
        i += 1;
    }
    if i == c.len() {
        return true;
    }
    if c.len() - i == 1 && "%‰".contains(c[i]) {
        return true;
    }
    (1..=8).contains(&(c.len() - i)) && c[i..].iter().all(char::is_ascii_alphabetic)
}
fn raw_char(c: &Value) -> String {
    super::super::super::table_native_multiline::spatial::raw_string(c.get("char"), false)
}
fn unruled(v: &Request, p: &Page, excluded: &[Box4], work: &mut Work) -> Result<Vec<Value>, Error> {
    let mut lines = Vec::new();
    let mut count = 0;
    for l in p.line_values() {
        work.charge(l["text"].as_str().unwrap().len() + 1)?;
        if !l["semantic_type"].is_null()
            || flag(l, "formula_candidate_only")?
            || flag(l, "compact_formula_cluster")?
            || flag(l, "caption_start")?
        {
            continue;
        }
        count += usize::from(token(l["text"].as_str().unwrap()));
        lines.push(l.clone());
    }
    if count < 9 {
        return Ok(vec![]);
    }
    let em = median(lines.iter().map(effective).collect::<Result<Vec<_>, _>>()?);
    let prepared = clustered(v, lines.clone(), em, work)?;
    let data = prepared
        .rows
        .iter()
        .filter(|r| {
            (3..=8).contains(&r.fragments.len())
                && r.fragments
                    .iter()
                    .all(|f| token(f.raw["text"].as_str().unwrap()))
        })
        .collect::<Vec<_>>();
    let mut by_source = std::collections::BTreeMap::new();
    for l in &lines {
        by_source.insert(integer(l, "source_index", 0)?, l);
    }
    let mut consumed = BTreeSet::new();
    let mut out: Vec<Value> = Vec::new();
    for first in &data {
        work.charge(data.len() + lines.len() + 1)?;
        if first.fragments.iter().any(|f| consumed.contains(&f.id)) {
            continue;
        }
        let columns = first
            .fragments
            .iter()
            .map(|f| f.local.n[0])
            .collect::<Vec<_>>();
        if columns.windows(2).any(|p| p[1] - p[0] < 1.5 * em) {
            continue;
        }
        let mut group = vec![*first];
        for row in &data {
            work.charge(columns.len() + 1)?;
            if row.center <= first.center {
                continue;
            }
            if row.b.n[1] - group.last().unwrap().b.n[3] > 1.5 * em
                || row.fragments.len() != columns.len()
                || row
                    .fragments
                    .iter()
                    .zip(&columns)
                    .any(|(f, x)| (f.local.n[0] - x).abs() > 0.3 * em)
            {
                break;
            }
            group.push(row);
        }
        if group.len() < 3 {
            continue;
        }
        let mut headers = Vec::new();
        for l in &lines {
            let b = bbox(&l["bbox"])?;
            let h = effective(l)?;
            let cov = l
                .get("font_coverage")
                .filter(|v| !v.is_null())
                .map_or(Ok(0.), num)?;
            if !l["font_signature"].is_null()
                && cov >= 0.75
                && bold(l)?
                && -0.15 * em <= first.b.n[1] - b.n[3]
                && first.b.n[1] - b.n[3] <= em
                && (b.n[0] - columns[0]).abs() <= 0.3 * em
                && 0.8 * em <= h
                && h <= 1.25 * em
                && b.n[2] >= first.b.n[2]
                && !flag(l, "paragraph_terminal")?
            {
                headers.push(l);
            }
        }
        if headers.len() != 1 {
            continue;
        }
        let header = headers[0];
        let chars = optional(header, "chars", 20000)?;
        let mut supported = true;
        for x in &columns {
            work.charge(chars.len() + 1)?;
            let mut found = false;
            for c in chars {
                let t = raw_char(c);
                if !t.is_empty()
                    && t.chars().all(rapiddoc_native::text_boundary::python_alpha)
                    && !c["bbox"].is_null()
                    && (bbox(&c["bbox"])?.n[0] - x).abs() <= 0.25 * em
                {
                    found = true;
                    break;
                }
            }
            supported &= found;
        }
        if !supported {
            continue;
        }
        let mut members = vec![header];
        for row in &group {
            for f in &row.fragments {
                members.push(*by_source.get(&f.id).ok_or_else(invalid)?);
            }
        }
        let right = members
            .iter()
            .map(|l| l["bbox"][2].as_f64().unwrap())
            .reduce(f64::max)
            .unwrap();
        let b = bbox(&json!([
            columns[0] - 0.25 * em,
            header["bbox"][1].as_f64().unwrap() - 0.25 * em,
            right + 0.25 * em,
            group.last().unwrap().b.n[3] + 0.25 * em
        ]))?;
        work.charge(excluded.len() + out.len() + 1)?;
        if excluded.iter().any(|e| overlap_smaller(&b, e) >= 0.5)
            || out
                .iter()
                .any(|c| overlap_smaller(&b, &bbox(&c["core_bbox"]).unwrap()) >= 0.5)
        {
            continue;
        }
        let mut xs = vec![b.n[0]];
        xs.extend(columns[1..].iter().map(|x| x - 0.1 * em));
        xs.push(b.n[2]);
        let mut ys = vec![
            b.n[1],
            (header["bbox"][3].as_f64().unwrap() + first.b.n[1]) / 2.,
        ];
        ys.extend(group.windows(2).map(|p| (p[0].b.n[3] + p[1].b.n[1]) / 2.));
        ys.push(b.n[3]);
        let mut c = candidate(b.clone(), &members, vec![], grid(&b, &xs, &ys), 12)?;
        c["preserve_inline_font_styles"] = json!(true);
        out.push(c);
        for l in members {
            consumed.insert(integer(l, "source_index", 0)?);
        }
    }
    Ok(out)
}
fn paired(v: &Request, p: &Page, excluded: &[Box4], work: &mut Work) -> Result<Vec<Value>, Error> {
    let em = p.em;
    let mut rules = Vec::new();
    for r in optional(v.source, "drawing_lines", 5000)? {
        if r["orientation"] == "horizontal" {
            let b = bbox(&r["bbox"])?;
            if b.n[2] - b.n[0] >= 8. * em {
                rules.push(b);
            }
        }
    }
    rules.sort_by(|a, b| cmp(a.n[1], b.n[1]));
    let mut out = Vec::new();
    for pair in rules.windows(2) {
        let (top, bottom) = (&pair[0], &pair[1]);
        let b = bbox(&json!([top.raw[0], top.raw[1], top.raw[2], bottom.raw[3]]))?;
        work.charge(excluded.len() + p.lines.len() * 2 + 1)?;
        if (top.n[0] - bottom.n[0]).abs() > 0.25 * em
            || (top.n[2] - bottom.n[2]).abs() > 0.25 * em
            || bottom.n[1] - top.n[3] < 4. * em
            || bottom.n[1] - top.n[3] > 15. * em
            || excluded.iter().any(|e| overlap_smaller(&b, e) >= 0.5)
        {
            continue;
        }
        let mut fills = Vec::new();
        for path in optional(v.source, "path_infos", 5000)? {
            work.charge(1)?;
            if !flag(path, "fill_visible")? || integer(path, "segment_count", 0)? != 5 {
                continue;
            }
            let rgba = optional(path, "fill_rgba", 4)?;
            if rgba.is_empty() {
                continue;
            }
            if rgba.len() < 3 {
                return Err(invalid());
            }
            let f = bbox(&path["bbox"])?;
            if rgba[..3]
                .iter()
                .map(num)
                .collect::<Result<Vec<_>, _>>()?
                .iter()
                .all(|x| *x <= 120.)
                && (f.n[0] - top.n[0]).abs() <= 0.25 * em
                && (f.n[2] - top.n[2]).abs() <= 0.25 * em
                && (f.n[1] - top.n[1]).abs() <= 0.25 * em
                && 0.75 * em <= f.n[3] - f.n[1]
                && f.n[3] - f.n[1] <= 2. * em
            {
                fills.push(f);
            }
        }
        let annotations = bounded(v, p, &b, work)?;
        if fills.len() != 1
            || annotations.len() != 1
            || annotations[0]["bbox"][1].as_f64().unwrap() < bottom.n[3]
        {
            continue;
        }
        let fill = &fills[0];
        let members = p
            .line_values()
            .filter(|l| overlap_first(&bbox(&l["bbox"]).unwrap(), &b) >= 0.9)
            .collect::<Vec<_>>();
        let header = members
            .iter()
            .copied()
            .filter(|l| overlap_first(&bbox(&l["bbox"]).unwrap(), fill) >= 0.9)
            .collect::<Vec<_>>();
        if header.len() != 1 {
            continue;
        }
        let data = members
            .iter()
            .copied()
            .filter(|l| !header.contains(l))
            .collect::<Vec<_>>();
        let prepared = clustered(v, data.iter().map(|l| (*l).clone()).collect(), em, work)?;
        let rows = &prepared.rows;
        let paired = rows
            .iter()
            .filter(|r| r.fragments.len() == 2)
            .collect::<Vec<_>>();
        if !(4..=10).contains(&rows.len()) || paired.len() < 2 {
            continue;
        }
        let left = median(paired.iter().map(|r| r.fragments[0].local.n[0]).collect());
        let right = median(paired.iter().map(|r| r.fragments[1].local.n[0]).collect());
        if right - left < 3. * em
            || paired.iter().any(|r| {
                (r.fragments[0].local.n[0] - left).abs() > 0.25 * em
                    || (r.fragments[1].local.n[0] - right).abs() > 0.25 * em
            })
        {
            continue;
        }
        let mut glyphs = Vec::new();
        for l in &data {
            for c in optional(l, "chars", 20000)? {
                work.charge(1)?;
                if !raw_char(c).trim_matches(python_space).is_empty() && !c["bbox"].is_null() {
                    glyphs.push(bbox(&c["bbox"])?);
                }
            }
            if glyphs.len() > rapiddoc_core::resources::limit(20000) {
                return Err(limit());
            }
        }
        work.charge(rows.len() * (glyphs.len() + 1))?;
        if !rows.iter().all(|row| {
            glyphs.iter().any(|g| {
                (g.n[0] - right).abs() <= 0.1 * em
                    && row.b.n[1] - 0.3 * em <= center(g, 1)
                    && center(g, 1) <= row.b.n[3] + 0.3 * em
            })
        }) {
            continue;
        }
        let left_glyphs = glyphs
            .iter()
            .filter(|g| g.n[0] < right - 0.2 * em)
            .collect::<Vec<_>>();
        let Some(left_end) = left_glyphs.iter().map(|g| g.n[2]).reduce(f64::max) else {
            continue;
        };
        if right - left_end < 0. || right - left_end > 3. * em {
            continue;
        }
        let split = (left_end + right) / 2.;
        let mut ys = vec![b.n[1], fill.n[3]];
        ys.extend(rows.windows(2).map(|p| (p[0].b.n[3] + p[1].b.n[1]) / 2.));
        ys.push(b.n[3]);
        let mut inferred = vec![
            axis(&b, b.n[0], false),
            axis(&b, b.n[2], false),
            json!({"bbox":[split-0.05,fill.raw[3],split+0.05,b.raw[3]],"width":0.1,"orientation":"vertical"}),
        ];
        inferred.extend(ys.into_iter().map(|y| axis(&b, y, true)));
        let mut c = candidate(b, &members, annotations, inferred, 12)?;
        c["preserve_inline_font_styles"] = json!(true);
        out.push(c);
    }
    Ok(out)
}
pub(super) fn detect(
    v: &Request,
    p: &Page,
    excluded: &[Box4],
    work: &mut Work,
) -> Result<Vec<Value>, Error> {
    if v.options["kind"] == "unruled_numeric" {
        unruled(v, p, excluded, work)
    } else {
        paired(v, p, excluded, work)
    }
}
