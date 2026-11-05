//! 旧有标题混合数值表：完整多层表头、原生年份切词、附近表注及重复表头单行续表。
use super::*;
use numeric_segments::{clustered, grid};
fn cjk(c: char) -> bool {
    ('\u{3400}'..='\u{9fff}').contains(&c)
}
fn numeric(s: &str) -> bool {
    let s = s.trim_matches(python_space);
    if let Some(s) = s.strip_prefix('*') {
        !s.starts_with(python_space) && text_numeric::token(s)
    } else {
        text_numeric::token(s)
    }
}
fn numbered(s: &str) -> bool {
    let c: Vec<_> = s.trim_start_matches(python_space).chars().collect();
    let mut i = if prefix(&c, "table") {
        5
    } else if prefix(&c, "tab") {
        if c.get(3) == Some(&'.') {
            4
        } else {
            3
        }
    } else if c.first() == Some(&'表') {
        1
    } else {
        return false;
    };
    while c.get(i).is_some_and(|c| python_space(*c)) {
        i += 1;
    }
    c.get(i).is_some_and(|c| python_decimal(*c).is_some())
}
fn year_word(s: &str) -> bool {
    let c: Vec<_> = s.chars().collect();
    if c.len() < 4 || !c[..4].iter().all(|c| python_decimal(*c).is_some()) {
        return false;
    }
    if c.len() == 4 {
        return true;
    }
    if "-–".contains(c[4]) {
        return c.len() == 9 && c[5..].iter().all(|c| python_decimal(*c).is_some());
    }
    c[4].is_ascii_alphabetic() && (c.len() == 5 || (c.len() == 6 && python_decimal(c[5]).is_some()))
}
fn parts(l: &Value, work: &mut Work) -> Result<Vec<Value>, Error> {
    let text = l["text"].as_str().unwrap();
    let words = text
        .split(python_space)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    if words.len() < 3 || !words.iter().all(|s| year_word(s)) {
        return Ok(vec![l.clone()]);
    }
    let chars = optional(l, "chars", 20000)?;
    if chars.is_empty() {
        return Ok(vec![l.clone()]);
    }
    let mut raw = Vec::new();
    for c in chars {
        let t = c
            .get("char")
            .map_or(Ok(""), |v| v.as_str().ok_or_else(invalid))?;
        raw.extend(t.chars());
    }
    work.charge(raw.len() + 1)?;
    let mut at = 0;
    let mut out = Vec::new();
    for word in words {
        let w: Vec<_> = word.chars().collect();
        let Some(start) = raw[at..].windows(w.len()).position(|s| s == w) else {
            return Ok(vec![l.clone()]);
        };
        let start = at + start;
        at = start + w.len();
        let selected = &chars[start.min(chars.len())..at.min(chars.len())];
        let mut boxes = Vec::new();
        for c in selected {
            if !c
                .get("char")
                .and_then(Value::as_str)
                .unwrap_or("")
                .trim_matches(python_space)
                .is_empty()
            {
                boxes.push(bbox(&c["bbox"])?);
            }
        }
        let mut b = boxes.first().ok_or_else(invalid)?.clone();
        for bb in &boxes[1..] {
            b = union(b, bb);
        }
        let mut part = l.clone();
        part["text"] = json!(word);
        part["bbox"] = json!(b.raw);
        out.push(part);
    }
    Ok(out)
}
fn annotation(kind: &str, lines: &[&Value]) -> Result<Value, Error> {
    let mut boxes: Vec<(i64, Box4)> = Vec::new();
    for l in lines {
        let id = integer(l, "source_index", 0)?;
        let b = bbox(&l["bbox"])?;
        if let Some((_, old)) = boxes.iter_mut().find(|(i, _)| *i == id) {
            *old = b;
        } else {
            boxes.push((id, b));
        }
    }
    let mut b = boxes.first().ok_or_else(invalid)?.1.clone();
    for (_, bb) in &boxes[1..] {
        b = union(b, bb);
    }
    let ids = boxes.iter().map(|(i, _)| *i).collect::<BTreeSet<_>>();
    Ok(
        json!({"kind":kind,"bbox":b.raw,"line_indices":ids,"line_bboxes":boxes.iter().map(|(i,b)|json!([i,b.raw])).collect::<Vec<_>>()}),
    )
}
fn note_marker(s: &str) -> bool {
    let c: Vec<_> = s
        .trim_matches(python_space)
        .chars()
        .map(|c| match c {
            'İ' | 'ı' => 'i',
            'ſ' => 's',
            'K' => 'k',
            _ => c.to_ascii_lowercase(),
        })
        .collect();
    if c.first() == Some(&'*') && c.get(1).is_some_and(char::is_ascii_alphabetic) {
        return true;
    }
    for prefix in ["source", "notes", "note", "来源", "注"] {
        let p: Vec<_> = prefix.chars().collect();
        if c.starts_with(&p) {
            let mut i = p.len();
            while c.get(i).is_some_and(|c| python_space(*c)) {
                i += 1;
            }
            if c.get(i).is_some_and(|c| ":：".contains(*c)) {
                return true;
            }
        }
    }
    false
}
fn near_note(lines: &[Value], b: &Box4, em: f64, work: &mut Work) -> Result<Option<Value>, Error> {
    let mut seeds = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        let t = l["text"].as_str().unwrap();
        work.charge(t.len() + 1)?;
        let lb = bbox(&l["bbox"])?;
        if note_marker(t)
            && (words(l) >= 4 || t.chars().filter(|c| cjk(*c)).count() >= 4)
            && 0. <= lb.n[1] - b.n[3]
            && lb.n[1] - b.n[3] <= 2. * em
            && super::super::super::graphic_captions::axis(&lb, b, 0) >= 0.5
        {
            seeds.push(i);
        }
    }
    if seeds.len() != 1 {
        return Ok(None);
    }
    let seed = seeds[0];
    let mut members = vec![&lines[seed]];
    let mut ordered = (0..lines.len()).collect::<Vec<_>>();
    ordered.sort_by(|a, b| {
        cmp(
            lines[*a]["bbox"][1].as_f64().unwrap(),
            lines[*b]["bbox"][1].as_f64().unwrap(),
        )
    });
    for i in ordered {
        if i == seed
            || lines[i]["bbox"][1].as_f64().unwrap() <= lines[seed]["bbox"][1].as_f64().unwrap()
        {
            continue;
        }
        work.charge(1)?;
        let l = &lines[i];
        let gap =
            l["bbox"][1].as_f64().unwrap() - members.last().unwrap()["bbox"][3].as_f64().unwrap();
        if !crate::legacy_page::eq(&l["font_signature"], &lines[seed]["font_signature"])
            || gap < -0.25 * em
            || gap > 0.75 * em
            || (l["bbox"][0].as_f64().unwrap() - lines[seed]["bbox"][0].as_f64().unwrap()).abs()
                > 0.5 * em
        {
            break;
        }
        members.push(l);
    }
    Ok(Some(annotation("footnote", &members)?))
}
fn repeated(
    v: &Request,
    lines: &[Value],
    rows: &[rowprep::Row],
    known: &[Vec<String>],
    used: &BTreeSet<i64>,
    em: f64,
    excluded: &[Box4],
    work: &mut Work,
) -> Result<Vec<Value>, Error> {
    let mut out = Vec::new();
    for header in lines {
        if used.contains(&integer(header, "source_index", 0)?) {
            continue;
        }
        let chars = optional(header, "chars", 20000)?;
        if chars.is_empty() {
            continue;
        }
        let mut visible = Vec::new();
        let mut raw = String::new();
        for c in chars {
            let t = c
                .get("char")
                .map_or(Ok(""), |v| v.as_str().ok_or_else(invalid))?;
            if !t.trim_matches(python_space).is_empty() {
                visible.push(c);
                raw.push_str(t);
            }
        }
        work.charge(
            raw.len()
                + known
                    .iter()
                    .map(|h| h.iter().map(String::len).sum::<usize>())
                    .sum::<usize>()
                + 1,
        )?;
        let matches = known
            .iter()
            .filter(|labels| {
                raw == labels
                    .iter()
                    .flat_map(|s| s.chars())
                    .filter(|c| !python_space(*c))
                    .collect::<String>()
            })
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            continue;
        }
        let mut at = 0;
        let mut parts = Vec::new();
        for label in matches[0] {
            let n = label.chars().filter(|c| !python_space(*c)).count();
            let selected = &visible[at.min(visible.len())..(at + n).min(visible.len())];
            at += n;
            let mut b = bbox(&selected.first().ok_or_else(invalid)?["bbox"])?;
            for c in &selected[1..] {
                b = union(b, &bbox(&c["bbox"])?);
            }
            parts.push(b);
        }
        let hb = bbox(&header["bbox"])?;
        let mut following = Vec::new();
        for row in rows {
            work.charge(
                row.fragments
                    .iter()
                    .map(|f| f.raw["text"].as_str().unwrap().len())
                    .sum::<usize>()
                    + 1,
            )?;
            if row.fragments.len() == parts.len()
                && 0. <= row.b.n[1] - hb.n[3]
                && row.b.n[1] - hb.n[3] <= 0.75 * em
                && row.fragments.iter().all(|f| {
                    let t = f.raw["text"].as_str().unwrap().trim_matches(python_space);
                    t.chars()
                        .next()
                        .is_some_and(|c| python_decimal(c).is_some())
                        && !t.contains(['%', '‰'])
                        && text_numeric::token(t)
                })
            {
                following.push(row);
            }
        }
        if following.len() != 1 {
            continue;
        }
        let row = following[0];
        let columns = parts
            .into_iter()
            .zip(&row.fragments)
            .map(|(a, f)| vec![a, f.local.clone()])
            .collect::<Vec<_>>();
        if columns.windows(2).any(|p| {
            p[0].iter().map(|b| b.n[2]).reduce(f64::max).unwrap()
                >= p[1].iter().map(|b| b.n[0]).reduce(f64::min).unwrap()
        }) {
            continue;
        }
        let mut xs = vec![columns[0].iter().map(|b| b.n[0]).reduce(f64::min).unwrap() - 0.2 * em];
        xs.extend(columns.windows(2).map(|p| {
            (p[0].iter().map(|b| b.n[2]).reduce(f64::max).unwrap()
                + p[1].iter().map(|b| b.n[0]).reduce(f64::min).unwrap())
                / 2.
        }));
        xs.push(
            columns
                .last()
                .unwrap()
                .iter()
                .map(|b| b.n[2])
                .reduce(f64::max)
                .unwrap()
                + 0.2 * em,
        );
        let b = bbox(&json!([
            xs[0],
            hb.n[1] - 0.2 * em,
            xs.last().unwrap(),
            row.b.n[3] + 0.2 * em
        ]))?;
        work.charge(excluded.len() + 1)?;
        if excluded.iter().any(|e| overlap_smaller(&b, e) >= 0.5) {
            continue;
        }
        let mut c = candidate(
            b.clone(),
            &[],
            vec![],
            grid(&b, &xs, &[b.n[1], (hb.n[3] + row.b.n[1]) / 2., b.n[3]]),
            12,
        )?;
        let mut ids = row.fragments.iter().map(|f| f.id).collect::<BTreeSet<_>>();
        ids.insert(integer(header, "source_index", 0)?);
        c["line_indices"] = json!(ids);
        c["preserve_inline_font_styles"] = json!(true);
        c["inferred_grid_authoritative"] = json!(true);
        out.push(c);
    }
    let _ = v;
    Ok(out)
}
pub(super) fn detect(
    v: &Request,
    p: &Page,
    excluded: &[Box4],
    work: &mut Work,
) -> Result<Vec<Value>, Error> {
    let mut lines = Vec::new();
    let mut numeric_count = 0;
    for l in p.line_values() {
        work.charge(l["text"].as_str().unwrap().len() + 1)?;
        if !l["semantic_type"].is_null()
            || flag(l, "formula_candidate_only")?
            || flag(l, "caption_start")?
        {
            continue;
        }
        numeric_count += usize::from(numeric(l["text"].as_str().unwrap()));
        lines.push(l.clone());
    }
    if lines.is_empty() || numeric_count < 6 {
        return Ok(vec![]);
    }
    let allowed = lines
        .iter()
        .map(|l| integer(l, "source_index", 0))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let original = optional(v.source, "lines", 4096)?;
    let typed = original.iter().all(|l| {
        l["text"].is_string()
            && l["angle"].as_i64().is_some()
            && l["source_index"].as_i64().is_some()
    });
    if typed
        && !lines.iter().any(|l| numbered(l["text"].as_str().unwrap()))
        && original.iter().all(|l| {
            l["angle"].as_i64() != Some(0) || allowed.contains(&l["source_index"].as_i64().unwrap())
        })
    {
        return Ok(vec![]);
    }
    let em = median(lines.iter().map(effective).collect::<Result<Vec<_>, _>>()?);
    let prepared = clustered(v, lines.clone(), em, work)?;
    let data = prepared
        .rows
        .iter()
        .filter(|r| {
            (3..=10).contains(&r.fragments.len())
                && r.fragments[0].raw["text"]
                    .as_str()
                    .unwrap()
                    .split(python_space)
                    .filter(|s| !s.is_empty())
                    .count()
                    <= 5
                && r.fragments[1..]
                    .iter()
                    .all(|f| numeric(f.raw["text"].as_str().unwrap()))
        })
        .collect::<Vec<_>>();
    let mut by_source = std::collections::BTreeMap::new();
    for l in &lines {
        by_source.insert(integer(l, "source_index", 0)?, l);
    }
    let mut used = BTreeSet::new();
    let mut out: Vec<Value> = Vec::new();
    let mut known = Vec::new();
    for first in &data {
        if first.fragments.iter().any(|f| used.contains(&f.id)) {
            continue;
        }
        let count = first.fragments.len();
        let mut group = vec![*first];
        for row in &data {
            work.charge(count + 1)?;
            if row.center <= first.center {
                continue;
            }
            let gap = row.b.n[1] - group.last().unwrap().b.n[3];
            if row.fragments.len() != count
                || gap < -0.15 * em
                || gap > 1.5 * em
                || row.fragments.iter().zip(&first.fragments).any(|(f, s)| {
                    (f.local.n[0] - s.local.n[0])
                        .abs()
                        .min((f.local.n[2] - s.local.n[2]).abs())
                        > 0.5 * em
                        && (center(&f.local, 0) - center(&s.local, 0)).abs() > 0.5 * em
                })
            {
                break;
            }
            group.push(row);
        }
        if group.len() < 3 {
            continue;
        }
        let cols = (0..count)
            .map(|i| {
                group
                    .iter()
                    .map(|r| r.fragments[i].local.clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        if cols.windows(2).any(|p| {
            p[1].iter().map(|b| b.n[0]).reduce(f64::min).unwrap()
                - p[0].iter().map(|b| b.n[2]).reduce(f64::max).unwrap()
                < 0.4 * em
        }) {
            continue;
        }
        let anchors = cols
            .iter()
            .map(|col| median(col.iter().map(|b| b.n[0]).collect()))
            .collect::<Vec<_>>();
        let mut headers = Vec::new();
        for l in &lines {
            work.charge(1)?;
            let b = bbox(&l["bbox"])?;
            let h = effective(l)?;
            let words = words(l);
            if 0. <= first.b.n[1] - b.n[3]
                && first.b.n[1] - b.n[3] <= 4. * em
                && 0.8 <= h / em
                && h / em <= 1.25
                && !flag(l, "paragraph_terminal")?
                && (1..=8).contains(&words)
                && !numbered(l["text"].as_str().unwrap())
                && first.b.n[0] - em <= b.n[0]
                && b.n[2] <= first.b.n[2] + 8. * em
            {
                headers.push(l);
            }
        }
        if headers.is_empty()
            || headers
                .iter()
                .filter(|l| {
                    l["text"]
                        .as_str()
                        .unwrap()
                        .chars()
                        .any(|c| c.is_ascii_alphabetic() || cjk(c))
                        && !numeric(l["text"].as_str().unwrap())
                })
                .count()
                < 2
        {
            continue;
        }
        let mut header_parts = Vec::new();
        for l in &headers {
            header_parts.extend(parts(l, work)?);
            if header_parts.len() > rapiddoc_core::resources::limit(20000) {
                return Err(limit());
            }
        }
        let part_boxes = header_parts
            .iter()
            .map(|l| bbox(&l["bbox"]))
            .collect::<Result<Vec<_>, _>>()?;
        let bottom = headers
            .iter()
            .map(|l| l["bbox"][3].as_f64().unwrap())
            .reduce(f64::max)
            .unwrap();
        let mut leaf = Vec::new();
        for (i, b) in part_boxes.iter().enumerate() {
            work.charge(count + 1)?;
            let mut index = 0;
            for j in 1..count {
                if (center(b, 0) - anchors[j]).abs() < (center(b, 0) - anchors[index]).abs() {
                    index = j;
                }
            }
            let distance = anchors
                .iter()
                .filter(|x| **x != anchors[index])
                .map(|x| (x - anchors[index]).abs())
                .reduce(f64::min)
                .unwrap_or(em);
            if bottom - b.n[3] <= 0.8 * em && b.n[2] - b.n[0] <= 1.8 * distance {
                leaf.push((i, index));
            }
        }
        if leaf.iter().map(|(_, i)| *i).collect::<BTreeSet<_>>().len() < count - 1 {
            continue;
        }
        let mut col_bounds = cols;
        for (part, col) in &leaf {
            col_bounds[*col].push(part_boxes[*part].clone());
        }
        if col_bounds.windows(2).any(|p| {
            p[0].iter().map(|b| b.n[2]).reduce(f64::max).unwrap()
                >= p[1].iter().map(|b| b.n[0]).reduce(f64::min).unwrap()
        }) {
            continue;
        }
        let mut edges = vec![
            col_bounds[0]
                .iter()
                .map(|b| b.n[0])
                .reduce(f64::min)
                .unwrap()
                - 0.25 * em,
        ];
        edges.extend(col_bounds.windows(2).map(|p| {
            (p[0].iter().map(|b| b.n[2]).reduce(f64::max).unwrap()
                + p[1].iter().map(|b| b.n[0]).reduce(f64::min).unwrap())
                / 2.
        }));
        edges.push(
            col_bounds
                .last()
                .unwrap()
                .iter()
                .map(|b| b.n[2])
                .reduce(f64::max)
                .unwrap()
                + 0.25 * em,
        );
        let top = headers
            .iter()
            .min_by(|a, b| {
                cmp(
                    a["bbox"][1].as_f64().unwrap(),
                    b["bbox"][1].as_f64().unwrap(),
                )
            })
            .unwrap()["bbox"][1]
            .clone();
        let b = bbox(&json!([
            edges[0],
            top,
            edges.last().unwrap(),
            group.last().unwrap().b.n[3] + 0.2 * em
        ]))?;
        work.charge(excluded.len() + out.len() + 1)?;
        if excluded.iter().any(|e| overlap_smaller(&b, e) >= 0.5)
            || out
                .iter()
                .any(|c| overlap_smaller(&b, &bbox(&c["core_bbox"]).unwrap()) >= 0.5)
        {
            continue;
        }
        let mut annotations = bounded_em(v, &b, em, 2.5, work)?;
        if annotations.is_empty() {
            continue;
        }
        let source_id = annotations[0]["line_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i.as_i64().unwrap())
            .min()
            .ok_or_else(invalid)?;
        if !numbered(
            by_source.get(&source_id).ok_or_else(invalid)?["text"]
                .as_str()
                .unwrap(),
        ) {
            continue;
        }
        let raw_rules = optional(v.source, "drawing_lines", 5000)?;
        for index in 1..count {
            let low = col_bounds[index - 1]
                .iter()
                .map(|b| b.n[2])
                .reduce(f64::max)
                .unwrap();
            let high = col_bounds[index]
                .iter()
                .map(|b| b.n[0])
                .reduce(f64::min)
                .unwrap();
            let mut native = Vec::new();
            for rule in raw_rules {
                work.charge(1)?;
                if rule["orientation"] == "vertical" {
                    let rb = bbox(&rule["bbox"])?;
                    let x = center(&rb, 0);
                    if low < x && x < high && rb.n[3] - rb.n[1] >= 0.7 * (b.n[3] - first.b.n[1]) {
                        native.push(x);
                    }
                }
            }
            if native.len() == 1 {
                edges[index] = native[0];
            }
        }
        let header_bottom = (bottom + first.b.n[1]) / 2.;
        let mut ys = vec![b.n[1], header_bottom];
        ys.extend(group.windows(2).map(|p| (p[0].b.n[3] + p[1].b.n[1]) / 2.));
        ys.push(b.n[3]);
        let mut inferred = vec![
            axis(&b, edges[0], false),
            axis(&b, *edges.last().unwrap(), false),
        ];
        let mut spans: Vec<(usize, usize, f64)> = Vec::new();
        for (upper_index, upper) in part_boxes.iter().enumerate() {
            work.charge(leaf.len() * (count * count + 1) + 1)?;
            let below = leaf
                .iter()
                .filter(|(part, _)| {
                    part_boxes[*part].n[1] >= upper.n[3] - 0.1 * em && *part != upper_index
                })
                .copied()
                .collect::<Vec<_>>();
            if below.len() < 2 {
                continue;
            }
            let mut best: Option<(usize, usize, f64)> = None;
            for start in 0..count - 1 {
                for end in start + 1..count {
                    let subset = below
                        .iter()
                        .filter(|(_, i)| start <= *i && *i <= end)
                        .collect::<Vec<_>>();
                    if subset
                        .iter()
                        .map(|(_, i)| *i)
                        .collect::<BTreeSet<_>>()
                        .len()
                        != end - start + 1
                    {
                        continue;
                    }
                    let mut lb = part_boxes[subset[0].0].clone();
                    for (part, _) in &subset[1..] {
                        lb = union(lb, &part_boxes[*part]);
                    }
                    if (center(upper, 0) - center(&lb, 0)).abs() <= 1.1 * em
                        && upper.n[2] - upper.n[0] >= 0.2 * (lb.n[2] - lb.n[0])
                    {
                        let split = (upper.n[3]
                            + subset
                                .iter()
                                .map(|(i, _)| part_boxes[*i].n[1])
                                .reduce(f64::min)
                                .unwrap())
                            / 2.;
                        if best.is_none_or(|b| end - start > b.1 - b.0) {
                            best = Some((start, end, split));
                        }
                    }
                }
            }
            if let Some(best) = best {
                spans.push(best);
            }
        }
        if let Some(&(start, end, split)) = spans.iter().min_by(|a, b| cmp(a.2, b.2)) {
            for (i, x) in edges.iter().enumerate().take(count).skip(1) {
                let top = if start < i && i <= end {
                    json!(split)
                } else {
                    b.raw[1].clone()
                };
                inferred.push(json!({"bbox":[x-0.05,top,x+0.05,b.raw[3]],"width":0.1,"orientation":"vertical"}));
            }
            inferred.push(json!({"bbox":[edges[start],split-0.05,edges[end+1],split+0.05],"width":0.1,"orientation":"horizontal"}));
            for rule in raw_rules {
                work.charge(1)?;
                if rule["orientation"] == "horizontal" {
                    let rb = bbox(&rule["bbox"])?;
                    if b.n[1] < rb.n[1]
                        && rb.n[1] < header_bottom
                        && rb.n[2] - rb.n[0] >= 0.9 * (b.n[2] - b.n[0])
                        && super::super::super::graphic_captions::axis(&rb, &b, 0) >= 0.9
                    {
                        inferred.push(json!({"bbox":rule["bbox"],"width":rule.get("width").cloned().unwrap_or(json!(0.)),"orientation":"horizontal"}));
                    }
                }
            }
        } else {
            inferred.extend(edges[1..count].iter().map(|x| axis(&b, *x, false)));
        }
        inferred.extend(ys.into_iter().map(|y| axis(&b, y, true)));
        let mut members = headers.clone();
        for row in &group {
            for f in &row.fragments {
                members.push(*by_source.get(&f.id).ok_or_else(invalid)?);
            }
        }
        if let Some(note) = near_note(&lines, &b, em, work)? {
            annotations.push(note);
        }
        let mut c = candidate(b, &members, annotations, inferred, 12)?;
        c["inferred_grid_authoritative"] = json!(true);
        c["preserve_inline_font_styles"] = json!(true);
        for l in &members {
            used.insert(integer(l, "source_index", 0)?);
        }
        out.push(c);
        if spans.is_empty() && headers.len() == count {
            headers.sort_by(|a, b| {
                cmp(
                    a["bbox"][0].as_f64().unwrap(),
                    b["bbox"][0].as_f64().unwrap(),
                )
            });
            known.push(
                headers
                    .iter()
                    .map(|l| l["text"].as_str().unwrap().to_owned())
                    .collect(),
            );
        }
    }
    out.extend(repeated(
        v,
        &lines,
        &prepared.rows,
        &known,
        &used,
        em,
        excluded,
        work,
    )?);
    Ok(out)
}
