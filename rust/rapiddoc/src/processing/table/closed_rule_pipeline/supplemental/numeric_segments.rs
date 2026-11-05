//! 旧部分物理网格、年份表头数值表、横线指标对表完整候选分支。
use super::*;
fn cjk(c: char) -> bool {
    ('\u{3400}'..='\u{9fff}').contains(&c)
}
fn digits(c: &[char], i: &mut usize) -> bool {
    let start = *i;
    while c.get(*i).is_some_and(|c| python_decimal(*c).is_some()) {
        *i += 1;
    }
    *i > start
}
fn decimal(c: &[char], i: &mut usize) -> bool {
    if !digits(c, i) {
        return false;
    }
    while c.get(*i).is_some_and(|c| ".,".contains(*c))
        && c.get(*i + 1).is_some_and(|c| python_decimal(*c).is_some())
    {
        *i += 1;
        digits(c, i);
    }
    true
}
fn numeric(s: &str, metric: bool) -> bool {
    let c: Vec<_> = s.trim_matches(python_space).chars().collect();
    let mut i = 0;
    if c.get(i).is_some_and(|c| "+−-".contains(*c)) {
        i += 1;
    }
    if !metric && c.get(i) == Some(&'(') {
        i += 1;
    }
    if !decimal(&c, &mut i) {
        return false;
    }
    if !metric && c.get(i) == Some(&')') {
        i += 1;
    }
    while c.get(i).is_some_and(|c| python_space(*c)) {
        i += 1;
    }
    if metric && c.get(i).is_some_and(|c| "-/".contains(*c)) {
        i += 1;
        while c.get(i).is_some_and(|c| python_space(*c)) {
            i += 1;
        }
        if !decimal(&c, &mut i) {
            return false;
        }
        while c.get(i).is_some_and(|c| python_space(*c)) {
            i += 1;
        }
    }
    if metric {
        let start = i;
        while c
            .get(i)
            .is_some_and(|c| "%‰".contains(*c) || c.is_ascii_alphabetic() || cjk(*c))
        {
            i += 1;
        }
        i == c.len() && i - start <= 6
    } else {
        if c.get(i).is_some_and(|c| "%‰".contains(*c)) {
            i += 1;
        }
        i == c.len()
    }
}
fn eligible(p: &Page, metric: bool) -> Result<Vec<Value>, Error> {
    let mut out = Vec::new();
    for l in p.line_values() {
        let semantic = l.get("semantic_type").filter(|v| !v.is_null());
        if semantic.is_some_and(|s| metric || s != "header" && s != "footer")
            || flag(l, "formula_candidate_only")?
            || (metric && flag(l, "caption_start")?)
        {
            continue;
        }
        out.push(l.clone());
    }
    Ok(out)
}
pub(super) fn clustered(
    v: &Request,
    lines: Vec<Value>,
    em: f64,
    work: &mut Work,
) -> Result<Page<'static>, Error> {
    // Row clustering reads only these three fields. Do not copy the entire
    // page character/path tree for every numeric subgroup.
    let mut source = value_object!({"page_size":v.source["page_size"]});
    source["lines"] = Value::Array(lines);
    source["drawing_lines"] = json!([]);
    prepare_source(
        &source,
        &value_object!({"angle":0,"median_height":em}),
        work,
    )
    .map(Page::into_owned)
}
pub(super) fn grid(b: &Box4, xs: &[f64], ys: &[f64]) -> Vec<Value> {
    xs.iter()
        .map(|x| axis(b, *x, false))
        .chain(ys.iter().map(|y| axis(b, *y, true)))
        .collect()
}
fn year(text: &str) -> Option<u32> {
    let c: Vec<_> = text.trim_matches(python_space).chars().collect();
    if !(c.len() == 4 || (c.len() == 5 && c[4].is_ascii_alphabetic())) {
        return None;
    }
    let mut n = 0;
    for c in &c[..4] {
        n = n * 10 + python_decimal(*c)? as u32;
    }
    Some(n)
}
fn year_tables(
    v: &Request,
    p: &Page,
    excluded: &[Box4],
    work: &mut Work,
) -> Result<Vec<Value>, Error> {
    let lines = eligible(p, false)?;
    if lines.is_empty() {
        return Ok(vec![]);
    }
    let em = median(lines.iter().map(effective).collect::<Result<Vec<_>, _>>()?);
    let mut rules = Vec::new();
    for r in optional(v.source, "drawing_lines", 5000)? {
        if r["orientation"] == "horizontal" {
            let b = bbox(&r["bbox"])?;
            if b.n[2] - b.n[0] >= 12. * em {
                rules.push(b);
            }
        }
    }
    let mut seeds = Vec::new();
    for rule in &rules {
        work.charge(lines.len() + 1)?;
        let mut header = lines
            .iter()
            .filter(|l| {
                let b = bbox(&l["bbox"]).unwrap();
                rule.n[0] - 0.3 * em <= b.n[0]
                    && b.n[2] <= rule.n[2] + 0.3 * em
                    && 0. <= rule.n[1] - b.n[3]
                    && rule.n[1] - b.n[3] <= 0.8 * em
            })
            .collect::<Vec<_>>();
        header.sort_by(|a, b| {
            cmp(
                a["bbox"][0].as_f64().unwrap(),
                b["bbox"][0].as_f64().unwrap(),
            )
        });
        if !(4..=10).contains(&header.len()) {
            continue;
        }
        let title = header[0]["text"].as_str().unwrap();
        if !title.chars().any(|c| c.is_ascii_alphabetic() || cjk(c))
            || title.chars().count() > 30
            || flag(header[0], "paragraph_terminal")?
            || header.windows(2).any(|p| {
                p[0]["bbox"][2].as_f64().unwrap() + 0.3 * em >= p[1]["bbox"][0].as_f64().unwrap()
            })
        {
            continue;
        }
        let years = header[1..]
            .iter()
            .map(|l| year(l["text"].as_str().unwrap()))
            .collect::<Option<Vec<_>>>();
        if years.is_some_and(|ys| ys.windows(2).all(|p| p[1] > p[0] && p[1] - p[0] <= 10)) {
            seeds.push((rule, header));
        }
    }
    let mut out = Vec::new();
    for (rule, header) in &seeds {
        work.charge(seeds.len() + lines.len() * 2 + 1)?;
        let ceiling = seeds
            .iter()
            .filter(|(r, _)| {
                r.n[1] > rule.n[3] && super::super::super::graphic_captions::axis(rule, r, 0) >= 0.9
            })
            .map(|(_, h)| {
                h.iter()
                    .map(|l| l["bbox"][1].as_f64().unwrap())
                    .reduce(f64::min)
                    .unwrap()
            })
            .reduce(f64::min)
            .unwrap_or(num(&p.size[1])?);
        let inside = lines
            .iter()
            .filter(|l| {
                let b = bbox(&l["bbox"]).unwrap();
                rule.n[3] <= b.n[1]
                    && b.n[1] < ceiling
                    && rule.n[0] - 0.3 * em <= b.n[0]
                    && b.n[2] <= rule.n[2] + 0.3 * em
            })
            .collect::<Vec<_>>();
        let numeric_lines = inside
            .iter()
            .copied()
            .filter(|l| {
                l["bbox"][0].as_f64().unwrap() >= header[0]["bbox"][2].as_f64().unwrap() + 0.3 * em
                    && numeric(l["text"].as_str().unwrap(), false)
            })
            .cloned()
            .collect();
        let prepared = clustered(v, numeric_lines, em, work)?;
        let mut group: Vec<&rowprep::Row> = Vec::new();
        let count = header.len() - 1;
        for row in &prepared.rows {
            work.charge(count + 1)?;
            if row.fragments.len() != count
                || group
                    .last()
                    .is_some_and(|r| row.b.n[1] - r.b.n[3] > 2. * em)
                || row
                    .fragments
                    .iter()
                    .zip(&header[1..])
                    .any(|(f, l)| (f.local.n[2] - l["bbox"][2].as_f64().unwrap()).abs() > 0.4 * em)
            {
                break;
            }
            group.push(row);
        }
        if group.len() < 3 || group[0].b.n[1] - rule.n[3] > 1.5 * em {
            continue;
        }
        let first_left = group
            .iter()
            .map(|r| r.fragments[0].local.n[0])
            .reduce(f64::min)
            .unwrap();
        let mut xs = vec![
            rule.n[0],
            (header[0]["bbox"][2].as_f64().unwrap() + first_left) / 2.,
        ];
        for i in 0..count - 1 {
            xs.push(
                (group
                    .iter()
                    .map(|r| r.fragments[i].local.n[2])
                    .reduce(f64::max)
                    .unwrap()
                    + group
                        .iter()
                        .map(|r| r.fragments[i + 1].local.n[0])
                        .reduce(f64::min)
                        .unwrap())
                    / 2.,
            );
        }
        xs.push(rule.n[2]);
        let top = header
            .iter()
            .min_by(|a, b| {
                cmp(
                    a["bbox"][1].as_f64().unwrap(),
                    b["bbox"][1].as_f64().unwrap(),
                )
            })
            .unwrap()["bbox"][1]
            .clone();
        let bottom = group.last().unwrap().b.n[3] + 0.3 * em;
        let mut ys = vec![num(&top)?, center(rule, 1)];
        ys.extend(group.windows(2).map(|p| (p[0].center + p[1].center) / 2.));
        ys.push(bottom);
        let b = bbox(&json!([rule.raw[0], top, rule.raw[2], bottom]))?;
        work.charge(excluded.len() + inside.len() + 1)?;
        if excluded.iter().any(|e| overlap_smaller(&b, e) >= 0.5) {
            continue;
        }
        let members = inside
            .iter()
            .copied()
            .filter(|l| {
                l["bbox"][1].as_f64().unwrap() < b.n[3]
                    && l["bbox"][2].as_f64().unwrap() <= xs[1]
                    && (l["bbox"][0].as_f64().unwrap() - header[0]["bbox"][0].as_f64().unwrap())
                        .abs()
                        <= 0.4 * em
            })
            .collect::<Vec<_>>();
        if members.len() < group.len() {
            continue;
        }
        let mut bad = false;
        for l in &members {
            let h = effective(l)?;
            bad |= h < 0.7 * em || h > 1.35 * em;
        }
        if bad {
            continue;
        }
        let mut ids = BTreeSet::new();
        for l in header.iter().copied().chain(members.iter().copied()) {
            ids.insert(integer(l, "source_index", 0)?);
        }
        for row in &group {
            for f in &row.fragments {
                ids.insert(f.id);
            }
        }
        let mut c = candidate(b.clone(), &[], vec![], grid(&b, &xs, &ys), 14)?;
        c["line_indices"] = json!(ids);
        c["preserve_inline_font_styles"] = json!(true);
        c["inferred_grid_authoritative"] = json!(true);
        out.push(c);
    }
    Ok(out)
}
fn metric_tables(
    v: &Request,
    p: &Page,
    excluded: &[Box4],
    work: &mut Work,
) -> Result<Vec<Value>, Error> {
    let lines = eligible(p, true)?;
    let mut out = Vec::new();
    for (hi, heading) in lines.iter().enumerate() {
        let title = heading["text"].as_str().unwrap().trim_matches(python_space);
        if !(4..=12).contains(&title.chars().count()) || !title.chars().all(cjk) {
            continue;
        }
        let em = effective(heading)? / 1.5;
        let hb = bbox(&heading["bbox"])?;
        let mut rules = Vec::new();
        for r in optional(v.source, "drawing_lines", 5000)? {
            work.charge(1)?;
            if r["orientation"] == "horizontal" {
                let b = bbox(&r["bbox"])?;
                if (b.n[0] - hb.n[2]).abs() <= em
                    && hb.n[1] <= b.n[1]
                    && b.n[1] <= hb.n[3] + 0.5 * em
                    && b.n[2] - b.n[0] >= 12. * em
                {
                    rules.push(b);
                }
            }
        }
        if rules.len() != 1 {
            continue;
        }
        let rule = &rules[0];
        let mut peers = Vec::new();
        for (i, l) in lines.iter().enumerate() {
            work.charge(1)?;
            if i == hi {
                continue;
            }
            let b = bbox(&l["bbox"])?;
            let h = effective(l)?;
            if hb.n[3] <= b.n[1]
                && b.n[1] < hb.n[3] + 15. * em
                && hb.n[0] - 0.4 * em <= b.n[0]
                && b.n[2] <= rule.n[2] + 0.3 * em
                && 0.85 * em <= h
                && h <= 1.15 * em
            {
                peers.push(l.clone());
            }
        }
        let prepared = clustered(v, peers, em, work)?;
        let mut group: Vec<&rowprep::Row> = Vec::new();
        for row in &prepared.rows {
            work.charge(
                row.fragments
                    .iter()
                    .map(|f| f.raw["text"].as_str().unwrap().len())
                    .sum::<usize>()
                    + 1,
            )?;
            if row.fragments.len() != 2 || group.last().is_some_and(|r| row.b.n[1] - r.b.n[3] > em)
            {
                break;
            }
            let label = row.fragments[0].raw["text"].as_str().unwrap();
            if !label.chars().any(cjk)
                || label.chars().filter(|c| !python_space(*c)).count() > 16
                || row.fragments[1].local.n[0] - row.fragments[0].local.n[2] < 2. * em
            {
                break;
            }
            group.push(row);
        }
        if group.len() < 5 || group[0].b.n[1] - hb.n[3] < 0. || group[0].b.n[1] - hb.n[3] > 1.5 * em
        {
            continue;
        }
        if (group
            .iter()
            .filter(|r| numeric(r.fragments[1].raw["text"].as_str().unwrap(), true))
            .count() as f64)
            < 0.8 * group.len() as f64
            || group
                .iter()
                .map(|r| r.fragments[1].local.n[2])
                .reduce(f64::max)
                .unwrap()
                - group
                    .iter()
                    .map(|r| r.fragments[1].local.n[2])
                    .reduce(f64::min)
                    .unwrap()
                > 0.4 * em
            || group
                .iter()
                .map(|r| r.fragments[0].local.n[0])
                .reduce(f64::max)
                .unwrap()
                - group
                    .iter()
                    .map(|r| r.fragments[0].local.n[0])
                    .reduce(f64::min)
                    .unwrap()
                > 0.4 * em
        {
            continue;
        }
        let left = group.iter().map(|r| r.b.n[0]).reduce(f64::min).unwrap() - 0.2 * em;
        let right = group.iter().map(|r| r.b.n[2]).reduce(f64::max).unwrap() + 0.2 * em;
        let split = (group
            .iter()
            .map(|r| r.fragments[0].local.n[2])
            .reduce(f64::max)
            .unwrap()
            + group
                .iter()
                .map(|r| r.fragments[1].local.n[0])
                .reduce(f64::min)
                .unwrap())
            / 2.;
        let b = bbox(&json!([
            left,
            group[0].b.n[1] - 0.2 * em,
            right,
            group.last().unwrap().b.n[3] + 0.2 * em
        ]))?;
        work.charge(excluded.len() + 1)?;
        if excluded.iter().any(|e| overlap_smaller(&b, e) >= 0.5) {
            continue;
        }
        let mut ys = vec![b.n[1]];
        ys.extend(group.windows(2).map(|p| (p[0].center + p[1].center) / 2.));
        ys.push(b.n[3]);
        let id = integer(heading, "source_index", 0)?;
        let annotation = json!({"kind":"caption","bbox":heading["bbox"],"line_indices":[id],"line_bboxes":[[id,heading["bbox"]]]});
        let ids: BTreeSet<_> = group
            .iter()
            .flat_map(|r| r.fragments.iter().map(|f| f.id))
            .collect();
        let mut c = candidate(
            b.clone(),
            &[],
            vec![annotation],
            grid(&b, &[left, split, right], &ys),
            14,
        )?;
        c["line_indices"] = json!(ids);
        c["preserve_inline_font_styles"] = json!(true);
        c["inferred_grid_authoritative"] = json!(true);
        out.push(c);
    }
    Ok(out)
}
fn partial(v: &Request, p: &Page, excluded: &[Box4], work: &mut Work) -> Result<Vec<Value>, Error> {
    let em = p.em;
    let tol = 0.75f64.max(0.15 * em);
    let source = v.source;
    let mut vertical = Vec::new();
    let mut horizontal = Vec::new();
    let raw = optional(source, "drawing_lines", 5000)?;
    for r in raw {
        let b = bbox(&r["bbox"])?;
        if r["orientation"] == "vertical" {
            vertical.push(b);
        } else if r["orientation"] == "horizontal" {
            horizontal.push(b);
        }
    }
    let forms = optional(source, "form_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let mut ordered = horizontal.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| cmp(a.n[1], b.n[1]).then(cmp(a.n[0], b.n[0])));
    let mut out: Vec<Value> = Vec::new();
    for rule in ordered {
        let (left, right, top) = (rule.n[0], rule.n[2], center(rule, 1));
        if right - left < 10. * em {
            continue;
        }
        work.charge(vertical.len() * 2 + excluded.len() + forms.len() + out.len() + 1)?;
        let outer = [left, right]
            .iter()
            .map(|x| {
                vertical.iter().find(|b| {
                    (center(b, 0) - x).abs() <= tol
                        && (b.n[1] - top).abs() <= tol
                        && b.n[3] - b.n[1] >= 5. * em
                })
            })
            .collect::<Vec<_>>();
        let (Some(a), Some(b)) = (outer[0], outer[1]) else {
            continue;
        };
        if (a.n[3] - b.n[3]).abs() > tol {
            continue;
        }
        let bottom = a.n[3].min(b.n[3]);
        let end = if a.n[3] >= b.n[3] {
            &a.raw[3]
        } else {
            &b.raw[3]
        };
        let b = bbox(&json!([rule.raw[0], rule.raw[1], rule.raw[2], end]))?;
        if excluded.iter().any(|e| overlap_smaller(&b, e) >= 0.5)
            || forms.iter().any(|f| overlap_first(&b, f) >= 0.9)
            || out
                .iter()
                .any(|c| overlap_first(&b, &bbox(&c["core_bbox"]).unwrap()) >= 0.9)
        {
            continue;
        }
        let mut xs = vertical
            .iter()
            .filter(|b| {
                left - tol <= center(b, 0)
                    && center(b, 0) <= right + tol
                    && top - tol <= b.n[1]
                    && b.n[1] <= top + 0.4 * (bottom - top)
                    && (b.n[3] - bottom).abs() <= tol
            })
            .map(|b| center(b, 0))
            .collect::<Vec<_>>();
        xs.sort_by(|a, b| cmp(*a, *b));
        let mut positions = Vec::new();
        for x in xs {
            if positions.last().is_none_or(|prior| x - prior > tol) {
                positions.push(x);
            }
        }
        if positions.len() < 4 {
            continue;
        }
        work.charge(horizontal.len() * (positions.len() + 1) + p.lines.len() + 1)?;
        let partial = horizontal
            .iter()
            .filter(|h| {
                top + tol < center(h, 1)
                    && center(h, 1) <= bottom + tol
                    && left + tol < h.n[0]
                    && h.n[0] < right - tol
                    && (h.n[2] - right).abs() <= tol
                    && positions.iter().any(|x| (h.n[0] - x).abs() <= tol)
            })
            .count();
        if partial < 2 {
            continue;
        }
        let mut members = p
            .line_values()
            .filter(|l| overlap_first(&bbox(&l["bbox"]).unwrap(), &b) >= 0.99)
            .collect::<Vec<_>>();
        members.sort_by(|a, b| {
            let a = bbox(&a["bbox"]).unwrap();
            let b = bbox(&b["bbox"]).unwrap();
            cmp(center(&a, 1), center(&b, 1)).then(cmp(a.n[0], b.n[0]))
        });
        let mut rows: Vec<Vec<&Value>> = Vec::new();
        for l in &members {
            if rows.last().is_some_and(|r| {
                (center(&bbox(&l["bbox"]).unwrap(), 1) - center(&bbox(&r[0]["bbox"]).unwrap(), 1))
                    .abs()
                    <= 0.5 * em
            }) {
                rows.last_mut().unwrap().push(l);
            } else {
                rows.push(vec![l]);
            }
        }
        let mut repeated = 0;
        for row in rows {
            let mut occupied = BTreeSet::new();
            for l in row {
                work.charge(positions.len() + 1)?;
                let x = center(&bbox(&l["bbox"])?, 0);
                for (i, p) in positions.windows(2).enumerate() {
                    if p[0] < x && x < p[1] {
                        occupied.insert(i);
                    }
                }
            }
            if occupied.len() >= 3 {
                repeated += 1;
            }
        }
        if repeated < 3 {
            continue;
        }
        let mut grid=raw.iter().map(|r|json!({"bbox":r["bbox"],"width":r.get("width").cloned().unwrap_or(json!(0.)),"orientation":r["orientation"]})).collect::<Vec<_>>();
        if !horizontal.iter().any(|h| {
            (center(h, 1) - bottom).abs() <= tol
                && (h.n[0] - left).abs() <= tol
                && (h.n[2] - right).abs() <= tol
        }) {
            grid.push(json!({"bbox":[rule.raw[0],bottom-0.05,rule.raw[2],bottom+0.05],"width":0.1,"orientation":"horizontal"}));
        }
        let mut c = candidate(b, &members, vec![], grid, 200)?;
        c["inferred_grid_authoritative"] = json!(true);
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
    match v.options["kind"].as_str() {
        Some("partial_grid") => partial(v, p, excluded, work),
        Some("year_header") => year_tables(v, p, excluded, work),
        _ => metric_tables(v, p, excluded, work),
    }
}
