//! 有标题的空答案表与页面右界裁切表单完整旧规则。
use super::*;
fn split(header: &Value, work: &mut Work) -> Result<Option<f64>, Error> {
    let mut chars = Vec::new();
    for c in optional(header, "chars", 20000)? {
        work.charge(1)?;
        let t = c.get("char").and_then(Value::as_str).unwrap_or("");
        if !t.trim_matches(python_space).is_empty() && !c["bbox"].is_null() {
            chars.push(bbox(&c["bbox"])?);
        }
    }
    chars.sort_by(|a, b| cmp(a.n[0], b.n[0]));
    let h = effective(header)?;
    let gaps = chars
        .windows(2)
        .filter_map(|p| {
            let gap = p[1].n[0] - p[0].n[2];
            (gap > 0.1 * h).then_some((gap, (p[1].n[0] + p[0].n[2]) / 2.))
        })
        .collect::<Vec<_>>();
    if gaps.len() < 2 {
        return Ok(None);
    }
    let typical = median(gaps.iter().map(|g| g.0).collect());
    let splits = gaps
        .iter()
        .filter(|g| g.0 >= (0.6 * h).max(2.5 * typical))
        .collect::<Vec<_>>();
    Ok(if splits.len() == 1 {
        Some(splits[0].1)
    } else {
        None
    })
}
fn blank(v: &Request, p: &Page, excluded: &[Box4], work: &mut Work) -> Result<Vec<Value>, Error> {
    let source = v.source;
    let mut hs = Vec::new();
    let mut vs = Vec::new();
    for r in optional(source, "drawing_lines", 5000)? {
        let b = bbox(&r["bbox"])?;
        if r["orientation"] == "horizontal" && b.n[2] - b.n[0] >= 6. * p.em {
            hs.push(b);
        } else if r["orientation"] == "vertical" {
            vs.push(b);
        }
    }
    hs.sort_by(|a, b| cmp(a.n[1], b.n[1]));
    let mut out = Vec::new();
    for pair in hs.windows(2) {
        let (top, bottom) = (&pair[0], &pair[1]);
        let b = bbox(&json!([top.raw[0], top.raw[1], top.raw[2], bottom.raw[3]]))?;
        work.charge(excluded.len() + vs.len() + p.lines.len() + 1)?;
        if (top.n[0] - bottom.n[0]).abs() > 0.25 * p.em
            || (top.n[2] - bottom.n[2]).abs() > 0.25 * p.em
            || bottom.n[1] - top.n[3] < 5. * p.em
            || bottom.n[1] - top.n[3] > 20. * p.em
            || excluded.iter().any(|e| overlap_smaller(&b, e) >= 0.5)
            || vs.iter().any(|v| {
                super::super::super::graphic_captions::axis(&b, v, 1) >= 0.5
                    && b.n[0] <= center(v, 0)
                    && center(v, 0) <= b.n[2]
            })
        {
            continue;
        }
        let annotations = bounded(v, p, &b, work)?;
        if annotations.len() != 1 || annotations[0]["bbox"][3].as_f64().unwrap() > top.n[1] {
            continue;
        }
        let mut members = p
            .line_values()
            .filter(|l| overlap_first(&bbox(&l["bbox"]).unwrap(), &b) >= 0.9)
            .collect::<Vec<_>>();
        members.sort_by(|a, b| {
            cmp(
                a["bbox"][1].as_f64().unwrap(),
                b["bbox"][1].as_f64().unwrap(),
            )
        });
        if !(6..=13).contains(&members.len()) {
            continue;
        }
        let header = members[0];
        let keys = &members[1..];
        if header["font_signature"].is_null()
            || !bold(header)?
            || header["bbox"][1].as_f64().unwrap() - top.n[3] > p.em
            || bottom.n[1] - keys.last().unwrap()["bbox"][3].as_f64().unwrap() > p.em
        {
            continue;
        }
        let Some(split) = split(header, work)? else {
            continue;
        };
        let hh = effective(header)?;
        let mut reject = split - b.n[0] < 3. * p.em || b.n[2] - split < 4. * p.em;
        for key in keys {
            work.charge(1)?;
            let kb = bbox(&key["bbox"])?;
            let h = effective(key)?;
            if words(key) > 3
                || flag(key, "paragraph_terminal")?
                || (kb.n[0] - header["bbox"][0].as_f64().unwrap()).abs() > 0.35 * p.em
                || kb.n[2] > split - 0.35 * p.em
                || h < 0.8 * hh
                || h > 1.2 * hh
            {
                reject = true;
            }
        }
        if reject
            || members.windows(2).any(|pair| {
                let gap =
                    pair[1]["bbox"][1].as_f64().unwrap() - pair[0]["bbox"][3].as_f64().unwrap();
                gap < 0.2 * p.em || gap > 1.5 * p.em
            })
        {
            continue;
        }
        let mut grid = [b.n[0], split, b.n[2]]
            .iter()
            .map(|x| axis(&b, *x, false))
            .collect::<Vec<_>>();
        let mut ys = vec![b.n[1]];
        for pair in members.windows(2) {
            ys.push(
                (pair[0]["bbox"][3].as_f64().unwrap() + pair[1]["bbox"][1].as_f64().unwrap()) / 2.,
            );
        }
        ys.push(b.n[3]);
        grid.extend(ys.into_iter().map(|y| axis(&b, y, true)));
        out.push(candidate(b, &members, annotations, grid, 12)?);
    }
    Ok(out)
}
fn clipped(v: &Request, p: &Page, excluded: &[Box4], work: &mut Work) -> Result<Vec<Value>, Error> {
    let width = num(&p.size[0])?;
    let mut hs = Vec::new();
    let mut vs = Vec::new();
    for r in optional(v.source, "drawing_lines", 5000)? {
        let b = bbox(&r["bbox"])?;
        if r["orientation"] == "horizontal"
            && (b.n[2] - width).abs() <= 0.1 * p.em
            && b.n[2] - b.n[0] >= 8. * p.em
        {
            hs.push(b.clone());
        }
        if r["orientation"] == "vertical" {
            vs.push(b);
        }
    }
    hs.sort_by(|a, b| cmp(a.n[1], b.n[1]));
    if hs.len() < 4 {
        return Ok(vec![]);
    }
    let first = &hs[0];
    let last = hs.last().unwrap();
    if last.n[3] - first.n[1] < 6. * p.em
        || last.n[3] - first.n[1] > 20. * p.em
        || hs.iter().any(|b| (b.n[0] - first.n[0]).abs() > 0.25 * p.em)
    {
        return Ok(vec![]);
    }
    let left = hs.iter().min_by(|a, b| cmp(a.n[0], b.n[0])).unwrap();
    let b = bbox(&json!([left.raw[0], first.raw[1], p.size[0], last.raw[3]]))?;
    work.charge(excluded.len() + vs.len() + 1)?;
    if excluded.iter().any(|e| overlap_smaller(&b, e) >= 0.5) {
        return Ok(vec![]);
    }
    let mut vertical = vs
        .iter()
        .filter(|v| {
            b.n[0] - 0.25 * p.em <= center(v, 0)
                && center(v, 0) < width - p.em
                && v.n[3].min(b.n[3]) - v.n[1].max(b.n[1]) >= 0.65 * (b.n[3] - b.n[1])
        })
        .collect::<Vec<_>>();
    vertical.sort_by(|a, b| cmp(a.n[0], b.n[0]));
    if vertical.len() != 3 || (center(vertical[0], 0) - b.n[0]).abs() > 0.25 * p.em {
        return Ok(vec![]);
    }
    let mut columns = vertical.iter().map(|v| center(v, 0)).collect::<Vec<_>>();
    columns.push(width);
    if columns.windows(2).any(|pair| pair[1] - pair[0] < 3. * p.em) {
        return Ok(vec![]);
    }
    let mut rules = Vec::new();
    for r in optional(v.source, "drawing_lines", 5000)? {
        work.charge(1)?;
        if r["orientation"] == "horizontal" {
            let rb = bbox(&r["bbox"])?;
            if b.n[1] <= center(&rb, 1)
                && center(&rb, 1) <= b.n[3]
                && (rb.n[0] - b.n[0]).abs() <= 0.25 * p.em
                && rb.n[2] >= columns[2] - 0.25 * p.em
            {
                rules.push(rb);
            }
        }
    }
    rules.sort_by(|a, b| cmp(a.n[1], b.n[1]));
    if !(6..=12).contains(&rules.len())
        || rules.windows(2).any(|pair| {
            let d = center(&pair[1], 1) - center(&pair[0], 1);
            d < 0.75 * p.em || d > 4. * p.em
        })
    {
        return Ok(vec![]);
    }
    work.charge(p.lines.len() + 1)?;
    let members = p
        .line_values()
        .filter(|l| overlap_first(&bbox(&l["bbox"]).unwrap(), &b) >= 0.9)
        .collect::<Vec<_>>();
    let header_bottom = center(&rules[1], 1);
    let header = members
        .iter()
        .copied()
        .filter(|l| l["bbox"][3].as_f64().unwrap() <= header_bottom)
        .collect::<Vec<_>>();
    let data = members
        .iter()
        .copied()
        .filter(|l| l["bbox"][1].as_f64().unwrap() >= header_bottom)
        .collect::<Vec<_>>();
    if header.len() != 4
        || data.is_empty()
        || data
            .iter()
            .any(|l| l["bbox"][2].as_f64().unwrap() > columns[1] - 0.5 * p.em)
    {
        return Ok(vec![]);
    }
    let mut left = Vec::new();
    let mut right = Vec::new();
    for l in &header {
        if center(&bbox(&l["bbox"])?, 0) < columns[2] {
            left.push(*l);
        } else {
            right.push(*l);
        }
    }
    for group in [&mut left, &mut right] {
        group.sort_by(|a, b| {
            cmp(
                a["bbox"][1].as_f64().unwrap(),
                b["bbox"][1].as_f64().unwrap(),
            )
        });
    }
    if left.len() != 2 || right.len() != 2 {
        return Ok(vec![]);
    }
    let left_right = left
        .iter()
        .map(|l| l["bbox"][2].as_f64().unwrap())
        .reduce(f64::max)
        .unwrap();
    let right_left = right
        .iter()
        .map(|l| l["bbox"][0].as_f64().unwrap())
        .reduce(f64::min)
        .unwrap();
    if left.iter().zip(&right).any(|(a, b)| {
        (a["bbox"][1].as_f64().unwrap() - b["bbox"][1].as_f64().unwrap()).abs() > 0.2 * p.em
    }) || left_right + 0.25 * p.em > right_left
    {
        return Ok(vec![]);
    }
    for l in &header {
        if flag(l, "paragraph_terminal")? {
            return Ok(vec![]);
        }
    }
    work.charge(rules.len() * (data.len() + 1))?;
    if !rules[1..].windows(2).all(|pair| {
        data.iter().any(|l| {
            let y = center(&bbox(&l["bbox"]).unwrap(), 1);
            center(&pair[0], 1) <= y && y <= center(&pair[1], 1)
        })
    }) {
        return Ok(vec![]);
    }
    columns[2] = columns[2].min(right_left - 0.1 * p.em);
    let top = center(&rules[0], 1);
    let bottom = center(rules.last().unwrap(), 1);
    let mut grid = columns
        .iter()
        .map(|x| json!({"bbox":[x-0.05,top,x+0.05,bottom],"width":0.1,"orientation":"vertical"}))
        .collect::<Vec<_>>();
    grid.extend(rules.iter().map(|r| {
        let y = center(r, 1);
        json!({"bbox":[columns[0],y-0.05,p.size[0],y+0.05],"width":0.1,"orientation":"horizontal"})
    }));
    let mut c = candidate(b, &members, vec![], grid, 120)?;
    c["preserve_inline_font_styles"] = json!(true);
    c["inferred_grid_authoritative"] = json!(true);
    Ok(vec![c])
}
pub(super) fn detect(
    v: &Request,
    p: &Page,
    excluded: &[Box4],
    work: &mut Work,
) -> Result<Vec<Value>, Error> {
    if v.options["kind"] == "blank_answer" {
        blank(v, p, excluded, work)
    } else {
        clipped(v, p, excluded, work)
    }
}
