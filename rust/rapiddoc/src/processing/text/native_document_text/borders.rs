//! Original rule-delimited headers/footers and outer page-number companions.
use super::*;
fn drawings(p: &Value) -> Result<&[Value], Error> {
    optional(p, "drawing_lines", 5000).and_then(|a| {
        if p.get("drawing_lines").is_some() {
            Ok(a)
        } else {
            optional(&p["source"], "drawing_lines", 5000)
        }
    })
}
fn blocked(r: &Value, p: &Value) -> Result<bool, Error> {
    let b = bbox(&r["original_bbox"])?;
    let d = num(&r["width"])?.max(1.);
    let b = bbox(&json!([b.n[0] - d, b.n[1] - d, b.n[2] + d, b.n[3] + d]))?;
    let s = sz(p)?;
    for block in optional(p, "fixed_blocks", 4000)? {
        if !matches!(
            block["type"].as_str(),
            Some("table" | "image" | "equation" | "code")
        ) {
            continue;
        }
        if let Some(c) = coerce(&block["bbox"])?
            .map(|b| clip(&b, s))
            .transpose()?
            .flatten()
        {
            if intersects(&b, &c) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
pub(super) fn classify(pages: &mut [Value], w: &mut Work) -> Result<(), Error> {
    // Keep whole-document pass order: all headers, then all footers, then companions.
    for p in pages.iter_mut() {
        headers(p, w)?
    }
    for p in pages.iter_mut() {
        footers(p, w)?
    }
    for p in pages {
        companions(p, w)?
    }
    Ok(())
}
fn headers(p: &mut Value, w: &mut Work) -> Result<(), Error> {
    let s = sz(p)?;
    let mut lines = ls(p)?.to_vec();
    let available = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l["semantic_type"].is_null())
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    if available.is_empty() || drawings(p)?.is_empty() {
        return Ok(());
    }
    let Some(a) = dominant(&lines, s)? else {
        return Ok(());
    };
    let lp = local_size(s, a);
    if lp[0] <= 0. || lp[1] <= 0. {
        return Ok(());
    }
    let rows = available
        .iter()
        .filter(|&&i| integer(&lines[i], "angle", 0).ok() == Some(a))
        .map(|&i| Ok((i, bs(&lines[i], s, a, true)?)))
        .collect::<Result<Vec<_>, Error>>()?;
    let evidence = lines
        .iter()
        .filter(|l| {
            integer(l, "angle", 0).ok() == Some(a)
                && (l["semantic_type"].is_null()
                    || matches!(l["semantic_type"].as_str(), Some("header" | "page_number")))
        })
        .map(|l| bs(l, s, a, true))
        .collect::<Result<Vec<_>, _>>()?;
    let em = if rows.is_empty() {
        1.
    } else {
        median(
            rows.iter()
                .map(|(i, b)| height(&lines[*i], b))
                .collect::<Result<Vec<_>, _>>()?,
        )
    };
    let scale = if rows.is_empty() {
        em
    } else {
        median(
            rows.iter()
                .map(|(i, b)| canonical(&lines[*i], b))
                .collect::<Result<Vec<_>, _>>()?,
        )
    };
    let mut separator = None;
    for r in super::super::native_footnotes::prepared_rules(p, a, w)? {
        w.charge(evidence.len() + 1)?;
        let b = bbox(&r["bbox"])?;
        if r["horizontal"] == true
            && cy(&b) <= 0.15 * lp[1]
            && b.n[2] - b.n[0] >= 0.6 * lp[0]
            && evidence.iter().any(|e| e.n[3] <= cy(&b))
            && r["table"] != true
            && !blocked(&r, p)?
            && separator.as_ref().is_none_or(|old: &Box4| cy(&b) < cy(old))
        {
            separator = Some(b)
        }
    }
    let Some(b) = separator else { return Ok(()) };
    let y = cy(&b);
    if !rows.iter().any(|(_, b)| cy(b) >= y + em) {
        return Ok(());
    }
    for (i, b) in rows {
        if b.n[3] <= y
            && lines[i]["caption_start"] != true
            && lines[i]["marginal_body_context"] != true
        {
            if canonical(&lines[i], &b)? >= 1.5 * scale {
                if lines[i]["numbered_heading_start"] == true {
                    lines[i]["semantic_type"] = json!("paragraph_title");
                    lines[i]["structural_title"] = json!(true);
                    lines[i]["explicit_section_title"] = json!(true)
                }
                continue;
            }
            lines[i]["semantic_type"] = json!("header")
        }
    }
    p["remaining_lines"] = json!(lines);
    Ok(())
}
fn footers(p: &mut Value, w: &mut Work) -> Result<(), Error> {
    let s = sz(p)?;
    let mut lines = ls(p)?.to_vec();
    let available = lines
        .iter()
        .enumerate()
        .filter(|(_, l)| l["semantic_type"].is_null())
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    if available.is_empty() || drawings(p)?.is_empty() {
        return Ok(());
    }
    let Some(a) = dominant(&lines, s)? else {
        return Ok(());
    };
    let lp = local_size(s, a);
    if lp[0] <= 0. || lp[1] <= 0. {
        return Ok(());
    }
    let mut rules = Vec::new();
    for r in super::super::native_footnotes::prepared_rules(p, a, w)? {
        let b = bbox(&r["bbox"])?;
        if r["horizontal"] == true
            && cy(&b) >= 0.85 * lp[1]
            && b.n[2] - b.n[0] >= 0.2 * lp[0]
            && r["table"] != true
            && !blocked(&r, p)?
        {
            rules.push(b)
        }
    }
    let rows = available
        .iter()
        .filter(|&&i| integer(&lines[i], "angle", 0).ok() == Some(a))
        .map(|&i| Ok((i, bs(&lines[i], s, a, false)?)))
        .collect::<Result<Vec<_>, Error>>()?;
    if rows.is_empty() {
        return Ok(());
    }
    let em = median(
        rows.iter()
            .map(|(i, b)| height(&lines[*i], b))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let lanes = super::super::table_text_styles::lanes::infer_layout(
        &json!({"line_geometry":rows.iter().map(|(i,b)|json!([lines[*i],b.raw])).collect::<Vec<_>>(),"local_page_width":lp[0],"median_height":em,"recalculate_intervals":false}),
        w,
    )?;
    let lanes = array(&lanes, 4096)?
        .iter()
        .filter(|l| l["is_span"] != true)
        .collect::<Vec<_>>();
    for (i, upper) in rules.iter().enumerate() {
        for lower in &rules[i + 1..] {
            w.charge(rows.len() + 1)?;
            let gap = lower.n[1] - upper.n[3];
            if gap < 2. * em || gap > 6. * em || axis(upper, lower, 0) < 0.9 {
                continue;
            }
            let left = upper.n[0].max(lower.n[0]);
            let right = upper.n[2].min(lower.n[2]);
            let members = rows
                .iter()
                .filter(|(_, b)| {
                    b.n[1] >= upper.n[3]
                        && b.n[3] <= lower.n[1]
                        && b.n[0] >= left - 0.5 * em
                        && b.n[2] <= right + 0.5 * em
                })
                .collect::<Vec<_>>();
            if !(1..=3).contains(&members.len())
                || members.iter().any(|(_, b)| {
                    (cx(b) - 0.5 * (left + right)).abs() > 0.15 * (right - left).max(0.1)
                })
            {
                continue;
            }
            for (i, _) in members {
                lines[*i]["semantic_type"] = json!("footer")
            }
            break;
        }
    }
    for rule in &rules {
        let rw = (rule.n[2] - rule.n[0]).max(0.1);
        let mut matches = Vec::new();
        for lane in &lanes {
            let left = num(&lane["left"])?;
            let right = num(&lane["right"])?;
            if (rule.n[2].min(right) - rule.n[0].max(left)).max(0.) / rw >= 0.8
                && left <= cx(rule)
                && cx(rule) <= right
            {
                matches.push((left, right))
            }
        }
        if matches.len() != 1 {
            continue;
        }
        let (left, right) = matches[0];
        if lanes.len() > 1
            && left
                < lanes
                    .iter()
                    .map(|l| num(&l["left"]))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .reduce(f64::max)
                    .unwrap()
                    - em
        {
            continue;
        }
        let mut below = rows
            .iter()
            .filter(|(i, b)| {
                lines[*i]["semantic_type"].is_null()
                    && b.n[1] >= rule.n[3]
                    && b.n[0] >= left - 0.5 * em
                    && b.n[2] <= right + 0.5 * em
            })
            .collect::<Vec<_>>();
        below.sort_by(|(i, a), (j, b)| {
            a.n[1]
                .total_cmp(&b.n[1])
                .then(a.n[0].total_cmp(&b.n[0]))
                .then(
                    integer(&lines[*i], "source_index", 0)
                        .unwrap()
                        .cmp(&integer(&lines[*j], "source_index", 0).unwrap()),
                )
        });
        let Some((i, first)) = below.first().copied() else {
            continue;
        };
        if first.n[1] - rule.n[3] > em || height(&lines[*i], first)? > 0.9 * em {
            continue;
        }
        let mut members = vec![(i, first)];
        for (i, b) in below.into_iter().skip(1) {
            w.charge(1)?;
            if b.n[1] - members.last().unwrap().1.n[3] > em
                || b.n[3] - first.n[1] > 5. * em
                || height(&lines[*i], b)? > 0.9 * em
            {
                break;
            }
            members.push((i, b))
        }
        if !(2..=8).contains(&members.len()) {
            continue;
        }
        let min = members
            .iter()
            .map(|(_, b)| b.n[0])
            .reduce(f64::min)
            .unwrap();
        let max = members
            .iter()
            .map(|(_, b)| b.n[0])
            .reduce(f64::max)
            .unwrap();
        if max - min <= 0.75 * em {
            for (i, _) in members {
                lines[*i]["semantic_type"] = json!("footer")
            }
        }
    }
    p["remaining_lines"] = json!(lines);
    Ok(())
}
fn companions(p: &mut Value, w: &mut Work) -> Result<(), Error> {
    let s = sz(p)?;
    let mut lines = ls(p)?.to_vec();
    let mut blocks = optional(p, "fixed_blocks", 4000)?.to_vec();
    let anchors = lines
        .iter()
        .filter(|l| l["semantic_type"] == "page_number")
        .cloned()
        .collect::<Vec<_>>();
    for anchor in anchors {
        let a = integer(&anchor, "angle", 0)?;
        let h = local_size(s, a)[1];
        if h <= 0. {
            continue;
        }
        let b = bs(&anchor, s, a, false)?;
        let y = cy(&b) / h;
        let (target, edge) = if y <= 0.3 {
            ("header", b.n[1])
        } else if y >= 0.7 {
            ("footer", b.n[3])
        } else {
            continue;
        };
        let matches = |c: &Box4| {
            (if target == "header" {
                c.n[3] <= edge
            } else {
                c.n[1] >= edge
            }) || (axis(c, &b, 1) >= 0.5 && c.n[3] - c.n[1] <= 0.15 * h)
        };
        for l in &mut lines {
            w.charge(1)?;
            if l["semantic_type"].is_null()
                && integer(l, "angle", 0)? == a
                && matches(&bs(l, s, a, false)?)
            {
                l["semantic_type"] = json!(target)
            }
        }
        for block in &mut blocks {
            w.charge(1)?;
            if block["type"] != "image" {
                continue;
            }
            let angle = if block["angle"].is_null() {
                0
            } else {
                super::super::table_native_text::number(&block["angle"])?.ok_or_else(invalid)?
                    as i64
            }
            .rem_euclid(360);
            if angle != a {
                continue;
            }
            if let Some(c) = coerce(&block["bbox"])?
                .map(|b| clip(&b, s))
                .transpose()?
                .flatten()
            {
                if matches(&local(&c, s, a)) {
                    block["type"] = json!(target)
                }
            }
        }
    }
    p["remaining_lines"] = json!(lines);
    p["fixed_blocks"] = json!(blocks);
    Ok(())
}
