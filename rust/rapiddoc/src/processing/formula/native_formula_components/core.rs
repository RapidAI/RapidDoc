//! Contiguous original formula pass, from split visual rows to lane components.
use super::evidence as e;
use super::*;
fn split_rows(p: &mut Page, w: &mut Work) -> Result<(Vec<Value>, BTreeSet<i64>), Error> {
    let mut groups = Vec::<((i64, i64), Vec<usize>)>::new();
    for (i, l) in p.lines.iter().enumerate() {
        if l["visual_row_id"].is_null() || l["split_from_row"] != true {
            continue;
        }
        let key = (integer(l, "angle", 0)?, integer(l, "visual_row_id", 0)?);
        if let Some((_, ids)) = groups.iter_mut().find(|(k, _)| *k == key) {
            ids.push(i)
        } else {
            groups.push((key, vec![i]))
        }
    }
    let mut blocks = Vec::new();
    let mut claimed = BTreeSet::new();
    for ((angle, _), ids) in groups {
        if ids.len() < 3 {
            continue;
        }
        let rows = ids
            .iter()
            .map(|&i| {
                Ok(Row {
                    index: i,
                    b: local(&bbox(&p.lines[i]["bbox"])?, p.size, angle),
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let markers = rows
            .iter()
            .filter(|r| standalone(p.t(r).unwrap()).is_some())
            .collect::<Vec<_>>();
        if markers.len() != 1 {
            continue;
        }
        let marker = markers[0];
        let mut table = false;
        for r in &rows {
            w.charge(p.tables.len() + 1)?;
            let b = bbox(&p.l(r)["bbox"])?;
            if p.tables.iter().any(|t| intersects(&b, t)) {
                table = true;
                break;
            }
        }
        if table {
            continue;
        }
        let body = rows
            .iter()
            .filter(|r| r.index != marker.index)
            .cloned()
            .collect::<Vec<_>>();
        if body.is_empty()
            || marker.b.n[0] <= body.iter().map(|r| cx(&r.b)).reduce(f64::max).unwrap()
        {
            continue;
        }
        let h = median(rows.iter().map(|r| p.h(r)).collect::<Result<Vec<_>, _>>()?);
        let center = median(rows.iter().map(|r| cy(&r.b)).collect());
        if rows.iter().any(|r| (cy(&r.b) - center).abs() > 0.75 * h) {
            continue;
        }
        let mut fonts = Vec::<Value>::new();
        for r in &body {
            let f = &p.l(r)["font_signature"];
            if !f.is_null() && !fonts.iter().any(|other| eq(f, other)) {
                fonts.push(f.clone())
            }
        }
        let mut typography = fonts.len() >= 2;
        for r in &body {
            typography |= p.l(r)["compact_formula_cluster"] == true || p.coverage(r)? < 0.8
        }
        if !typography {
            continue;
        }
        let b = p.bounds(&body)?;
        let width = (0.65 * (b.n[2] - b.n[0]).max(0.1)).max(3. * h);
        let mut nearby = false;
        for i in 0..p.lines.len() {
            w.charge(1)?;
            if ids.contains(&i) || integer(&p.lines[i], "angle", 0)? != angle {
                continue;
            }
            let other = local(&bbox(&p.lines[i]["bbox"])?, p.size, angle);
            let vg = (other.n[1].max(b.n[1]) - other.n[3].min(b.n[3])).max(0.);
            let hg = (other.n[0].max(b.n[0]) - other.n[2].min(b.n[2])).max(0.);
            if vg <= 0.75 * h && other.n[2] - other.n[0] <= width && hg <= h {
                nearby = true;
                break;
            }
        }
        if nearby {
            continue;
        }
        let Some(block) = materialize::block(p, &rows, angle, p.id(marker)?, w)? else {
            continue;
        };
        let b = p.bounds(&rows)?;
        if margin(
            &b,
            if angle == 90 || angle == 270 {
                p.size[0]
            } else {
                p.size[1]
            },
        ) {
            continue;
        }
        blocks.push(block);
        for r in &rows {
            claimed.insert(p.id(r)?);
        }
    }
    Ok((blocks, claimed))
}
fn fraction(
    p: &Page,
    members: &[Row],
    lane: &Lane,
    em: f64,
    rules: &[Box4],
) -> Result<bool, Error> {
    if members.len() < 3 || rules.is_empty() {
        return Ok(false);
    }
    let markers = members
        .iter()
        .filter(|r| standalone(p.t(r).unwrap()).is_some())
        .collect::<Vec<_>>();
    if markers.len() != 1 {
        return Ok(false);
    }
    let marker = markers[0];
    let width = (lane.right - lane.left).max(0.1);
    if marker.b.n[2] < lane.right - 3f64.max(0.08 * width)
        || marker.b.n[2] - marker.b.n[0] > 0.12 * width
    {
        return Ok(false);
    }
    let body = members
        .iter()
        .filter(|r| r.index != marker.index)
        .cloned()
        .collect::<Vec<_>>();
    if body.len() < 2 {
        return Ok(false);
    }
    let b = p.bounds(&body)?;
    let ids = members
        .iter()
        .map(|r| p.id(r))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let above = lane
        .rows
        .iter()
        .filter(|r| !ids.contains(&p.id(r).unwrap()) && r.b.n[3] <= b.n[1])
        .map(|r| r.b.n[3])
        .reduce(f64::max);
    let below = lane
        .rows
        .iter()
        .filter(|r| !ids.contains(&p.id(r).unwrap()) && r.b.n[1] >= b.n[3])
        .map(|r| r.b.n[1])
        .reduce(f64::min);
    let (Some(above), Some(below)) = (above, below) else {
        return Ok(false);
    };
    if (b.n[1] - above).min(below - b.n[3]) < 0.75 * em {
        return Ok(false);
    }
    for rule in rules {
        let rw = rule.n[2] - rule.n[0];
        if rw < 3. * em
            || rw > 0.75 * width
            || (rule.n[2].min(b.n[2]) - rule.n[0].max(b.n[0])).max(0.) < 0.6 * rw
        {
            continue;
        }
        let y = cy(rule);
        if body.iter().any(|r| cy(&r.b) <= y - 0.1 * em)
            && body.iter().any(|r| cy(&r.b) >= y + 0.1 * em)
        {
            return Ok(true);
        }
    }
    Ok(false)
}
fn paragraph(p: &mut Page, members: &[Row], em: f64, w: &mut Work) -> Result<Value, Error> {
    let mut ordered = members.to_vec();
    ordered.sort_by(|a, b| {
        a.b.n[1]
            .total_cmp(&b.b.n[1])
            .then(a.b.n[0].total_cmp(&b.b.n[0]))
    });
    let mut rows = Vec::<Vec<Row>>::new();
    for r in ordered {
        if let Some(row) = rows
            .iter_mut()
            .find(|row| row.iter().any(|other| axis(&r.b, &other.b, 1) >= 0.55))
        {
            row.push(r)
        } else {
            rows.push(vec![r])
        }
    }
    let raw = members.iter().map(|r| p.l(r).clone()).collect::<Vec<_>>();
    let geometry = members.iter().map(|r| json!(r.b.raw)).collect::<Vec<_>>();
    let mut merged = super::super::table_text_styles::clusters::materialize_page_members(
        &raw, &geometry, p.size, em, false, w,
    )?;
    let mut contents = Vec::new();
    for row in rows {
        contents.push(materialize::join(
            p,
            &row.into_iter().map(|r| (r, None)).collect::<Vec<_>>(),
            w,
        )?)
    }
    merged["text"] = json!(contents.join(" "));
    merged["paragraph_formula_context"] = json!(true);
    merged["formula_candidate_only"] = json!(false);
    Ok(merged)
}
pub(super) fn build(
    p: &mut Page,
    drawings: &[Value],
    w: &mut Work,
) -> Result<(Vec<Value>, BTreeSet<i64>, Vec<Value>), Error> {
    let (mut blocks, mut claimed) = split_rows(p, w)?;
    let mut paragraphs = Vec::new();
    let angles = p
        .lines
        .iter()
        .map(|l| integer(l, "angle", 0))
        .collect::<Result<BTreeSet<_>, _>>()?;
    for angle in angles {
        let mut rows = Vec::new();
        for (i, l) in p.lines.iter().enumerate() {
            if integer(l, "angle", 0)? == angle
                && !claimed.contains(&integer(l, "source_index", 0)?)
            {
                rows.push(Row {
                    index: i,
                    b: local(&bbox(&l["bbox"])?, p.size, angle),
                })
            }
        }
        if rows.len() < 2 {
            continue;
        }
        let em = median(rows.iter().map(|r| p.h(r)).collect::<Result<Vec<_>, _>>()?);
        let lp = if angle == 90 || angle == 270 {
            [p.size[1], p.size[0]]
        } else {
            p.size
        };
        let mut rules = Vec::new();
        for rule in drawings {
            let horizontal = if angle == 90 || angle == 270 {
                rule["orientation"] == "vertical"
            } else {
                rule["orientation"] == "horizontal"
            };
            if horizontal {
                rules.push(local(&bbox(&rule["bbox"])?, p.size, angle))
            }
        }
        for mut lane in p.lanes(&rows, lp[0], em, w)? {
            let font = e::body_font(p, &lane, em)?;
            for r in lane.rows.clone() {
                w.charge(lane.rows.len() + 1)?;
                if !e::single_numbered(p, &r, &lane, em)? {
                    continue;
                }
                let members = anchors::expand_single(p, &r, &lane, &claimed, &font, em, w)?;
                if let Some(block) = materialize::block(p, &members, angle, p.id(&r)?, w)? {
                    blocks.push(block);
                    for r in members {
                        claimed.insert(p.id(&r)?);
                    }
                }
            }
            lane.rows.retain(|r| !claimed.contains(&p.id(r).unwrap()));
            for r in lane.rows.clone() {
                w.charge(lane.rows.len() + 1)?;
                let isolated = p.l(&r)["compact_formula_cluster"] == true
                    && !e::nearby_number(p, &r, &lane, em)?
                    && e::compact(p, &r, &lane, em)?;
                if (isolated || e::unnumbered(p, &r, &lane, em, &font)?) && !margin(&r.b, lp[1]) {
                    if let Some(block) = materialize::single(p, &r, angle)? {
                        blocks.push(block);
                        claimed.insert(p.id(&r)?);
                    }
                }
            }
            lane.rows.retain(|r| !claimed.contains(&p.id(r).unwrap()));
            if lane.rows.len() < 2 {
                continue;
            }
            let anchors = anchors::find(p, &lane, em, &font, w)?;
            if anchors.is_empty() {
                continue;
            }
            let top = lane.rows.iter().map(|r| r.b.n[1]).reduce(f64::min).unwrap();
            let bottom = lane.rows.iter().map(|r| r.b.n[3]).reduce(f64::max).unwrap();
            for (k, anchor) in anchors.iter().enumerate() {
                if claimed.contains(&p.id(&anchor.row)?) {
                    continue;
                }
                let bt = if k > 0 {
                    top.max((cy(&anchors[k - 1].row.b) + cy(&anchor.row.b)) / 2.)
                } else {
                    top
                };
                let bb = if k + 1 < anchors.len() {
                    bottom.min((cy(&anchors[k + 1].row.b) + cy(&anchor.row.b)) / 2.)
                } else {
                    bottom
                };
                let members = anchors::grow(p, &lane, anchor, bt, bb, &claimed, &font, em, w)?;
                let isolated = fraction(p, &members, &lane, em, &rules)?;
                let (left, fragmented) = e::left_prose(p, &members, &lane, em)?;
                if left && !isolated {
                    for r in &members {
                        p.lines[r.index]["paragraph_formula_context"] = json!(true)
                    }
                    if fragmented {
                        paragraphs.push(paragraph(p, &members, em, w)?);
                        for r in &members {
                            claimed.insert(p.id(r)?);
                        }
                    }
                    continue;
                }
                if members.len() < 2 {
                    continue;
                }
                if members.len() == 2 && axis(&members[0].b, &members[1].b, 1) < 0.2 {
                    let mut wide = false;
                    for r in &members {
                        if r.index != anchor.row.index
                            && e::wide(p, &anchor.row, r, (lane.right - lane.left).max(0.1))?
                        {
                            wide = true
                        }
                    }
                    if !wide {
                        continue;
                    }
                }
                if margin(&p.bounds(&members)?, lp[1]) {
                    continue;
                }
                if let Some(block) = materialize::block(p, &members, angle, p.id(&anchor.row)?, w)?
                {
                    blocks.push(block);
                    for r in members {
                        claimed.insert(p.id(&r)?);
                    }
                }
            }
        }
    }
    Ok((blocks, claimed, paragraphs))
}
