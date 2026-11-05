use super::*;
fn baseline(a: &Sample, b: &Sample) -> bool {
    (a.base[1] - b.base[1]).abs()
        <= 0.5f64.max(0.25 * (a.ink[3] - a.ink[1]).max(b.ink[3] - b.ink[1]))
}
fn next<'a>(
    s: &Sample,
    ids: &[usize],
    samples: &'a [Sample],
    keys: &[RunKey],
    style: bool,
) -> Option<&'a Sample> {
    ids.iter().map(|&i| &samples[i]).find(|c| {
        c.position > s.position
            && (if style {
                keys[c.run].style(&keys[s.run])
            } else {
                c.run == s.run
            })
            && baseline(s, c)
            && c.base[0] - s.base[0] > 0.1
    })
}
fn build(
    s: &Sample,
    r: &Run,
    n: Option<&Sample>,
    keys: &[RunKey],
    pages: &[Page],
    bearing: f64,
    donor: bool,
) -> Option<Value> {
    let advance = if donor || n.is_none() {
        r.median_advance?
    } else {
        n.unwrap().base[0] - s.base[0]
    };
    if advance <= 0. {
        return None;
    }
    let left = s.ink[0].min(s.base[0] + bearing);
    let mut right = s.ink[2].max(s.base[0] + advance);
    if let Some(n) = n {
        right = right.min(s.ink[2].max(n.base[0]))
    }
    if right <= left {
        return None;
    }
    let layout = clipped(
        world(
            [left, s.local[1], right, s.local[3]],
            pages[s.page].size,
            keys[s.run].angle,
        ),
        pages[s.page].size,
    )?;
    Some(
        json!({"source_bbox":s.source,"tight_bbox":s.tight,"origin":s.origin,"layout_bbox":layout,"ink_bbox":s.tight,"baseline":s.base[1],"advance":advance,"em_height":s.size.max(s.ink[3]-s.ink[1]),"x_state":"abnormal","y_state":"healthy","confidence":(r.ratios.len()as f64/30.).min(1.)}),
    )
}
fn sparse(
    s: &Sample,
    ids: &[usize],
    samples: &[Sample],
    r: &Run,
    keys: &[RunKey],
    pages: &[Page],
    w: &mut Work,
) -> Result<bool, Error> {
    let Some(advance) = r.median_advance.filter(|a| *a > 0.) else {
        return Ok(false);
    };
    let tight = (s.ink[2] - s.ink[0]).max(0.1);
    let canonical = advance.max(tight);
    let right = s.ink[2].max(s.base[0] + advance);
    let p = pages[s.page].size;
    let width = if [90, 270].contains(&keys[s.run].angle) {
        p[1]
    } else {
        p[0]
    };
    if s.local[2] - s.local[0] <= 1.35 * canonical
        || s.local[2] - right < (2. * canonical).max(0.01 * width)
    {
        return Ok(false);
    }
    w.charge(ids.len())?;
    let mut peers = ids
        .iter()
        .map(|&i| &samples[i])
        .filter(|c| c.position != s.position && baseline(s, c))
        .collect::<Vec<_>>();
    peers.sort_by_key(|c| c.position);
    let following = peers.iter().copied().find(|c| c.position > s.position);
    let Some(following) = following else {
        let right = peers.iter().map(|c| c.local[2]).fold(s.local[2], f64::max);
        return Ok((s.local[2] - right).abs() <= 0.25);
    };
    let mut widths = vec![tight];
    widths.extend(peers.iter().map(|c| (c.ink[2] - c.ink[0]).max(0.1)));
    let hard = 15f64.max(3. * med(widths)).max(0.02 * width);
    Ok(following.ink[0] - right >= hard && s.local[2] >= following.base[0])
}
pub(super) fn repair(
    plan: &mut Plan,
    samples: &[Sample],
    by_line: &[(Key, Vec<usize>)],
    runs: &[Run],
    keys: &[RunKey],
    pages: &[Page],
    w: &mut Work,
) -> Result<(), Error> {
    let mut donors = (0..runs.len())
        .filter(|&i| {
            (runs[i].strong || runs[i].sibling) && runs[i].median_advance.is_some_and(|a| a > 0.)
        })
        .collect::<Vec<_>>();
    donors.sort_by(|&a, &b| {
        (!runs[a].strong)
            .cmp(&(!runs[b].strong))
            .then(runs[b].ratios.len().cmp(&runs[a].ratios.len()))
            .then(keys[a].cmp(&keys[b]))
    });
    for (_, ids) in by_line {
        for &i in ids {
            let s = &samples[i];
            let r = &runs[s.run];
            if !(r.strong || r.sibling) {
                continue;
            }
            w.charge(ids.len() + 1)?;
            if let Some(repair) = build(
                s,
                r,
                next(s, ids, samples, keys, false),
                keys,
                pages,
                r.median_bearing,
                false,
            ) {
                plan.chars.insert((s.page, s.index), repair);
            }
        }
        for &i in ids {
            let s = &samples[i];
            let r = &runs[s.run];
            w.charge(donors.len() + ids.len() + 1)?;
            let donor = donors
                .iter()
                .copied()
                .find(|&d| keys[d].style(&keys[s.run]));
            let Some(d) = donor else { continue };
            if plan.chars.contains_key(&(s.page, s.index))
                || r.ratios.len() >= 10
                || !ids.iter().map(|&j| &samples[j]).any(|c| {
                    c.position.abs_diff(s.position) == 1
                        && keys[c.run].style(&keys[s.run])
                        && baseline(s, c)
                        && plan.chars.contains_key(&(s.page, c.index))
                })
                || !sparse(s, ids, samples, &runs[d], keys, pages, w)?
            {
                continue;
            }
            if let Some(repair) = build(
                s,
                &runs[d],
                next(s, ids, samples, keys, true),
                keys,
                pages,
                s.ink[0] - s.base[0],
                true,
            ) {
                plan.chars.insert((s.page, s.index), repair);
            }
        }
    }
    for (page, physical) in pages.iter().enumerate() {
        for line in physical.lines {
            let line_key = (page, integer(line, "source_index", 0)?);
            let Some((_, ids)) = by_line.iter().find(|(k, _)| *k == line_key) else {
                continue;
            };
            let ss = ids.iter().map(|&i| &samples[i]).collect::<Vec<_>>();
            let repaired = ss
                .iter()
                .filter_map(|s| plan.chars.get(&(s.page, s.index)))
                .collect::<Vec<_>>();
            if repaired.is_empty() {
                continue;
            }
            let a = integer(line, "angle", 0)?;
            let p = physical.size;
            let b = bbox(&line["bbox"])?.n;
            let local = g::rotate(b, p, a as i32);
            let explicit = union(ss.iter().map(|s| s.local));
            if (explicit[0] - local[0]).abs() > 0.25 || (explicit[2] - local[2]).abs() > 0.25 {
                continue;
            }
            w.charge(ss.len() + 1)?;
            let mut boxes = Vec::new();
            for s in &ss {
                let b = if let Some(r) = plan.chars.get(&(s.page, s.index)) {
                    bbox(&r["layout_bbox"])?.n
                } else {
                    s.source
                };
                boxes.push(g::rotate(b, p, a as i32))
            }
            let bounds = union(boxes);
            let Some(layout) = clipped(world([bounds[0], local[1], bounds[2], local[3]], p, a), p)
            else {
                continue;
            };
            let mut repair = line_repair(
                b,
                layout,
                Some(union(ss.iter().map(|s| s.tight))),
                None,
                line.get("effective_height").map_or(Ok(0.), num)?,
            );
            repair["state"] = json!("repair_x");
            repair["confidence"] = json!(repaired
                .iter()
                .map(|r| r["confidence"].as_f64().unwrap())
                .fold(f64::INFINITY, f64::min));
            repair["repaired_char_count"] = json!(repaired.len());
            plan.lines.insert(line_key, repair);
        }
    }
    Ok(())
}
