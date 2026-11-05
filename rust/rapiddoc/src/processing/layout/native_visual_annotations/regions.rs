//! Original visual region merging and annotation-row ordering, retaining block identity indices.
use super::*;
pub(super) fn key(i: usize, bs: &[Value], p: [f64; 2]) -> Result<(f64, f64, usize), Error> {
    if let Some(band) = bs[i].get("_legend_sort_band") {
        let band = array(band, 2)?;
        if band.len() != 2 {
            return Err(invalid());
        }
        return Ok((num(&band[0])?, num(&band[1])?, i));
    }
    Ok(own_box(&bs[i], p)?.map_or((f64::INFINITY, f64::INFINITY, i), |b| (b.n[1], b.n[0], i)))
}
fn visual_rows(
    ids: &[usize],
    bs: &[Value],
    p: [f64; 2],
    rs: &Assignments,
) -> Result<Vec<usize>, Error> {
    let mut positioned = vec![];
    let mut fallback = vec![];
    for &i in ids {
        if let Some(b) = first_row(&bs[i], p, angle(&bs[i])?)? {
            positioned.push((i, b, height(&bs[i], p)?))
        } else {
            fallback.push(i)
        }
    }
    positioned.sort_by(|a, b| {
        a.1.n[1]
            .total_cmp(&b.1.n[1])
            .then(a.1.n[0].total_cmp(&b.1.n[0]))
            .then(a.0.cmp(&b.0))
    });
    let mut rows: Vec<Vec<(usize, Box4, f64)>> = vec![];
    for item in positioned {
        let mut append = false;
        if let Some(row) = rows.last() {
            if let Some((_, r)) = rs.iter().find(|(i, _)| *i == item.0) {
                let shared = ["above", "below"].contains(&r.direction)
                    && row.iter().all(|member| {
                        rs.iter()
                            .find(|(i, _)| *i == member.0)
                            .is_some_and(|(_, mr)| {
                                mr.direction == r.direction && distinct(&bs[member.0], &bs[item.0])
                            })
                    });
                let top = median(row.iter().map(|m| m.1.n[1]).collect());
                let h = median(row.iter().map(|m| m.2).collect());
                append = shared && (item.1.n[1] - top).abs() <= 0.25 * median(vec![h, item.2]);
            }
        }
        if append {
            rows.last_mut().unwrap().push(item)
        } else {
            rows.push(vec![item])
        }
    }
    let mut out = vec![];
    for mut row in rows {
        row.sort_by(|a, b| {
            a.1.n[0]
                .total_cmp(&b.1.n[0])
                .then(a.1.n[1].total_cmp(&b.1.n[1]))
                .then(a.0.cmp(&b.0))
        });
        out.extend(row.into_iter().map(|m| m.0))
    }
    fallback.sort_by(|i, j| {
        key(*i, bs, p)
            .unwrap()
            .partial_cmp(&key(*j, bs, p).unwrap())
            .unwrap()
    });
    out.extend(fallback);
    Ok(out)
}
pub(super) fn build(
    rs: &Assignments,
    bs: &[Value],
    p: [f64; 2],
    w: &mut Work,
) -> Result<Vec<Vec<usize>>, Error> {
    let mut groups: Vec<(BTreeSet<usize>, Assignments)> = vec![];
    for (i, r) in rs {
        let mut members = r.parent.members.iter().copied().collect::<BTreeSet<_>>();
        let overlapping = groups
            .iter()
            .enumerate()
            .filter(|(_, g)| !g.0.is_disjoint(&members))
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let mut relations = vec![(*i, r.clone())];
        for k in overlapping.into_iter().rev() {
            let (m, rs) = groups.remove(k);
            members.extend(m);
            for (i, r) in rs {
                assign(&mut relations, i, r)
            }
        }
        groups.push((members, relations))
    }
    let mut regions = vec![];
    for (members, rs) in groups {
        let leading = rs
            .iter()
            .filter(|(i, r)| {
                bs[*i]["type"] == "caption" && ["above", "left"].contains(&r.direction)
            })
            .map(|(i, _)| *i)
            .collect::<Vec<_>>();
        let mut trailing = rs
            .iter()
            .filter(|(i, r)| {
                bs[*i]["type"] == "caption" && ["below", "right"].contains(&r.direction)
            })
            .map(|(i, _)| *i)
            .collect::<Vec<_>>();
        let mut footnotes = rs
            .iter()
            .filter(|(i, _)| bs[*i]["type"] == "footnote")
            .map(|(i, _)| *i)
            .collect::<Vec<_>>();
        let a = angle(&bs[*members.first().ok_or_else(invalid)?])?;
        let mut proxies = vec![];
        let mut proxy_ids = vec![];
        for &i in &members {
            if let Some(b) = local_box(&bs[i], p, a)? {
                proxies.push(json!({"bbox":b.raw}));
                proxy_ids.push(i)
            }
        }
        let bodies = xycut::sort(&proxies, w)?;
        let mut ordered = vec![];
        let mut panel_notes = BTreeSet::new();
        for idx in bodies {
            let i = proxy_ids[idx];
            ordered.push(i);
            if bs[i]["type"] != "table" {
                continue;
            }
            let notes = trailing
                .iter()
                .copied()
                .filter(|j| {
                    rs.iter()
                        .find(|(k, _)| k == j)
                        .is_some_and(|(_, r)| r.parent.members == vec![i])
                        && regex(r"^\s*\([a-z]\)\s+\S").is_match(text(&bs[*j]))
                })
                .collect::<Vec<_>>();
            for j in visual_rows(&notes, bs, p, &rs)? {
                ordered.push(j);
                panel_notes.insert(j);
            }
        }
        trailing.retain(|i| !panel_notes.contains(i));
        let mut out = visual_rows(&leading, bs, p, &rs)?;
        out.extend(ordered);
        out.extend(visual_rows(&trailing, bs, p, &rs)?);
        footnotes.sort_by(|i, j| {
            key(*i, bs, p)
                .unwrap()
                .partial_cmp(&key(*j, bs, p).unwrap())
                .unwrap()
        });
        out.extend(footnotes);
        regions.push(out)
    }
    Ok(regions)
}
