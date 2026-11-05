//! Full existing _shared.xycut, including cross-layout reinsertion and narrow-element retry.
use super::*;
fn order(ids: &mut [usize], boxes: &BTreeMap<usize, Box4>) {
    ids.sort_by(|i, j| {
        boxes[i].n[1]
            .total_cmp(&boxes[j].n[1])
            .then(boxes[i].n[0].total_cmp(&boxes[j].n[0]))
            .then(i.cmp(j))
    });
}
fn cut(ids: &[usize], boxes: &BTreeMap<usize, Box4>, k: usize) -> (f64, f64) {
    let mut ids = ids.to_vec();
    ids.sort_by(|i, j| {
        boxes[i].n[k]
            .total_cmp(&boxes[j].n[k])
            .then(boxes[i].n[k + 2].total_cmp(&boxes[j].n[k + 2]))
            .then(i.cmp(j))
    });
    let (mut gap, mut position, mut end) = (0., 0., None::<f64>);
    for i in ids {
        let b = &boxes[&i];
        if let Some(e) = end {
            if b.n[k] > e && b.n[k] - e > gap {
                gap = b.n[k] - e;
                position = (e + b.n[k]) * 0.5
            }
        }
        end = Some(end.map_or(b.n[k + 2], |e| e.max(b.n[k + 2])))
    }
    (position, gap)
}
fn recursive(
    ids: &[usize],
    boxes: &BTreeMap<usize, Box4>,
    beta: f64,
    density: f64,
    w: &mut Work,
    depth: usize,
) -> Result<Vec<usize>, Error> {
    if ids.len() <= 1 {
        return Ok(ids.to_vec());
    }
    if depth > rapiddoc_core::resources::limit(4096) {
        return Err(limit());
    }
    w.charge(ids.len() * ids.len())?;
    let threshold = beta * median(ids.iter().map(|i| boxes[i].n[2] - boxes[i].n[0]).collect());
    let cross = ids
        .iter()
        .copied()
        .filter(|i| {
            ids.len() >= 3
                && boxes[i].n[2] - boxes[i].n[0] >= threshold
                && ids
                    .iter()
                    .filter(|j| *j != i && axis(&boxes[i], &boxes[j], 0) >= 0.1)
                    .count()
                    >= 2
        })
        .collect::<Vec<_>>();
    let main = ids
        .iter()
        .copied()
        .filter(|i| !cross.contains(i))
        .collect::<Vec<_>>();
    let mut sorted = if main.is_empty() {
        vec![]
    } else if main.len() == 1 {
        main.clone()
    } else {
        let bounds = combined(main.iter().map(|i| &boxes[i]))?;
        let area = (bounds.n[2] - bounds.n[0]) * (bounds.n[3] - bounds.n[1]);
        let sum = main
            .iter()
            .map(|i| {
                let b = &boxes[i];
                (b.n[2] - b.n[0]) * (b.n[3] - b.n[1])
            })
            .sum::<f64>();
        let dense = if area <= 0. { 1. } else { (sum / area).min(1.) };
        let horizontal = cut(&main, boxes, 1);
        let mut vertical = cut(&main, boxes, 0);
        if vertical.1 < 5. && main.len() >= 3 {
            let threshold = (bounds.n[2] - bounds.n[0]) * 0.1;
            let filtered = main
                .iter()
                .copied()
                .filter(|i| boxes[i].n[2] - boxes[i].n[0] >= threshold)
                .collect::<Vec<_>>();
            if filtered.len() >= 2 && filtered.len() != main.len() {
                let retry = cut(&filtered, boxes, 0);
                if retry.1 > vertical.1 && retry.1 >= 5. {
                    vertical = retry
                }
            }
        }
        let mut k = if dense > density { 1 } else { 0 };
        let (mut preferred, fallback) = if k == 1 {
            (horizontal, vertical)
        } else {
            (vertical, horizontal)
        };
        if preferred.1 < 5. && fallback.1 >= 5. {
            preferred = fallback;
            k = 1 - k
        }
        if preferred.1 < 5. {
            let mut out = main.clone();
            order(&mut out, boxes);
            out
        } else {
            let mut groups = [vec![], vec![]];
            for &i in &main {
                let b = &boxes[&i];
                let center = (b.n[k] + b.n[k + 2]) * 0.5;
                groups[usize::from(center >= preferred.0)].push(i)
            }
            if groups.iter().any(Vec::is_empty) {
                let mut out = main.clone();
                order(&mut out, boxes);
                out
            } else {
                let mut out = vec![];
                for g in groups {
                    out.extend(recursive(&g, boxes, beta, density, w, depth + 1)?)
                }
                out
            }
        }
    };
    let mut cross = cross;
    order(&mut cross, boxes);
    if sorted.is_empty() {
        return Ok(cross);
    }
    let mut out = vec![];
    for c in cross {
        out.extend(
            sorted
                .iter()
                .copied()
                .filter(|i| boxes[i].n[3] <= boxes[&c].n[1]),
        );
        sorted.retain(|i| boxes[i].n[3] > boxes[&c].n[1]);
        out.push(c)
    }
    out.extend(sorted);
    Ok(out)
}
pub(super) fn sort(bs: &[Value], w: &mut Work) -> Result<Vec<usize>, Error> {
    let mut boxes = BTreeMap::new();
    let mut invalids = vec![];
    for (i, b) in bs.iter().enumerate() {
        let raw = &b["bbox"];
        let Some(bb) = coerce(raw)? else {
            invalids.push(i);
            continue;
        };
        let vals = raw.as_array().unwrap();
        let left = super::super::table_native_text::number(&vals[0])?.ok_or_else(invalid)?;
        let top = super::super::table_native_text::number(&vals[1])?.ok_or_else(invalid)?;
        let right = super::super::table_native_text::number(&vals[2])?.ok_or_else(invalid)?;
        let bottom = super::super::table_native_text::number(&vals[3])?.ok_or_else(invalid)?;
        if right <= left || bottom <= top {
            invalids.push(i)
        } else {
            boxes.insert(i, bb);
        }
    }
    let ids = boxes.keys().copied().collect::<Vec<_>>();
    let mut out = recursive(&ids, &boxes, 1.3, 0.9, w, 0)?;
    out.extend(invalids);
    Ok(out)
}
