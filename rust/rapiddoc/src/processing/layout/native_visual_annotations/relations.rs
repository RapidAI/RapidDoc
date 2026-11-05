//! Full original visual parents, directional relations and intervening-block arbitration.
use super::*;
pub(super) fn metrics(a: &Box4, b: &Box4, k: usize, margin: f64) -> (f64, f64, f64) {
    let start = b.n[k] - margin;
    let end = b.n[k + 2] + margin;
    let overlap = (a.n[k + 2].min(end) - a.n[k].max(start)).max(0.);
    let al = (a.n[k + 2] - a.n[k]).max(0.1);
    let bl = (end - start).max(0.1);
    (
        overlap / al.min(bl),
        overlap / al,
        ((a.n[k] + a.n[k + 2]) * 0.5 - (b.n[k] + b.n[k + 2]) * 0.5).abs()
            / al.max(b.n[k + 2] - b.n[k]).max(0.1),
    )
}
pub(super) fn direction(
    parent: &Parent,
    a: &Box4,
    h: f64,
    dir: &'static str,
    max: f64,
) -> Result<Option<Relation>, Error> {
    let b = &parent.b;
    let (gap, k) = match dir {
        "above" => {
            if cy(a) > cy(b) {
                return Ok(None);
            }
            (b.n[1] - a.n[3], 0)
        }
        "below" => {
            if cy(a) < cy(b) {
                return Ok(None);
            }
            (a.n[1] - b.n[3], 0)
        }
        "left" => {
            if cx(a) > cx(b) {
                return Ok(None);
            }
            (b.n[0] - a.n[2], 1)
        }
        _ => {
            if cx(a) < cx(b) {
                return Ok(None);
            }
            (a.n[0] - b.n[2], 1)
        }
    };
    let penetration = (-gap).max(0.);
    let gap = gap.max(0.);
    if gap > max * h || penetration > h {
        return Ok(None);
    }
    let (projection, coverage, offset) = metrics(a, b, k, h);
    let centered = k == 0 && offset <= 0.08 && projection >= 0.95;
    let inset =
        k == 0 && b.n[2] - b.n[0] >= 1.5 * (a.n[2] - a.n[0]) && gap <= 2. * h && coverage >= 0.5;
    if projection < (if inset { 0.5 } else { 0.8 }) || (coverage < 0.65 && !centered) {
        return Ok(None);
    }
    if h == 0. {
        return Err(invalid());
    }
    Ok(Some(Relation {
        parent: parent.clone(),
        direction: dir,
        gap: gap / h,
        projection,
        coverage,
        offset,
    }))
}
fn cost(a: &Relation, b: &Relation) -> std::cmp::Ordering {
    a.gap
        .total_cmp(&b.gap)
        .then(b.coverage.total_cmp(&a.coverage))
        .then(b.projection.total_cmp(&a.projection))
        .then(a.offset.total_cmp(&b.offset))
}
pub(super) fn best(
    parent: &Parent,
    a: &Box4,
    h: f64,
    kind: &str,
) -> Result<Option<Relation>, Error> {
    let mut rs = vec![];
    let dirs = if kind == "footnote" {
        &["below"][..]
    } else {
        &["above", "below", "left", "right"][..]
    };
    for &d in dirs {
        if let Some(r) = direction(parent, a, h, d, if kind == "footnote" { 6. } else { 4. })? {
            rs.push(r)
        }
    }
    Ok(rs.into_iter().min_by(cost))
}
fn component(a: &Box4, b: &Box4, h: f64) -> bool {
    let dx = 0f64.max(a.n[0] - b.n[2]).max(b.n[0] - a.n[2]);
    let dy = 0f64.max(a.n[1] - b.n[3]).max(b.n[1] - a.n[3]);
    let x = (a.n[2].min(b.n[2]) - a.n[0].max(b.n[0])).max(0.);
    let y = (a.n[3].min(b.n[3]) - a.n[1].max(b.n[1])).max(0.);
    let minw = (a.n[2] - a.n[0]).min(b.n[2] - b.n[0]).max(0.1);
    let minh = (a.n[3] - a.n[1]).min(b.n[3] - b.n[1]).max(0.1);
    dx <= 2. * h && y / minh >= 0.5 || dy <= 2. * h && x / minw >= 0.5
}
pub(super) fn parents(
    bs: &[Value],
    p: [f64; 2],
    h: f64,
    w: &mut Work,
) -> Result<Vec<Parent>, Error> {
    let mut out = vec![];
    let mut angles: Vec<(i64, Vec<usize>)> = vec![];
    let mut boxes = BTreeMap::new();
    for (i, b) in bs.iter().enumerate() {
        if !["image", "table", "code"].iter().any(|t| b["type"] == *t) {
            continue;
        }
        let a = angle(b)?;
        let Some(bb) = local_box(b, p, a)? else {
            continue;
        };
        boxes.insert(i, bb.clone());
        out.push(Parent {
            members: vec![i],
            angle: a,
            b: bb,
            numeric: truth(&b["_native_numeric_grid"]),
        });
        if b["type"] == "image" {
            if let Some((_, ids)) = angles.iter_mut().find(|(k, _)| *k == a) {
                ids.push(i)
            } else {
                angles.push((a, vec![i]))
            }
        }
    }
    for (a, ids) in angles {
        let mut remaining = super::super::tiled_tables::pending::Pending::from_indices(&ids);
        while !remaining.ordered().is_empty() {
            let seed = *remaining.ordered().iter().min().unwrap();
            remaining.remove(seed);
            let mut group = vec![seed];
            loop {
                let mut changed = false;
                for i in remaining.ordered() {
                    w.charge(group.len())?;
                    if group.iter().any(|j| component(&boxes[j], &boxes[&i], h)) {
                        group.push(i);
                        remaining.remove(i);
                        changed = true
                    }
                }
                if !changed {
                    break;
                }
            }
            if group.len() < 2 {
                continue;
            }
            group.sort_unstable();
            let b = combined(group.iter().map(|i| &boxes[i]))?;
            out.push(Parent {
                members: group,
                angle: a,
                b,
                numeric: false,
            });
        }
    }
    Ok(out)
}
pub(super) fn intervening(
    r: &Relation,
    index: usize,
    a: &Box4,
    h: f64,
    bs: &[Value],
    p: [f64; 2],
    ignored: &BTreeSet<usize>,
    w: &mut Work,
) -> Result<bool, Error> {
    if r.gap <= 0. {
        return Ok(false);
    }
    let b = &r.parent.b;
    let n = match r.direction {
        "above" => [
            a.n[0].max(b.n[0]) - 0.25 * h,
            a.n[3],
            a.n[2].min(b.n[2]) + 0.25 * h,
            b.n[1],
        ],
        "below" => [
            a.n[0].max(b.n[0]) - 0.25 * h,
            b.n[3],
            a.n[2].min(b.n[2]) + 0.25 * h,
            a.n[1],
        ],
        "left" => [
            a.n[2],
            a.n[1].max(b.n[1]) - 0.25 * h,
            b.n[0],
            a.n[3].min(b.n[3]) + 0.25 * h,
        ],
        _ => [
            b.n[2],
            a.n[1].max(b.n[1]) - 0.25 * h,
            a.n[0],
            a.n[3].min(b.n[3]) + 0.25 * h,
        ],
    };
    if n[2] <= n[0] || n[3] <= n[1] {
        return Ok(false);
    }
    for (i, b) in bs.iter().enumerate() {
        w.charge(1)?;
        if i == index
            || r.parent.members.contains(&i)
            || ignored.contains(&i)
            || angle(b)? != r.parent.angle
        {
            continue;
        }
        if let Some(b) = local_box(b, p, r.parent.angle)? {
            if b.n[2].min(n[2]) > b.n[0].max(n[0]) && b.n[3].min(n[3]) > b.n[1].max(n[1]) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
pub(super) fn table_panels(
    bs: &[Value],
    p: [f64; 2],
    parents: &mut Vec<Parent>,
    candidates: &mut Candidates,
    w: &mut Work,
) -> Result<(), Error> {
    let tables = parents
        .iter()
        .filter(|v| v.members.len() == 1 && bs[v.members[0]]["type"] == "table")
        .cloned()
        .collect::<Vec<_>>();
    for (i, k) in candidates.clone() {
        let caption = &bs[i];
        if k != "caption" || !regex(r"(?i)^\s*tab(?:le)?[.\s0-9]").is_match(&fold(text(caption))) {
            continue;
        }
        let h = height(caption, p)?;
        let a = angle(caption)?;
        let Some(cb) = local_box(caption, p, a)? else {
            continue;
        };
        if h <= 0. {
            continue;
        }
        let mut nearby = tables
            .iter()
            .filter(|t| {
                t.angle == a
                    && (0.0..=4. * h).contains(&(cb.n[1] - t.b.n[3]))
                    && cb.n[0] - h <= t.b.n[0]
                    && t.b.n[2] <= cb.n[2] + h
            })
            .cloned()
            .collect::<Vec<_>>();
        nearby.sort_by(|a, b| a.b.n[0].total_cmp(&b.b.n[0]));
        if nearby.len() != 2 {
            continue;
        }
        let ab = &nearby[0].b;
        let bb = &nearby[1].b;
        if (ab.n[1] - bb.n[1]).abs() > 0.5 * h
            || (ab.n[3] - bb.n[3]).abs() > 0.5 * h
            || !(0.0..=8. * h).contains(&(bb.n[0] - ab.n[2]))
            || !(0.7..=1.43).contains(&((ab.n[2] - ab.n[0]) / (bb.n[2] - bb.n[0]).max(1e-6)))
        {
            continue;
        }
        let mut notes = vec![];
        for (t, letter) in nearby.iter().zip(['a', 'b']) {
            let mut matches = vec![];
            for (j, b) in bs.iter().enumerate() {
                w.charge(1)?;
                if b["type"] != "text"
                    || angle(b)? != a
                    || !regex(&format!(r"^\s*\({letter}\)\s+\S")).is_match(text(b))
                {
                    continue;
                }
                if let Some(nb) = local_box(b, p, a)? {
                    if (0.0..=2. * h).contains(&(nb.n[1] - t.b.n[3]))
                        && nb.n[3] <= cb.n[1] + 0.1 * h
                        && t.b.n[0] - 0.25 * h <= nb.n[0]
                        && nb.n[2] <= t.b.n[2] + 0.25 * h
                        && (0.75 * h..=1.25 * h).contains(&height(b, p)?)
                    {
                        matches.push(j)
                    }
                }
            }
            if matches.len() != 1 {
                break;
            }
            notes.extend(matches)
        }
        if notes.len() != 2 {
            continue;
        }
        let parent = Parent {
            members: nearby.iter().map(|p| p.members[0]).collect(),
            angle: a,
            b: combined([ab, bb])?,
            numeric: false,
        };
        let Some(r) = direction(&parent, &cb, h, "below", 4.)? else {
            continue;
        };
        let mut ignored = candidates.iter().map(|(i, _)| *i).collect::<BTreeSet<_>>();
        ignored.extend(&notes);
        if intervening(&r, i, &cb, h, bs, p, &ignored, w)? {
            continue;
        }
        parents.push(parent);
        for j in notes {
            if let Some((_, k)) = candidates.iter_mut().find(|(i, _)| *i == j) {
                *k = "caption"
            } else {
                candidates.push((j, "caption"))
            }
        }
    }
    Ok(())
}
pub(super) fn choose(
    i: usize,
    kind: &str,
    bs: &[Value],
    p: [f64; 2],
    parents: &[Parent],
    annotations: &BTreeSet<usize>,
    w: &mut Work,
) -> Result<Option<Relation>, Error> {
    let a = angle(&bs[i])?;
    let Some(ab) = local_box(&bs[i], p, a)? else {
        return Ok(None);
    };
    let h = height(&bs[i], p)?;
    let mut rs = vec![];
    for parent in parents {
        w.charge(1)?;
        if parent.angle == a {
            if let Some(r) = best(parent, &ab, h, kind)? {
                rs.push(r)
            }
        }
    }
    if kind == "caption" && rs.is_empty() && strong_caption(text(&bs[i])) {
        let mut out = vec![];
        for parent in parents {
            if parent.angle != a
                || parent.members.len() != 1
                || bs[parent.members[0]]["type"] != "image"
            {
                continue;
            }
            let b = &parent.b;
            let dir = if ab.n[3] <= b.n[1] { "above" } else { "below" };
            let gap = if dir == "above" {
                b.n[1] - ab.n[3]
            } else {
                ab.n[1] - b.n[3]
            };
            let (projection, coverage, offset) = metrics(&ab, b, 0, h);
            if (0.0..=3. * h).contains(&gap)
                && projection >= 0.55
                && coverage >= 0.55
                && ab.n[0] >= b.n[0] - 0.45 * (b.n[2] - b.n[0])
                && ab.n[2] <= b.n[2] + 0.45 * (b.n[2] - b.n[0])
            {
                if h == 0. {
                    return Err(invalid());
                }
                out.push(Relation {
                    parent: parent.clone(),
                    direction: dir,
                    gap: gap / h,
                    projection,
                    coverage,
                    offset,
                })
            }
        }
        if out.len() == 1 {
            rs.extend(out)
        }
    }
    if kind == "footnote" {
        for parent in parents {
            if parent.angle != a
                || parent.members.len() != 1
                || bs[parent.members[0]]["type"] != "image"
            {
                continue;
            }
            let b = &parent.b;
            let width = b.n[2] - b.n[0];
            if !(0.0..=6. * h).contains(&(ab.n[1] - b.n[3]))
                || !(b.n[0] - 0.25 * width..=b.n[2]).contains(&ab.n[0])
                || ab.n[2] > b.n[2] + 0.25 * width
            {
                continue;
            }
            let mut supported = false;
            for &index in annotations {
                if strong_caption(text(&bs[index])) && angle(&bs[index])? == a {
                    if let Some(cb) = local_box(&bs[index], p, a)? {
                        if best(parent, &cb, height(&bs[index], p)?, "caption")?.is_some() {
                            supported = true;
                            break;
                        }
                    }
                }
            }
            if supported {
                if h == 0. {
                    return Err(invalid());
                }
                rs.push(Relation {
                    parent: parent.clone(),
                    direction: "below",
                    gap: (ab.n[1] - b.n[3]) / h,
                    projection: 1.,
                    coverage: 1.,
                    offset: 0.,
                })
            }
        }
    }
    let singles = rs
        .iter()
        .filter(|r| r.parent.members.len() == 1)
        .cloned()
        .collect::<Vec<_>>();
    let mut eligible = vec![];
    for r in rs {
        let comparable = singles
            .iter()
            .filter(|s| {
                s.direction == r.direction && r.parent.members.contains(&s.parent.members[0])
            })
            .collect::<Vec<_>>();
        let improved = r.parent.members.len() < 2
            || comparable.is_empty()
            || r.coverage
                >= comparable
                    .iter()
                    .map(|s| s.coverage)
                    .fold(f64::NEG_INFINITY, f64::max)
                    + 0.15;
        if improved && !intervening(&r, i, &ab, h, bs, p, annotations, w)? {
            eligible.push(r)
        }
    }
    if kind == "caption"
        && graph(text(&bs[i]))
        && eligible.iter().any(|r| r.parent.numeric)
        && eligible
            .iter()
            .any(|r| !r.parent.numeric && r.direction == "above")
    {
        eligible.retain(|r| !r.parent.numeric && r.direction == "above")
    }
    Ok(eligible.into_iter().min_by(|a, b| {
        cost(a, b)
            .then(a.parent.members.len().cmp(&b.parent.members.len()))
            .then(a.parent.members.cmp(&b.parent.members))
    }))
}
