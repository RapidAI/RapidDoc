//! Original bilingual/cross-lane expansion and hanging-indent table-footnote closure.
use super::*;
fn identifier(t: &str) -> Option<String> {
    let original = normalized(t);
    let t = fold(&original);
    // IGNORECASE matching may project Turkish I; casefold the captured original.
    let captured = |start: usize, end: usize| {
        original
            .chars()
            .skip(t[..start].chars().count())
            .take(t[start..end].chars().count())
            .flat_map(char::to_lowercase)
            .collect()
    };
    let pattern=r"(?: (?:[A-Z]\s*[.-]?\s*)?\d+(?:\s*[-./–—]\s*[A-Z]?\d+)*[A-Z]? |[IVXLCDM]+|[零〇一二三四五六七八九十百千两]+)".replace(' ',"");
    let en = regex(&format!(
        r"(?is)^\s*(?:fig(?:ure)?|tab(?:le)?|alg(?:orithm)?|listing|chart|scheme|diagram)([.\s0-9])"
    ));
    let re = if let Some(m) = en.find(&t) {
        let prefix = &t[..m.end() - m.as_str().chars().next_back().unwrap().len_utf8()];
        let tail = &t[prefix.len()..];
        let re = regex(&format!(
            r"(?is)^\.?\s*(?P<identifier>{pattern})(?P<tail>.*)$"
        ));
        return re.captures(tail).map(|c| {
            let m = c.name("identifier").unwrap();
            captured(prefix.len() + m.start(), prefix.len() + m.end())
        });
    } else {
        regex(&format!(
            r"(?is)^\s*(?:程序清单|图表|表格|算法|图|表)\s*(?P<identifier>{pattern})(?P<tail>.*)$"
        ))
    };
    re.captures(&t).map(|c| {
        let m = c.name("identifier").unwrap();
        captured(m.start(), m.end())
    })
}
pub(super) fn bilingual(
    bs: &[Value],
    p: [f64; 2],
    cs: &Candidates,
    assignments: &Assignments,
) -> Result<Assignments, Error> {
    let mut out = vec![];
    let assigned = assignments
        .iter()
        .filter(|(i, _)| kind(cs, *i) == Some("caption"))
        .collect::<Vec<_>>();
    for &(i, k) in cs {
        if k != "caption" || assignments.iter().any(|(j, _)| *j == i) {
            continue;
        }
        let Some(id) = identifier(text(&bs[i])) else {
            continue;
        };
        let Some(cb) = own_box(&bs[i], p)? else {
            continue;
        };
        let ch = height(&bs[i], p)?;
        let mut matches = vec![];
        for &&(j, ref r) in &assigned {
            if identifier(text(&bs[j])) != Some(id.clone()) || angle(&bs[j])? != angle(&bs[i])? {
                continue;
            }
            let Some(ab) = own_box(&bs[j], p)? else {
                continue;
            };
            let ah = height(&bs[j], p)?;
            let gap = 0f64.max(cb.n[1] - ab.n[3]).max(ab.n[1] - cb.n[3]);
            if gap <= 1.5 * ch.max(ah) && axis(&cb, &ab, 0) >= 0.35 {
                matches.push((gap, j, r.clone()))
            }
        }
        if let Some((_, _, r)) = matches
            .into_iter()
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
        {
            out.push((i, r))
        }
    }
    Ok(out)
}
pub(super) fn cross(
    bs: &[Value],
    p: [f64; 2],
    parents: &[Parent],
    cs: &Candidates,
    assignments: &Assignments,
    w: &mut Work,
) -> Result<Assignments, Error> {
    let annotations = cs.iter().map(|(i, _)| *i).collect::<BTreeSet<_>>();
    let mut proposals: Vec<(usize, Vec<Relation>)> = vec![];
    for (i, r) in assignments {
        if kind(cs, *i) != Some("caption") || !["above", "below"].contains(&r.direction) {
            continue;
        }
        let anchor = &bs[*i];
        let mut matches = vec![];
        for (j, b) in bs.iter().enumerate() {
            w.charge(1)?;
            if annotations.contains(&j)
                || b["type"] != "text"
                || angle(b)? != r.parent.angle
                || !distinct(anchor, b)
            {
                continue;
            }
            let ah = height(anchor, p)?;
            let bh = height(b, p)?;
            if ah <= 0. || bh <= 0. || !(0.85..=1.15).contains(&(bh / ah)) {
                continue;
            }
            let af = optional(anchor, "_font_signatures", 4096)?;
            let bf = optional(b, "_font_signatures", 4096)?;
            if af.is_empty() || bf.is_empty() || font_conflict(anchor, b) {
                continue;
            }
            let (Some(ar), Some(br)) = (
                first_row(anchor, p, r.parent.angle)?,
                first_row(b, p, r.parent.angle)?,
            ) else {
                continue;
            };
            if (ar.n[1] - br.n[1]).abs() > 0.25 * median(vec![ah, bh]) {
                continue;
            }
            let mut ignored = annotations.clone();
            ignored.insert(j);
            if let Some(relation) = relations::choose(j, "caption", bs, p, parents, &ignored, w)? {
                if same_parent(&relation.parent, &r.parent) && relation.direction == r.direction {
                    matches.push((j, relation))
                }
            }
        }
        if matches.len() != 1 {
            continue;
        }
        let (j, r) = matches.remove(0);
        if let Some((_, rs)) = proposals.iter_mut().find(|(i, _)| *i == j) {
            rs.push(r)
        } else {
            proposals.push((j, vec![r]))
        }
    }
    Ok(proposals
        .into_iter()
        .filter(|(_, r)| r.len() == 1)
        .map(|(i, mut r)| (i, r.remove(0)))
        .collect())
}
fn shares(a: &Value, b: &Value, h: f64) -> bool {
    eq(&a["_lane_is_span"], &b["_lane_is_span"])
        && interval(a)
            .zip(interval(b))
            .is_some_and(|(a, b)| (a.0 - b.0).abs() <= 0.5 * h && (a.1 - b.1).abs() <= 0.5 * h)
}
fn fits(b: &Box4, parent: &Box4, h: f64) -> bool {
    if cy(b) < cy(parent) {
        return false;
    }
    let (projection, coverage, _) = relations::metrics(b, parent, 0, h);
    projection >= 0.8 && coverage >= 0.65
}
fn compatible(
    a: &Value,
    prev: &Value,
    b: &Value,
    parent: &Box4,
    p: [f64; 2],
    angle: i64,
    left: Option<f64>,
) -> Result<(bool, Option<f64>), Error> {
    let (Some(pb), Some(bb), Some(ar), Some(br)) = (
        local_box(prev, p, angle)?,
        local_box(b, p, angle)?,
        first_row(a, p, angle)?,
        first_row(b, p, angle)?,
    ) else {
        return Ok((false, left));
    };
    let ph = height(prev, p)?;
    let bh = height(b, p)?;
    if ph <= 0. || bh <= 0. {
        return Ok((false, left));
    }
    let pair = median(vec![ph, bh]);
    if !(0.75..=1.25).contains(&(bh / ph))
        || (bb.n[1] - pb.n[3]).max(0.) > 1.5 * pair
        || !shares(a, b, pair)
        || font_conflict(prev, b)
        || !fits(&bb, parent, pair)
    {
        return Ok((false, left));
    }
    if let Some(left) = left {
        Ok(((br.n[0] - left).abs() <= 0.5 * pair, Some(left)))
    } else {
        let indent = br.n[0] - ar.n[0];
        if !(0.5 * pair..=2. * pair).contains(&indent) {
            Ok((false, None))
        } else {
            Ok((true, Some(br.n[0])))
        }
    }
}
pub(super) fn footnotes(
    bs: &mut [Value],
    p: [f64; 2],
    cs: &Candidates,
    assignments: &Assignments,
    w: &mut Work,
) -> Result<BTreeSet<usize>, Error> {
    let annotations = cs.iter().map(|(i, _)| *i).collect::<BTreeSet<_>>();
    let mut used = BTreeSet::new();
    let mut ordered = assignments.clone();
    ordered.sort_by(|(i, _), (j, _)| {
        regions::key(*i, bs, p)
            .unwrap()
            .partial_cmp(&regions::key(*j, bs, p).unwrap())
            .unwrap()
    });
    for (i, r) in ordered {
        let anchor = &bs[i];
        if kind(cs, i) != Some("footnote")
            || anchor["_table_annotation_complete"] == true
            || anchor["type"] != "text"
            || !anchor["content"].is_string()
            || !candidates::footnote(text(anchor))
            || r.direction != "below"
            || r.parent.members.len() != 1
            || bs[r.parent.members[0]]["type"] != "table"
        {
            continue;
        }
        let Some(ab) = local_box(anchor, p, r.parent.angle)? else {
            continue;
        };
        let Some(lane) = interval(anchor) else {
            continue;
        };
        let mut ids = vec![];
        for (j, b) in bs.iter().enumerate() {
            w.charge(1)?;
            if j == i
                || r.parent.members.contains(&j)
                || used.contains(&j)
                || angle(b)? != r.parent.angle
            {
                continue;
            }
            if let Some(bb) = local_box(b, p, r.parent.angle)? {
                let overlap = (bb.n[2].min(lane.1) - bb.n[0].max(lane.0)).max(0.);
                let shorter = (bb.n[2] - bb.n[0]).min(lane.1 - lane.0).max(0.1);
                if cy(&bb) > cy(&ab) && overlap / shorter >= 0.5 {
                    ids.push((j, bb))
                }
            }
        }
        ids.sort_by(|a, b| {
            a.1.n[1]
                .total_cmp(&b.1.n[1])
                .then(a.1.n[0].total_cmp(&b.1.n[0]))
                .then(a.0.cmp(&b.0))
        });
        let mut collected = vec![i];
        let mut prev = i;
        let mut left = None;
        for (j, _) in ids {
            if annotations.contains(&j) || bs[j]["type"] != "text" {
                break;
            }
            let (ok, next) = compatible(
                anchor,
                &bs[prev],
                &bs[j],
                &r.parent.b,
                p,
                r.parent.angle,
                left,
            )?;
            left = next;
            if !ok {
                break;
            }
            collected.push(j);
            prev = j
        }
        if collected.len() > 1 {
            let merged = merge_blocks(bs, &collected)?;
            bs[i] = merged;
            used.extend(collected.into_iter().skip(1));
        }
    }
    Ok(used)
}
