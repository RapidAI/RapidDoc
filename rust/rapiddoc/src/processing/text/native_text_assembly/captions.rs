//! Original image/numeric-grid caption recovery and labeled visual-note splitting.
use super::*;
use super::{block_common as c, member_blocks as m};
pub(super) fn marker(t: &str) -> bool {
    regex(r"(?i)^(?:图\s*[0-9０-９一二三四五六七八九十]|fig(?:ure)?\.?\s*[0-9])").is_match(&fold(t))
}
fn terminal(t: &str) -> bool {
    regex(r#"[.!?。！？][\])’"']*$"#).is_match(t.trim_end_matches(python_space))
}
fn graph(t: &str) -> bool {
    let t = fold(t);
    regex(r"(?i)graph|chart|plot").find_iter(&t).any(|m| {
        let word = |c: char| c == '_' || python_alnum(c);
        !t[..m.start()].chars().next_back().is_some_and(word)
            && !t[m.end()..].chars().next().is_some_and(word)
    })
}
fn image_groups(images: &[Box4], h: f64, w: &mut Work) -> Result<Vec<Box4>, Error> {
    let mut remaining = images.to_vec();
    let mut out = images.to_vec();
    while !remaining.is_empty() {
        let mut group = vec![remaining.remove(0)];
        let mut changed = true;
        while changed {
            changed = false;
            for b in remaining.clone() {
                w.charge(group.len())?;
                let aligned = group.iter().any(|a| {
                    let overlap = (a.n[3].min(b.n[3]) - a.n[1].max(b.n[1])).max(0.);
                    let min = (a.n[3] - a.n[1]).min(b.n[3] - b.n[1]).max(0.1);
                    let gap = (a.n[0].max(b.n[0]) - a.n[2].min(b.n[2])).max(0.);
                    overlap / min >= 0.7 && gap <= 2. * h
                });
                if aligned {
                    group.push(b.clone());
                    let i = remaining
                        .iter()
                        .position(|a| eq(&json!(a.raw), &json!(b.raw)))
                        .ok_or_else(invalid)?;
                    remaining.remove(i);
                    changed = true
                }
            }
        }
        if group.len() >= 2 {
            out.push(combined(group.iter())?)
        }
    }
    Ok(out)
}
fn seed_matches(block: &Value, image: &Box4, h: f64, max_gap: f64) -> Result<bool, Error> {
    let b = bbox(&block["bbox"])?;
    let iw = (image.n[2] - image.n[0]).max(0.1);
    let bw = (b.n[2] - b.n[0]).max(0.1);
    Ok(cy(&b) >= image.n[3] - 0.25 * h
        && (b.n[1] - image.n[3]).max(0.) <= max_gap * h
        && axis(&b, image, 0) >= 0.35
        && (cx(&b) - cx(image)).abs() <= 0.35 * iw.max(bw)
        && bw <= 1.75 * iw)
}
fn structural(seed: &Value, tail: &Value) -> Result<bool, Error> {
    let ar = c::rows(seed)?;
    let br = c::rows(tail)?;
    let ah = c::heights(seed, false)?;
    let bh = c::heights(tail, false)?;
    if ar.is_empty() || br.is_empty() || ar.len() != ah.len() || br.len() != bh.len() {
        return Ok(false);
    }
    let mut a = ar.into_iter().zip(ah).collect::<Vec<_>>();
    let mut b = br.into_iter().zip(bh).collect::<Vec<_>>();
    let order = |a: &(Box4, f64), b: &(Box4, f64)| {
        a.0.n[1]
            .total_cmp(&b.0.n[1])
            .then(a.0.n[0].total_cmp(&b.0.n[0]))
    };
    a.sort_by(order);
    b.sort_by(order);
    let gaps = a
        .windows(2)
        .map(|p| (p[1].0.n[1] - (p[0].0.n[1] + p[0].1)).max(0.))
        .collect::<Vec<_>>();
    let reg = if gaps.is_empty() {
        0.
    } else {
        median(gaps.clone())?
    };
    let mad = if gaps.is_empty() {
        0.
    } else {
        median(gaps.iter().map(|g| (g - reg).abs()).collect())?
    };
    let previous = a.last().unwrap();
    let current = &b[0];
    let h = previous.1.max(current.1);
    let min = previous.1.min(current.1).max(0.1);
    let gap = current.0.n[1] - (previous.0.n[1] + previous.1);
    Ok(gap > reg + (0.75 * h).max(3. * mad) && (h / min >= 1.12 || c::conflict(seed, tail)))
}
fn tail_matches(seed: &Value, tail: &Value, h: f64) -> Result<bool, Error> {
    if m::truth(&tail["_protected_hard_break_before"])
        && m::text(seed)
            .trim_end_matches(python_space)
            .ends_with([')', '）'])
    {
        return Ok(false);
    }
    let a = bbox(&seed["bbox"])?;
    let b = bbox(&tail["bbox"])?;
    let ah = c::heights(seed, false)?;
    let bh = c::heights(tail, false)?;
    let em = if ah.is_empty() { h } else { median(ah)? };
    if tail["type"] == "paragraph_title" {
        if bh.is_empty()
            || bh.len() > 2
            || bh.iter().copied().fold(f64::NEG_INFINITY, f64::max) > 1.1 * em
        {
            return Ok(false);
        }
        let af = optional(seed, "_font_signatures", 4096)?;
        let bf = optional(tail, "_font_signatures", 4096)?;
        if !af.iter().any(|a| bf.iter().any(|b| eq(a, b))) {
            return Ok(false);
        }
        for l in m::lines(tail)? {
            if !l["dominant_font_weight"].is_null() && num(&l["dominant_font_weight"])? >= 600. {
                return Ok(false);
            }
        }
    }
    let centered = !bh.is_empty()
        && bh.len() <= 2
        && bh.iter().copied().fold(f64::NEG_INFINITY, f64::max) <= 1.1 * em
        && b.n[0] >= a.n[0] - 0.25 * em
        && b.n[2] <= a.n[2] + 0.25 * em
        && (cx(&b) - cx(&a)).abs() <= 2. * em
        && (-0.25 * em..=0.5 * em).contains(&(b.n[1] - a.n[3]));
    if b.n[0] < a.n[0] - h
        && (cx(&b) - cx(&a)).abs() > 0.25 * (b.n[2] - b.n[0]).max(a.n[2] - a.n[0])
    {
        return Ok(false);
    }
    if !centered
        && !c::role(seed, tail, h)
        && (axis(&a, &b, 0) < 0.75 || (a.n[0] - b.n[0]).abs() > h)
    {
        return Ok(false);
    }
    if cy(&b) <= cy(&a) || structural(seed, tail)? {
        return Ok(false);
    }
    let gap = b.n[1] - a.n[3];
    if !bh.is_empty()
        && bh.iter().copied().fold(f64::INFINITY, f64::min) >= 1.15 * em
        && gap >= 0.5 * em
    {
        return Ok(false);
    }
    let parenthetical = m::text(tail)
        .trim_start_matches(python_space)
        .starts_with(['(', '（']);
    if gap.max(0.) > (if parenthetical { em } else { 0.5 * h }) || axis(&a, &b, 0) < 0.35 {
        return Ok(false);
    }
    Ok(centered || !c::conflict(seed, tail))
}
fn pass(
    bs: Vec<Value>,
    images: &[Box4],
    grids: &[Box4],
    continuation: bool,
    w: &mut Work,
) -> Result<Vec<Value>, Error> {
    if images.is_empty() {
        return Ok(bs);
    }
    let ids = bs
        .iter()
        .enumerate()
        .filter(|(_, b)| {
            ["text", "paragraph_title"].iter().any(|t| b["type"] == *t)
                && b["content"].is_string()
                && b["bbox"].is_array()
        })
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    let hs = ids
        .iter()
        .map(|&i| c::heights(&bs[i], true))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let h = if hs.is_empty() { 1. } else { median(hs)? };
    let images = image_groups(images, h, w)?;
    let mut seeds = vec![];
    for &i in &ids {
        let t = m::text(&bs[i]);
        if !marker(t.trim_matches(python_space)) || continuation && terminal(t) {
            continue;
        }
        let mut matched = false;
        for image in &images {
            if seed_matches(&bs[i], image, h, 2.5)? {
                matched = true;
                break;
            }
        }
        if !matched && graph(t) {
            for image in grids {
                if seed_matches(&bs[i], image, h, 4.)? {
                    matched = true;
                    break;
                }
            }
        }
        if matched {
            seeds.push(i)
        }
    }
    if seeds.is_empty() {
        return Ok(bs);
    }
    let mut assignments: BTreeMap<usize, Vec<usize>> = seeds.iter().map(|&i| (i, vec![])).collect();
    for i in ids {
        if seeds.contains(&i) {
            continue;
        }
        let mut matches = vec![];
        let b = bbox(&bs[i]["bbox"])?;
        for &j in &seeds {
            w.charge(1)?;
            if tail_matches(&bs[j], &bs[i], h)? {
                let a = bbox(&bs[j]["bbox"])?;
                matches.push((cy(&b) - cy(&a), (cx(&b) - cx(&a)).abs(), j))
            }
        }
        if let Some((_, _, j)) = matches.into_iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
            assignments.get_mut(&j).unwrap().push(i)
        }
    }
    let (mut used, mut replacements) = (BTreeSet::new(), BTreeMap::new());
    for (i, tails) in assignments {
        if tails.is_empty() {
            continue;
        }
        let mut group = vec![i];
        group.extend_from_slice(&tails);
        replacements.insert(i, c::merge(&bs, &group, false)?);
        used.extend(tails);
    }
    Ok(c::replace(bs, replacements, used))
}
pub(super) fn merge(
    mut bs: Vec<Value>,
    images: &[Box4],
    grids: &[Box4],
    w: &mut Work,
) -> Result<Vec<Value>, Error> {
    let mut continuation = false;
    loop {
        let len = bs.len();
        bs = pass(bs, images, grids, continuation, w)?;
        if bs.len() == len {
            return Ok(bs);
        }
        continuation = true
    }
}
fn note(t: &str, source: bool) -> bool {
    regex(if source {
        r"(?i)^\s*(?:sources?|资料来源|数据来源|来源)\s*[:：]"
    } else {
        r"(?i)^\s*(?:notes?|注|备注|sources?|资料来源|数据来源|来源)\s*[:：]"
    })
    .is_match(&fold(t))
}
pub(super) fn split_notes(bs: Vec<Value>, images: &[Box4]) -> Result<Vec<Value>, Error> {
    let mut out = vec![];
    for b in bs {
        let mut ls = m::lines(&b)?.to_vec();
        ls.sort_by(|a, b| {
            let a = m::raw(a).unwrap();
            let b = m::raw(b).unwrap();
            a.n[1].total_cmp(&b.n[1]).then(a.n[0].total_cmp(&b.n[0]))
        });
        if b["type"] != "text"
            || ls.len() < 2
            || !note(ls[0]["text"].as_str().ok_or_else(invalid)?, false)
        {
            out.push(b);
            continue;
        }
        let h = median(ls.iter().map(m::height).collect::<Result<Vec<_>, _>>()?)?;
        let bb = bbox(&b["bbox"])?;
        if !images.iter().any(|image| {
            (0.0..=6. * h).contains(&(bb.n[1] - image.n[3])) && axis(&bb, image, 0) >= 0.65
        }) {
            out.push(b);
            continue;
        }
        let mut cuts = vec![0];
        for (i, l) in ls.iter().enumerate().skip(1) {
            if note(l["text"].as_str().ok_or_else(invalid)?, true) {
                cuts.push(i)
            }
        }
        cuts.push(ls.len());
        let nh = m::height(&ls[0])?;
        let mut start = None;
        for (i, l) in ls.iter().enumerate().skip(1) {
            if m::height(l)? >= 1.2 * nh && !note(l["text"].as_str().ok_or_else(invalid)?, false) {
                start = Some(i);
                break;
            }
        }
        if let Some(i) = start {
            cuts.push(i);
            cuts.sort_unstable();
            cuts.dedup()
        }
        if cuts.len() == 2 {
            out.push(b);
            continue;
        }
        for p in cuts.windows(2) {
            let members = &ls[p[0]..p[1]];
            let mut child = b.clone();
            let boxes = members.iter().map(m::raw).collect::<Result<Vec<_>, _>>()?;
            child["bbox"] = json!(combined(boxes.iter())?.raw);
            child["content"] = json!(m::reassemble(members, "")?);
            child["_text_lines"] = json!(members);
            child["_local_line_bboxes"] =
                json!(boxes.iter().map(|b| b.raw.clone()).collect::<Vec<_>>());
            child["_local_output_line_bboxes"] = json!(members
                .iter()
                .map(|l| m::ink(l).map(|b| b.raw))
                .collect::<Result<Vec<_>, _>>()?);
            child["_line_heights"] = json!(members
                .iter()
                .map(m::height)
                .collect::<Result<Vec<_>, _>>()?);
            let mut fonts = vec![];
            for l in members {
                let f = &l["font_signature"];
                if !f.is_null() && !fonts.iter().any(|v| eq(v, f)) {
                    fonts.push(f.clone())
                }
            }
            c::sort_set(&mut fonts);
            child["_font_signatures"] = json!(fonts);
            out.push(child)
        }
    }
    Ok(out)
}
