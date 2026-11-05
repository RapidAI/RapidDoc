//! Original numbered/author-year regions and sequential reference start/end events.
use super::*;
struct Candidate {
    index: usize,
    number: u32,
    bracket: bool,
}
fn numbered(p: &mut Value, active: bool, w: &mut Work) -> Result<Vec<Box4>, Error> {
    let lines = ls(p)?;
    let boxes = lines
        .iter()
        .map(|l| bbox(&l["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let size = sz(p)?;
    let mut candidates = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        w.charge(txt(l)?.len() + 1)?;
        if integer(l, "angle", 0)? == 0 && !edge(l) {
            if let Some((number, bracket)) = text::number(txt(l)?) {
                candidates.push(Candidate {
                    index: i,
                    number,
                    bracket,
                })
            }
        }
    }
    candidates.sort_by(|a, b| boxes[a.index].n[0].total_cmp(&boxes[b.index].n[0]));
    let mut clusters = Vec::<Vec<Candidate>>::new();
    for c in candidates {
        let em = height(&lines[c.index], &boxes[c.index])?;
        let same = if let Some(last) = clusters.last() {
            w.charge(last.len() + 1)?;
            (boxes[c.index].n[0] - median(last.iter().map(|c| boxes[c.index].n[0]).collect())).abs()
                <= 0.8 * em
        } else {
            false
        };
        if same {
            clusters.last_mut().unwrap().push(c)
        } else {
            clusters.push(vec![c])
        }
    }
    let mut accepted = Vec::new();
    for mut cluster in clusters {
        cluster.sort_by(|a, b| boxes[a.index].n[1].total_cmp(&boxes[b.index].n[1]));
        let first = &cluster[0];
        let first_box = &boxes[first.index];
        if cluster.len() == 1
            && first.bracket
            && lines.iter().zip(&boxes).any(|(l, b)| {
                text::heading(txt(l).unwrap())
                    && b.n[3] <= first_box.n[1]
                    && axis(b, first_box) >= 0.5
            })
        {
            accepted.push(cluster);
            continue;
        }
        let min = if active { 2 } else { 3 };
        let consecutive = cluster
            .windows(2)
            .filter(|c| c[1].number == c[0].number + 1)
            .count();
        if cluster.len() < min
            || consecutive < min - 1
            || !active && !cluster.iter().all(|c| c.bracket)
        {
            continue;
        }
        accepted.push(cluster)
    }
    if accepted.is_empty() {
        return Ok(vec![]);
    }
    let bracketed = accepted.iter().flatten().filter(|c| c.bracket).count() as f64
        >= 0.75 * accepted.iter().map(Vec::len).sum::<usize>() as f64;
    accepted.sort_by(|a, b| boxes[a[0].index].n[0].total_cmp(&boxes[b[0].index].n[0]));
    let mut regions = Vec::new();
    for (i, c) in accepted.iter().enumerate() {
        let em = median(
            c.iter()
                .map(|c| height(&lines[c.index], &boxes[c.index]))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let left = median(c.iter().map(|c| boxes[c.index].n[0]).collect());
        let next = if i + 1 < accepted.len() {
            accepted[i + 1]
                .iter()
                .map(|c| boxes[c.index].n[0])
                .reduce(f64::min)
                .unwrap()
        } else {
            size[0]
        };
        let mut top = c
            .iter()
            .map(|c| boxes[c.index].n[1])
            .reduce(f64::min)
            .unwrap();
        let headings = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| text::heading(txt(l).unwrap()))
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let floor = headings
            .iter()
            .filter(|&&j| left - em <= boxes[j].n[0] && boxes[j].n[0] < next - em)
            .map(|&j| boxes[j].n[3])
            .reduce(f64::max)
            .unwrap_or(0.);
        if active || i > 0 {
            for (j, l) in lines.iter().enumerate() {
                w.charge(1)?;
                let b = &boxes[j];
                if left - em <= b.n[0]
                    && b.n[0] < left + 3. * em
                    && b.n[2] < next
                    && floor <= b.n[1]
                    && b.n[1] < top
                    && !edge(l)
                    && l["structural_title"] != true
                    && l["explicit_section_title"] != true
                    && !text::heading(txt(l)?)
                    && (0.85..=1.15).contains(&(height(l, b)? / em))
                {
                    top = top.min(b.n[1])
                }
            }
        }
        let mut endings = Vec::new();
        for (j, l) in lines.iter().enumerate() {
            let b = &boxes[j];
            if b.n[1] <= top || b.n[0] < left - em || b.n[0] >= next - em {
                continue;
            }
            let mut end = text::appendix(txt(l)?);
            if !end
                && l["semantic_type"] == "paragraph_title"
                && (l["structural_title"] == true || l["explicit_section_title"] == true)
                && text::number(txt(l)?).is_none()
            {
                end = true;
                for (k, a) in boxes.iter().enumerate() {
                    w.charge(1)?;
                    if j != k
                        && left - em <= a.n[0]
                        && a.n[0] < next - em
                        && a.n[1] < b.n[1]
                        && axis(a, b) >= 0.2
                        && b.n[1] - a.n[3] < 0.6 * em
                    {
                        end = false;
                        break;
                    }
                }
            }
            if end {
                endings.push(b.n[1])
            }
        }
        let abstracts = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| text::abstract_line(txt(l).unwrap()))
            .map(|(i, _)| boxes[i].n[1])
            .collect::<Vec<_>>();
        for (j, l) in lines.iter().enumerate() {
            w.charge(abstracts.len() + 1)?;
            let b = &boxes[j];
            if b.n[1] > top
                && b.n[2] - b.n[0] >= 0.6 * size[0]
                && (cx(b) - size[0] / 2.).abs() <= 0.05 * size[0]
                && text::alpha_count(txt(l)?) >= 20
                && text::number(txt(l)?).is_none()
                && abstracts
                    .iter()
                    .any(|&y| y - b.n[3] >= 2. * em && y - b.n[3] <= 20. * em)
            {
                endings.push(b.n[1])
            }
        }
        regions.push(bbox(&json!([
            left - em,
            top - 0.2 * em,
            next - em,
            endings
                .into_iter()
                .reduce(f64::min)
                .unwrap_or(0.94 * size[1])
        ]))?);
    }
    p["bracketed_reference_numbers"] = json!(bracketed);
    Ok(regions)
}
fn standalone(p: &Value, w: &mut Work) -> Result<Vec<Box4>, Error> {
    let lines = ls(p)?
        .iter()
        .filter(|l| {
            integer(l, "angle", 0).ok() == Some(0)
                && !edge(l)
                && l["semantic_type"] != "page_footnote"
        })
        .cloned()
        .collect::<Vec<_>>();
    if lines.len() < 15 {
        return Ok(vec![]);
    }
    let mut regions = Vec::new();
    for (left, right) in corridors(&lines, p, w)? {
        let members = lines
            .iter()
            .filter(|l| {
                let b = bbox(&l["bbox"]).unwrap();
                left <= cx(&b) && cx(&b) < right
            })
            .collect::<Vec<_>>();
        if members.len() < 15 {
            continue;
        }
        let bs = members
            .iter()
            .map(|l| bbox(&l["bbox"]))
            .collect::<Result<Vec<_>, _>>()?;
        let em = median(
            members
                .iter()
                .zip(&bs)
                .map(|(l, b)| height(l, b))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let edge = bs.iter().map(|b| b.n[0]).reduce(f64::min).unwrap();
        let mut starts = 0;
        let mut dates = 0;
        let mut indented = 0;
        for (l, b) in members.iter().zip(&bs) {
            w.charge(txt(l)?.len() + 1)?;
            let t = txt(l)?;
            if (b.n[0] - edge).abs() <= 0.4 * em
                && t.chars().filter(|&c| c == ',').count() >= 2
                && text::author(t)
            {
                starts += 1
            }
            if b.n[0] - edge > 0.5 * em && b.n[0] - edge < 2. * em {
                indented += 1
            }
            if text::year(t) {
                dates += 1
            }
        }
        if starts >= 4 && dates >= 4 && indented as f64 >= members.len() as f64 / 3. {
            regions.push(bbox(&json!([
                left,
                bs.iter().map(|b| b.n[1]).reduce(f64::min).unwrap() - 0.2 * em,
                right,
                bs.iter().map(|b| b.n[3]).reduce(f64::max).unwrap() + 0.2 * em
            ]))?)
        }
    }
    Ok(regions)
}
fn extend(p: &mut Value, regions: &[Box4]) -> Result<(), Error> {
    let mut all = optional(p, "reference_regions", 4096)?.to_vec();
    all.extend(regions.iter().map(|b| json!(b.raw)));
    if !all.is_empty() || p.get("reference_regions").is_some() {
        p["reference_regions"] = json!(all)
    }
    Ok(())
}
pub(super) fn mark(pages: &mut [Value], w: &mut Work) -> Result<(), Error> {
    let mut active = false;
    let mut numbered_active = false;
    for p in pages {
        let size = sz(p)?;
        let headings = ls(p)?
            .iter()
            .filter(|l| text::heading(txt(l).unwrap()))
            .map(sid)
            .collect::<Result<BTreeSet<_>, _>>()?;
        let numbered = numbered(p, numbered_active || !headings.is_empty(), w)?;
        if !numbered.is_empty() {
            p["numbered_references"] = json!(true);
            extend(p, &numbered)?;
            if !headings.is_empty() {
                let mut lines = ls(p)?.to_vec();
                for l in &mut lines {
                    if headings.contains(&sid(l)?) {
                        l["semantic_type"] = json!("paragraph_title");
                        l["structural_title"] = json!(true)
                    }
                }
                p["remaining_lines"] = json!(lines);
                prune(p, &headings)?
            }
            active = numbered.last().unwrap().n[3] == 0.94 * size[1];
            numbered_active = active;
            continue;
        }
        if numbered_active {
            active = false
        }
        numbered_active = false;
        if !active && headings.is_empty() {
            let regions = standalone(p, w)?;
            extend(p, &regions)?;
            continue;
        }
        let lines = ls(p)?.to_vec();
        let bs = lines
            .iter()
            .map(|l| bbox(&l["bbox"]))
            .collect::<Result<Vec<_>, _>>()?;
        let mut regions = Vec::new();
        for (left, right) in corridors(&lines, p, w)? {
            let mut top = if active { Some(0.055 * size[1]) } else { None };
            let mut members = (0..lines.len())
                .filter(|&i| left <= cx(&bs[i]) && cx(&bs[i]) < right)
                .collect::<Vec<_>>();
            members.sort_by(|&i, &j| bs[i].n[1].total_cmp(&bs[j].n[1]));
            for i in members {
                w.charge(1)?;
                let l = &lines[i];
                let b = &bs[i];
                if headings.contains(&sid(l)?) {
                    if let Some(y) = top {
                        if b.n[1] > y {
                            regions.push(bbox(&json!([left, y, right, b.n[1]]))?)
                        }
                    }
                    top = Some(b.n[3]);
                    active = true
                } else if active
                    && (text::appendix(txt(l)?)
                        || l["semantic_type"] == "paragraph_title"
                            && (l["structural_title"] == true
                                || l["explicit_section_title"] == true))
                {
                    if let Some(y) = top {
                        if b.n[1] > y {
                            regions.push(bbox(&json!([left, y, right, b.n[1]]))?)
                        }
                    }
                    top = None;
                    active = false
                }
            }
            if let Some(top) = top {
                if top < 0.94 * size[1] {
                    regions.push(bbox(&json!([left, top, right, 0.94 * size[1]]))?)
                }
            }
        }
        extend(p, &regions)?;
    }
    Ok(())
}
