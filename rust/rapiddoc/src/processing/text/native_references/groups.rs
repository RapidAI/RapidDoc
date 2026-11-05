//! Original reference-region grouping, stable left edges and biography barriers.
use super::*;
fn starts(t: &str, bracketed: bool) -> bool {
    text::number(t).is_some_and(|(_, b)| !bracketed || b)
}
pub(super) fn group(p: &mut Value, w: &mut Work) -> Result<(), Error> {
    let mut lines = ls(p)?.to_vec();
    let boxes = lines
        .iter()
        .map(|l| bbox(&l["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let numbered_page = p["numbered_references"] == true;
    let bracketed = p["bracketed_reference_numbers"] == true;
    let regions = optional(p, "reference_regions", 4096)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let mut group = 0;
    let mut claimed = BTreeSet::new();
    for region in regions {
        let mut members = Vec::new();
        for (i, l) in lines.iter().enumerate() {
            w.charge(txt(l)?.len() + 1)?;
            let b = &boxes[i];
            if integer(l, "angle", 0)? == 0
                && region.n[0] <= cx(b)
                && cx(b) < region.n[2]
                && region.n[1] <= cy(b)
                && cy(b) <= region.n[3]
                && !matches!(
                    l["semantic_type"].as_str(),
                    Some("page_number" | "doc_title" | "header" | "footer")
                )
                && (numbered_page || l["semantic_type"] != "page_footnote")
                && (!numbered_page || b.n[2] - b.n[0] <= 1.15 * (region.n[2] - region.n[0]))
                && (!numbered_page || text::alnum(txt(l)?))
            {
                members.push(i)
            }
        }
        if members.len() < if numbered_page { 1 } else { 3 } {
            continue;
        }
        let em = median(
            members
                .iter()
                .map(|&i| height(&lines[i], &boxes[i]))
                .collect::<Result<Vec<_>, _>>()?,
        );
        let left = members
            .iter()
            .map(|&i| boxes[i].n[0])
            .reduce(f64::min)
            .unwrap();
        members.sort_by(|&i, &j| {
            cy(&boxes[i])
                .total_cmp(&cy(&boxes[j]))
                .then(boxes[i].n[0].total_cmp(&boxes[j].n[0]))
        });
        let mut biography = None;
        for &i in &members {
            w.charge(txt(&lines[i])?.len() + 1)?;
            if text::biography(txt(&lines[i])?) {
                biography = Some(biography.map_or(boxes[i].n[1], |old: f64| old.min(boxes[i].n[1])))
            }
        }
        if let Some(y) = biography {
            members.retain(|&i| boxes[i].n[1] < y)
        }
        if !members
            .iter()
            .any(|&i| (boxes[i].n[0] - left).abs() <= 0.5 * em)
        {
            continue;
        }
        let numbered = numbered_page
            && members
                .iter()
                .any(|&i| starts(txt(&lines[i]).unwrap(), bracketed));
        if !numbered
            && members
                .iter()
                .filter(|&&i| boxes[i].n[0] - left > 0.5 * em && boxes[i].n[0] - left < 4. * em)
                .count()
                < 2
        {
            continue;
        }
        let mut rows = Vec::<Vec<usize>>::new();
        for &i in &members {
            let same = if let Some(last) = rows.last() {
                w.charge(last.len() + 1)?;
                (cy(&boxes[i]) - median(last.iter().map(|&j| cy(&boxes[j])).collect())).abs()
                    < 0.6 * em
            } else {
                false
            };
            if same {
                rows.last_mut().unwrap().push(i)
            } else {
                rows.push(vec![i])
            }
        }
        for (index, mut row) in rows.into_iter().enumerate() {
            row.sort_by(|&i, &j| boxes[i].n[0].total_cmp(&boxes[j].n[0]));
            let first = row[0];
            let start = starts(txt(&lines[first])?, bracketed)
                || !numbered && boxes[first].n[0] - left <= 0.5 * em;
            if index == 0 || start {
                group += 1
            }
            for i in row {
                lines[i]["paragraph_group"] = json!(group);
                if numbered {
                    lines[i]["reference_start"] = json!(start)
                }
                lines[i]["semantic_type"] = Value::Null;
                lines[i]["title_suppressed"] = json!(true)
            }
        }
        // Each region prunes immediately, matching the original live mutation order.
        claimed.extend(members.iter().map(|&i| sid(&lines[i]).unwrap()));
        prune(p, &claimed)?;
    }
    p["remaining_lines"] = json!(lines);
    Ok(())
}
