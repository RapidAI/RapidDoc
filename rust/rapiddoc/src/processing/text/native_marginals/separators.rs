use super::*;
pub(super) fn detect(pages: &[Page], w: &mut Work) -> Result<Vec<Vec<[Value; 4]>>, Error> {
    let mut groups: Vec<([f64; 3], Vec<(usize, Box4)>)> = Vec::new();
    for (i, p) in pages.iter().enumerate() {
        if p.size[0] <= 0. || p.size[1] <= 0. {
            continue;
        }
        let mut source = p.source.as_ref().clone();
        source["lines"] = json!(p.lines);
        let connected =
            crate::legacy_model::closed_rule_pipeline::connected_horizontal_rules(&source, w)?;
        for line in optional(&p.source, "drawing_lines", 5000)? {
            let b = bbox(&line["bbox"])?;
            let cy = center(&b, 1);
            w.charge(p.lines.len() + connected.len() + groups.len() + 1)?;
            if line["orientation"] != "horizontal"
                || cy > 0.15 * p.size[1]
                || b.n[2] - b.n[0] < 0.6 * p.size[0]
                || p.lines
                    .iter()
                    .filter(|l| {
                        l["angle"].as_i64().unwrap_or(0) == 0
                            && l["bbox"][3].as_f64().unwrap_or(f64::INFINITY) <= cy
                    })
                    .count()
                    < 2
                || connected.iter().any(|c| c.n == b.n)
            {
                continue;
            }
            let signature = [
                format!("{:.2}", b.n[0] / p.size[0]).parse().unwrap(),
                format!("{:.2}", b.n[2] / p.size[0]).parse().unwrap(),
                format!("{:.3}", cy / p.size[1]).parse().unwrap(),
            ];
            if let Some((_, members)) = groups.iter_mut().find(|(s, _)| *s == signature) {
                members.push((i, b))
            } else {
                groups.push((signature, vec![(i, b)]))
            }
        }
    }
    let mut result = vec![Vec::<Box4>::new(); pages.len()];
    for (_, members) in groups {
        if members
            .iter()
            .map(|(p, _)| p)
            .collect::<BTreeSet<_>>()
            .len()
            < 3
        {
            continue;
        }
        let mut supported = BTreeSet::new();
        let mut candidates = Vec::new();
        for (i, b) in &members {
            let p = &pages[*i];
            for (j, l) in p.lines.iter().enumerate() {
                w.charge(1)?;
                if integer(l, "angle", 0)? != 0 || bbox(&l["bbox"])?.n[3] > center(b, 1) {
                    continue;
                }
                if l["semantic_type"] == "header" {
                    supported.insert(*i);
                    continue;
                }
                if let Some(c) = candidate(*i, j, p, w)? {
                    if c.region == "header" {
                        candidates.push(c)
                    }
                }
            }
        }
        for (i, a) in candidates.iter().enumerate() {
            for b in &candidates[i + 1..] {
                let delta = b.page as i64 - a.page as i64;
                if delta > 2 {
                    break;
                }
                w.charge(1)?;
                if delta > 0
                    && geometry(a, b, pages)?
                    && text::matches(
                        txt(&pages[a.page].lines[a.line])?,
                        txt(&pages[b.page].lines[b.line])?,
                        w,
                    )?
                {
                    supported.insert(a.page);
                    supported.insert(b.page);
                }
            }
        }
        if supported.len() < 2 {
            continue;
        }
        for (i, b) in members {
            if !result[i].iter().any(|a| a.n == b.n) {
                result[i].push(b)
            }
        }
    }
    for bs in &mut result {
        bs.sort_by(|a, b| a.n.partial_cmp(&b.n).unwrap())
    }
    Ok(result
        .into_iter()
        .map(|bs| bs.into_iter().map(|b| b.raw).collect())
        .collect())
}
