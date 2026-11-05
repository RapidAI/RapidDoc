//! Original repeated unknown-weight heading recovery using barrier-aware layout evidence.
use super::*;
pub(super) fn stable_bounds(
    s: &State,
    rows: &[usize],
    start: usize,
    h: f64,
) -> Result<Option<Box4>, Error> {
    let mut groups = Vec::<Vec<usize>>::new();
    for &i in rows.iter().skip(start).take(20) {
        let l = &s.lines[i];
        if !s.unresolved(i) || !l["paragraph_group"].is_null() || l["title_suppressed"] == true {
            break;
        }
        if !groups.is_empty()
            && !eq(
                &l["font_signature"],
                &s.lines[groups[0][0]]["font_signature"],
            )
        {
            break;
        }
        if let Some(g) = groups.last_mut() {
            if (cy(&s.boxes[i]) - cy(&s.boxes[g[0]])).abs() < 0.2 * h {
                g.push(i);
                continue;
            }
        }
        if let Some(g) = groups.last() {
            let bottom = g.iter().map(|&j| s.boxes[j].n[3]).reduce(f64::max).unwrap();
            if s.boxes[i].n[1] - bottom > 0.8 * h || groups.len() >= 5 {
                break;
            }
        }
        groups.push(vec![i])
    }
    if groups.len() < 2 {
        return Ok(None);
    }
    let bs = groups
        .iter()
        .map(|g| s.bounds(g))
        .collect::<Result<Vec<_>, _>>()?;
    let max = bs.iter().map(|b| b.n[2] - b.n[0]).reduce(f64::max).unwrap();
    let wide = bs
        .iter()
        .filter(|b| b.n[2] - b.n[0] >= 0.8 * max)
        .collect::<Vec<_>>();
    if wide.len() < 2 {
        return Ok(None);
    }
    Ok(Some(bbox(&json!([
        median(wide.iter().map(|b| b.n[0]).collect()),
        bs[0].n[1],
        median(wide.iter().map(|b| b.n[2]).collect()),
        bs.last().unwrap().n[3]
    ]))?))
}
pub(super) fn classify(pages: &mut [Value], w: &mut Work) -> Result<(), Error> {
    let mut candidates = Vec::<(usize, Value, Vec<usize>)>::new();
    let mut body_styles = Vec::<Value>::new();
    let english = crate::cached_regex!(r"[A-Za-z]{2,}");
    for (page, p) in pages.iter().enumerate().skip(1) {
        let size = size(p.get("page_size").unwrap_or(&p["source"]["page_size"]))?;
        let lines = array(&p["remaining_lines"], 4096)?.to_vec();
        let boxes = lines
            .iter()
            .map(|l| bbox(&l["bbox"]))
            .collect::<Result<Vec<_>, _>>()?;
        let s = State { lines, boxes };
        let containers = optional(p, "fixed_blocks", 4000)?
            .iter()
            .map(|b| bbox(&b["bbox"]))
            .collect::<Result<Vec<_>, _>>()?;
        let mut free = Vec::new();
        for (i, l) in s.lines.iter().enumerate() {
            w.charge(containers.len() + 1)?;
            if !containers
                .iter()
                .any(|c| super::super::path_graphics::overlap_first(&s.boxes[i], c) >= 0.8)
            {
                free.push(lane_line(l))
            }
        }
        let lanes = super::super::page_containers::layout_lanes(&free, size, w)?;
        for lane in &lanes {
            let left = num(&lane["left"])?;
            let right = num(&lane["right"])?;
            let bounds = bbox(&json!([left, 0., right, 1.]))?;
            let Some((lo, hi)) =
                super::super::page_containers::layout_corridor(&bounds, &lanes, size[0])
            else {
                return Err(invalid());
            };
            let mut rows = Vec::new();
            for (i, l) in s.lines.iter().enumerate() {
                w.charge(containers.len() + 1)?;
                let b = &s.boxes[i];
                if integer(l, "angle", 0)? == 0
                    && lo <= cx(b)
                    && cx(b) < hi
                    && b.n[1] >= 0.06 * size[1]
                    && b.n[3] <= 0.93 * size[1]
                    && (s.unresolved(i) || l["semantic_type"] == "paragraph_title")
                    && !inside(b, &containers)
                {
                    rows.push(i)
                }
            }
            rows.sort_by(|&i, &j| {
                s.boxes[i].n[1]
                    .total_cmp(&s.boxes[j].n[1])
                    .then(s.boxes[i].n[0].total_cmp(&s.boxes[j].n[0]))
            });
            if rows.len() < 4 {
                continue;
            }
            let h = median(
                rows.iter()
                    .map(|&i| s.h(i))
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let mut k = 0;
            while k < rows.len() {
                let first = rows[k];
                let mut end = k + 1;
                while end < rows.len() {
                    w.charge(1)?;
                    let i = rows[end];
                    let (a, b) = (&s.boxes[first], &s.boxes[i]);
                    let gap = b.n[1] - s.boxes[rows[end - 1]].n[3];
                    if !eq(
                        &s.lines[i]["font_signature"],
                        &s.lines[first]["font_signature"],
                    ) || !((b.n[0] - a.n[0]).abs() <= 0.5 * h
                        || (cx(b) - cx(a)).abs() <= 0.5 * h)
                        || gap < -0.25 * h
                        || gap > 0.8 * h
                    {
                        break;
                    }
                    end += 1
                }
                let group = rows[k..end].to_vec();
                let signature = &s.lines[first]["font_signature"];
                if group.len() >= 4 && !body_styles.iter().any(|f| eq(f, signature)) {
                    body_styles.push(signature.clone())
                }
                let texts = group
                    .iter()
                    .map(|&i| s.lines[i]["text"].as_str().ok_or_else(invalid))
                    .collect::<Result<Vec<_>, _>>()?
                    .join(" ");
                let words = texts.split(python_space).filter(|s| !s.is_empty()).count();
                let text = s.lines[first]["text"]
                    .as_str()
                    .ok_or_else(invalid)?
                    .trim_matches(python_space);
                let heading = (2..=30).contains(&words)
                    || words == 1
                        && (5..=25).contains(&text.chars().count())
                        && text
                            .chars()
                            .all(rapiddoc_native::text_boundary::python_alpha);
                let body_bounds = stable_bounds(&s, &rows, end, h)?;
                if !signature.is_null()
                    && group.len() <= 3
                    && end < rows.len()
                    && heading
                    && group.iter().all(|&i| {
                        s.lines[i]["dominant_font_weight"].is_null()
                            && coverage(&s.lines[i]).unwrap() >= 0.75
                    })
                    && !group
                        .iter()
                        .any(|&i| s.lines[i]["paragraph_terminal"] == true)
                {
                    let j = rows[end];
                    let b = &s.boxes[j];
                    let a = &s.boxes[first];
                    let gap = b.n[1] - s.boxes[*group.last().unwrap()].n[3];
                    let align = (b.n[0] - a.n[0]).abs() <= h
                        || (cx(a) - (left + right) / 2.).abs() <= h
                        || body_bounds.as_ref().is_some_and(|body| {
                            (cx(&s.bounds(&group).unwrap()) - cx(body)).abs() <= 0.5 * h
                        });
                    if !eq(&s.lines[j]["font_signature"], signature)
                        && b.n[2] - b.n[0] >= 0.25 * (hi - lo)
                        && english
                            .find_iter(s.lines[j]["text"].as_str().ok_or_else(invalid)?)
                            .count()
                            >= 4
                        && align
                        && gap >= -0.1 * h
                        && gap <= 1.5 * h
                        && (k == 0 || a.n[1] - s.boxes[rows[k - 1]].n[3] >= 0.3 * h)
                    {
                        candidates.push((page, signature.clone(), group))
                    }
                }
                k = end;
            }
        }
    }
    let mut groups = Vec::<(Value, Vec<(usize, Vec<usize>)>)>::new();
    for (p, signature, ids) in candidates {
        if let Some((_, g)) = groups.iter_mut().find(|(s, _)| eq(s, &signature)) {
            g.push((p, ids))
        } else {
            groups.push((signature, vec![(p, ids)]))
        }
    }
    for (signature, groups) in groups {
        if body_styles.iter().any(|s| eq(s, &signature))
            || groups.len() < 3
            || groups
                .iter()
                .map(|(p, _)| *p)
                .collect::<BTreeSet<_>>()
                .len()
                < 2
        {
            continue;
        }
        for (p, ids) in groups {
            for i in ids {
                pages[p]["remaining_lines"][i]["semantic_type"] = json!("paragraph_title");
                pages[p]["remaining_lines"][i]["structural_title"] = json!(true)
            }
        }
    }
    Ok(())
}
