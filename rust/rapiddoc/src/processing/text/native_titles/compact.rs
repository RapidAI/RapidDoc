//! Original small emphasized headings after reference-region grouping.
use super::*;
pub(super) fn classify(p: &mut Value, page: i64, w: &mut Work) -> Result<(), Error> {
    if page == 0 {
        return Ok(());
    }
    let size = size(p.get("page_size").unwrap_or(&p["source"]["page_size"]))?;
    let lines = array(&p["remaining_lines"], 4096)?.to_vec();
    let boxes = lines
        .iter()
        .map(|l| bbox(&l["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut s = State { lines, boxes };
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
    let mut processed = BTreeSet::new();
    for lane in &lanes {
        let b = bbox(&json!([num(&lane["left"])?, 0., num(&lane["right"])?, 1.]))?;
        let (left, right) = super::super::page_containers::layout_corridor(&b, &lanes, size[0])
            .ok_or_else(invalid)?;
        let mut rows = s
            .lines
            .iter()
            .enumerate()
            .filter(|(i, l)| {
                integer(l, "angle", 0).ok() == Some(0)
                    && left <= cx(&s.boxes[*i])
                    && cx(&s.boxes[*i]) < right
            })
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        rows.sort_by(|&i, &j| {
            s.boxes[i].n[1]
                .total_cmp(&s.boxes[j].n[1])
                .then(s.boxes[i].n[0].total_cmp(&s.boxes[j].n[0]))
        });
        for (k, &first) in rows.iter().enumerate() {
            let l = &s.lines[first];
            if processed.contains(&integer(l, "source_index", 0)?)
                || !s.unresolved(first)
                || l["title_suppressed"] == true
                || !l["paragraph_group"].is_null()
                || weight(l)?.is_none()
                || coverage(l)? < 0.9
                || l["caption_start"] == true
                || l["paragraph_terminal"] == true
            {
                continue;
            }
            let scale = s.h(first)?;
            let mut j = k + 1;
            while j < rows.len() && j - k < 3 {
                w.charge(1)?;
                let i = rows[j];
                let l = &s.lines[i];
                let gap = s.boxes[i].n[1] - s.boxes[rows[j - 1]].n[3];
                if !s.unresolved(i)
                    || l["title_suppressed"] == true
                    || !l["paragraph_group"].is_null()
                    || !eq(&l["font_signature"], &s.lines[first]["font_signature"])
                    || l["paragraph_terminal"] == true
                    || (s.boxes[i].n[0] - s.boxes[first].n[0]).abs() > 0.5 * scale
                    || gap < -0.1 * scale
                    || gap > 0.8 * scale
                {
                    break;
                }
                j += 1
            }
            let group = &rows[k..j];
            if j >= rows.len() {
                continue;
            }
            let body = rows[j];
            let em = s.h(body)?;
            let Some(bounds) = super::recurrent::stable_bounds(&s, &rows, j, em)? else {
                continue;
            };
            if weight(&s.lines[body])?.is_none() || coverage(&s.lines[body])? < 0.9 {
                continue;
            }
            let band = s.bounds(group)?;
            let preceding = rows[..k]
                .iter()
                .copied()
                .filter(|&i| {
                    !matches!(
                        s.lines[i]["semantic_type"].as_str(),
                        Some("header" | "footer" | "page_number")
                    )
                })
                .last();
            let ga = preceding.map(|i| band.n[1] - s.boxes[i].n[3]);
            let gb = s.boxes[body].n[1] - band.n[3];
            let mut top = None;
            for (i, l) in s.lines.iter().enumerate() {
                w.charge(1)?;
                if integer(l, "angle", 0)? == 0
                    && s.unresolved(i)
                    && l["paragraph_group"].is_null()
                    && eq(&l["font_signature"], &s.lines[body]["font_signature"])
                    && (0.9..=1.1).contains(&(s.h(i)? / em))
                    && s.boxes[i].n[2] - s.boxes[i].n[0] >= 0.6 * (bounds.n[2] - bounds.n[0])
                {
                    top = Some(top.map_or(s.boxes[i].n[1], |old: f64| old.min(s.boxes[i].n[1])))
                }
            }
            let body_top =
                top.is_some_and(|y| band.n[1] - y >= -3.5 * em && band.n[1] - y <= 0.5 * em);
            let fw = weight(&s.lines[first])?.unwrap();
            let bw = weight(&s.lines[body])?.unwrap();
            if !(0.65..=0.85).contains(&(scale / em))
                || fw < (bw + 100.).max(1.15 * bw)
                || band.n[2] > bounds.n[2] + 0.5 * em
                || band.n[0] - bounds.n[0] < -0.5 * em
                || band.n[0] - bounds.n[0] > 1.5 * em
                || ga.map_or(!body_top, |g| g < 0.65 * em || g > 2.5 * em)
                || gb < 0.35 * em
                || gb > 2.5 * em
                || !(0.07 * size[1] < band.n[1]
                    && band.n[1] < band.n[3]
                    && band.n[3] < 0.93 * size[1])
                || inside(&band, &containers)
                || near(&band, &containers, em)
            {
                continue;
            }
            let source = integer(&s.lines[first], "source_index", 0)?;
            for &i in group {
                s.mark(i, "paragraph_title");
                s.lines[i]["structural_title"] = json!(true);
                s.lines[i]["title_band_id"] = json!(source);
                processed.insert(integer(&s.lines[i], "source_index", 0)?);
            }
        }
    }
    p["remaining_lines"] = json!(s.lines);
    Ok(())
}
