//! Original page-title expansion and demotion passes in their original order.
use super::*;
fn same_family(a: &Value, b: &Value) -> Result<bool, Error> {
    Ok(a["font_signature"].is_null()
        || b["font_signature"].is_null()
        || share_family(&a["font_signature"], &b["font_signature"])?)
}
fn terminal(l: &Value) -> Result<bool, Error> {
    let t = l["text"]
        .as_str()
        .ok_or_else(invalid)?
        .trim_end_matches(python_space)
        .trim_end_matches(|c| "])}）】》”’'\"".contains(c));
    Ok(t.ends_with(['.', '!', '?', '。', '！', '？']))
}
pub(super) fn finish(
    s: &mut State,
    ids: &[usize],
    lanes: &mut [Lane],
    containers: &[Box4],
    caption_containers: &[Box4],
    body: &Value,
    page: i64,
    angle: i64,
    w: &mut Work,
) -> Result<(), Error> {
    expand(s, ids, w)?;
    hanging(s, lanes, body, page, w)?;
    continuations(s, ids, lanes, w)?;
    captions(s, ids, caption_containers, w)?;
    let peers = (0..s.lines.len())
        .filter(|&i| integer(&s.lines[i], "angle", 0).ok() == Some(angle))
        .filter(|&i| {
            (s.unresolved(i) || s.lines[i]["semantic_type"] == "paragraph_title")
                && s.lines[i]["title_suppressed"] != true
        })
        .collect::<Vec<_>>();
    sentence_tails(s, &peers, w)?;
    if body["has_style_scale_repairs"] == true {
        for &i in ids {
            if s.lines[i]["semantic_type"] == "paragraph_title"
                && s.lines[i]["structural_title"] != true
            {
                s.lines[i]["semantic_type"] = Value::Null
            }
        }
    }
    let peers = (0..s.lines.len())
        .filter(|&i| integer(&s.lines[i], "angle", 0).ok() == Some(angle))
        .filter(|&i| {
            (s.unresolved(i) || s.lines[i]["semantic_type"] == "paragraph_title")
                && s.lines[i]["title_suppressed"] != true
        })
        .collect::<Vec<_>>();
    labels(s, &peers, containers, w)
}
fn expand(s: &mut State, ids: &[usize], w: &mut Work) -> Result<(), Error> {
    loop {
        let titles = ids
            .iter()
            .copied()
            .filter(|&i| {
                s.lines[i]["semantic_type"] == "paragraph_title"
                    && s.lines[i]["explicit_section_title"] != true
            })
            .collect::<Vec<_>>();
        let mut chosen = None;
        for i in titles {
            let h = s.h(i)?;
            let a = &s.boxes[i];
            for &j in ids {
                w.charge(1)?;
                let l = &s.lines[j];
                if !s.unresolved(j)
                    || l["title_suppressed"] == true
                    || !l["paragraph_group"].is_null()
                {
                    continue;
                }
                let ch = s.h(j)?;
                if !(0.85..=1.2).contains(&(ch / h.max(0.1))) {
                    continue;
                }
                let b = &s.boxes[j];
                let tw = (a.n[2] - a.n[0]).max(0.1);
                let cw = b.n[2] - b.n[0];
                let pair = h.max(ch).max(0.1);
                let leading = cw / tw > 2.
                    && cw / tw <= 2.5
                    && cy(b) < cy(a)
                    && a.n[1] - b.n[3] >= -0.1 * pair
                    && a.n[1] - b.n[3] <= 0.15 * pair
                    && axis(a, b, 0) >= 0.8
                    && (cx(b) - cx(a)).abs() <= 0.5 * pair;
                if !(0.25..=2.).contains(&(cw / tw)) && !leading {
                    continue;
                }
                if !s.lines[i]["font_signature"].is_null()
                    && !l["font_signature"].is_null()
                    && !eq(&s.lines[i]["font_signature"], &l["font_signature"])
                {
                    continue;
                }
                if (b.n[1] - a.n[3]).max(a.n[1] - b.n[3]).max(0.) > 0.35 * h.max(ch)
                    || s.lines[i]["paragraph_terminal"] == true
                    || l["paragraph_terminal"] == true
                {
                    continue;
                }
                if axis(a, b, 0) < 0.2 && (b.n[0] - a.n[0]).abs() > 2. * h {
                    continue;
                }
                chosen = Some(j);
                break;
            }
            if chosen.is_some() {
                break;
            }
        }
        let Some(j) = chosen else { break };
        s.mark(j, "paragraph_title")
    }
    Ok(())
}
fn hanging(
    s: &mut State,
    lanes: &[Lane],
    body: &Value,
    page: i64,
    w: &mut Work,
) -> Result<(), Error> {
    if page != 0 || body.is_null() {
        return Ok(());
    }
    let body_h = num(&body["body_height"])?.max(0.1);
    for lane in lanes {
        let mut ids = lane.ids.clone();
        s.order(&mut ids);
        let width = (lane.right - lane.left).max(0.1);
        for pair in ids.windows(2) {
            w.charge(1)?;
            let (i, j) = (pair[0], pair[1]);
            let (a, b) = (&s.lines[i], &s.lines[j]);
            if a["semantic_type"] != "paragraph_title"
                || b["semantic_type"] != "paragraph_title"
                || a["font_signature"].is_null()
                || b["font_signature"].is_null()
                || coverage(a)? < 0.75
                || coverage(b)? < 0.75
                || !share_family(&a["font_signature"], &b["font_signature"])?
            {
                continue;
            }
            let (ah, bh) = (s.h(i)?, s.h(j)?);
            let ph = ah.max(bh);
            let (a, b) = (&s.boxes[i], &s.boxes[j]);
            let gap = gap(s, i, j)?;
            if ph <= 1.1 * body_h
                && ah.min(bh) >= 0.85 * ph
                && a.n[2] - a.n[0] >= 0.9 * width
                && a.n[0] - lane.left >= 0.75 * ph
                && (b.n[0] - lane.left).abs() <= 0.5 * ph
                && gap >= -0.25 * ph
                && gap <= 0.5 * ph
            {
                for k in [i, j] {
                    s.lines[k]["semantic_type"] = Value::Null;
                    s.lines[k]["title_suppressed"] = json!(true)
                }
            }
        }
    }
    Ok(())
}
fn continuations(
    s: &mut State,
    ids: &[usize],
    lanes: &mut [Lane],
    w: &mut Work,
) -> Result<(), Error> {
    for &i in ids {
        if s.lines[i]["semantic_type"] != "paragraph_title" {
            continue;
        }
        let h = s.h(i)?;
        let b = &s.boxes[i];
        let mut global = None;
        for &j in ids {
            w.charge(1)?;
            let c = &s.boxes[j];
            if i != j
                && s.unresolved(j)
                && eq(&s.lines[j]["angle"], &s.lines[i]["angle"])
                && cy(c) < cy(b)
                && (c.n[0] - b.n[0]).abs() <= h
                && global.is_none_or(|old| cy(c) > cy(&s.boxes[old]))
            {
                global = Some(j)
            }
        }
        if let Some(j) = global {
            let ph = s.h(j)?;
            let a = &s.boxes[j];
            if same_family(&s.lines[j], &s.lines[i])?
                && (0.9..=1.1).contains(&(h / ph.max(0.1)))
                && b.n[1] - a.n[3] <= 0.2 * h.max(ph)
                && (b.n[0] - a.n[0]).abs() <= 0.75 * h.max(ph)
                && a.n[2] - a.n[0] >= 2. * (b.n[2] - b.n[0])
            {
                s.lines[i]["semantic_type"] = Value::Null;
                continue;
            }
        }
        let mut chosen = None;
        for (k, lane) in lanes.iter().enumerate() {
            if lane.span || lane.ids.len() < 4 || lane.left > cx(b) || cx(b) > lane.right {
                continue;
            }
            let d = (cx(b) - 0.5 * (lane.left + lane.right)).abs();
            if chosen.is_none_or(|(_, old)| d < old) {
                chosen = Some((k, d))
            }
        }
        let Some((k, _)) = chosen else { continue };
        let profile = s.profile(&mut lanes[k], w)?;
        let lane = &lanes[k];
        let width = (lane.right - lane.left).max(0.1);
        if b.n[2] - b.n[0] > 0.55 * width {
            continue;
        }
        let mut previous: Option<usize> = None;
        for &j in ids {
            w.charge(1)?;
            let c = &s.boxes[j];
            if j != i
                && s.unresolved(j)
                && cy(c) < cy(b)
                && lane.left <= cx(c)
                && cx(c) <= lane.right
                && previous.is_none_or(|old| cy(c) > cy(&s.boxes[old]))
            {
                previous = Some(j)
            }
        }
        let Some(j) = previous else { continue };
        let a = &s.boxes[j];
        let ph = s.h(j)?;
        if !(0.9..=1.1).contains(&(h / ph.max(0.1)))
            || !same_family(&s.lines[j], &s.lines[i])?
            || a.n[2] - a.n[0] < 0.5 * width
        {
            continue;
        }
        let mut max = 0.15 * h.max(ph);
        if terminal(&s.lines[i])? {
            max = max.max(profile.gap + 0.35 * profile.h)
        }
        if b.n[1] - a.n[3] > max || (b.n[0] - lane.left).abs() > 1f64.max(0.75 * h) {
            continue;
        }
        s.lines[i]["semantic_type"] = Value::Null;
    }
    Ok(())
}
fn captions(s: &mut State, ids: &[usize], containers: &[Box4], w: &mut Work) -> Result<(), Error> {
    for &i in ids {
        if s.lines[i]["semantic_type"] != "paragraph_title" {
            continue;
        }
        let h = s.h(i)?;
        let b = &s.boxes[i];
        for c in containers {
            w.charge(1)?;
            let gap = b.n[1] - c.n[3];
            if gap >= -0.25 * h
                && gap <= 1.5 * h
                && axis(b, c, 0) >= 0.35
                && (cx(b) - cx(c)).abs() <= 0.35 * (c.n[2] - c.n[0]).max(b.n[2] - b.n[0])
            {
                s.lines[i]["semantic_type"] = Value::Null;
                break;
            }
        }
    }
    Ok(())
}
fn sentence_tails(s: &mut State, ids: &[usize], w: &mut Work) -> Result<(), Error> {
    for &i in ids {
        if s.lines[i]["semantic_type"] != "paragraph_title" || !terminal(&s.lines[i])? {
            continue;
        }
        let h = s.h(i)?;
        let b = &s.boxes[i];
        let mut previous: Option<usize> = None;
        for &j in ids {
            w.charge(1)?;
            let a = &s.boxes[j];
            if j != i
                && s.unresolved(j)
                && a.n[3] <= b.n[1]
                && (a.n[0] - b.n[0]).abs() <= 0.75 * h
                && previous.is_none_or(|old| a.n[3] > s.boxes[old].n[3])
            {
                previous = Some(j)
            }
        }
        let Some(j) = previous else { continue };
        let a = &s.boxes[j];
        let ph = s.h(j)?;
        let bh = (b.n[3] - b.n[1]).max(0.1);
        let abh = (a.n[3] - a.n[1]).max(0.1);
        let compatible = same_family(&s.lines[j], &s.lines[i])?
            && (0.75..=1.25).contains(&(h / ph.max(0.1)))
            || (0.8..=1.2).contains(&(bh / abh));
        if compatible
            && b.n[1] - a.n[3] <= 0.9 * h.max(ph).max(bh).max(abh)
            && a.n[2] - a.n[0] >= 1.5 * (b.n[2] - b.n[0])
            && !terminal(&s.lines[j])?
        {
            s.lines[i]["semantic_type"] = Value::Null
        }
    }
    Ok(())
}
fn labels(s: &mut State, ids: &[usize], containers: &[Box4], w: &mut Work) -> Result<(), Error> {
    use rapiddoc_native::python_decimal;
    let text_count = crate::cached_regex!(r"[A-Za-z]{2,}|[\x{3400}-\x{9fff}]");
    let mut candidates = Vec::new();
    for &i in ids {
        let l = &s.lines[i];
        let b = &s.boxes[i];
        let h = s.h(i)?;
        let t = l["text"].as_str().ok_or_else(invalid)?;
        let words = t.split(python_space).filter(|s| !s.is_empty()).count();
        if l["paragraph_terminal"] == true
            || l["caption_start"] == true
            || l["font_signature"].is_null()
            || !(1..=6).contains(&words)
            || t.chars().any(|c| python_decimal(c).is_some())
            || t.trim_matches(python_space)
                .ends_with(['.', '!', '?', ':', '：', ';', ','])
            || b.n[2] - b.n[0] > 9. * h
            || inside(b, containers)
        {
            continue;
        }
        let mut followers = 0;
        for &j in ids {
            w.charge(1)?;
            let a = &s.boxes[j];
            let g = a.n[1] - b.n[3];
            if i != j
                && s.unresolved(j)
                && (a.n[0] - b.n[0]).abs() <= 0.25 * h
                && g >= 0.5 * h
                && g <= 2. * h
                && (1.2 * h..=1.6 * h).contains(&s.h(j)?)
                && a.n[2] - a.n[0] >= 1.3 * (b.n[2] - b.n[0])
                && text_count
                    .find_iter(s.lines[j]["text"].as_str().ok_or_else(invalid)?)
                    .count()
                    >= 2
            {
                followers += 1
            }
        }
        if followers == 1 {
            candidates.push(i)
        }
    }
    for &i in &candidates {
        let h = s.h(i)?;
        let mut band = Vec::new();
        for &j in &candidates {
            w.charge(1)?;
            if eq(&s.lines[j]["font_signature"], &s.lines[i]["font_signature"])
                && (0.9 * h..=1.1 * h).contains(&s.h(j)?)
                && (cy(&s.boxes[j]) - cy(&s.boxes[i])).abs() <= 0.15 * h
            {
                band.push(j)
            }
        }
        band.sort_by(|&a, &b| s.boxes[a].n[0].total_cmp(&s.boxes[b].n[0]));
        if band.len() < 3
            || !band
                .iter()
                .any(|&j| s.lines[j]["semantic_type"] == "paragraph_title")
            || band
                .windows(2)
                .any(|p| s.boxes[p[1]].n[0] - s.boxes[p[0]].n[2] < 4. * h)
        {
            continue;
        }
        for j in band {
            s.mark(j, "paragraph_title");
            s.lines[j]["structural_title"] = json!(true);
            s.lines[j]["explicit_section_title"] = json!(true);
            s.lines[j]["title_band_id"] = s.lines[j]["source_index"].clone()
        }
    }
    Ok(())
}
