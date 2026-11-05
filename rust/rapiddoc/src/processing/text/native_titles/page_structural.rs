//! Original final-page explicit/reset/body-height/noninitial-title classifiers.
use super::*;
use unicode_normalization::UnicodeNormalization;
fn text(l: &Value) -> Result<&str, Error> {
    l["text"].as_str().ok_or_else(invalid)
}
fn normalized(t: &str) -> String {
    t.nfkc()
        .collect::<String>()
        .split(python_space)
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}
fn plausible(n: &str, label: &str) -> bool {
    let digit = crate::cached_regex!(r"\d+");
    let parts = digit
        .find_iter(n)
        .map(|m| {
            m.as_str().chars().try_fold(0u64, |n, c| {
                n.checked_mul(10)?
                    .checked_add(rapiddoc_native::python_decimal(c)? as u64)
            })
        })
        .collect::<Option<Vec<_>>>();
    let Some(parts) = parts else { return false };
    if parts.is_empty()
        || parts.iter().any(|&p| p > 99)
        || parts.len() > 1 && parts[0] == 0
        || parts.len() == 1 && parts[0] > 12
    {
        return false;
    }
    let label = label.trim_start_matches(python_space);
    if let Some(c) = label.chars().next() {
        if !rapiddoc_native::text_boundary::python_alpha(c)
            || c.is_ascii_alphabetic() && !c.is_ascii_uppercase()
        {
            return false;
        }
    }
    crate::cached_regex!(r"\d+(?:\.\d+)?")
        .find_iter(label)
        .count()
        < 2
        && !label.chars().any(|c| ",，;；:：。!?！？".contains(c))
}
fn geometry(
    lines: &[Value],
    p: [f64; 2],
    a: i64,
    _filter: impl Fn(&Value) -> bool,
) -> Result<State, Error> {
    Ok(State {
        lines: lines.to_vec(),
        boxes: lines
            .iter()
            .map(|l| bbox(&l["bbox"]).map(|b| local(&b, p, a)))
            .collect::<Result<Vec<_>, _>>()?,
    })
}
fn ids(s: &State, a: i64, filter: impl Fn(&Value) -> bool) -> Result<Vec<usize>, Error> {
    let mut ids = Vec::new();
    for (i, l) in s.lines.iter().enumerate() {
        if integer(l, "angle", 0)? == a && filter(l) {
            ids.push(i)
        }
    }
    s.order(&mut ids);
    Ok(ids)
}
fn available(l: &Value) -> bool {
    l["semantic_type"].is_null() && l["title_suppressed"] != true
}
fn local_size(p: [f64; 2], a: i64) -> [f64; 2] {
    if a == 90 || a == 270 {
        [p[1], p[0]]
    } else {
        p
    }
}
fn followers(
    s: &State,
    ids: &[usize],
    b: &Box4,
    h: f64,
    width: f64,
    count: usize,
    w: &mut Work,
) -> Result<bool, Error> {
    let mut n = 0;
    for &i in ids {
        w.charge(1)?;
        let c = &s.boxes[i];
        if c.n[1] <= b.n[1] + 0.4 * h {
            continue;
        }
        if c.n[1] - b.n[3] > 8. * h {
            break;
        }
        if s.unresolved(i)
            && (axis(b, c, 0) >= 0.15 || (c.n[0] - b.n[0]).abs() <= 2.5 * h)
            && c.n[2] - c.n[0] >= 0.25 * width
            && (0.7..=1.4).contains(&(s.h(i)? / h))
        {
            n += 1;
            if n >= count {
                return Ok(true);
            }
        }
    }
    Ok(false)
}
fn gap_above(s: &State, ids: &[usize], members: &[usize], b: &Box4, h: f64) -> f64 {
    let preceding = ids
        .iter()
        .copied()
        .filter(|i| !members.contains(i))
        .map(|i| &s.boxes[i])
        .filter(|c| {
            c.n[3] <= b.n[1] && (axis(c, b, 0) >= 0.15 || (c.n[0] - b.n[0]).abs() <= 2.5 * h)
        })
        .map(|b| b.n[3])
        .reduce(f64::max);
    preceding.map_or(h, |bottom| b.n[1] - bottom)
}
fn blocked(b: &Box4, cs: &[Box4]) -> bool {
    cs.iter()
        .any(|c| axis(b, c, 0) >= 0.8 && axis(b, c, 1) >= 0.8)
}
fn mark(s: &mut State, ids: &[usize], band: Option<i64>) {
    for &i in ids {
        s.mark(i, "paragraph_title");
        s.lines[i]["structural_title"] = json!(true);
        s.lines[i]["explicit_section_title"] = json!(true);
        if let Some(b) = band {
            s.lines[i]["title_band_id"] = json!(b)
        }
    }
}
fn lanes(s: &State, ids: &[usize], width: f64, w: &mut Work) -> Result<Vec<Lane>, Error> {
    if ids.is_empty() {
        return Ok(vec![]);
    }
    let h = median(ids.iter().map(|&i| s.h(i)).collect::<Result<Vec<_>, _>>()?);
    let raw = super::super::table_text_styles::lanes::infer_formula_layout(
        &json!({"line_geometry":ids.iter().map(|&i|json!([lane_line(&s.lines[i]),s.boxes[i].raw])).collect::<Vec<_>>(),"local_page_width":width,"median_height":h}),
        w,
    )?;
    array(&raw, 4096)?
        .iter()
        .map(|l| {
            Ok(Lane {
                left: num(&l["left"])?,
                right: num(&l["right"])?,
                span: l["is_span"] == true,
                ids: array(&l["line_indices"], 4096)?
                    .iter()
                    .map(|i| {
                        ids.get(i.as_u64().ok_or_else(invalid)? as usize)
                            .copied()
                            .ok_or_else(invalid)
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            })
        })
        .collect()
}
fn explicit(
    s: &mut State,
    indices: &[usize],
    lp: [f64; 2],
    cs: &[Box4],
    h: f64,
    w: &mut Work,
) -> Result<(), Error> {
    let numbered = crate::cached_regex!(
        r"^(?P<number>(?:[A-Z]\s*\.\s*)?\d+(?:\s*\.\s*\d+)*)\s+(?P<label>\S.*)$"
    );
    let only = crate::cached_regex!(r"^\d+(?:\s*\.\s*\d+)*\.?$");
    let terminal = crate::cached_regex!(r"[.!?。！？:：;；,，]$");
    let cjk = crate::cached_regex!(r"\A[\u{4e00}-\u{9fff}]{2,16}\z");
    let mut groups = Vec::new();
    let mut sources = BTreeSet::new();
    for &i in indices {
        let t = normalized(text(&s.lines[i])?);
        if numbered
            .captures(&t)
            .is_some_and(|m| plausible(&m["number"], &m["label"]))
        {
            groups.push(vec![i]);
            sources.insert(integer(&s.lines[i], "source_index", 0)?);
            continue;
        }
        if !only.is_match(&t) || !plausible(&t, "") {
            continue;
        }
        let b = &s.boxes[i];
        let mut companions = Vec::new();
        for &j in indices {
            w.charge(1)?;
            let c = &s.boxes[j];
            if i != j
                && !sources.contains(&integer(&s.lines[j], "source_index", 0)?)
                && c.n[0] >= b.n[2]
                && c.n[0] - b.n[2] <= 4. * h.max(s.h(i)?)
                && axis(b, c, 1) >= 0.5
                && c.n[2] - c.n[0] <= 0.55 * lp[0]
                && !terminal.is_match(&normalized(text(&s.lines[j])?))
            {
                companions.push(j)
            }
        }
        companions.sort_by(|&a, &b| {
            s.boxes[a].n[0]
                .total_cmp(&s.boxes[b].n[0])
                .then(s.boxes[a].n[1].total_cmp(&s.boxes[b].n[1]))
        });
        if let Some(&j) = companions.first() {
            groups.push(vec![i, j]);
            sources.insert(integer(&s.lines[i], "source_index", 0)?);
            sources.insert(integer(&s.lines[j], "source_index", 0)?);
        }
    }
    for mut group in groups {
        let first = group[0];
        let fb = &s.boxes[first];
        let fonts = optional(&s.lines[first], "chars", 20000)?
            .iter()
            .filter(|c| {
                c["char"].as_str().is_some_and(|s| {
                    s.chars().count() == 1
                        && s.chars().all(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
                })
            })
            .map(|c| c["font"]["name"].clone())
            .collect::<Vec<_>>();
        let mut continuations = Vec::new();
        for &i in indices {
            w.charge(1)?;
            let b = &s.boxes[i];
            if i == first
                || !(0.0..=h).contains(&(b.n[1] - fb.n[3]))
                || !(0.5 * h..=4. * h).contains(&(b.n[0] - fb.n[0]))
                || b.n[2] - b.n[0] > 0.5 * (fb.n[2] - fb.n[0])
                || !cjk.is_match(text(&s.lines[i])?.trim_matches(python_space))
            {
                continue;
            }
            let style = |l: &Value| {
                l["native_cjk_style"]
                    .as_array()
                    .filter(|a| !a.is_empty())
                    .map(|a| a.iter().take(3).cloned().collect::<Vec<_>>())
            };
            let same_style = style(&s.lines[first])
                .zip(style(&s.lines[i]))
                .is_some_and(|(a, b)| a == b);
            if same_style
                || optional(&s.lines[i], "chars", 20000)?
                    .iter()
                    .any(|c| fonts.iter().any(|f| eq(f, &c["font"]["name"])))
            {
                continuations.push(i)
            }
        }
        continuations.sort_by(|&a, &b| s.boxes[a].n[1].total_cmp(&s.boxes[b].n[1]));
        if let Some(&i) = continuations.first() {
            group.push(i)
        }
        let b = s.bounds(&group)?;
        let gap = gap_above(s, indices, &group, &b, h);
        if b.n[2] - b.n[0] > 0.7 * lp[0]
            || cy(&b) < 0.1 * lp[1]
            || cy(&b) > 0.93 * lp[1]
            || blocked(&b, cs)
            || !followers(s, indices, &b, h, lp[0], 1, w)?
            || gap < (if continuations.is_empty() { 0.4 } else { 0.25 }) * h
        {
            continue;
        }
        let band = if continuations.is_empty() {
            None
        } else {
            Some(integer(&s.lines[first], "source_index", 0)?)
        };
        mark(s, &group, band)
    }
    let heading = crate::cached_regex!(
        r"(?i)\A(?:introduction|references?|bibliography|acknowledg(?:e)?ments?|引言|绪论|参考文献|参考资料)\z"
    );
    for &i in indices {
        if !s.unresolved(i) {
            continue;
        }
        let t = normalized(text(&s.lines[i])?);
        let canonical = t
            .trim_matches(|c| "[]［］【】()（）".contains(c))
            .replace(' ', "");
        let b = &s.boxes[i];
        let canonical = canonical
            .chars()
            .map(|c| match c {
                'İ' | 'ı' => 'i',
                'ſ' => 's',
                'K' => 'k',
                _ => c,
            })
            .collect::<String>();
        if !heading.is_match(&canonical)
            || !(2..=24).contains(&t.chars().count())
            || terminal.is_match(&t)
            || t.chars().any(|c| "[]［］".contains(c))
            || b.n[2] - b.n[0] > 0.3 * lp[0]
            || cy(b) < 0.35 * lp[1]
            || !(0.75..=1.4).contains(&(s.h(i)? / h))
            || blocked(b, cs)
            || !followers(s, indices, b, h, lp[0], 2, w)?
        {
            continue;
        }
        if gap_above(s, indices, &[i], b, h) >= 0.5 * h {
            mark(s, &[i], None)
        }
    }
    Ok(())
}
fn reset(
    s: &mut State,
    indices: &[usize],
    width: f64,
    cs: &[Box4],
    w: &mut Work,
) -> Result<(), Error> {
    if indices.len() < 3 {
        return Ok(());
    }
    for mut lane in lanes(s, indices, width, w)? {
        s.order(&mut lane.ids);
        for ids in lane.ids.windows(3) {
            let (a, i, b) = (ids[0], ids[1], ids[2]);
            w.charge(1)?;
            if ids
                .iter()
                .any(|&i| !s.unresolved(i) || s.lines[i]["font_signature"].is_null())
            {
                continue;
            }
            if ids
                .iter()
                .map(|&i| coverage(&s.lines[i]))
                .collect::<Result<Vec<_>, _>>()?
                .iter()
                .any(|&c| c < 0.65)
                || !share_family(&s.lines[a]["font_signature"], &s.lines[b]["font_signature"])?
                || share_family(&s.lines[i]["font_signature"], &s.lines[a]["font_signature"])?
            {
                continue;
            }
            let ah = s.h(a)?;
            let ih = s.h(i)?;
            let bh = s.h(b)?;
            let ph = ah.max(ih).max(bh);
            let (aa, ib, bb) = (&s.boxes[a], &s.boxes[i], &s.boxes[b]);
            let lw = (lane.right - lane.left).max(0.1);
            if aa.n[2] - aa.n[0] <= 0.45 * lw
                && (0.2 * lw..=0.65 * lw).contains(&(ib.n[2] - ib.n[0]))
                && bb.n[2] - bb.n[0] >= 0.75 * lw
                && (ib.n[0] - lane.left).abs() <= 0.75 * ph
                && (0.75 * ph..=3. * ph).contains(&(bb.n[0] - ib.n[0]))
                && (0.85..=1.15).contains(&(ih / ((ah + bh) * 0.5).max(0.1)))
                && (-0.25 * ph..=0.5 * ph).contains(&gap(s, a, i)?)
                && (-0.25 * ph..=0.5 * ph).contains(&gap(s, i, b)?)
                && !inside(ib, cs)
            {
                s.mark(i, "paragraph_title");
                s.lines[i]["structural_title"] = json!(true)
            }
        }
    }
    Ok(())
}
fn body_height(
    s: &mut State,
    indices: &[usize],
    lp: [f64; 2],
    cs: &[Box4],
    h: f64,
    w: &mut Work,
) -> Result<(), Error> {
    if indices.len() < 8 {
        return Ok(());
    }
    let mut lanes = lanes(s, indices, lp[0], w)?;
    let mut by_source = BTreeMap::new();
    let mut gaps = Vec::new();
    for (k, lane) in lanes.iter_mut().enumerate() {
        s.order(&mut lane.ids);
        gaps.push(s.profile(lane, w)?.gap);
        for &i in &lane.ids {
            by_source.insert(integer(&s.lines[i], "source_index", 0)?, k);
        }
    }
    if by_source.is_empty() {
        return Ok(());
    }
    let regular = if gaps.is_empty() {
        0.2 * h
    } else {
        median(gaps)
    };
    let physical = evidence::physical(s, indices, w)?;
    let mut pg = BTreeMap::new();
    for g in physical {
        pg.insert(
            integer(&g, "source_index", 0)?,
            if g["above"].is_null() {
                None
            } else {
                Some(num(&g["above"])?)
            },
        );
    }
    let mut candidates = Vec::new();
    for &i in indices {
        let Some(&k) = by_source.get(&integer(&s.lines[i], "source_index", 0)?) else {
            continue;
        };
        let lane = &lanes[k];
        let b = &s.boxes[i];
        let lw = (lane.right - lane.left).max(0.1);
        if !(0.9..=1.1).contains(&(s.h(i)? / h))
            || b.n[2] - b.n[0] > 0.22 * lw
            || (b.n[0] - lane.left).abs() > 0.75 * h
            || inside(b, cs)
        {
            continue;
        }
        let previous = lane
            .ids
            .iter()
            .copied()
            .filter(|&j| {
                s.boxes[j].n[1] < b.n[1]
                    && eq(&s.lines[j]["font_signature"], &s.lines[i]["font_signature"])
                    && s.boxes[j].n[2] - s.boxes[j].n[0] >= 0.75 * lw
            })
            .max_by(|&a, &b| s.boxes[a].n[1].total_cmp(&s.boxes[b].n[1]));
        if previous.is_some_and(|j| {
            b.n[1] - s.boxes[j].n[3] <= 0.5 * h && s.lines[j]["paragraph_terminal"] != true
        }) {
            continue;
        }
        let mut followers = Vec::new();
        let mut top = b.n[1];
        for &j in indices {
            w.charge(1)?;
            let c = &s.boxes[j];
            if i == j
                || c.n[1] <= b.n[1] + 0.4 * h
                || c.n[0] < b.n[0] - 0.75 * h
                || c.n[0] > b.n[0] + 1.5 * h
            {
                continue;
            }
            let pitch = c.n[1] - top;
            if pitch < 0.5 * h {
                continue;
            }
            if pitch > 1.8 * h
                || !(0.9..=1.1).contains(&(s.h(j)? / h))
                || !by_source.contains_key(&integer(&s.lines[j], "source_index", 0)?)
            {
                break;
            }
            followers.push(j);
            top = c.n[1];
            if followers.len() == 3 {
                break;
            }
        }
        if followers.len() < 3 {
            continue;
        }
        let above = pg
            .get(&integer(&s.lines[i], "source_index", 0)?)
            .copied()
            .flatten();
        if above.is_some_and(|g| g <= 0.25 * h) {
            continue;
        }
        let full = followers.iter().any(|&j| {
            by_source
                .get(&integer(&s.lines[j], "source_index", 0).unwrap())
                .is_some_and(|&k| {
                    s.boxes[j].n[2] - s.boxes[j].n[0]
                        >= 0.75 * (lanes[k].right - lanes[k].left).max(0.1)
                })
        });
        if above.is_none() && b.n[1] <= 0.2 * lp[1]
            || above.is_some_and(|g| g - regular >= 0.75 * h)
            || full
        {
            candidates.push(i)
        }
    }
    for &i in &candidates {
        let mut n = 0;
        for &j in &candidates {
            w.charge(1)?;
            if (s.boxes[j].n[0] - s.boxes[i].n[0]).abs() <= h
                && (0.9..=1.1).contains(&(s.h(j)? / s.h(i)?))
                && fonts_ok(&s.lines[i], &s.lines[j])?
            {
                n += 1
            }
        }
        if n >= 2 {
            s.mark(i, "paragraph_title")
        }
    }
    Ok(())
}
fn promote(
    s: &mut State,
    indices: &[usize],
    lp: [f64; 2],
    cs: &[Box4],
    h: f64,
    w: &mut Work,
) -> Result<Vec<([f64; 4], Vec<usize>)>, Error> {
    let mut candidates = Vec::new();
    for (k, &i) in indices.iter().enumerate() {
        let b = &s.boxes[i];
        let lh = s.h(i)?;
        let wr = (b.n[2] - b.n[0]) / lp[0].max(0.1);
        if lh < 1.25 * h
            || !(0.45..=0.85).contains(&wr)
            || (cx(b) - 0.5 * lp[0]).abs() > 0.08 * lp[0]
            || cy(b) < 0.12 * lp[1]
            || cy(b) > 0.75 * lp[1]
            || inside(b, cs)
        {
            continue;
        }
        let mut members = vec![i];
        for &j in &indices[k + 1..] {
            w.charge(1)?;
            let c = &s.boxes[j];
            let ch = s.h(j)?;
            if ch < 1.2 * h
                || (c.n[1] - s.boxes[*members.last().unwrap()].n[3]).max(0.) > 1.5 * lh.max(ch)
                || (cx(c) - 0.5 * lp[0]).abs() > 0.1 * lp[0]
                || c.n[2] - c.n[0] > 0.85 * lp[0]
                || inside(c, cs)
                || !fonts_ok(&s.lines[*members.last().unwrap()], &s.lines[j])?
            {
                break;
            }
            members.push(j)
        }
        let b = s.bounds(&members)?;
        let score = [
            median(
                members
                    .iter()
                    .map(|&i| s.h(i))
                    .collect::<Result<Vec<_>, _>>()?,
            ) / h,
            members
                .iter()
                .map(|&i| s.boxes[i].n[2] - s.boxes[i].n[0])
                .sum::<f64>()
                / lp[0],
            members.len() as f64,
            -cy(&b) / lp[1],
        ];
        candidates.push((score, members));
    }
    Ok(candidates)
}
pub(super) fn run(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let mut lines = array(&v["lines"], 4096)?.to_vec();
    let p = size(&v["page_size"])?;
    let body = &v["document_body_profile"];
    let kind = v["kind"].as_str().ok_or_else(invalid)?;
    let h = if body.is_null() {
        0.
    } else {
        num(&body["body_height"])?
    };
    if body.is_null()
        || ((kind == "explicit" || kind == "body") && h <= 0.)
        || kind == "promote" && integer(v, "page_index", 0)? == 0
    {
        return Ok(json!({"lines":lines}));
    }
    let containers = optional(v, "container_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let selected = optional(v, "title_candidate_source_indices", 4096)?
        .iter()
        .map(|i| i.as_i64().ok_or_else(invalid))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let predicate = |l: &Value| {
        if kind == "promote" {
            selected.contains(&integer(l, "source_index", 0).unwrap())
                && (l["semantic_type"].is_null() || l["semantic_type"] == "paragraph_title")
                && l["title_suppressed"] != true
        } else {
            available(l)
        }
    };
    let angles = lines
        .iter()
        .filter(|l| {
            predicate(l)
                || kind == "explicit"
                    && l["explicit_section_title"] == true
                    && l["title_suppressed"] != true
        })
        .map(|l| integer(l, "angle", 0))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut promoted = Vec::new();
    for a in angles {
        let mut s = geometry(&lines, p, a, |_| true)?;
        let mut indices = if kind == "reset" {
            lines
                .iter()
                .enumerate()
                .filter(|(_, l)| integer(l, "angle", 0).ok() == Some(a) && available(l))
                .map(|(i, _)| i)
                .collect::<Vec<_>>()
        } else {
            ids(&s, a, predicate)?
        };
        if kind != "reset" {
            s.order(&mut indices)
        }
        let lp = local_size(p, a);
        let cs = containers
            .iter()
            .map(|b| local(b, p, a))
            .collect::<Vec<_>>();
        match kind {
            "explicit" => explicit(&mut s, &indices, lp, &cs, h, w)?,
            "reset" => reset(&mut s, &indices, lp[0], &cs, w)?,
            "body" => body_height(&mut s, &indices, lp, &cs, h, w)?,
            "promote" => promoted.extend(promote(&mut s, &indices, lp, &cs, h.max(0.1), w)?),
            _ => return Err(invalid()),
        }
        lines = s.lines;
    }
    if let Some((_, members)) = promoted.into_iter().max_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then(std::cmp::Ordering::Greater)
    }) {
        for i in members {
            lines[i]["semantic_type"] = json!("doc_title")
        }
    }
    Ok(json!({"lines":lines}))
}
