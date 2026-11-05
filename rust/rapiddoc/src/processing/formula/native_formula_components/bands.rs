//! Original right-edge numbered-band recovery, including CPython component order.
use super::evidence::display_prose;
use super::*;
fn body(p: &Page, r: &Row, width: f64) -> Result<bool, Error> {
    let t = p.t(r)?;
    if !display_prose(t) {
        return Ok(false);
    }
    let words = regex_words(t, 3, None);
    if words.len() >= 2 && p.coverage(r)? >= 0.75 && !math(t) {
        return Ok(true);
    }
    let keywords = regex_words(t, 1, None).iter().any(|s| {
        [
            "where", "with", "when", "which", "the", "this", "these", "is", "are", "from", "that",
            "then",
        ]
        .contains(&s.to_ascii_lowercase().as_str())
    });
    let raw = bbox(&p.l(r)["bbox"])?;
    Ok(sentence(t)
        && (raw.n[2] - raw.n[0] >= 0.6 * width
            || words.len() >= 3
            || t.chars()
                .filter(|c| ('\u{3400}'..='\u{9fff}').contains(c))
                .count()
                >= 3
            || keywords))
}
pub(super) fn build(p: &mut Page, rules: &[Value], w: &mut Work) -> Result<Vec<Value>, Error> {
    let mut prose = Vec::new();
    for (i, l) in p.lines.iter().enumerate() {
        if integer(l, "angle", 0)? == 0
            && l["text"].as_str().ok_or_else(invalid)?.chars().count() >= 25
            && sentence(l["text"].as_str().unwrap())
        {
            let b = bbox(&l["bbox"])?;
            if !p
                .tables
                .iter()
                .any(|t| super::super::path_graphics::overlap_first(&b, t) >= 0.8)
            {
                prose.push(i)
            }
        }
    }
    let lanes = layout(p, &prose, w)?;
    let mut blocks = Vec::new();
    let mut claimed = BTreeSet::new();
    let digit = crate::cached_regex!(r"\d");
    let marker_re =
        crate::cached_regex!(r"\A[^A-Za-z0-9][A-Za-z]?[.]?\d+(?:[.\-]\d+)*[^A-Za-z0-9]\z");
    for lane in lanes {
        let left = num(&lane["left"])?;
        let right = num(&lane["right"])?;
        let mut heights = Vec::new();
        for id in array(&lane["line_indices"], 4096)? {
            let i = *prose
                .get(id.as_u64().ok_or_else(invalid)? as usize)
                .ok_or_else(invalid)?;
            heights.push(p.h(&ink(p, i, 0)?)?)
        }
        let em = median(heights);
        let mut local = Vec::new();
        for (i, l) in p.lines.iter().enumerate() {
            let b = bbox(&l["bbox"])?;
            if integer(l, "angle", 0)? == 0
                && l["paragraph_group"].is_null()
                && l["semantic_type"].is_null()
                && left - em <= b.n[0]
                && b.n[2] <= right + em
            {
                local.push(ink(p, i, 0)?)
            }
        }
        let mut markers = Vec::new();
        for r in &local {
            let t = p.t(r)?;
            let s = t.trim_matches(python_space);
            let cs = s.chars().collect::<Vec<_>>();
            let b = bbox(&p.l(r)["bbox"])?;
            if (2..=8).contains(&cs.len())
                && !cs.iter().copied().any(python_space)
                && digit.is_match(t)
                && !rapiddoc_native::text_boundary::python_digit(cs[0])
                && !rapiddoc_native::text_boundary::python_digit(*cs.last().unwrap())
                && (standalone(t).is_some() || marker_re.is_match(s))
                && (b.n[2] - right).abs() < 1.5 * em
                && b.n[2] - b.n[0] <= 4. * em
            {
                markers.push(r.clone())
            }
        }
        for marker in &markers {
            if claimed.contains(&p.id(marker)?) {
                continue;
            }
            let mb = &marker.b;
            let mut nearby = Vec::new();
            for r in &local {
                if r.index != marker.index
                    && !claimed.contains(&p.id(r)?)
                    && !markers.iter().any(|m| eq(p.l(m), p.l(r)))
                    && !body(p, r, right - left)?
                    && r.b.n[3] >= mb.n[1] - 4.5 * em
                    && r.b.n[1] <= mb.n[3] + 1.1 * em
                {
                    nearby.push(r.clone())
                }
            }
            let mut pending = super::super::tiled_tables::pending::Pending::new(nearby.len());
            let mut components = Vec::new();
            while let Some(seed) = pending.pop() {
                let mut component = vec![seed];
                let mut frontier = vec![seed];
                while let Some(current) = frontier.pop() {
                    let b = &nearby[current].b;
                    let mut connected = Vec::new();
                    for i in pending.ordered() {
                        w.charge(1)?;
                        let c = &nearby[i].b;
                        let xgap = (b.n[0] - c.n[2]).max(c.n[0] - b.n[2]).max(0.);
                        let ygap = (b.n[1] - c.n[3]).max(c.n[1] - b.n[3]).max(0.);
                        if ygap <= 0.8 * em && axis(b, c, 0) > 0.1
                            || xgap <= 3. * em && axis(b, c, 1) > 0.15
                        {
                            connected.push(i)
                        }
                    }
                    pending.difference(&connected);
                    frontier.extend(&connected);
                    component.extend(connected)
                }
                components.push(
                    component
                        .into_iter()
                        .map(|i| nearby[i].clone())
                        .collect::<Vec<_>>(),
                )
            }
            let mut candidates = Vec::new();
            for members in components {
                let b = p.bounds(&members)?;
                if (b.n[2] >= mb.n[0] + em && b.n[3] > mb.n[1]) || b.n[2] - b.n[0] < 3. * em {
                    continue;
                }
                let gap = (mb.n[1] - b.n[3]).max(b.n[1] - mb.n[3]).max(0.);
                if gap > 2. * em
                    || b.n[3] - b.n[1] > 6. * em
                    || p.tables.iter().any(|t| intersects(&b, t))
                {
                    continue;
                }
                let mut blocked = false;
                for r in &local {
                    w.charge(1)?;
                    if body(p, r, right - left)? && axis(&b, &r.b, 1) > 0.1 {
                        blocked = true;
                        break;
                    }
                }
                if blocked {
                    continue;
                }
                let mut has_math = false;
                let mut typography = false;
                for r in &members {
                    let t = p.t(r)?;
                    has_math |= math(t);
                    let f = &p.l(r)["font_signature"];
                    typography |= !f.is_null()
                        && (f[1]
                            .as_i64()
                            .or_else(|| f[1].as_bool().map(i64::from))
                            .ok_or_else(invalid)?
                            & 64
                            != 0
                            || p.coverage(r)? < 0.75)
                        && !display_prose(t)
                        && regex_words(t, 1, Some(2)).len() >= 3;
                }
                let mut fraction = false;
                for r in rules {
                    if r["orientation"] == "horizontal" {
                        let rb = bbox(&r["bbox"])?;
                        if b.n[0] <= rb.n[0]
                            && rb.n[0] < rb.n[2]
                            && rb.n[2] <= b.n[2]
                            && b.n[1] < rb.n[1]
                            && rb.n[1] < b.n[3]
                        {
                            fraction = true
                        }
                    }
                }
                if !has_math
                    && !fraction
                    && !typography
                    && !((members.len() >= 3
                        || members
                            .iter()
                            .any(|r| p.l(r)["compact_formula_cluster"] == true))
                        && b.n[3] - b.n[1] >= 1.3 * em)
                {
                    continue;
                }
                candidates.push((gap, members))
            }
            candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
            if candidates.is_empty()
                || candidates.len() > 1 && (candidates[0].0 - candidates[1].0).abs() < 0.25 * em
            {
                continue;
            }
            let mut members = candidates.remove(0).1;
            members.push(marker.clone());
            if let Some(mut block) = materialize::block(p, &members, 0, p.id(marker)?, w)? {
                block["_spatial_display_band"] = json!(true);
                blocks.push(block);
                for r in members {
                    claimed.insert(p.id(&r)?);
                }
            }
        }
    }
    Ok(blocks)
}
