//! Original formula visual rows, detached sidecars, tags and tight output envelopes.
use super::*;
pub(super) fn sanitize(t: &str, newlines: bool) -> String {
    let t = t.replace("\r\n", "\n").replace('\r', "\n");
    let mut cs = Vec::new();
    for c in t.chars() {
        if matches!(c, '\u{200b}' | '\u{2060}' | '\u{feff}') {
            continue;
        }
        cs.push(match c {
            '\u{85}' | '\u{2028}' | '\u{2029}' => '\n',
            '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}' => ' ',
            _ => c,
        });
    }
    let mut out = String::new();
    for (i, &c) in cs.iter().enumerate() {
        if matches!(c, '\u{2}' | '\u{ad}') && i > 0 && cs[i - 1].is_ascii_alphabetic() {
            let mut j = i + 1;
            while cs.get(j).is_some_and(|c| matches!(c, '\t' | ' ')) {
                j += 1
            }
            if cs.get(j).is_none_or(|c| *c == '\n') {
                out.push('-');
                continue;
            }
        }
        if c == '\u{ad}' {
            continue;
        }
        let c = if c == '\t' { ' ' } else { c };
        if !newlines && c == '\n'
            || matches!(c,'\u{0}'..='\u{8}'|'\u{b}'|'\u{c}'|'\u{e}'..='\u{1f}'|'\u{7f}'..='\u{9f}')
        {
            continue;
        }
        out.push(c)
    }
    out
}
pub(super) fn tight(p: &Page, rows: &[Row]) -> Result<Option<Box4>, Error> {
    let mut boxes = Vec::new();
    let mut changed = false;
    for r in rows {
        let mut candidate = None;
        if let Some(b) = coerce(&p.l(r)["ink_bbox"])? {
            if let Some(b) = clip(
                &bbox(&json!([b.n[0] - 1., b.n[1] - 1., b.n[2] + 1., b.n[3] + 1.]))?,
                p.size,
            )? {
                candidate = Some(b);
                changed = true;
            }
        }
        boxes.push(candidate.unwrap_or(bbox(&p.l(r)["bbox"])?));
    }
    if !changed || boxes.is_empty() {
        Ok(None)
    } else {
        Ok(Some(combined(boxes.iter())?))
    }
}
pub(super) fn join(
    p: &mut Page,
    rows: &[(Row, Option<String>)],
    w: &mut Work,
) -> Result<String, Error> {
    let mut ordered = rows.to_vec();
    ordered.sort_by(|(a, _), (b, _)| {
        a.b.n[0]
            .total_cmp(&b.b.n[0])
            .then(a.b.n[1].total_cmp(&b.b.n[1]))
            .then(p.id(a).unwrap().cmp(&p.id(b).unwrap()))
    });
    let mut values = ordered
        .iter()
        .map(|(r, text)| {
            let mut v = p.l(r).clone();
            if let Some(t) = text {
                v["text"] = json!(t)
            }
            v
        })
        .collect::<Vec<_>>();
    if ordered.is_empty() {
        return Ok(String::new());
    }
    let mut out = values[0]["text"]
        .as_str()
        .ok_or_else(invalid)?
        .trim_matches(python_space)
        .to_owned();
    for k in 1..ordered.len() {
        w.charge(1)?;
        let (a, b) = (&ordered[k - 1].0, &ordered[k].0);
        let gap = b.b.n[0] - a.b.n[2];
        let h = p.h(a)?.max(p.h(b)?);
        let mut widths = Vec::new();
        for i in [k - 1, k] {
            if let Some(v) =
                super::super::page_containers::media::glyph_width(&mut values[i], p.size, w)?
            {
                widths.push(v)
            }
            if ordered[i].1.is_none() {
                p.lines[ordered[i].0.index]["median_glyph_width"] =
                    values[i]["median_glyph_width"].clone();
            }
        }
        let width = if widths.is_empty() {
            (0.5 * h).max(1.)
        } else {
            median(widths)
        };
        if gap > 0.5f64.max(0.2 * h) {
            if width == 0. {
                return Err(invalid());
            }
            let n = (gap / width).round_ties_even().clamp(1., 8.) as usize;
            out.push_str(&" ".repeat(n))
        }
        out.push_str(
            values[k]["text"]
                .as_str()
                .ok_or_else(invalid)?
                .trim_matches(python_space),
        );
    }
    Ok(out.trim_matches(python_space).to_owned())
}
fn sidecar(p: &Page, anchor: &Row, members: &[Row], em: f64) -> Result<bool, Error> {
    let id = p.id(anchor)?;
    let body = members
        .iter()
        .filter(|r| p.id(r).ok() != Some(id))
        .cloned()
        .collect::<Vec<_>>();
    if body.is_empty() {
        return Ok(false);
    }
    let b = p.bounds(&body)?;
    let component = combined([&b, &anchor.b])?;
    let h = em.max(0.1);
    let width = (component.n[2] - component.n[0]).max(0.1);
    Ok(anchor.b.n[0] >= b.n[2]
        && anchor.b.n[2] >= component.n[2] - 0.5f64.max(0.1 * h)
        && (anchor.b.n[2] - anchor.b.n[0]).max(0.) <= 2. * h
        && anchor.b.n[0] - b.n[2] > (2.5 * h).max(0.08 * width))
}
fn tagged(formula: &str, tag: &str) -> Option<String> {
    let f = formula.strip_prefix("\\[").unwrap_or(formula);
    let f = f
        .strip_suffix("\\]")
        .unwrap_or(f)
        .trim_matches(python_space);
    let t = tag
        .trim_matches(python_space)
        .chars()
        .map(|c| {
            if ('\u{ff01}'..='\u{ff5e}').contains(&c) {
                char::from_u32(c as u32 - 0xfee0).unwrap()
            } else {
                c
            }
        })
        .collect::<String>();
    let t = t
        .strip_prefix(['(', '﹙'])
        .unwrap_or(&t)
        .trim_matches(python_space);
    let t = t
        .strip_suffix([')', '﹚'])
        .unwrap_or(t)
        .trim_matches(python_space);
    (!f.is_empty() && !t.is_empty()).then(|| format!("{f}\\tag{{{t}}}"))
}
pub(super) fn block(
    p: &mut Page,
    members: &[Row],
    angle: i64,
    anchor_id: i64,
    w: &mut Work,
) -> Result<Option<Value>, Error> {
    let anchor = members.iter().find(|r| p.id(r).ok() == Some(anchor_id));
    let parts = anchor
        .map(|r| p.t(r).map(split_number))
        .transpose()?
        .flatten();
    let h = if members.is_empty() {
        1.
    } else {
        median(
            members
                .iter()
                .map(|r| p.h(r))
                .collect::<Result<Vec<_>, _>>()?,
        )
    };
    let tolerance = 1.5f64.max(0.35 * h);
    let mut ordered = members.to_vec();
    ordered.sort_by(|a, b| {
        cy(&a.b)
            .total_cmp(&cy(&b.b))
            .then(a.b.n[0].total_cmp(&b.b.n[0]))
            .then(p.id(a).unwrap().cmp(&p.id(b).unwrap()))
    });
    let mut rows = Vec::<Vec<(Row, Option<String>)>>::new();
    for r in ordered {
        let same = rows.last().is_some_and(|row| {
            (cy(&r.b) - median(row.iter().map(|(r, _)| cy(&r.b)).collect())).abs() <= tolerance
        });
        if same {
            rows.last_mut().unwrap().push((r, None))
        } else {
            rows.push(vec![(r, None)])
        }
    }
    let mut trailing = None;
    for k in 0..rows.len().saturating_sub(1) {
        let a = rows[k]
            .iter()
            .find(|(r, _)| p.id(r).ok() == Some(anchor_id))
            .map(|(r, _)| r.clone());
        let Some(a) = a else { continue };
        if let Some((prefix, marker)) = split_number(p.t(&a)?) {
            rows[k].retain(|(r, _)| p.id(r).ok() != Some(anchor_id) || !prefix.is_empty());
            for (r, text) in &mut rows[k] {
                if p.id(r)? == anchor_id {
                    *text = Some(prefix.clone())
                }
            }
            trailing = Some(marker)
        } else if sidecar(p, &a, members, h)? {
            rows[k].retain(|(r, _)| p.id(r).ok() != Some(anchor_id));
            trailing = Some(p.t(&a)?.trim_matches(python_space).to_owned())
        }
        break;
    }
    let mut contents = Vec::new();
    for row in &rows {
        if !row.is_empty() {
            let content = join(p, row, w)?;
            if !content.is_empty() {
                contents.push(content)
            }
        }
    }
    if let Some(t) = trailing {
        contents.push(t)
    }
    let mut content = sanitize(&contents.join("\n"), true);
    if let Some((_, tag)) = parts {
        let stripped = content.trim_end_matches(python_space);
        if let Some(formula) = stripped.strip_suffix(&tag) {
            if let Some(t) = tagged(formula.trim_end_matches(python_space), &tag) {
                content = t
            }
        }
    }
    if content.trim_matches(python_space).is_empty() {
        return Ok(None);
    }
    let original = members
        .iter()
        .map(|r| bbox(&p.l(r)["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let mut out = json!({"type":"equation","bbox":combined(original.iter())?.raw,"angle":angle,"content":content,"_formula_members":members.iter().map(|r|p.id(r)).collect::<Result<Vec<_>,_>>()?});
    if let Some(b) = tight(p, members)? {
        out["_tight_output_bbox"] = json!(b.raw)
    }
    Ok(Some(out))
}
pub(super) fn single(p: &Page, r: &Row, angle: i64) -> Result<Option<Value>, Error> {
    let text = sanitize(p.t(r)?, false)
        .trim_matches(python_space)
        .to_owned();
    if text.is_empty() {
        return Ok(None);
    }
    let mut out = json!({"type":"equation","bbox":p.l(r)["bbox"],"angle":angle,"content":text,"_formula_members":[p.id(r)?]});
    if let Some(b) = tight(p, std::slice::from_ref(r))? {
        out["_tight_output_bbox"] = json!(b.raw)
    }
    Ok(Some(out))
}
