//! Full existing strong-caption/footnote and adjacent italic/source candidate rules.
use super::*;
pub(super) fn footnote(t: &str) -> bool {
    let original = normalized(t);
    let t = fold(&original);
    regex(r"(?is)^\s*(?:source(?:s|\(s\))?|data\s+source|note(?:s|\(s\))?|资料来源|数据来源|来源|注|备注)\s*[:：]\s*\S").is_match(&t)||regex(r"(?is)^[A-Za-z][A-Za-z0-9 -]{0,30}\s*=\s*\S.+?\s+Source(?:s)?\s*:\s*\S").is_match(&t)||regex(r"^\(?[A-Z][A-Za-z’'-]+\s+(?:(?:and|&)\s+[A-Z][A-Za-z’'-]+\s+|et\s+al\.?\s+)?[12]\d{3}\)?\.?$").is_match(&original)||regex(r"(?i)^(?:table|figure)\s+(?:adapted|reprinted|reproduced)\s+from\s+\S").is_match(&t)
}
fn italic(i: usize, bs: &[Value]) -> Result<bool, Error> {
    let b = &bs[i];
    let Some(bb) = coerce(&b["bbox"])? else {
        return Ok(false);
    };
    let Some(fonts) = b["_font_signatures"].as_array() else {
        return Ok(false);
    };
    if angle(b)? != 0
        || fonts.is_empty()
        || b["_local_line_bboxes"]
            .as_array()
            .is_none_or(|r| r.len() != 1)
        || !(4..=24).contains(&words(text(b)))
    {
        return Ok(false);
    }
    for f in fonts {
        let Some(a) = f.as_array().filter(|a| a.len() == 2) else {
            return Ok(false);
        };
        let Some(n) = a[1].as_i64().or_else(|| a[1].as_bool().map(i64::from)) else {
            return Ok(false);
        };
        if n & 64 == 0 {
            return Ok(false);
        }
    }
    let em = height(b, [0., 0.])?;
    let mut images = 0;
    for image in bs {
        if image["type"] != "image" {
            continue;
        }
        let Some(ib) = coerce(&image["bbox"])? else {
            continue;
        };
        if ib.n[2] - ib.n[0] >= 10. * em
            && ib.n[3] - ib.n[1] >= 8. * em
            && (0.0..=0.75 * em).contains(&(bb.n[1] - ib.n[3]))
            && (bb.n[0] - ib.n[0]).abs() <= 0.75 * em
            && bb.n[2] <= ib.n[2] + 0.5 * em
        {
            images += 1
        }
    }
    if images != 1 {
        return Ok(false);
    }
    let mut following = vec![];
    for (j, other) in bs.iter().enumerate() {
        if i == j || other["type"] != "text" {
            continue;
        }
        let Some(ob) = coerce(&other["bbox"])? else {
            continue;
        };
        if ob.n[1] >= bb.n[3] && (ob.n[0] - bb.n[0]).abs() <= em && words(text(other)) >= 15 {
            following.push(ob.n[1])
        }
    }
    Ok(!following.is_empty()
        && following.into_iter().fold(f64::INFINITY, f64::min) - bb.n[3] >= 1.5 * em)
}
pub(super) fn collect(bs: &[Value], w: &mut Work) -> Result<Candidates, Error> {
    let mut out = vec![];
    for (i, b) in bs.iter().enumerate() {
        w.charge(1)?;
        let k = match b["type"].as_str() {
            Some("caption") => Some("caption"),
            Some("footnote") => Some("footnote"),
            Some("text" | "paragraph_title") if b["content"].is_string() => {
                if strong_caption(text(b)) {
                    Some("caption")
                } else if b["type"] == "text" && footnote(text(b)) {
                    Some("footnote")
                } else if b["type"] == "text" && italic(i, bs)? {
                    Some("caption")
                } else {
                    None
                }
            }
            _ => None,
        };
        if let Some(k) = k {
            out.push((i, k))
        }
    }
    for (i, k) in out.clone() {
        let anchor = &bs[i];
        if k != "footnote" || angle(anchor)? != 0 || !footnote(text(anchor)) {
            continue;
        }
        let h = height(anchor, [0., 0.])?;
        let Some(b) = coerce(&anchor["bbox"])? else {
            continue;
        };
        let mut images = vec![];
        for image in bs {
            if image["type"] != "image" {
                continue;
            }
            if let Some(bb) = coerce(&image["bbox"])? {
                if (0.0..=6. * h).contains(&(b.n[1] - bb.n[3]))
                    && (bb.n[0] - 0.25 * (bb.n[2] - bb.n[0])..=bb.n[2]).contains(&b.n[0])
                {
                    images.push(bb)
                }
            }
        }
        let Some(image) = images
            .iter()
            .max_by(|a, b| a.n[3].total_cmp(&b.n[3]).then(std::cmp::Ordering::Greater))
        else {
            continue;
        };
        let mut lower = b.n[1];
        let mut preceding = vec![];
        for (j, other) in bs.iter().enumerate() {
            w.charge(1)?;
            if kind(&out, j).is_some() || other["type"] != "text" || angle(other)? != 0 {
                continue;
            }
            if let Some(bb) = coerce(&other["bbox"])? {
                if image.n[3] - 0.25 * h <= bb.n[1] && bb.n[3] <= lower + 0.25 * h {
                    preceding.push((j, bb))
                }
            }
        }
        preceding.sort_by(|a, b| b.1.n[3].total_cmp(&a.1.n[3]));
        for (j, bb) in preceding {
            let candidate = &bs[j];
            if lower - bb.n[3] > 1.5 * h
                || (bb.n[0] - b.n[0]).abs() > h
                || !regex(r"^[A-Za-z][A-Za-z0-9 -]{0,30}\s*(?:=|:)\s*\S").is_match(text(candidate))
                || !(0.85 * h..=1.15 * h).contains(&height(candidate, [0., 0.])?)
                || bb.n[2] > image.n[2] + 0.25 * (image.n[2] - image.n[0])
                || font_conflict(anchor, candidate)
                || strong_caption(text(candidate))
            {
                break;
            }
            out.push((j, "footnote"));
            lower = bb.n[1];
        }
    }
    Ok(out)
}
