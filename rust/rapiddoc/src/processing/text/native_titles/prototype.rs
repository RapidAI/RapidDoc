//! Original prototype matching, alignment and size-conflict rules.
use super::*;
fn alignment(b: &Box4, lane: &Lane, h: f64) -> Option<(&'static str, f64)> {
    let width = (lane.right - lane.left).max(0.1);
    let center = (cx(b) - (lane.left + lane.right) / 2.) / width;
    let left = (b.n[0] - lane.left) / width;
    if center.abs() <= 0.12 && b.n[2] - b.n[0] <= 0.85 * width {
        Some(("center", center))
    } else if (b.n[0] - lane.left).abs() <= 3. * h {
        Some(("left", left))
    } else {
        None
    }
}
fn families(a: &str, b: &str) -> bool {
    a == b
        || [",bold", "bold", ",regular", "regular", ",medium", "medium"]
            .iter()
            .any(|suffix| {
                a.strip_suffix(suffix).unwrap_or(a) == b || b.strip_suffix(suffix).unwrap_or(b) == a
            })
}
pub(super) fn matching(
    s: &State,
    i: usize,
    lane: &Lane,
    body: &Value,
    titles: &Value,
) -> Result<(bool, bool), Error> {
    let l = &s.lines[i];
    if body.is_null() || titles.is_null() || l["font_signature"].is_null() || coverage(l)? < 0.75 {
        return Ok((false, false));
    }
    let Some(f) = family(&l["font_signature"])? else {
        return Ok((false, false));
    };
    let h = num(&body["body_height"])?;
    let ratio = s.h(i)? / h.max(0.1);
    let a = alignment(&s.boxes[i], lane, h);
    let flags = &l["font_signature"][1];
    let mut matched = false;
    let mut conflicts = false;
    for p in optional(titles, "prototypes", 4096)? {
        let pf = p["font_family"].as_str().ok_or_else(invalid)?;
        let ph = num(&p["height_ratio"])?;
        if !eq(&p["font_flags"], flags) {
            continue;
        }
        if let (Some(pw), Some(lw)) = (
            if p["weight"].is_null() {
                None
            } else {
                Some(num(&p["weight"])?)
            },
            weight(l)?,
        ) {
            if (pw - lw).abs() >= 100. && pw.max(lw) >= 1.15 * pw.min(lw) {
                continue;
            }
        }
        let Some((mode, anchor)) = a else { continue };
        if p["alignment"] != mode || (anchor - num(&p["anchor_offset"])?).abs() > 0.15 {
            continue;
        }
        if pf == f && (ratio - ph).abs() <= 0.1 * ph {
            matched = true
        }
        if families(pf, &f) && (ratio - ph).abs() > 0.15 * ph {
            conflicts = true
        }
    }
    Ok((matched, conflicts))
}
