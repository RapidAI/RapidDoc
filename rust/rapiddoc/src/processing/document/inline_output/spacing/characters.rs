//! Original generated Chinese and expanded ligature exclusions, retaining JSON font equality.
use super::*;
use crate::legacy_model::table_native_multiline::spatial::raw_string;
fn number(v: &Value) -> Option<f64> {
    crate::legacy_model::table_native_text::number(v)
        .ok()
        .flatten()
        .filter(|n| n.is_finite())
}
fn vals<const N: usize>(v: &Value) -> Option<[f64; N]> {
    let a = v.as_array().filter(|a| a.len() == N)?;
    let mut out = [0.; N];
    for (i, v) in a.iter().enumerate() {
        // Python geometry requires real finite numbers, not numeric strings.
        if !v.is_number() && !v.is_boolean() {
            return None;
        }
        out[i] = number(v)?;
    }
    Some(out)
}
fn integer(v: &Value) -> Option<i128> {
    v.as_i64()
        .map(i128::from)
        .or_else(|| v.as_u64().map(i128::from))
        .or_else(|| v.as_bool().map(i128::from))
}
pub(super) fn consecutive(a: Option<&Value>, b: &Value, default: Option<i128>) -> bool {
    a.and_then(integer)
        .or(default)
        .zip(integer(b))
        .is_some_and(|(a, b)| a + 1 == b)
}
fn angle(v: &Value) -> Option<i64> {
    let degrees = number(&v["writing_angle"])?.to_degrees().rem_euclid(360.);
    Some(((degrees / 90.).round_ties_even() * 90.).rem_euclid(360.) as i64)
}
fn aligned(v: &Value, a: i64) -> bool {
    number(&v["writing_angle"])
        .is_some_and(|n| ((n.to_degrees() - a as f64 + 180.).rem_euclid(360.) - 180.).abs() <= 0.1)
}
fn local(b: [f64; 4], a: i64) -> [f64; 4] {
    match a {
        90 => [b[1], -b[2], b[3], -b[0]],
        180 => [-b[2], -b[3], -b[0], -b[1]],
        270 => [-b[3], b[0], -b[1], b[2]],
        _ => b,
    }
}
fn han(t: &str) -> bool {
    let mut cs = t.chars();
    let Some(c) = cs.next() else { return false };
    cs.next().is_none()
        && matches!(c as u32,0x3400..=0x4dbf|0x4e00..=0x9fff|0xf900..=0xfaff|0x20000..=0x323af)
}
pub(super) fn hyphen(t: &str) -> bool {
    let cs = t.trim_end_matches(python_space).chars().collect::<Vec<_>>();
    cs.len() >= 2
        && "-\u{ad}\u{2010}\u{2011}\u{2043}".contains(cs[cs.len() - 1])
        && cs[cs.len() - 2].is_ascii_alphabetic()
}
pub(super) fn cjk(a: &Value, s: &Value, b: &Value) -> bool {
    if s["char"] != " " || s["is_generated"] != true {
        return false;
    }
    let at = raw_string(a.get("char"), false);
    let bt = raw_string(b.get("char"), false);
    let punctuation = |t: &str| {
        t.chars().count() == 1
            && "、。，；：？！“”‘’（）【】《》〈〉「」『』〔〕［］｛｝—…·".contains(t)
    };
    if !((han(&at) && (han(&bt) || punctuation(&bt))) || (punctuation(&at) && han(&bt)))
        || !consecutive(a.get("char_idx"), &s["char_idx"], None)
        || !consecutive(s.get("char_idx"), &b["char_idx"], None)
    {
        return false;
    }
    let Some(angle) = angle(a) else { return false };
    if !aligned(a, angle) || !aligned(b, angle) {
        return false;
    }
    let bounds = |v: &Value| {
        vals::<4>(
            if crate::legacy_model::native_text_assembly::python_truth(&v["tight_bbox"]) {
                &v["tight_bbox"]
            } else {
                &v["bbox"]
            },
        )
    };
    let (Some(ab), Some(bb)) = (bounds(a), bounds(b)) else {
        return false;
    };
    let (ab, bb) = (local(ab, angle), local(bb, angle));
    let (ah, bh) = (ab[3] - ab[1], bb[3] - bb[1]);
    if ah.min(bh).min(ab[2] - ab[0]).min(bb[2] - bb[0]) <= 0.
        || ab[3].min(bb[3]) - ab[1].max(bb[1]) < 0.5 * ah.min(bh)
        || bb[0] - ab[2] > 0.5 * ah.max(bh)
    {
        return false;
    }
    let tolerance = 0.15 * ah.max(bh);
    if !a["origin"].is_null() && !b["origin"].is_null() {
        let (Some(ao), Some(bo)) = (vals::<2>(&a["origin"]), vals::<2>(&b["origin"])) else {
            return false;
        };
        let axis = usize::from(angle != 90 && angle != 270);
        (ao[axis] - bo[axis]).abs() <= tolerance
    } else {
        (ab[3] - bb[3]).abs() <= tolerance
    }
}
pub(super) fn ligature(cs: &[Value], i: usize) -> bool {
    if i == 0 || i + 1 >= cs.len() || cs[i]["char"] != " " {
        return false;
    }
    let t = raw_string(cs[i + 1].get("char"), false);
    if t.len() != 1 || !t.as_bytes()[0].is_ascii_alphabetic() {
        return false;
    }
    for count in 1..=i.min(3) {
        let glyph = &cs[i - count..i];
        let text = glyph
            .iter()
            .map(|v| raw_string(v.get("char"), false))
            .collect::<String>();
        if ![
            "fi", "fl", "ff", "ffi", "ffl", "ft", "st", "ﬁ", "ﬂ", "ﬀ", "ﬃ", "ﬄ", "ﬅ", "ﬆ",
        ]
        .contains(&text.as_str())
        {
            continue;
        }
        let first = &glyph[0];
        let records = &cs[i - count..=i + 1];
        if records
            .windows(2)
            .any(|p| !consecutive(p[0].get("char_idx"), &p[1]["char_idx"], None))
            || first["text_object_id"].is_null()
            || records.iter().any(|c| {
                !crate::legacy_page::eq(&c["text_object_id"], &first["text_object_id"])
                    || !crate::legacy_page::eq(&c["font"], &first["font"])
            })
        {
            continue;
        }
        let Some(a) = angle(first) else { continue };
        if records.iter().any(|v| !aligned(v, a)) {
            continue;
        }
        let Some(boxes) = records
            .iter()
            .map(|v| vals::<4>(&v["tight_bbox"]).map(|b| local(b, a)))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let Some(points) = records
            .iter()
            .map(|v| vals::<2>(&v["origin"]).map(|p| local([p[0], p[1], p[0], p[1]], a)))
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let b = boxes[0];
        let h = b[3] - b[1];
        if h.min(b[2] - b[0]) <= 0.
            || (1..count).any(|i| {
                (0..4).any(|j| (b[j] - boxes[i][j]).abs() > 0.01 * h)
                    || (0..2).any(|j| (points[0][j] - points[i][j]).abs() > 0.01 * h)
            })
        {
            continue;
        }
        let rb = boxes[count + 1];
        if b[0] <= points[count][0]
            && points[count][0] < b[2]
            && points[count..]
                .iter()
                .all(|p| (p[1] - points[0][1]).abs() <= 0.15 * h)
            && (-0.15 * h..=0.15 * h).contains(&(rb[0] - b[2]))
            && (points[count + 1][0] - b[2]).abs() <= 0.15 * h
            && b[3].min(rb[3]) - b[1].max(rb[1]) >= 0.5 * h.min(rb[3] - rb[1])
            && rb[3] > rb[1]
        {
            return true;
        }
    }
    false
}
