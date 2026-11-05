use super::super::table_native_multiline::spatial::raw_string;
use super::*;
use rapiddoc_native::{
    python_printable,
    text_boundary::{python_alnum, python_digit},
};
fn visible(c: &Value) -> bool {
    !raw_string(c.get("char"), false)
        .trim_matches(python_space)
        .is_empty()
}
fn font_number(c: &Value, k: &str) -> Result<f64, Error> {
    if c["font"][k].is_null() || c["font"][k] == false {
        Ok(0.)
    } else {
        super::super::table_native_text::number(&c["font"][k])?.ok_or_else(invalid)
    }
}
fn footnote_size(c: &Value) -> Result<f64, Error> {
    if c.get("font").is_some_and(|v| !v.is_object()) {
        return Err(invalid());
    }
    match c.get("font").and_then(|f| f.get("size")) {
        None => Ok(0.),
        Some(v) => super::super::table_native_text::number(v)?.ok_or_else(invalid),
    }
}
pub(super) fn footnote(l: &Value) -> Result<bool, Error> {
    let chars = optional(l, "chars", 20000)?
        .iter()
        .filter(|c| visible(c))
        .collect::<Vec<_>>();
    let n = chars
        .iter()
        .take_while(|c| {
            let t = raw_string(c.get("char"), false);
            !t.is_empty() && t.chars().all(python_digit)
        })
        .count();
    if !(1..=3).contains(&n) || chars.len() < n + 3 {
        return Ok(false);
    }
    let marker = &chars[..n];
    let body = &chars[n..chars.len().min(n + 8)];
    let ms = median(
        marker
            .iter()
            .map(|c| footnote_size(c))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let bs = median(
        body.iter()
            .map(|c| footnote_size(c))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let heights = |a: &[&Value]| -> Result<f64, Error> {
        Ok(median(
            a.iter()
                .map(|c| {
                    let b = bbox(&c["bbox"])?;
                    Ok(b.n[3] - b.n[1])
                })
                .collect::<Result<Vec<_>, Error>>()?,
        ))
    };
    let mh = heights(marker)?;
    let bh = heights(body)?;
    let origins = |a: &[&Value]| -> Result<Vec<f64>, Error> {
        a.iter()
            .filter(|c| !c["origin"].is_null())
            .map(|c| num(&c["origin"][1]))
            .collect()
    };
    let mo = origins(marker)?;
    let bo = origins(body)?;
    Ok(integer(l, "angle", 0)? == 0
        && bh > 0.
        && (bs > 0. && ms <= 0.82 * bs || mh <= 0.75 * bh)
        && !mo.is_empty()
        && !bo.is_empty()
        && median(bo) - median(mo) >= 0.15 * bh)
}
pub(super) fn math_words(l: &Value) -> Result<Value, Error> {
    let mut words = optional(l, "native_math_words", 20000)?
        .iter()
        .map(|v| v.as_str().map(str::to_owned).ok_or_else(invalid))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let chars = optional(l, "chars", 20000)?
        .iter()
        .filter(|c| {
            let t = raw_string(c.get("char"), false);
            !t.is_empty() && t.chars().all(python_printable)
        })
        .collect::<Vec<_>>();
    let t = chars
        .iter()
        .map(|c| raw_string(c.get("char"), false))
        .collect::<String>();
    let re = crate::cached_regex!(r"[a-z]{3,4}");
    for m in re.find_iter(&t) {
        let word = |c: char| python_alnum(c) || c == '_';
        if t[..m.start()].chars().next_back().is_some_and(word)
            || t[m.end()..].chars().next().is_some_and(word)
        {
            continue;
        }
        let start = t[..m.start()].chars().count();
        let end = start + m.as_str().chars().count();
        let members = &chars[start.min(chars.len())..end.min(chars.len())];
        if members.is_empty() {
            continue;
        }
        let mut italic = true;
        let mut sizes = Vec::new();
        for c in members {
            italic &= (font_number(c, "flags")? as i64) & 64 != 0;
            sizes.push(font_number(c, "size")?);
        }
        let min = sizes.iter().copied().reduce(f64::min).unwrap();
        let max = sizes.iter().copied().reduce(f64::max).unwrap();
        if italic || min > 0. && min <= 0.85 * max {
            words.insert(m.as_str().to_owned());
        }
    }
    Ok(json!(words))
}
pub(super) fn freeze(l: &mut Value, p: [f64; 2], w: &mut Work) -> Result<(), Error> {
    let chars = optional(l, "chars", 20000)?.to_vec();
    let re = crate::cached_regex!(
        r"^(?:[A-Za-z]|[0-9]{1,2})[.)][\u{9}-\u{d}\u{1c}-\u{20}\u{85}\u{a0}\u{1680}\u{2000}-\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}]+"
    );
    let mapped = text(l)?
        .chars()
        .map(|c| python_decimal(c).map_or(c, |d| (b'0' + d) as char))
        .collect::<String>();
    let mut left = l["native_title_label_left"].clone();
    if left.is_null() {
        left = l["bbox"][0].clone();
        if re.is_match(&mapped) && !chars.is_empty() {
            let mut ended = false;
            for c in &chars {
                let t = raw_string(c.get("char"), true);
                if ended && !t.trim_matches(python_space).is_empty() {
                    left = json!(num(&c["bbox"][0])?);
                    break;
                }
                if t == "." || t == ")" {
                    ended = true
                }
            }
        }
    }
    l["native_title_label_left"] = left;
    let mut n = 0;
    for c in &chars {
        let weight = c["font"].get("weight").unwrap_or(&json!(0)).clone();
        if num(&weight)? < 600. {
            break;
        }
        n += 1
    }
    let prefix = &chars[..n];
    let rest = &chars[n..];
    let tail = prefix
        .iter()
        .map(|c| raw_string(c.get("char"), true))
        .collect::<String>();
    let tail = tail.trim_matches(python_space);
    let body = rest
        .iter()
        .map(|c| raw_string(c.get("char"), true))
        .collect::<String>();
    let body = body.trim_matches(python_space);
    let wc = |s: &str| s.split(python_space).filter(|s| !s.is_empty()).count();
    if prefix.is_empty()
        || rest.is_empty()
        || !(1..=3).contains(&wc(tail))
        || !tail.ends_with(['.', ':', '。', '：'])
        || wc(body) < 4
    {
        return Ok(());
    }
    for c in rest.iter().filter(|c| visible(c)) {
        let weight = c["font"].get("weight").cloned().unwrap_or(json!(0));
        if num(&weight)? > 500. {
            return Ok(());
        }
    }
    let mut parts = Vec::new();
    for (t, cs) in [(tail, prefix), (body, rest)] {
        w.charge(cs.len() + 1)?;
        let boxes = cs
            .iter()
            .filter(|c| visible(c))
            .map(|c| bbox(&c["bbox"]))
            .collect::<Result<Vec<_>, _>>()?;
        let b = combined(boxes.iter())?;
        let mut part = l.clone();
        part["text"] = json!(t);
        part["bbox"] = json!(b.raw);
        part["source_bbox"] = part["bbox"].clone();
        part["ink_bbox"] = Value::Null;
        part["chars"] = json!(cs);
        part["wrapped_title_parts"] = Value::Null;
        let metrics = preparation::page_metrics(&part, p)?;
        part.as_object_mut()
            .unwrap()
            .extend(metrics.as_object().unwrap().clone());
        part["chars"] = json!([]);
        parts.push(part)
    }
    l["wrapped_title_parts"] = json!(parts);
    Ok(())
}
pub(super) fn lines(ls: &mut [Value], p: [f64; 2], w: &mut Work) -> Result<(), Error> {
    for l in ls {
        w.charge(text(l)?.len() + optional(l, "chars", 20000)?.len() + 1)?;
        let cjk = optional(l, "chars", 20000)?
            .iter()
            .filter(|c| {
                c["char"].as_str().is_some_and(|t| {
                    t.chars().count() == 1
                        && t.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
                })
            })
            .collect::<Vec<_>>();
        if !cjk.is_empty() {
            let first = cjk[0];
            let name = raw_string(first["font"].get("name"), false);
            let flags = font_number(first, "flags")? as i64;
            let weight = font_number(first, "weight")?;
            let hs = cjk
                .iter()
                .map(|c| {
                    let b = bbox(&c["bbox"])?;
                    Ok(b.n[3] - b.n[1])
                })
                .collect::<Result<Vec<_>, Error>>()?;
            l["native_cjk_style"] = json!([name, flags, weight, median(hs)]);
        }
        if l["median_glyph_width"].is_null() {
            l["median_glyph_width"] = glyph_width(l, p, w)?.map_or(Value::Null, |v| json!(v))
        }
        l["footnote_marker_start"] = json!(l["footnote_marker_start"] == true || footnote(l)?);
        l["native_math_words"] = math_words(l)?;
        freeze(l, p, w)?;
        l["chars"] = json!([]);
    }
    Ok(())
}
pub(super) fn unmapped(chars: &[Value], w: &mut Work) -> Result<Value, Error> {
    let mut out = Vec::new();
    for c in chars {
        w.charge(1)?;
        let t = raw_string(c.get("char"), false);
        if !t.is_empty() && t.chars().all(python_printable) && !t.chars().all(python_space) {
            continue;
        }
        let b = super::super::page_containers::coerce(&c["tight_bbox"])?;
        let size = font_number(c, "size")?;
        if let Some(b) = b {
            if size > 0.
                && !c["text_object_id"].is_null()
                && c["text_render_mode"] != 3
                && c["text_render_mode"] != 7
                && b.n[2] - b.n[0] > 0.05 * size
                && b.n[3] - b.n[1] > 1.5 * size
            {
                out.push(json!(b.raw))
            }
        }
    }
    Ok(json!(out))
}
