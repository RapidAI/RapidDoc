//! Entire original glyph_weight grouping, conditional rendering and same-character ink comparison.
use super::{
    page_containers::{coerce, optional},
    path_graphics::array,
    table_text_styles,
};
use crate::legacy_page::{assets::Pixels, eq, invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
fn metadata(c: &Value) -> Result<Value, Error> {
    let f = &c["font"];
    if !f.is_object() {
        return Ok(json!(["", 0, null]));
    }
    let name = super::table_native_multiline::spatial::raw_string(f.get("name"), true);
    let b = name.as_bytes();
    let name = if b.len() >= 7 && b[..6].iter().all(u8::is_ascii_uppercase) && b[6] == b'+' {
        &name[7..]
    } else {
        &name
    };
    let raw = &f["flags"];
    let flags = if raw.is_null() || raw == false || raw == "" {
        0
    } else if let Some(n) = raw.as_i64() {
        n
    } else if let Some(n) = raw.as_bool() {
        i64::from(n)
    } else if let Some(n) = raw.as_f64() {
        n.trunc() as i64
    } else {
        let s = raw.as_str().unwrap_or("");
        super::table_native_text::numeric_string(s)?
            .filter(|s| {
                s.trim_start_matches(['+', '-'])
                    .bytes()
                    .all(|b| b.is_ascii_digit())
            })
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(0)
    };
    let weight = table_text_styles::numeric(f.get("weight")).filter(|w| *w > 0.);
    Ok(json!([name, flags, weight]))
}
#[derive(Clone)]
struct Group {
    font: Value,
    count: usize,
    letters: Vec<(char, Vec<Value>)>,
}
fn groups(lines: &Value, tight_bboxes: &Value, w: &mut Work) -> Result<Vec<Group>, Error> {
    let mut out: Vec<Group> = vec![];
    let mut total = 0;
    for l in array(lines, 4096)? {
        if l.get("angle").cloned().unwrap_or(json!(0)) != 0 {
            continue;
        }
        let cs = optional(l, "chars", 20000)?;
        total += cs.len();
        if total > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        for c in cs {
            w.charge(1)?;
            let t = super::table_native_multiline::spatial::raw_string(c.get("char"), false);
            if t.len() != 1 || !t.as_bytes()[0].is_ascii_alphabetic() {
                continue;
            }
            let font = metadata(c)?;
            if font[1].as_i64().unwrap() & (1 << 6) != 0 || table_text_styles::font_styles(c).0 {
                continue;
            }
            let key = c["char_idx"].as_i64().map(|i| i.to_string());
            let mapped = key.as_ref().and_then(|k| tight_bboxes.get(k));
            let b = if mapped.is_some_and(super::native_text_assembly::python_truth) {
                coerce(mapped.unwrap())?
            } else {
                coerce(&c["tight_bbox"])?
            };
            let Some(b) = b else { continue };
            let i = if let Some(i) = out.iter().position(|g| eq(&g.font, &font)) {
                i
            } else {
                out.push(Group {
                    font,
                    count: 0,
                    letters: vec![],
                });
                out.len() - 1
            };
            let g = &mut out[i];
            g.count += 1;
            let ch = t.chars().next().unwrap();
            let k = if let Some(k) = g.letters.iter().position(|(c, _)| *c == ch) {
                k
            } else {
                g.letters.push((ch, vec![]));
                g.letters.len() - 1
            };
            if g.letters[k].1.len() < 3 {
                g.letters[k].1.push(json!(b.raw))
            }
        }
    }
    Ok(out)
}
fn selection(gs: &[Group]) -> Option<(usize, Vec<usize>)> {
    let eligible = gs
        .iter()
        .enumerate()
        .filter(|(_, g)| g.count >= 20 && g.letters.len() >= 8)
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    if eligible.len() < 2 {
        return None;
    }
    let mut reference = eligible[0];
    for &i in &eligible[1..] {
        if gs[i].count > gs[reference].count {
            reference = i
        }
    }
    let candidates = eligible
        .into_iter()
        .filter(|&i| {
            i != reference
                && gs[i].font[2].is_null()
                && gs[i]
                    .letters
                    .iter()
                    .filter(|(c, _)| gs[reference].letters.iter().any(|(r, _)| r == c))
                    .count()
                    >= 8
        })
        .collect::<Vec<_>>();
    (!candidates.is_empty()).then_some((reference, candidates))
}
fn plan_value(gs: &[Group]) -> Value {
    let selection = selection(gs);
    json!({"groups":gs.iter().map(|g|json!({"font":g.font,"count":g.count,"letters":g.letters.iter().map(|(c,bs)|json!([c.to_string(),bs])).collect::<Vec<_>>()})).collect::<Vec<_>>(),"reference":selection.as_ref().map(|(r,_)|*r),"candidates":selection.map(|(_,cs)|cs).unwrap_or_default()})
}
pub fn plan(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    plan_with_work(v, &mut Work(0))
}
pub(super) fn plan_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    plan_parts(&v["lines"], &v["tight_bboxes"], w)
}
pub(super) fn plan_parts(
    lines: &Value,
    tight_bboxes: &Value,
    w: &mut Work,
) -> Result<Value, Error> {
    Ok(plan_value(&groups(lines, tight_bboxes, w)?))
}
#[derive(Clone)]
struct Sample {
    bits: [u64; 16],
    count: u32,
    aspect: f64,
}
fn sample(image: &Pixels, b: &Value, scale: f64, w: &mut Work) -> Result<Option<Sample>, Error> {
    let b = super::path_graphics::bbox(b)?.n;
    let edge = [
        (b[0] * scale).floor(),
        (b[1] * scale).floor(),
        (b[2] * scale).ceil(),
        (b[3] * scale).ceil(),
    ];
    if edge.iter().any(|v| v.abs() > 1e9) {
        return Err(limit());
    }
    let edge = edge.map(|v| v as i64);
    let (cw, ch) = (edge[2] - edge[0], edge[3] - edge[1]);
    if cw < 6 || ch < 9 {
        return Ok(None);
    }
    w.charge(
        (cw as usize)
            .saturating_mul(ch as usize)
            .saturating_add(1024),
    )?;
    let mut bits = [0u64; 16];
    for y in 0..32 {
        for x in 0..32 {
            let sx = edge[0] + ((2 * x + 1) * cw) / 64;
            let sy = edge[1] + ((2 * y + 1) * ch) / 64;
            let gray = if sx < 0 || sy < 0 || sx >= image.width as i64 || sy >= image.height as i64
            {
                0
            } else {
                let p = (sy as usize * image.width + sx as usize) * image.channels;
                match image.channels {
                    1 | 2 => u32::from(image.data[p]),
                    3 | 4 => {
                        (19595 * u32::from(image.data[p])
                            + 38470 * u32::from(image.data[p + 1])
                            + 7471 * u32::from(image.data[p + 2])
                            + 32768)
                            >> 16
                    }
                    _ => return Err(invalid()),
                }
            };
            if gray < 128 {
                let index = (y * 32 + x) as usize;
                bits[index / 64] |= 1 << (index % 64)
            }
        }
    }
    let count = bits.iter().map(|b| b.count_ones()).sum();
    if !(0.08..=0.85).contains(&(f64::from(count) / 1024.)) {
        return Ok(None);
    }
    Ok(Some(Sample {
        bits,
        count,
        aspect: cw as f64 / ch as f64,
    }))
}
pub fn detect(v: &Value, png: Option<&[u8]>) -> Result<Value, Error> {
    validate_tree(v)?;
    detect_with_work(v, png, &mut Work(0))
}
pub(super) fn detect_with_work(
    v: &Value,
    png: Option<&[u8]>,
    w: &mut Work,
) -> Result<Value, Error> {
    detect_parts(&v["lines"], &v["tight_bboxes"], v, png, w)
}
pub(super) fn detect_parts(
    lines: &Value,
    tight_bboxes: &Value,
    options: &Value,
    png: Option<&[u8]>,
    w: &mut Work,
) -> Result<Value, Error> {
    let gs = groups(lines, tight_bboxes, w)?;
    let Some((reference, candidates)) = selection(&gs) else {
        return Ok(json!([]));
    };
    let Some(png) = png else { return Ok(json!([])) };
    let scale = super::path_graphics::num(&options["scale"])?;
    if scale <= 0. {
        return Err(invalid());
    }
    let image = Pixels::decode(png)?;
    let mut masks: Vec<(usize, Vec<(char, Sample)>)> = vec![];
    for i in std::iter::once(reference).chain(candidates.iter().copied()) {
        let mut chars = vec![];
        for (c, bs) in &gs[i].letters {
            let mut samples = vec![];
            for b in bs {
                if let Some(s) = sample(&image, b, scale, w)? {
                    samples.push(s)
                }
            }
            samples.sort_by_key(|s| s.count);
            if !samples.is_empty() {
                chars.push((*c, samples[samples.len() / 2].clone()))
            }
        }
        masks.push((i, chars))
    }
    let normal = &masks[0].1;
    let mut bold = vec![];
    for (i, chars) in &masks[1..] {
        let mut ratios = vec![];
        for (c, target) in chars {
            let Some((_, normal)) = normal.iter().find(|(n, _)| n == c) else {
                continue;
            };
            let aspect = target.aspect / normal.aspect;
            let intersection = target
                .bits
                .iter()
                .zip(&normal.bits)
                .map(|(a, b)| (a & b).count_ones())
                .sum::<u32>();
            let union = target
                .bits
                .iter()
                .zip(&normal.bits)
                .map(|(a, b)| (a | b).count_ones())
                .sum::<u32>();
            if !(0.75..=1.33).contains(&aspect) || f64::from(intersection) / f64::from(union) < 0.4
            {
                continue;
            }
            ratios.push(f64::from(target.count) / f64::from(normal.count));
        }
        if ratios.len() >= 8
            && super::page_containers::median(ratios.clone()) >= 1.4
            && ratios.iter().filter(|&&r| r >= 1.3).count() as f64 / ratios.len() as f64 >= 0.75
        {
            bold.push(*i)
        }
    }
    if bold.is_empty() {
        return Ok(json!([]));
    }
    let mut patched = vec![];
    for l in array(lines, 4096)? {
        let cs = optional(l, "chars", 20000)?;
        let keys = cs.iter().map(metadata).collect::<Result<Vec<_>, _>>()?;
        if !keys
            .iter()
            .any(|k| bold.iter().any(|&i| eq(k, &gs[i].font)))
        {
            continue;
        }
        let mut l = l.clone();
        let Value::Array(mut cs) = l["chars"].take() else {
            return Err(invalid());
        };
        for (c, key) in cs.iter_mut().zip(keys) {
            if bold.iter().any(|&i| eq(&key, &gs[i].font)) {
                c["font"]["weight"] = json!(700)
            }
        }
        l["chars"] = Value::Array(cs);
        patched.push(l)
    }
    Ok(
        super::inline_evidence::detect_parts(&Value::Array(patched), None, None, w)?["style_lines"]
            .take(),
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_python_glyph_weight() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let f: Value = serde_json::from_slice(
            &std::fs::read(root.join("build/migration/native-glyph-weight.json")).unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            let p = plan(&c["request"]).unwrap();
            if let Some(d) = super::super::native_document::tests::diff(&p, &c["plan"], "") {
                panic!("{} plan: {d}", c["label"])
            }
            let png = c["png"]
                .as_str()
                .map(|p| std::fs::read(root.join(p)).unwrap());
            let got = detect(&c["request"], png.as_deref())
                .unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = super::super::native_document::tests::diff(&got, &c["expected"], "") {
                panic!("{} output: {d}", c["label"])
            }
        }
    }
}
