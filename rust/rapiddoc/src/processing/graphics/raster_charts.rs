//! 旧栅格坐标图刻度/图体证明和上方同式说明续行归组。
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{python_decimal, raster_charts::raster_axis_groups, text_boundary::python_space};
use serde_json::{json, Value};
type B = [f64; 4];
fn num(v: &Value) -> Result<f64, Error> {
    let n = v.as_f64().filter(|v| v.is_finite()).ok_or_else(invalid)?;
    if n.abs() > 1e9 {
        return Err(limit());
    }
    Ok(n)
}
fn array(v: &Value, max: usize) -> Result<&[Value], Error> {
    let a = v.as_array().ok_or_else(invalid)?;
    let max = if max > 8 {
        crate::legacy_page::budget::records(max)
    } else {
        max
    };
    if a.len() > max {
        return Err(limit());
    }
    Ok(a)
}
fn bbox(v: &Value) -> Result<B, Error> {
    let a = array(v, 4)?;
    if a.len() != 4 {
        return Err(invalid());
    }
    Ok([num(&a[0])?, num(&a[1])?, num(&a[2])?, num(&a[3])?])
}
fn boolean(v: &Value, k: &str) -> Result<bool, Error> {
    v.get(k)
        .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))
}
fn font(v: &Value) -> Result<Option<String>, Error> {
    if v.is_null() {
        return Ok(None);
    }
    let a = array(v, 32)?;
    if a.is_empty() {
        return Ok(None);
    }
    let mut key = Vec::new();
    for p in a {
        key.push(match p {
            Value::String(s) if s.len() <= 256 => p.clone(),
            Value::Null => Value::Null,
            Value::Bool(b) => json!(i32::from(*b)),
            Value::Number(_) if p.as_i64().is_some() => p.clone(),
            _ => return Err(invalid()),
        });
    }
    Ok(Some(serde_json::to_string(&key).map_err(|_| invalid())?))
}
fn numeric(s: &str) -> Option<f64> {
    let chars: Vec<_> = s.trim_matches(python_space).chars().collect();
    if chars.is_empty() || chars.len() > 8 {
        return None;
    }
    let mut i = 0;
    let mut out = String::new();
    if matches!(chars[0], '+' | '-' | '−') {
        out.push(if chars[0] == '−' { '-' } else { chars[0] });
        i = 1;
    }
    let start = i;
    while i < chars.len() {
        let Some(d) = python_decimal(chars[i]) else {
            break;
        };
        out.push(char::from(b'0' + d as u8));
        i += 1;
    }
    if i == start {
        return None;
    }
    if i < chars.len() && chars[i] == '.' {
        out.push('.');
        i += 1;
        let start = i;
        while i < chars.len() {
            let Some(d) = python_decimal(chars[i]) else {
                break;
            };
            out.push(char::from(b'0' + d as u8));
            i += 1;
        }
        if i == start {
            return None;
        }
    }
    if i != chars.len() {
        return None;
    }
    out.parse().ok()
}
fn list_start(s: &str) -> bool {
    let mut it = s.trim_start_matches(python_space).chars().peekable();
    if it.peek().is_some_and(|c| matches!(c, '•' | '▪')) {
        return true;
    }
    let mut digits = 0;
    while it.peek().is_some_and(|c| python_decimal(*c).is_some()) {
        digits += 1;
        it.next();
    }
    digits > 0 && it.next().is_some_and(|c| matches!(c, '.' | ')' | '、'))
}
struct Line {
    b: B,
    h: f64,
    upright: bool,
    free: bool,
    caption: bool,
    font: Option<String>,
    group: Option<i64>,
    terminal: bool,
    list: bool,
}
fn area(b: B) -> f64 {
    (b[2] - b[0]).max(0.) * (b[3] - b[1]).max(0.)
}
#[derive(Clone, Copy)]
enum Area {
    Integer(i128),
    Float(f64),
}
fn area_key(raw: &Value, b: B) -> Area {
    let ints: Option<Vec<_>> = raw.as_array().unwrap().iter().map(Value::as_i64).collect();
    match ints {
        Some(v) => Area::Integer(
            (v[2] as i128 - v[0] as i128).max(0) * (v[3] as i128 - v[1] as i128).max(0),
        ),
        None => Area::Float(area(b)),
    }
}
fn compare_area(a: Area, b: Area) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let mixed = |i: i128, f: f64| match i.cmp(&(f as i128)) {
        Ordering::Equal if f.fract() > 0. => Ordering::Less,
        other => other,
    };
    match (a, b) {
        (Area::Integer(a), Area::Integer(b)) => a.cmp(&b),
        (Area::Float(a), Area::Float(b)) => a.partial_cmp(&b).unwrap(),
        (Area::Integer(i), Area::Float(f)) => mixed(i, f),
        (Area::Float(f), Area::Integer(i)) => mixed(i, f).reverse(),
    }
}
fn center(b: B, axis: usize) -> f64 {
    (b[axis] + b[axis + 2]) / 2.
}
fn overlap(a: B, b: B, axis: usize) -> f64 {
    let shorter = (a[axis + 2] - a[axis]).min(b[axis + 2] - b[axis]);
    if shorter > 0. {
        (a[axis + 2].min(b[axis + 2]) - a[axis].max(b[axis])).max(0.) / shorter
    } else {
        0.
    }
}
fn in_first(a: B, b: B) -> f64 {
    let size = area(a);
    if size > 0. {
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.) / size
    } else {
        0.
    }
}
fn union(a: B, b: B) -> B {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}
pub fn group(v: &Value) -> Result<Value, Error> {
    group_with_work(v, &mut Work(0))
}
pub(crate) fn group_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    core_impl(v, work, true)
}
pub(super) fn cores_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    core_impl(v, work, false)
}
fn core_impl(v: &Value, work: &mut Work, descriptions: bool) -> Result<Value, Error> {
    validate_tree(v)?;
    core_source(&v["source"], work, descriptions)
}
pub(super) fn cores_source(source: &Value, work: &mut Work) -> Result<Value, Error> {
    core_source(source, work, false)
}
fn core_source(source: &Value, work: &mut Work, descriptions: bool) -> Result<Value, Error> {
    core_source_owned(source.clone(), work, descriptions)
}
pub(crate) fn group_source_owned(source: Value, work: &mut Work) -> Result<Value, Error> {
    core_source_owned(source, work, true)
}
fn core_source_owned(
    mut source: Value,
    work: &mut Work,
    descriptions: bool,
) -> Result<Value, Error> {
    crate::legacy_page::validate_source_request(&source, &json!({}))?;
    source.as_object().ok_or_else(invalid)?;
    let raw = array(&source["lines"], 4096)?;
    let mut lines = Vec::new();
    let mut records = Vec::new();
    let mut indices = Vec::new();
    let mut next = None;
    for (i, r) in raw.iter().enumerate() {
        r.as_object().ok_or_else(invalid)?;
        let text = r["text"].as_str().ok_or_else(invalid)?;
        work.charge(text.len() + 1)?;
        if text.chars().count() > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        let b = bbox(&r["bbox"])?;
        let h = r.get("effective_height").map_or(Ok(0.), num)?;
        let group = if r["paragraph_group"].is_null() {
            None
        } else {
            Some(
                r["paragraph_group"]
                    .as_i64()
                    .filter(|n| n.unsigned_abs() <= rapiddoc_core::resources::limit(1_000_000_000))
                    .ok_or_else(invalid)?,
            )
        };
        if let Some(g) = group {
            next = Some(next.map_or(g, |v: i64| v.max(g)));
        }
        let l = Line {
            b,
            h,
            upright: r.get("angle").map_or(Ok(0.), num)? == 0.,
            free: r["semantic_type"].is_null(),
            caption: boolean(r, "caption_start")?,
            font: font(&r["font_signature"])?,
            group,
            terminal: boolean(r, "paragraph_terminal")?,
            list: list_start(text),
        };
        if l.upright && l.free && h > 0. && b[2] - b[0] <= 4. * h {
            if let Some(value) = numeric(text) {
                records.push((b, h, value));
                indices.push(i);
            }
        }
        lines.push(l);
    }
    let images = source.get("image_bboxes").map_or(Ok(Vec::new()), |v| {
        array(v, 5000)?
            .iter()
            .map(bbox)
            .collect::<Result<Vec<_>, _>>()
    })?;
    let areas: Vec<_> = images
        .iter()
        .enumerate()
        .map(|(i, b)| area_key(&source["image_bboxes"][i], *b))
        .collect();
    let n = records.len();
    let log = if n < 2 {
        1
    } else {
        usize::BITS as usize - (n - 1).leading_zeros() as usize + 1
    };
    work.charge(
        n.checked_mul(n + 1)
            .and_then(|n| n.checked_mul(2 * log))
            .ok_or_else(limit)?,
    )?;
    let groups = raster_axis_groups(&records);
    let mut cores = Vec::new();
    for (vertical, ticks, em) in &groups {
        work.charge(ticks.len() + images.len() * 2 + lines.len())?;
        let axis = ticks.iter().map(|i| records[*i].0).reduce(union).unwrap();
        let mut best = None;
        for (i, &b) in images.iter().enumerate() {
            let (w, h) = (b[2] - b[0], b[3] - b[1]);
            if w < 8. * em || h < 8. * em {
                continue;
            }
            let compatible = if *vertical {
                let gap = b[0] - axis[2];
                gap >= 0.
                    && gap <= 4. * em
                    && 0.7 * h <= axis[3] - axis[1]
                    && axis[3] - axis[1] <= 1.25 * h
                    && (center(axis, 1) - center(b, 1)).abs() <= *em
            } else {
                let gap = axis[1] - b[3];
                gap >= 0.
                    && gap <= 2. * em
                    && 0.7 * w <= axis[2] - axis[0]
                    && axis[2] - axis[0] <= 1.25 * w
                    && (center(axis, 0) - center(b, 0)).abs() <= *em
            };
            if compatible
                && best.is_none_or(|old: usize| compare_area(areas[i], areas[old]).is_gt())
            {
                best = Some(i);
            }
        }
        let Some(image) = best.map(|i| images[i]) else {
            continue;
        };
        let window = [
            if *vertical {
                axis[0] - 0.4 * em
            } else {
                image[0] - 10. * em
            },
            image[1] - 1.5 * em,
            image[2] + if *vertical { 2. * em } else { 11. * em },
            image[3] + if *vertical { 6.5 * em } else { 2.3 * em },
        ];
        let mut bounds = union(image, axis);
        for l in &lines {
            if l.upright
                && !l.caption
                && l.free
                && 0.6 * em <= l.h
                && l.h <= 1.8 * em
                && in_first(l.b, window) >= 0.95
            {
                bounds = union(bounds, l.b);
            }
        }
        for &b in &images {
            if !vertical
                && 0. <= b[0] - image[2]
                && b[0] - image[2] <= 4. * em
                && b[2] - b[0] <= 10. * em
                && b[3] - b[1] <= 6. * em
                && overlap(b, image, 1) >= 0.8
            {
                bounds = union(bounds, b);
            }
        }
        cores.push([
            bounds[0] - 0.2 * em,
            bounds[1] - 0.2 * em,
            bounds[2] + 0.2 * em,
            bounds[3] + 0.2 * em,
        ]);
    }
    if !descriptions {
        return Ok(crate::processing::with_source(
            source,
            value_object!({"graphic_bboxes":cores}),
        ));
    }
    let mut next = next.unwrap_or(0) + 1;
    for core in &cores {
        work.charge(lines.len() * (log_for(lines.len()) + 1))?;
        let mut above: Vec<_> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| {
                l.upright
                    && l.free
                    && l.group.is_none()
                    && !l.caption
                    && l.font.is_some()
                    && 0. <= core[1] - l.b[3]
                    && core[1] - l.b[3] <= 5. * l.h
                    && overlap(l.b, *core, 0) >= 0.3
            })
            .map(|(i, _)| i)
            .collect();
        above.sort_by(|a, b| lines[*a].b[1].partial_cmp(&lines[*b].b[1]).unwrap());
        for pair in above.windows(2) {
            let (i, j) = (pair[0], pair[1]);
            let (first, tail) = (&lines[i], &lines[j]);
            let em = first.h;
            work.charge(
                first.font.as_ref().map_or(0, String::len)
                    + tail.font.as_ref().map_or(0, String::len)
                    + 1,
            )?;
            if em <= 0.
                || first.font != tail.font
                || tail.h < 0.9 * em
                || tail.h > 1.1 * em
                || (first.b[0] - tail.b[0]).abs() > 0.2 * em
                || first.b[2] - first.b[0] < 10. * em
                || tail.b[2] - tail.b[0] > 0.6 * (first.b[2] - first.b[0])
                || tail.b[1] - first.b[3] < 0.
                || tail.b[1] - first.b[3] > 0.75 * em
                || first.terminal
                || tail.list
                || core[1] - tail.b[3] < 0.
                || core[1] - tail.b[3] > 3. * em
            {
                continue;
            }
            // above为原列表快照；同一core的相邻链允许覆盖中间行的刚赋分组。
            for k in [i, j] {
                lines[k].group = Some(next);
                source["lines"][k]["paragraph_group"] = json!(next);
            }
            next += 1;
        }
    }
    let groups:Vec<_>=groups.into_iter().map(|(vertical,ids,_)|json!({"vertical":vertical,"line_indices":ids.into_iter().map(|i|indices[i]).collect::<Vec<_>>()})).collect();
    let result = crate::processing::with_source(
        source,
        value_object!({"graphic_bboxes":cores,"axis_groups":groups}),
    );
    if crate::legacy_page::budget::output_exceeds_limit(
        &result,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(result)
}
fn log_for(n: usize) -> usize {
    if n < 2 {
        1
    } else {
        usize::BITS as usize - (n - 1).leading_zeros() as usize + 1
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_raster_charts() {
        for name in ["native-raster-charts", "native-raster-charts-edges"] {
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../build/migration/{name}.json"));
            let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
            for c in f["cases"].as_array().unwrap() {
                assert_eq!(
                    group(&c["request"]).unwrap(),
                    c["expected"],
                    "{}",
                    c["label"]
                );
            }
        }
    }
    #[test]
    fn shared_budget() {
        let q = json!({"source":{"lines":[{"text":"body","bbox":[0,0,10,10]}]}});
        assert!(group_with_work(&q, &mut Work(9_999_999)).is_err());
        assert!(group(&q).is_ok());
    }
}
