//! 旧spatial_text.project_pdf_table_text及其紧字框间距，仅用于分组标题复核。
use super::super::table_native_text::{normalize, normalize_bbox, number, source_index};
use super::*;
use rapiddoc_native::{
    python_decimal,
    text_boundary::{python_alnum, python_alpha, python_space},
};
type Bbox = [f64; 4];
struct Item {
    text: String,
    bbox: Bbox,
}
pub(in crate::processing) fn raw_string(v: Option<&Value>, empty_false: bool) -> String {
    match v {
        None => String::new(),
        Some(Value::Null) => {
            if empty_false {
                String::new()
            } else {
                "None".into()
            }
        }
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(b)) => {
            if empty_false && !*b {
                String::new()
            } else if *b {
                "True".into()
            } else {
                "False".into()
            }
        }
        Some(n) if n.is_number() => {
            if empty_false && n.as_f64() == Some(0.) {
                String::new()
            } else {
                n.to_string()
            }
        }
        _ => String::new(),
    }
}
fn rotate(b: Bbox, w: f64, h: f64, a: i64) -> Bbox {
    match a {
        90 => [b[1], w - b[2], b[3], w - b[0]],
        180 => [w - b[2], h - b[3], w - b[0], h - b[1]],
        270 => [h - b[3], b[0], h - b[1], b[2]],
        _ => b,
    }
}
fn union(a: Bbox, b: Bbox) -> Bbox {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}
fn values(v: &Value, n: usize) -> Option<Vec<f64>> {
    let a = v.as_array()?;
    if a.len() != n {
        return None;
    }
    a.iter().map(|v| number(v).ok().flatten()).collect()
}
fn ordinary(s: &str) -> bool {
    let mut cs = s.chars();
    let Some(c) = cs.next() else {
        return false;
    };
    if cs.next().is_some() {
        return false;
    }
    let n = c as u32;
    !matches!(n,0x1100..=0x11ff|0x2e80..=0xa4cf|0xa960..=0xa97f|0xac00..=0xd7ff|0xf900..=0xfaff|0xff66..=0xff9f|0x1aff0..=0x1b16f|0x20000..=0x323af)
        && (python_alpha(c) || python_decimal(c).is_some())
}
fn integer(v: &Value) -> Option<i64> {
    if let Some(s) = v.as_str() {
        return crate::legacy_model::table_native_text::numeric_string(s)
            .ok()
            .flatten()
            .and_then(|s| s.parse().ok());
    }
    if let Some(b) = v.as_bool() {
        Some(i64::from(b))
    } else {
        number(v).ok().flatten().map(|n| n.trunc() as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Value {
        serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-table-multiline-extra.json"),
            )
            .unwrap(),
        )
        .unwrap()
    }
    #[test]
    fn old_spatial_table_projection() {
        for (i, c) in fixture()["projection"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            let r = &c["request"];
            let chars = r["chars"].as_array().unwrap().iter().collect::<Vec<_>>();
            let a = crate::legacy_model::table_native_text::angle(&r["angle"]).unwrap();
            assert_eq!(
                project(&chars, &r["table_bbox"], a as i64, &mut Work(0)).unwrap(),
                c["expected"].as_str().unwrap(),
                "projection {i}"
            );
        }
    }
    #[test]
    fn old_tight_space_conditions() {
        for (i, c) in fixture()["spacing"].as_array().unwrap().iter().enumerate() {
            assert_eq!(
                tight(&c["left"], &c["right"]),
                c["expected"].as_bool().unwrap(),
                "tight {i}"
            );
        }
    }
}
pub(in crate::processing) fn tight(left: &Value, right: &Value) -> bool {
    let (ls, rs) = (
        raw_string(left.get("char"), false),
        raw_string(right.get("char"), false),
    );
    if !ordinary(&ls) || !ordinary(&rs) {
        return false;
    }
    if ls.chars().all(|c| python_decimal(c).is_some())
        && rs.chars().all(|c| python_decimal(c).is_some())
    {
        return false;
    }
    let source = |c: &Value| {
        let v = &c["char_idx"];
        if v.is_i64() || v.is_u64() || v.is_boolean() {
            integer(v)
        } else {
            None
        }
    };
    let (Some(li), Some(ri)) = (source(left), source(right)) else {
        return false;
    };
    if ri != li + 1 {
        return false;
    }
    for c in [left, right] {
        let Some(font) = c["font"].as_object() else {
            return false;
        };
        let name = raw_string(font.get("name"), false).to_lowercase();
        let Some(flags) = integer(font.get("flags").unwrap_or(&json!(0))) else {
            return false;
        };
        if flags & 1 != 0
            || [
                "mono",
                "courier",
                "consolas",
                "menlo",
                "typewriter",
                "fixed",
            ]
            .iter()
            .any(|s| name.contains(s))
        {
            return false;
        }
    }
    let (Some(ls), Some(rs)) = (
        number(&left["font"]["size"]).ok().flatten(),
        number(&right["font"]["size"]).ok().flatten(),
    ) else {
        return false;
    };
    let em = ls.max(rs);
    if ls.min(rs) <= 0. || ls.min(rs) < 0.8 * em {
        return false;
    }
    let (Some(la), Some(ra)) = (
        number(&left["writing_angle"]).ok().flatten(),
        number(&right["writing_angle"]).ok().flatten(),
    ) else {
        return false;
    };
    let degrees = la.to_degrees().rem_euclid(360.);
    let angle = ((degrees / 90.).round_ties_even() * 90.).rem_euclid(360.) as i64;
    if ((degrees - angle as f64 + 180.).rem_euclid(360.) - 180.).abs() > 0.1
        || ((ra.to_degrees() - angle as f64 + 180.).rem_euclid(360.) - 180.).abs() > 0.1
    {
        return false;
    }
    let (Some(lb), Some(rb)) = (
        values(&left["tight_bbox"], 4),
        values(&right["tight_bbox"], 4),
    ) else {
        return false;
    };
    let local = |b: Vec<f64>| {
        let b = [b[0], b[1], b[2], b[3]];
        rotate(b, 0., 0., angle)
    };
    let (a, b) = (local(lb), local(rb));
    let (ah, bh) = (a[3] - a[1], b[3] - b[1]);
    if a[2] <= a[0]
        || b[2] <= b[0]
        || ah <= 0.
        || bh <= 0.
        || ah > 1.5 * ls
        || bh > 1.5 * rs
        || b[0] - a[2] <= 0.25 * em
        || a[3].min(b[3]) - a[1].max(b[1]) < 0.5 * ah.min(bh)
    {
        return false;
    }
    if !left["origin"].is_null() && !right["origin"].is_null() {
        let (Some(lo), Some(ro)) = (values(&left["origin"], 2), values(&right["origin"], 2)) else {
            return false;
        };
        let perpendicular = if [90, 270].contains(&angle) { 0 } else { 1 };
        if (lo[perpendicular] - ro[perpendicular]).abs() > 0.15 * em {
            return false;
        }
    } else if (a[3] - b[3]).abs() > 0.15 * em {
        return false;
    }
    true
}
#[derive(Default)]
struct Segment<'a> {
    parts: String,
    bbox: Option<Bbox>,
    last: Option<Bbox>,
    widths: Vec<f64>,
    pending: bool,
    source: Option<&'a Value>,
}
impl Segment<'_> {
    fn flush(&mut self, items: &mut Vec<Item>, table: Bbox, angle: i64) {
        let text = normalize(&self.parts).trim_matches(python_space).to_owned();
        if !text.is_empty() {
            if let Some(b) = self.bbox {
                items.push(Item {
                    text,
                    bbox: rotate(b, table[2] - table[0], table[3] - table[1], angle),
                });
            }
        }
        *self = Self::default();
    }
}
fn median(mut a: Vec<f64>) -> f64 {
    if a.is_empty() {
        return 1.;
    }
    a.sort_by(f64::total_cmp);
    let m = a.len() / 2;
    if a.len() % 2 == 1 {
        a[m]
    } else {
        (a[m - 1] + a[m]) / 2.
    }
}
pub(in crate::processing) fn project(
    chars: &[&Value],
    table: &Value,
    angle: i64,
    work: &mut Work,
) -> Result<String, Error> {
    project_options(chars, table, angle, true, false, work)
}
pub(in crate::processing) fn project_options(
    chars: &[&Value],
    table: &Value,
    angle: i64,
    infer_tight_spaces: bool,
    preserve_blank_rows: bool,
    work: &mut Work,
) -> Result<String, Error> {
    let Some(table) = normalize_bbox(table)? else {
        return Ok(String::new());
    };
    if chars.iter().any(|c| {
        !c.is_object()
            || c.get("char")
                .is_some_and(|s| !s.is_null() && !s.is_string())
    }) {
        return Err(invalid());
    }
    let mut selected = Vec::new();
    for (i, c) in chars.iter().enumerate() {
        let Some(b) = values(&c["bbox"], 4) else {
            continue;
        };
        let (cx, cy) = ((b[0] + b[2]) / 2., (b[1] + b[3]) / 2.);
        if table[0] <= cx && cx <= table[2] && table[1] <= cy && cy <= table[3] {
            selected.push((source_index(c, i)?, i, *c));
        }
    }
    selected.sort_by_key(|(source, i, _)| (*source, *i));
    let mut segment = Segment::default();
    let mut items = Vec::new();
    for (_, _, c) in selected {
        work.charge(1)?;
        let raw = raw_string(c.get("char"), true);
        if matches!(raw.as_str(), "\r" | "\n") {
            segment.flush(&mut items, table, angle);
            continue;
        }
        if !raw.is_empty() && raw.chars().all(python_space) {
            if !segment.parts.is_empty() {
                segment.pending = true;
            }
            continue;
        }
        let Some(b) = normalize_bbox(&c["bbox"])? else {
            continue;
        };
        let b = [
            b[0] - table[0],
            b[1] - table[1],
            b[2] - table[0],
            b[3] - table[1],
        ];
        let height = b[3] - b[1];
        if height < 0.5 {
            continue;
        }
        let text = normalize(&raw);
        if text.is_empty() {
            continue;
        }
        if let (Some(union), Some(last)) = (segment.bbox, segment.last) {
            work.charge(segment.widths.len())?;
            let gap = b[0] - last[2];
            let overlap = b[1] < union[3] + 2. && b[3] > union[1] - 2.;
            let width = union[2] - union[0];
            let changed = b[1] > last[3] + 2.
                || (b[1] > last[3] && gap < -5.)
                || (width > 20. && gap < -(width * 0.5));
            let average =
                table_native_candidate::sum(&segment.widths) / segment.widths.len() as f64;
            if !overlap || changed || gap >= 15. || (segment.pending && gap > average * 2.2) {
                segment.flush(&mut items, table, angle);
            } else if segment.pending {
                segment.parts.push(' ');
                segment.pending = false;
            } else {
                let previous = segment.parts.chars().last();
                let current = text.chars().next();
                if (previous.is_some_and(python_alnum)
                    && current.is_some_and(python_alnum)
                    && gap > height * 0.35)
                    || (infer_tight_spaces && segment.source.is_some_and(|p| tight(p, c)))
                {
                    segment.parts.push(' ');
                }
            }
        }
        segment.bbox = Some(segment.bbox.map_or(b, |a| union(a, b)));
        segment.parts.push_str(&text);
        segment.last = Some(b);
        segment.source = Some(c);
        segment.widths.push(b[2] - b[0]);
    }
    segment.flush(&mut items, table, angle);
    project_ready(items, preserve_blank_rows, work)
}
/// Project already assembled text items without changing their text or segmentation.
pub(in crate::processing) fn project_items(
    items: &[(String, [f64; 4])],
    work: &mut Work,
) -> Result<String, Error> {
    work.charge(items.len())?;
    project_ready(
        items
            .iter()
            .map(|(text, bbox)| Item {
                text: text.clone(),
                bbox: *bbox,
            })
            .collect(),
        false,
        work,
    )
}
fn project_ready(
    mut items: Vec<Item>,
    preserve_blank_rows: bool,
    work: &mut Work,
) -> Result<String, Error> {
    items.retain(|i| !i.text.is_empty() && i.bbox[2] > i.bbox[0] && i.bbox[3] > i.bbox[1]);
    if items.is_empty() {
        return Ok(String::new());
    }
    let mw = median(
        items
            .iter()
            .map(|i| (i.bbox[2] - i.bbox[0]) / i.text.chars().count().max(1) as f64)
            .collect(),
    )
    .max(0.1);
    let mh = median(items.iter().map(|i| i.bbox[3] - i.bbox[1]).collect()).max(0.1);
    let ytol = 5f64.max(mh * 0.5);
    items.sort_by(|a, b| {
        (a.bbox[1] / ytol)
            .round_ties_even()
            .partial_cmp(&(b.bbox[1] / ytol).round_ties_even())
            .unwrap()
            .then(a.bbox[0].partial_cmp(&b.bbox[0]).unwrap())
    });
    let mut lines: Vec<Vec<usize>> = Vec::new();
    for (i, item) in items.iter().enumerate() {
        if lines.is_empty() {
            lines.push(vec![i]);
            continue;
        }
        let line = lines.last_mut().unwrap();
        work.charge(line.len())?;
        let top = line
            .iter()
            .map(|i| items[*i].bbox[1])
            .fold(f64::INFINITY, f64::min);
        let bottom = line
            .iter()
            .map(|i| items[*i].bbox[3])
            .fold(f64::NEG_INFINITY, f64::max);
        let collision = line.iter().any(|i| {
            items[*i].bbox[2].min(item.bbox[2]) - items[*i].bbox[0].max(item.bbox[0])
                > 5f64.max(mw / 3.)
        });
        let center = (item.bbox[1] + item.bbox[3]) / 2.;
        let compatible =
            (top <= center && center <= bottom) || (top <= item.bbox[1] && item.bbox[1] <= bottom);
        if !collision && compatible && bottom.max(item.bbox[3]) - top.min(item.bbox[1]) <= mh * 1.8
        {
            line.push(i);
        } else {
            lines.push(vec![i]);
        }
    }
    for line in &mut lines {
        line.sort_by(|a, b| items[*a].bbox[0].partial_cmp(&items[*b].bbox[0]).unwrap());
    }
    lines.sort_by(|a, b| {
        let top = |l: &Vec<usize>| {
            l.iter()
                .map(|i| items[*i].bbox[1])
                .fold(f64::INFINITY, f64::min)
        };
        top(a).partial_cmp(&top(b)).unwrap()
    });
    let mut projected = Vec::new();
    let mut total = 0;
    let mut previous_bottom: Option<f64> = None;
    for line in lines {
        let top = line
            .iter()
            .map(|&i| items[i].bbox[1])
            .reduce(f64::min)
            .unwrap();
        let bottom = line
            .iter()
            .map(|&i| items[i].bbox[3])
            .reduce(f64::max)
            .unwrap();
        if preserve_blank_rows {
            if let Some(previous) = previous_bottom {
                let gap = top - previous;
                if gap > 1.25 * mh {
                    let count = ((gap / mh).round_ties_even() - 1.).max(1.) as usize;
                    if count > rapiddoc_core::resources::limit(20000) {
                        return Err(limit());
                    }
                    work.charge(count)?;
                    projected.extend(std::iter::repeat_n(String::new(), count));
                }
            }
        }
        previous_bottom = Some(bottom);
        let mut output = String::new();
        let mut length = 0;
        for i in line {
            let item = &items[i];
            let minimum = length + usize::from(!output.is_empty() && !output.ends_with(' '));
            let column = ((item.bbox[0] / mw).round_ties_even().max(0.) as usize).max(minimum);
            let item_len = item.text.chars().count();
            if column > rapiddoc_core::resources::limit(1_000_000)
                || total + column + item_len > rapiddoc_core::resources::limit(8_000_000)
            {
                return Err(limit());
            }
            work.charge(column - length + item_len)?;
            output.extend(std::iter::repeat_n(' ', column - length));
            output.push_str(&item.text);
            length = column + item_len;
        }
        let output = output.trim_end_matches(python_space).to_owned();
        total += output.chars().count();
        projected.push(output);
    }
    while projected.first().is_some_and(String::is_empty) {
        projected.remove(0);
    }
    while projected.last().is_some_and(String::is_empty) {
        projected.pop();
    }
    if projected.is_empty() {
        return Ok(String::new());
    }
    let indent = projected
        .iter()
        .filter(|s| !s.is_empty())
        .map(|s| s.chars().take_while(|c| *c == ' ').count())
        .min()
        .unwrap_or(0);
    Ok(projected
        .iter()
        .map(|s| s.chars().skip(indent).collect::<String>())
        .collect::<Vec<_>>()
        .join("\n"))
}
