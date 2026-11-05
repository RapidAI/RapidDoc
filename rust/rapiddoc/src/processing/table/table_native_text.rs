//! 旧_table_recovery/text.py字符选择→局部字形→视觉行/token；尚不生成表格候选。
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{python_decimal, python_printable, text_boundary::python_space};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
#[path = "table_native_text/rotation.rs"]
mod rotation;
use rotation::zero_angle;
pub(crate) use rotation::{reading_rows as cell_text_rows, writing_angle as cell_writing_angle};
type Box4 = [f64; 4];
#[derive(Clone, Serialize)]
pub struct Pending {
    pub glyph_id: usize,
    pub source_index: i64,
    pub text: String,
    pub bbox: Box4,
    pub explicit_space_before: bool,
    pub explicit_break_before: bool,
    pub writing_angle: Option<i32>,
    pub advance_bbox: Option<Box4>,
    pub origin: Option<[f64; 2]>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Glyph {
    pub glyph_id: usize,
    pub source_index: i64,
    pub text: String,
    pub bbox: Box4,
    pub visual_row: i64,
    pub explicit_space_before: bool,
    pub explicit_break_before: bool,
    pub spacing_bbox: Option<Box4>,
    #[serde(default = "zero_angle")]
    pub writing_angle: Option<i32>,
    #[serde(default)]
    pub advance_bbox: Option<Box4>,
    #[serde(default)]
    pub origin: Option<[f64; 2]>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Token {
    pub text: String,
    pub bbox: Box4,
    pub glyph_ids: Vec<usize>,
    pub source_char_indices: Vec<i64>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct Row {
    pub row_index: i64,
    pub bbox: Box4,
    pub tokens: Vec<Token>,
    pub glyph_ids: Vec<usize>,
}
#[derive(Serialize, Deserialize)]
pub struct Text {
    pub glyphs: Vec<Glyph>,
    pub rows: Vec<Row>,
    pub median_glyph_width: f64,
    pub median_glyph_height: f64,
}

pub(super) fn numeric_string(s: &str) -> Result<Option<String>, Error> {
    if s.len() > rapiddoc_core::resources::limit(256) {
        return Err(limit());
    }
    let s = s.trim_matches(|c| python_space(c) && !matches!(c, '\u{1c}'..='\u{1f}'));
    let mapped: String = s
        .chars()
        .map(|c| python_decimal(c).map_or(c, |n| (b'0' + n) as char))
        .collect();
    let bytes = mapped.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        if b == b'_'
            && (i == 0
                || i + 1 == bytes.len()
                || !bytes[i - 1].is_ascii_digit()
                || !bytes[i + 1].is_ascii_digit())
        {
            return Ok(None);
        }
    }
    Ok(Some(mapped.replace('_', "")))
}
pub(super) fn number(v: &Value) -> Result<Option<f64>, Error> {
    let n = if let Some(b) = v.as_bool() {
        Some(u8::from(b) as f64)
    } else if let Some(s) = v.as_str() {
        numeric_string(s)?.and_then(|s| s.parse::<f64>().ok())
    } else {
        v.as_f64()
    };
    if n.is_some_and(|x| !x.is_finite() || x.abs() > 1e12) {
        return Err(limit());
    }
    Ok(n)
}
pub(super) fn source_index(c: &Value, fallback: usize) -> Result<i64, Error> {
    let Some(v) = c.get("char_idx") else {
        return Ok(fallback as i64);
    };
    let parsed = if let Some(s) = v.as_str() {
        numeric_string(s)?.and_then(|s| s.parse::<i64>().ok())
    } else {
        number(v)?.map(|x| x.trunc() as i64)
    };
    let value = parsed.unwrap_or(fallback as i64);
    if value.unsigned_abs() > rapiddoc_core::resources::limit(1_000_000_000_000) {
        return Err(limit());
    }
    Ok(value)
}
pub(super) fn normalize_bbox(v: &Value) -> Result<Option<Box4>, Error> {
    let Some(a) = v.as_array().filter(|a| a.len() == 4) else {
        return Ok(None);
    };
    let mut b = [0.; 4];
    for (i, x) in a.iter().enumerate() {
        let Some(n) = number(x)? else { return Ok(None) };
        if n.abs() > 1e9 {
            return Err(limit());
        }
        b[i] = n;
    }
    if b[2] < b[0] {
        b.swap(0, 2);
    }
    if b[3] < b[1] {
        b.swap(1, 3);
    }
    Ok((b[2] > b[0] && b[3] > b[1]).then_some(b))
}
pub(super) fn angle(v: &Value) -> Result<i32, Error> {
    let a = number(v)?.unwrap_or(0.).trunc().rem_euclid(360.) as i32;
    Ok(if matches!(a, 0 | 90 | 180 | 270) {
        a
    } else {
        0
    })
}
pub(super) fn local(b: Box4, t: Box4, a: i32) -> Option<Box4> {
    let c = [
        b[0].max(t[0]),
        b[1].max(t[1]),
        b[2].min(t[2]),
        b[3].min(t[3]),
    ];
    if c[2] <= c[0] || c[3] <= c[1] {
        return None;
    }
    let [x, y, r, d] = [c[0] - t[0], c[1] - t[1], c[2] - t[0], c[3] - t[1]];
    let (w, h) = (t[2] - t[0], t[3] - t[1]);
    Some(match a {
        90 => [y, w - r, d, w - x],
        180 => [w - r, h - d, w - x, h - y],
        270 => [h - d, x, h - y, r],
        _ => [x, y, r, d],
    })
}
fn cjk(s: &str) -> bool {
    s.chars().any(|c| ('\u{3400}'..='\u{9fff}').contains(&c))
}
pub fn normalize(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        out.push_str(match c {
            '\u{2018}' | '\u{2019}' | '\u{201a}' | '\u{2032}' => "'",
            '\u{201c}' | '\u{201d}' | '\u{201e}' | '\u{2033}' => "\"",
            '\u{2010}'..='\u{2015}' | '\u{2212}' | '\u{0002}' => "-",
            '\u{00a0}' => " ",
            '\u{fb00}' => "ff",
            '\u{fb01}' => "fi",
            '\u{fb02}' => "fl",
            '\u{fb03}' => "ffi",
            '\u{fb04}' => "ffl",
            '\u{fb05}' => "ft",
            '\u{fb06}' => "st",
            _ => {
                out.push(c);
                continue;
            }
        });
    }
    out
}
fn chars(v: &Value) -> Result<&Vec<Value>, Error> {
    v["chars"]
        .as_array()
        .filter(|a| a.len() <= crate::legacy_page::budget::records(20000))
        .ok_or_else(invalid)
}
fn select(v: &Value, t: Box4, a: i32) -> Result<Vec<Pending>, Error> {
    let mut selected = Vec::new();
    for (i, c) in chars(v)?.iter().enumerate() {
        let Some(b) = normalize_bbox(&c["bbox"])? else {
            continue;
        };
        let (cx, cy) = ((b[0] + b[2]) / 2., (b[1] + b[3]) / 2.);
        if t[0] <= cx && cx <= t[2] && t[1] <= cy && cy <= t[3] {
            selected.push((source_index(c, i)?, i, b));
        }
    }
    selected.sort_by_key(|(index, i, _)| (*index, *i));
    let mut output = Vec::new();
    let (mut space, mut line_break) = (false, false);
    for (index, i, b) in selected {
        let c = &v["chars"][i];
        let raw = if c["char"].is_null() {
            ""
        } else {
            c["char"].as_str().ok_or_else(invalid)?
        };
        if raw.is_empty() {
            continue;
        }
        if matches!(raw, "\r" | "\n") {
            line_break = true;
            space = false;
            continue;
        }
        if raw.chars().all(python_space) {
            if !line_break {
                space = true;
            }
            continue;
        }
        let text = normalize(raw);
        if text.is_empty() || text.chars().all(python_space) {
            continue;
        }
        let Some(bbox) = local(b, t, a) else { continue };
        if bbox[3] - bbox[1] < 0.5 && !cjk(&text) {
            continue;
        }
        if output.len() >= crate::legacy_page::budget::records(3000) {
            return Err(limit());
        }
        let (writing_angle, advance_bbox, origin) = rotation::metadata(c, t, a)?;
        output.push(Pending {
            glyph_id: output.len(),
            source_index: index,
            text,
            bbox,
            explicit_space_before: space,
            explicit_break_before: line_break,
            writing_angle,
            advance_bbox,
            origin,
        });
        space = false;
        line_break = false;
    }
    Ok(output)
}
pub fn pending(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let out = match normalize_bbox(&v["table_bbox"])? {
        None => Vec::new(),
        Some(t) => select(v, t, angle(&v["angle"])?)?,
    };
    serde_json::to_value(out).map_err(|_| invalid())
}
fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 1.;
    }
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2]
    } else {
        (values[n / 2 - 1] + values[n / 2]) / 2.
    }
}
fn union<'a>(boxes: impl Iterator<Item = &'a Box4>) -> Box4 {
    boxes
        .copied()
        .reduce(|a, b| {
            [
                a[0].min(b[0]),
                a[1].min(b[1]),
                a[2].max(b[2]),
                a[3].max(b[3]),
            ]
        })
        .unwrap()
}
fn glyph_parts(glyphs: &[Glyph], height: f64, use_advance: bool) -> Vec<(String, Option<i64>)> {
    let Some(first) = glyphs.first() else {
        return Vec::new();
    };
    let mut out = vec![(first.text.clone(), Some(first.source_index))];
    let mut previous = first;
    for g in &glyphs[1..] {
        let (mut gap, mut spacing_height) = (g.bbox[0] - previous.bbox[2], height);
        if use_advance {
            let current = g.advance_bbox.unwrap_or(g.bbox);
            let prior = previous.advance_bbox.unwrap_or(previous.bbox);
            gap = current[0] - prior[2];
        } else if let (Some(p), Some(c)) = (previous.spacing_bbox, g.spacing_bbox) {
            gap = c[0] - p[2];
            spacing_height = (p[3] - p[1]).min(c[3] - c[1]);
        }
        let tail = previous.text.chars().last();
        let head = g.text.chars().next();
        let needs = g.explicit_space_before
            || (!g.explicit_break_before
                && gap > 0.6f64.max(spacing_height * 0.20)
                && !tail.is_some_and(|c| ('\u{3400}'..='\u{9fff}').contains(&c))
                && !head.is_some_and(|c| ('\u{3400}'..='\u{9fff}').contains(&c))
                && tail.is_some_and(python_printable)
                && head.is_some_and(python_printable));
        if needs && !out.last().unwrap().0.ends_with(' ') {
            out.push((" ".into(), None));
        }
        out.push((g.text.clone(), Some(g.source_index)));
        previous = g;
    }
    out
}
fn line_parts(glyphs: &[Glyph], height: f64) -> String {
    glyph_parts(glyphs, height, false)
        .into_iter()
        .map(|(s, _)| s)
        .collect()
}

fn ascii_word_run(s: &str, minimum: usize) -> bool {
    let mut n = 0;
    for c in s.bytes() {
        if c.is_ascii_alphabetic() {
            n += 1;
            if n >= minimum {
                return true;
            }
        } else {
            n = 0;
        }
    }
    false
}
pub fn row_separator(context: &str, previous: &str, next: &str) -> &'static str {
    use rapiddoc_native::{
        python_latin,
        text_boundary::{python_alpha, python_digit, resolve},
    };
    let before = previous.trim_end_matches(python_space);
    let after = next.trim_start_matches(python_space);
    if before.ends_with([',', ';'])
        && ascii_word_run(previous, 3)
        && after.chars().next().is_some_and(python_digit)
    {
        return " ";
    }
    let left = before.chars().rev().find(|c| python_alpha(*c));
    let right = after.chars().find(|c| python_alpha(*c));
    if !left.is_some_and(python_latin) || !right.is_some_and(python_latin) {
        return "";
    }
    let (_, separator) = resolve(context, next);
    if separator != " " {
        return separator;
    }
    let folded: String = after
        .chars()
        .flat_map(|c| {
            if c == 'ſ' {
                "s".chars().collect::<Vec<_>>()
            } else {
                c.to_lowercase().collect()
            }
        })
        .collect();
    let url = ["http://", "https://", "ftp://", "www."]
        .iter()
        .any(|p| folded.starts_with(p));
    let previous = previous.trim_matches(python_space);
    let next = next.trim_matches(python_space);
    let prefix = next.bytes().take_while(u8::is_ascii_alphabetic).count();
    let phrase = previous.len() >= 4
        && previous.bytes().all(|c| c.is_ascii_alphabetic())
        && prefix >= 4
        && next
            .as_bytes()
            .get(prefix)
            .is_some_and(|c| matches!(c, b'/' | b'%'));
    let compact = !phrase
        && previous
            .chars()
            .chain(next.chars())
            .all(|c| !python_space(c))
        && previous
            .chars()
            .chain(next.chars())
            .any(|c| python_digit(c) || "/_^%°µμ".contains(c));
    if !url && compact {
        ""
    } else {
        separator
    }
}
pub fn cell_parts(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    cell_parts_with_work(v, &mut Work(0))
}
pub(super) fn cell_parts_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    let glyphs: Vec<Glyph> = serde_json::from_value(v["glyphs"].clone()).map_err(|_| invalid())?;
    if glyphs.len() > crate::legacy_page::budget::records(3000)
        || glyphs.iter().any(|g| {
            g.source_index.unsigned_abs() > rapiddoc_core::resources::limit(1_000_000_000_000)
                || g.glyph_id > rapiddoc_core::resources::limit(1_000_000)
                || g.visual_row.unsigned_abs() > rapiddoc_core::resources::limit(10000)
                || g.spacing_bbox
                    .is_some_and(|b| b.iter().any(|x| !x.is_finite() || x.abs() > 1e9))
        })
    {
        return Err(limit());
    }
    let height = number(&v["median_height"])?
        .filter(|h| *h > 0. && *h <= 1e9)
        .ok_or_else(invalid)?;
    work.charge(glyphs.len() * glyphs.len() * 2)?;
    let use_advance = cell_writing_angle(&glyphs).is_some_and(|a| a != 0);
    let (glyphs, groups, height) = cell_text_rows(glyphs, height)?;
    let mut parts: Vec<(String, Option<i64>)> = Vec::new();
    let mut context = String::new();
    let mut previous = String::new();
    for group in &groups {
        let mut row: Vec<_> = group.iter().map(|i| glyphs[*i].clone()).collect();
        row.sort_by(|a, b| {
            a.bbox[0]
                .partial_cmp(&b.bbox[0])
                .unwrap()
                .then(a.bbox[1].partial_cmp(&b.bbox[1]).unwrap())
                .then(a.glyph_id.cmp(&b.glyph_id))
        });
        let row_parts = glyph_parts(&row, height, use_advance);
        let row_text: String = row_parts.iter().map(|(s, _)| s.as_str()).collect();
        if row_text.is_empty() {
            continue;
        }
        if !parts.is_empty() {
            work.charge(context.len() + row_text.len() + previous.len())?;
            let separator = row_separator(&context, &previous, &row_text);
            if !separator.is_empty() && !context.ends_with(' ') && !row_text.starts_with(' ') {
                parts.push((separator.into(), None));
                context.push_str(separator);
            }
        }
        parts.extend(row_parts);
        context.push_str(&row_text);
        previous = row_text;
        if context.len() > rapiddoc_core::resources::limit(1_000_000) {
            return Err(limit());
        }
    }
    Ok(serde_json::json!({"parts":parts,"text":context,"groups":groups}))
}
pub(super) fn tokens(glyphs: &[Glyph], width: f64, height: f64) -> Vec<Token> {
    if glyphs.is_empty() {
        return Vec::new();
    }
    let gap = (2. * width).max(0.70 * height).max(2.);
    let mut groups = Vec::new();
    let mut begin = 0;
    for i in 1..glyphs.len() {
        if glyphs[i].bbox[0] - glyphs[i - 1].bbox[2] > gap {
            groups.push(&glyphs[begin..i]);
            begin = i;
        }
    }
    groups.push(&glyphs[begin..]);
    groups
        .into_iter()
        .filter_map(|g| {
            let text = line_parts(g, height);
            (!text.is_empty()).then(|| Token {
                text,
                bbox: union(g.iter().map(|g| &g.bbox)),
                glyph_ids: g.iter().map(|g| g.glyph_id).collect(),
                source_char_indices: g.iter().map(|g| g.source_index).collect(),
            })
        })
        .collect()
}
pub fn prepare(v: &Value) -> Result<Option<Text>, Error> {
    prepare_with_work(v, &mut Work(0))
}
pub(super) fn prepare_with_work(v: &Value, work: &mut Work) -> Result<Option<Text>, Error> {
    validate_tree(v)?;
    let Some(t) = normalize_bbox(&v["table_bbox"])? else {
        return Ok(None);
    };
    let a = angle(&v["angle"])?;
    let pending = select(v, t, a)?;
    if pending.is_empty() {
        return Ok(None);
    }
    // 旧代码只在字形选择时归一化table_bbox；advance裁剪仍使用原四坐标。
    let mut original_table = t;
    for (i, x) in v["table_bbox"].as_array().unwrap().iter().enumerate() {
        original_table[i] = number(x)?.ok_or_else(invalid)?;
    }
    let width = median(
        pending
            .iter()
            .filter_map(|g| (g.bbox[2] > g.bbox[0]).then_some(g.bbox[2] - g.bbox[0]))
            .collect(),
    )
    .max(0.1);
    let height = median(
        pending
            .iter()
            .filter_map(|g| (g.bbox[3] > g.bbox[1]).then_some(g.bbox[3] - g.bbox[1]))
            .collect(),
    )
    .max(0.1);
    let selected: BTreeSet<_> = pending.iter().map(|g| g.source_index).collect();
    let mut spacing = BTreeMap::new();
    for (i, c) in chars(v)?.iter().enumerate() {
        let index = source_index(c, i)?;
        if !selected.contains(&index) {
            continue;
        }
        if let Some(b) = normalize_bbox(&c["loose_bbox"])? {
            spacing.insert(index, local(b, original_table, a));
        }
    }
    let row_boxes = pending
        .iter()
        .map(|g| {
            if let Some(Some(s)) = spacing.get(&g.source_index) {
                if matches!(g.text.as_str(), "." | "," | ";" | ":")
                    && g.bbox[3] - g.bbox[1] < 0.6 * height
                {
                    return [g.bbox[0], s[1], g.bbox[2], s[3]];
                }
            }
            g.bbox
        })
        .collect();
    work.charge(pending.len() * pending.len() * 2)?;
    let kernel = if crate::legacy_page::budget::current().host {
        rapiddoc_native::table_visual_rows_page
    } else {
        rapiddoc_native::table_visual_rows
    };
    let rows = kernel(
        row_boxes,
        pending.iter().map(|g| g.glyph_id).collect(),
        height,
    )?;
    let (mut glyphs, mut result) = (Vec::new(), Vec::new());
    for (row_index, row) in rows.into_iter().enumerate() {
        let values: Vec<_> = row
            .into_iter()
            .map(|i| {
                let g = &pending[i];
                Glyph {
                    glyph_id: g.glyph_id,
                    source_index: g.source_index,
                    text: g.text.clone(),
                    bbox: g.bbox,
                    visual_row: row_index as i64,
                    explicit_space_before: g.explicit_space_before,
                    explicit_break_before: g.explicit_break_before,
                    spacing_bbox: spacing.get(&g.source_index).copied().flatten(),
                    writing_angle: rotation::baseline(g),
                    advance_bbox: g.advance_bbox,
                    origin: g.origin,
                }
            })
            .collect();
        result.push(Row {
            row_index: row_index as i64,
            bbox: union(values.iter().map(|g| &g.bbox)),
            tokens: tokens(&values, width, height),
            glyph_ids: values.iter().map(|g| g.glyph_id).collect(),
        });
        glyphs.extend(values);
    }
    glyphs.sort_by_key(|g| g.glyph_id);
    Ok(Some(Text {
        glyphs,
        rows: result,
        median_glyph_width: width,
        median_glyph_height: height,
    }))
}
pub fn build(v: &Value) -> Result<Value, Error> {
    serde_json::to_value(prepare(v)?).map_err(|_| invalid())
}
