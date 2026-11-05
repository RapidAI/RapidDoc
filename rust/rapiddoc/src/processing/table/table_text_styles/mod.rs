//! 旧_render_styled_cell：给定稳定来源角色的单元格片段渲染；不代替整表自动角色分类。
pub mod clusters;
pub mod lanes;
pub mod materialization;
pub mod preparation;
use super::{table_native_multiline::spatial, table_native_text};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
use std::collections::BTreeSet;
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
pub(super) fn numeric(v: Option<&Value>) -> Option<f64> {
    let v = v?;
    let number = if let Some(b) = v.as_bool() {
        f64::from(u8::from(b))
    } else if let Some(n) = v.as_f64() {
        n
    } else {
        table_native_text::numeric_string(v.as_str()?)
            .ok()
            .flatten()?
            .parse()
            .ok()?
    };
    number.is_finite().then_some(number)
}
fn flag(c: &Value) -> u64 {
    const MASK: u64 = (1 << 19) - 1;
    let v = &c["font"]["flags"];
    if let Some(n) = v.as_i64() {
        return n as u64 & MASK;
    }
    if let Some(n) = v.as_u64() {
        return n & MASK;
    }
    if let Some(b) = v.as_bool() {
        return u64::from(b);
    }
    if let Some(n) = v.as_f64() {
        return n.trunc().rem_euclid((MASK + 1) as f64) as u64;
    }
    let Some(text) = v
        .as_str()
        .and_then(|s| table_native_text::numeric_string(s).ok().flatten())
    else {
        return 0;
    };
    let (negative, digits) = if let Some(s) = text.strip_prefix('-') {
        (true, s)
    } else {
        (false, text.strip_prefix('+').unwrap_or(&text))
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return 0;
    }
    let mut n = 0u64;
    for b in digits.bytes() {
        n = (n * 10 + u64::from(b - b'0')) & MASK;
    }
    if negative {
        n.wrapping_neg() & MASK
    } else {
        n
    }
}
pub(super) fn font_styles(c: &Value) -> (bool, bool) {
    let raw = spatial::raw_string(c["font"].get("name"), true);
    let bytes = raw.as_bytes();
    let raw =
        if bytes.len() >= 7 && bytes[..6].iter().all(u8::is_ascii_uppercase) && bytes[6] == b'+' {
            &raw[7..]
        } else {
            &raw
        };
    // Python re.IGNORECASE 对ASCII字母还接受这四个Unicode变体，不做全串NFKC。
    let mut name = String::with_capacity(raw.len());
    for c in raw.chars() {
        match c {
            'İ' | 'ı' => name.push('i'),
            'ſ' => name.push('s'),
            'K' => name.push('k'),
            _ => name.extend(c.to_lowercase()),
        }
    }
    let terminal = name.strip_suffix('\n').unwrap_or(&name);
    let suffix = terminal.rsplit(['-', '_', ',']).next().unwrap();
    let marker = matches!(
        suffix,
        "bd" | "bi" | "bdmt" | "bimt" | "w6" | "w7" | "w8" | "w9"
    ) || terminal.ends_with("gbi");
    let bold = flag(c) & (1 << 18) != 0
        || numeric(c["font"].get("weight")).is_some_and(|w| w >= 600.)
        || ["bold", "demi", "black", "heavy"]
            .iter()
            .any(|s| name.contains(s))
        || marker;
    (bold, flag(c) & (1 << 6) != 0)
}
fn source_value(map: &Value, source: Option<i64>) -> Option<&Value> {
    map.as_object()?.get(&source?.to_string())
}
fn spacing_char(c: &Value, tight: &Value, origins: &Value) -> Value {
    let mut c = c.clone();
    let index = c["char_idx"]
        .as_i64()
        .or_else(|| c["char_idx"].as_bool().map(i64::from));
    if let Some(b) = source_value(tight, index) {
        c["tight_bbox"] = b.clone();
    }
    if let Some(p) = source_value(origins, index) {
        c["origin"] = p.clone();
    }
    c
}
fn flush(parts: &mut String, role: &str, styles: (bool, bool), output: &mut String) {
    if parts.is_empty() {
        return;
    }
    let mut text = escape(parts);
    if styles.1 {
        text = format!("<i>{text}</i>");
    }
    if styles.0 {
        text = format!("<b>{text}</b>");
    }
    match role {
        "sup" => output.push_str(&format!("<sup>{text}</sup>")),
        "sub" => output.push_str(&format!("<sub>{text}</sub>")),
        _ => output.push_str(&text),
    }
    parts.clear();
}
pub fn refine_tokens(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    rapiddoc_native::script_geometry::refine::refine(v)
}
pub fn classify_runs(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    rapiddoc_native::script_geometry::refine::classify_runs(v)
}
pub fn classify_line(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    rapiddoc_native::script_geometry::refine::classify_line(v)
}
pub fn fraction_sources(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    rapiddoc_native::script_geometry::refine::fraction_members(v)
}
pub fn classify_geometry(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    rapiddoc_native::script_geometry::classify(v)
}
/// Full-page side maps contain unrelated characters. Cell kernels only look up
/// char_idx values present in their input; keep those values, including malformed
/// entries, so their original validation and error behavior remain unchanged.
fn cell_side_map(map: &Value, chars: &[Value]) -> Value {
    if !crate::legacy_page::budget::current().host || !map.is_object() {
        return map.clone();
    }
    let mut selected = serde_json::Map::new();
    for c in chars {
        if let Some(id) = c["char_idx"]
            .as_i64()
            .or_else(|| c["char_idx"].as_bool().map(i64::from))
        {
            let key = id.to_string();
            if let Some(value) = map.get(&key) {
                selected.insert(key, value.clone());
            }
        }
    }
    Value::Object(selected)
}
pub fn numeric_indices(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    rapiddoc_native::script_geometry::numeric_indices(v)
}
pub fn render_cell(v: &Value) -> Result<String, Error> {
    validate_tree(v)?;
    render_cell_with_work(v, &mut Work(0))
}
fn render_cell_with_work(v: &Value, work: &mut Work) -> Result<String, Error> {
    render_cell_parts_with_work(
        CellRender {
            cell: &v["cell"],
            glyphs: &v["glyphs"],
            median_height: &v["median_height"],
            chars: &v["chars_by_source"],
            roles: &v["roles"],
            tight: &v["tight_bboxes"],
            origins: &v["origins"],
            preserve: v.get("preserve_font_styles"),
            possible: v.get("possible_space_indices"),
        },
        work,
    )
}
// Borrow page-wide maps; materializing a JSON request for each cell deep-copies
// every unrelated character. The public JSON adapter and validation stay intact.
struct CellRender<'a> {
    cell: &'a Value,
    glyphs: &'a Value,
    median_height: &'a Value,
    chars: &'a Value,
    roles: &'a Value,
    tight: &'a Value,
    origins: &'a Value,
    preserve: Option<&'a Value>,
    possible: Option<&'a Value>,
}
fn render_cell_parts_with_work(v: CellRender<'_>, work: &mut Work) -> Result<String, Error> {
    let glyphs = v
        .glyphs
        .as_array()
        .filter(|g| g.len() <= rapiddoc_core::resources::limit(1500))
        .ok_or_else(limit)?;
    let content = v.cell["content"].as_str().ok_or_else(invalid)?;
    let preserve = match v.preserve {
        None => false,
        Some(b) => b.as_bool().ok_or_else(invalid)?,
    };
    let parts = table_native_text::cell_parts_with_work(
        &json!({"glyphs":glyphs,"median_height":v.median_height}),
        work,
    )?;
    if parts["text"].as_str() != Some(content) {
        let escaped = escape(content);
        if escaped.len() > rapiddoc_core::resources::limit(4_000_000) {
            return Err(limit());
        }
        return Ok(escaped);
    }
    let chars = v.chars;
    let roles = v.roles;
    for map in [chars, roles, v.tight, v.origins] {
        if !map.is_null() && !map.is_object() {
            return Err(invalid());
        }
    }
    let possible = match v.possible {
        None | Some(Value::Null) => None,
        Some(a) => Some(
            a.as_array()
                .filter(|a| a.len() <= rapiddoc_core::resources::limit(20000))
                .ok_or_else(limit)?
                .iter()
                .map(|n| n.as_i64().ok_or_else(invalid))
                .collect::<Result<BTreeSet<_>, _>>()?,
        ),
    };
    let (mut output, mut active_parts) = (String::new(), String::new());
    let mut active_role = "body";
    let mut active_styles = (false, false);
    let mut previous = None;
    for part in parts["parts"].as_array().ok_or_else(invalid)? {
        let text = part[0].as_str().ok_or_else(invalid)?;
        let source = part[1].as_i64();
        let role = source_value(roles, source).map_or(Ok("body"), |r| {
            r.as_str()
                .filter(|s| ["body", "sup", "sub"].contains(s))
                .ok_or_else(invalid)
        })?;
        let mut styles = if source.is_none() {
            active_styles
        } else {
            (false, false)
        };
        if preserve {
            if let Some(c) = source_value(chars, source) {
                styles = font_styles(c);
            }
        }
        if role == "body"
            && active_role == "body"
            && possible
                .as_ref()
                .is_none_or(|set| source.is_some_and(|s| set.contains(&s)))
        {
            if let (Some(left), Some(right)) =
                (source_value(chars, previous), source_value(chars, source))
            {
                let l = spacing_char(left, v.tight, v.origins);
                let r = spacing_char(right, v.tight, v.origins);
                if spatial::tight(&l, &r) {
                    active_parts.push(' ');
                }
            }
        }
        if role != active_role || styles != active_styles {
            flush(&mut active_parts, active_role, active_styles, &mut output);
            active_role = role;
            active_styles = styles;
        }
        active_parts.push_str(text);
        previous = source;
        if output.len() + active_parts.len() > rapiddoc_core::resources::limit(4_000_000) {
            return Err(limit());
        }
    }
    flush(&mut active_parts, active_role, active_styles, &mut output);
    if output.len() > rapiddoc_core::resources::limit(4_000_000) {
        return Err(limit());
    }
    Ok(output)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_styled_cell_html() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/table-styled-cell.json");
        let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                render_cell(&c["request"]).unwrap(),
                c["expected"].as_str().unwrap(),
                "{}",
                c["label"]
            );
        }
        let extra: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/table-style-metadata.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in extra["cases"].as_array().unwrap() {
            assert_eq!(
                render_cell(&c["request"]).unwrap(),
                c["expected"].as_str().unwrap(),
                "{}",
                c["label"]
            );
        }
    }
}
