//! Port of the current local omml.py, including its malformed-input behavior.
//! Lookup data includes pylatexenc 2.10 per-scalar conversion, not an approximate
//! math-symbol subset. See crate licenses and office-native-provenance.json.
//! The local Python adaptation credits https://github.com/xiilei/dwml.
use crate::{Error, Result};
use roxmltree::Node;
use serde::Deserialize;
use std::{collections::BTreeMap, sync::OnceLock};

#[derive(Deserialize)]
struct Data {
    unicode: BTreeMap<char, String>,
    maps: BTreeMap<String, BTreeMap<String, String>>,
    templates: BTreeMap<String, String>,
}
fn data() -> &'static Data {
    static DATA: OnceLock<Data> = OnceLock::new();
    DATA.get_or_init(|| {
        serde_json::from_str(include_str!("omml_data.json")).expect("generated OMML data")
    })
}
fn template(key: &str) -> &'static str {
    &data().templates[key]
}
fn mapped<'a>(map: &str, value: Option<&'a str>, fallback: Option<&'a str>) -> Option<&'a str> {
    value
        .map(|v| data().maps[map].get(v).map(String::as_str).unwrap_or(v))
        .or(fallback)
}
pub fn math_namespace(ns: Option<&str>) -> bool {
    crate::package::namespace_family(ns, "officeDocument/math")
}
fn math(node: Node<'_, '_>) -> bool {
    node.is_element() && math_namespace(node.tag_name().namespace())
}
fn attr<'a>(node: Node<'a, '_>) -> Option<&'a str> {
    node.attributes()
        .find(|a| a.name() == "val" && math_namespace(a.namespace()))
        .map(|a| a.value())
}
fn child<'a, 'i>(node: Node<'a, 'i>, name: &str) -> Option<Node<'a, 'i>> {
    node.children()
        .find(|n| math(*n) && n.tag_name().name() == name)
}
pub(super) fn whitespace(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}
#[derive(Debug)]
struct Properties {
    values: BTreeMap<String, Option<String>>,
    text: String,
}
impl Properties {
    fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).and_then(|s| s.as_deref())
    }
}
#[derive(Debug)]
enum Value {
    Text(String),
    Properties(Properties),
}
impl Value {
    fn string(&self) -> &str {
        match self {
            Self::Text(s) => s,
            Self::Properties(p) => &p.text,
        }
    }
    fn property(&self) -> Result<&Properties> {
        match self {
            Self::Properties(p) => Ok(p),
            _ => Err(Error::malformed("OMML property object required")),
        }
    }
    fn as_text(&self) -> Result<&str> {
        match self {
            Self::Text(s) => Ok(s),
            _ => Err(Error::malformed("OMML string required")),
        }
    }
}

/// Raw .latex property of oMath2Latex, before its final double-space collapse.
pub fn raw(node: Node<'_, '_>) -> Result<String> {
    children(node, None).map(|c| joined(&c))
}
/// str(oMath2Latex(node)); caller decides whether to strip surrounding space.
pub fn convert(node: Node<'_, '_>) -> Result<String> {
    Ok(raw(node)?.replace("  ", " "))
}
fn properties(node: Node<'_, '_>) -> Properties {
    let mut value = Properties {
        values: BTreeMap::new(),
        text: String::new(),
    };
    for n in node.children().filter(|n| math(*n)) {
        let tag = n.tag_name().name();
        if tag == "brk" {
            value.text.push_str(template("BRK"));
        } else if matches!(tag, "chr" | "pos" | "begChr" | "endChr" | "type") {
            value.values.insert(tag.into(), attr(n).map(str::to_owned));
        }
    }
    value
}
fn children(node: Node<'_, '_>, include: Option<&[&str]>) -> Result<Vec<(String, Value)>> {
    let mut out = Vec::new();
    for n in node.children().filter(|n| math(*n)) {
        let tag = n.tag_name().name();
        if include.is_some_and(|names| !names.contains(&tag)) {
            continue;
        }
        if let Some(value) = process(n)? {
            out.push((tag.into(), value));
        }
    }
    Ok(out)
}
fn joined(values: &[(String, Value)]) -> String {
    values.iter().map(|(_, v)| v.string()).collect()
}
fn strict_join(values: impl IntoIterator<Item = Value>, separator: &str) -> Result<String> {
    let mut output = Vec::new();
    for value in values {
        output.push(value.as_text()?.to_owned());
    }
    Ok(output.join(separator))
}
fn dict(node: Node<'_, '_>, include: Option<&[&str]>) -> Result<BTreeMap<String, Value>> {
    Ok(children(node, include)?.into_iter().collect())
}
fn required<'a>(values: &'a BTreeMap<String, Value>, key: &str) -> Result<&'a Value> {
    values
        .get(key)
        .ok_or_else(|| Error::malformed(format!("missing OMML {key}")))
}
fn optional(values: &BTreeMap<String, Value>, key: &str) -> String {
    values
        .get(key)
        .map(|v| v.string().to_owned())
        .unwrap_or_else(|| "None".into())
}
// Python string format on the trusted templates and raw property fallback:
// escaped braces are literals; missing fields and malformed braces are errors.
fn format(input: &str, values: &[(&str, &str)]) -> Result<String> {
    let mut out = String::new();
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
                out.push('{');
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
                out.push('}');
            }
            '{' => {
                let mut key = String::new();
                let mut closed = false;
                for c in chars.by_ref() {
                    if c == '}' {
                        closed = true;
                        break;
                    }
                    key.push(c);
                }
                if !closed {
                    return Err(Error::malformed("unclosed OMML format field"));
                }
                let key = if key.is_empty() { "0" } else { &key };
                out.push_str(
                    values
                        .iter()
                        .find(|(k, _)| *k == key)
                        .map(|(_, v)| *v)
                        .ok_or_else(|| Error::malformed("unknown OMML format field"))?,
                );
            }
            '}' => return Err(Error::malformed("unmatched OMML format brace")),
            _ => out.push(c),
        }
    }
    Ok(out)
}
pub fn unicode(c: char) -> String {
    data()
        .unicode
        .get(&c)
        .cloned()
        .unwrap_or_else(|| c.to_string())
}
fn escape(input: &str) -> String {
    let input = input.replace("\\\\", "\\");
    let mut previous = None;
    let mut out = String::new();
    for c in input.chars() {
        if matches!(c, '{' | '}' | '_' | '^' | '#' | '&' | '$' | '%') && previous != Some('\\') {
            out.push('\\');
        }
        out.push(c);
        previous = Some(c);
    }
    out
}
fn run(node: Node<'_, '_>) -> Result<String> {
    let base = child(node, "t").and_then(|n| n.text()).unwrap_or("");
    let mut text = escape(&base.chars().map(unicode).collect::<String>());
    if !base.contains('{') && text.contains("\\{") {
        text = text.replace("\\{", "{");
    }
    if !base.contains('}') && text.contains("\\}") {
        text = text.replace("\\}", "}");
    }
    if let Some(script) = child(node, "rPr")
        .and_then(|n| child(n, "scr"))
        .and_then(attr)
    {
        if let Some(t) = data().maps["SCR_TO_LATEX"].get(script) {
            if !text.trim_matches(whitespace).is_empty() {
                text = format(t, &[("0", text.trim_matches(whitespace))])?;
            }
        }
    }
    Ok(text)
}
fn limit(base: &str, limit: &str, upper: bool) -> Result<String> {
    if let Some(t) = data().maps["CHR"].get(limit).filter(|s| s.contains("{0}")) {
        return format(t, &[("0", base)]);
    }
    let group = if upper {
        ["\\overbrace{", "\\overbracket{", "\\overparen{"]
    } else {
        ["\\underbrace{", "\\underbracket{", "\\underparen{"]
    };
    if !upper {
        if let Some(t) = data().maps["LIM_FUNC"].get(base) {
            return format(t, &[("lim", limit)]);
        }
    }
    if group
        .iter()
        .any(|s| base.trim_start_matches(whitespace).starts_with(s))
    {
        return Ok(format!(
            "{base}{}",
            format(template(if upper { "SUP" } else { "SUB" }), &[("0", limit)])?
        ));
    }
    if upper {
        format(template("LIM_UPP"), &[("lim", limit), ("text", base)])
    } else {
        Ok(format!("\\underset{{{limit}}}{{{base}}}"))
    }
}
fn process(node: Node<'_, '_>) -> Result<Option<Value>> {
    let tag = node.tag_name().name();
    let output = match tag {
        "r" => run(node)?,
        "box" | "sSub" | "sSup" | "sSubSup" | "num" | "den" | "deg" | "e" => {
            joined(&children(node, None)?)
        }
        "sub" | "sup" => format(
            template(if tag == "sub" { "SUB" } else { "SUP" }),
            &[("0", &joined(&children(node, None)?))],
        )?,
        "acc" | "bar" | "d" | "groupChr" => {
            let values = dict(node, None)?;
            let p = required(&values, &format!("{tag}Pr"))?.property()?;
            let e = required(&values, "e")?.string();
            match tag {
                "acc" => format(
                    mapped(
                        "CHR",
                        p.get("chr"),
                        Some(&data().maps["CHR_DEFAULT"]["ACC_VAL"]),
                    )
                    .unwrap(),
                    &[("0", e)],
                )?,
                "bar" => format!(
                    "{}{}",
                    p.text,
                    format(
                        mapped(
                            "POS",
                            p.get("pos"),
                            Some(&data().maps["POS_DEFAULT"]["BAR_VAL"])
                        )
                        .unwrap(),
                        &[("0", e)]
                    )?
                ),
                "groupChr" => format!(
                    "{}{}",
                    p.text,
                    format(
                        mapped("CHR", p.get("chr"), None)
                            .ok_or_else(|| Error::malformed("missing group character"))?,
                        &[("0", e)]
                    )?
                ),
                _ => {
                    let left = mapped(
                        "T",
                        p.get("begChr"),
                        Some(&data().maps["D_DEFAULT"]["left"]),
                    )
                    .unwrap();
                    let right = mapped(
                        "T",
                        p.get("endChr"),
                        Some(&data().maps["D_DEFAULT"]["right"]),
                    )
                    .unwrap();
                    let norm = |s: &str| {
                        if s.is_empty() {
                            data().maps["D_DEFAULT"]["null"].clone()
                        } else {
                            escape(if matches!(s, "∥" | "‖") { "\\|" } else { s })
                        }
                    };
                    format!(
                        "{}{}",
                        p.text,
                        format(
                            template("D"),
                            &[("left", &norm(left)), ("right", &norm(right)), ("text", e)]
                        )?
                    )
                }
            }
        }
        "f" => {
            let values = dict(node, None)?;
            let num = optional(&values, "num");
            let den = optional(&values, "den");
            if let Some(p) = values.get("fPr") {
                let p = p.property()?;
                format!(
                    "{}{}",
                    p.text,
                    format(
                        mapped("F", p.get("type"), Some(template("F_DEFAULT"))).unwrap(),
                        &[("num", &num), ("den", &den)]
                    )?
                )
            } else {
                format(template("F_DEFAULT"), &[("num", &num), ("den", &den)])?
            }
        }
        "func" => {
            let values = dict(node, None)?;
            let name = required(&values, "fName")?.as_text()?;
            name.replace(template("FUNC_PLACE"), required(&values, "e")?.as_text()?)
        }
        "fName" => {
            let mut output = String::new();
            for (name, value) in children(node, None)? {
                if let Value::Text(value) = value {
                    output.push_str(if name == "r" {
                        data().maps["FUNC"]
                            .get(&value)
                            .map(String::as_str)
                            .unwrap_or(&value)
                    } else {
                        &value
                    });
                }
            }
            if !output.contains(template("FUNC_PLACE")) {
                output.push_str(template("FUNC_PLACE"));
            }
            output
        }
        "rad" => {
            let values = dict(node, None)?;
            let text = optional(&values, "e");
            if let Some(degree) = values.get("deg").filter(|v| !v.string().is_empty()) {
                format(
                    template("RAD"),
                    &[("deg", degree.string()), ("text", &text)],
                )?
            } else {
                format(template("RAD_DEFAULT"), &[("text", &text)])?
            }
        }
        "eqArr" => {
            let rows = children(node, Some(&["e"]))?;
            if rows.len() == 1 {
                let row = rows[0].1.as_text()?;
                static TAG: OnceLock<regex::Regex> = OnceLock::new();
                let regex = TAG.get_or_init(|| {
                    regex::Regex::new(r"\\#[\s\x1c-\x1f]*\(([^)]*)\)[\s\x1c-\x1f]*$").unwrap()
                });
                if let Some(c) = regex.captures(row) {
                    format!(
                        "{}\\tag{{{}}}",
                        row[..c.get(0).unwrap().start()].trim_end_matches(whitespace),
                        &c[1]
                    )
                } else {
                    row.into()
                }
            } else {
                format(
                    template("ARR"),
                    &[(
                        "text",
                        &strict_join(rows.into_iter().map(|(_, v)| v), template("BRK"))?,
                    )],
                )?
            }
        }
        "limLow" | "limUpp" => {
            let values = dict(node, Some(&["e", "lim"]))?;
            limit(
                values.get("e").map(Value::string).unwrap_or(""),
                values.get("lim").map(Value::string).unwrap_or(""),
                tag == "limUpp",
            )?
        }
        "lim" => joined(&children(node, None)?).replace("\\rightarrow", "\\to"),
        "m" => {
            let rows = children(node, None)?
                .into_iter()
                .filter(|(tag, _)| tag == "mr")
                .map(|(_, v)| v);
            format(
                template("M"),
                &[("text", &strict_join(rows, template("BRK"))?)],
            )?
        }
        "mr" => strict_join(
            children(node, Some(&["e"]))?.into_iter().map(|(_, v)| v),
            template("ALN"),
        )?,
        "nary" => {
            let mut operator = String::new();
            let mut values = Vec::new();
            for (tag, value) in children(node, None)? {
                if tag == "naryPr" {
                    operator = mapped("CHR_BO", value.property()?.get("chr"), Some("\\int"))
                        .unwrap()
                        .into();
                } else {
                    values.push(value);
                }
            }
            operator + &strict_join(values, "")?
        }
        _ if tag.ends_with("Pr") => return Ok(Some(Value::Properties(properties(node)))),
        _ => return Ok(None),
    };
    Ok(Some(Value::Text(output)))
}
