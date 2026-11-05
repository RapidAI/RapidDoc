use super::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct Formatting {
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
    pub underline_style: String,
    pub emphasis: bool,
    pub strikethrough: bool,
    pub script: Script,
}
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Script {
    #[default]
    Baseline,
    Sub,
    Super,
}
impl Formatting {
    pub fn visible(&self) -> bool {
        self.underline || self.emphasis || self.strikethrough
    }
    pub fn nonvisible(&self) -> bool {
        self.bold || self.italic
    }
    pub fn for_text(&self, text: &str, preserve: bool) -> Option<Self> {
        if !text.trim_matches(space).is_empty() {
            return Some(self.clone());
        }
        let mut output = self.clone();
        if output.underline_style == "words" {
            output.underline = false;
            output.underline_style.clear();
        }
        if output.nonvisible() && !preserve {
            output.bold = false;
            output.italic = false;
        }
        if output.visible() || (preserve && output.nonvisible()) {
            Some(output)
        } else {
            None
        }
    }
}

/// Keep values lazy: unused malformed styles must not fail the document early.
#[derive(Default)]
struct FontProperties {
    values: [Option<Option<String>>; 4],
}
fn on_off(value: Option<&str>) -> Result<bool> {
    match value.unwrap_or("1") {
        "1" | "true" | "on" => Ok(true),
        "0" | "false" | "off" => Ok(false),
        _ => Err(Error::malformed("invalid Word on/off value")),
    }
}
fn underline(value: Option<&str>) -> Result<bool> {
    match value {
        Some("none") => Ok(false),
        Some(
            "single" | "words" | "double" | "dotted" | "thick" | "dash" | "dotDash" | "dotDotDash"
            | "wave" | "dottedHeavy" | "dashedHeavy" | "dashLong" | "dashLongHeavy"
            | "dotDashHeavy" | "dotDotDashHeavy" | "wavyHeavy" | "wavyDouble",
        ) => Ok(true),
        _ => Err(Error::malformed("invalid or missing Word underline value")),
    }
}
impl FontProperties {
    fn read(node: Option<Node<'_, '_>>) -> Self {
        Self {
            values: std::array::from_fn(|i| {
                node.and_then(|node| child(node, ["b", "i", "u", "strike"][i]))
                    .map(|n| attr(n, "val").map(str::to_owned))
            }),
        }
    }
    fn get(&self, index: usize) -> Result<Option<bool>> {
        if index == 2 && self.values[index] == Some(None) {
            return Ok(None);
        }
        self.values[index]
            .as_ref()
            .map(|value| {
                if index == 2 {
                    underline(value.as_deref())
                } else {
                    on_off(value.as_deref())
                }
            })
            .transpose()
    }
}
struct Style {
    id: Option<String>,
    name: Option<String>,
    kind: String,
    default: Option<String>,
    base: Option<String>,
    font: FontProperties,
    numbering: Option<NumberingProperties>,
    outline: Option<Option<i64>>,
    cache: [OnceLock<Result<Option<bool>>>; 4],
}
#[derive(Default)]
pub struct Styles {
    styles: Vec<Style>,
    ids: BTreeMap<String, usize>,
    defaults: [OnceLock<Result<Option<usize>>>; 2],
}
impl Styles {
    pub fn parse(xml: Option<&str>) -> Result<Self> {
        let mut output = Self::default();
        let Some(xml) = xml else {
            return Ok(output);
        };
        let document = crate::xml::parse(xml)?;
        for node in document
            .root_element()
            .children()
            .filter(|n| is_word(*n, "style"))
        {
            let ppr = child(node, "pPr");
            let id = attr(node, "styleId").map(str::to_owned);
            if let Some(id) = &id {
                output.ids.entry(id.clone()).or_insert(output.styles.len());
            }
            output.styles.push(Style {
                id,
                name: child(node, "name")
                    .and_then(|n| attr(n, "val"))
                    .map(ui_name),
                kind: attr(node, "type").unwrap_or("paragraph").to_owned(),
                default: attr(node, "default").map(str::to_owned),
                base: child(node, "basedOn")
                    .and_then(|n| attr(n, "val"))
                    .map(str::to_owned),
                font: FontProperties::read(child(node, "rPr")),
                numbering: ppr
                    .and_then(|p| child(p, "numPr"))
                    .map(NumberingProperties::read),
                outline: ppr
                    .and_then(|p| child(p, "outlineLvl"))
                    .map(|n| integer(attr(n, "val"))),
                cache: std::array::from_fn(|_| OnceLock::new()),
            });
        }
        Ok(output)
    }
    fn get(&self, id: Option<&str>, kind: &str) -> Result<Option<usize>> {
        if let Some(index) = id
            .filter(|id| !id.is_empty())
            .and_then(|id| self.ids.get(id))
            .copied()
        {
            if self.styles[index].kind == kind {
                return Ok(Some(index));
            }
        }
        self.defaults[usize::from(kind == "character")]
            .get_or_init(|| {
                let mut result = None;
                for (index, style) in self
                    .styles
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.kind == kind)
                {
                    if style
                        .default
                        .as_deref()
                        .map(|value| on_off(Some(value)))
                        .transpose()?
                        .unwrap_or(false)
                    {
                        result = Some(index);
                    }
                }
                Ok(result)
            })
            .clone()
    }

    fn paragraph_style(&self, paragraph: Node<'_, '_>) -> Result<Option<usize>> {
        self.get(
            child(paragraph, "pPr")
                .and_then(|p| child(p, "pStyle"))
                .and_then(|s| attr(s, "val")),
            "paragraph",
        )
    }
    fn chain(&self, start: Option<usize>) -> Vec<usize> {
        let mut output = Vec::new();
        let mut visited = BTreeSet::new();
        let mut current = start;
        while let Some(index) = current {
            if !visited.insert(index) {
                break;
            }
            output.push(index);
            current = self.styles[index]
                .base
                .as_ref()
                .and_then(|id| self.ids.get(id))
                .copied();
        }
        output
    }
    fn inherited(&self, start: Option<usize>, property: usize) -> Result<Option<bool>> {
        let Some(index) = start else {
            return Ok(None);
        };
        self.styles[index].cache[property]
            .get_or_init(|| {
                for index in self.chain(Some(index)) {
                    if let Some(value) = self.styles[index].font.get(property)? {
                        return Ok(Some(value));
                    }
                }
                Ok(None)
            })
            .clone()
    }
    pub fn format(&self, run: Node<'_, '_>, paragraph: Node<'_, '_>) -> Result<Formatting> {
        let rpr = child(run, "rPr");
        let direct = FontProperties::read(rpr);
        let mut values = [false; 4];
        // Resolve each property lazily in the same precedence and call order.
        for property in [0, 1, 3, 2] {
            if let Some(value) = direct.get(property)? {
                values[property] = value;
                continue;
            }
            let character = self.get(
                rpr.and_then(|p| child(p, "rStyle"))
                    .and_then(|s| attr(s, "val")),
                "character",
            )?;
            let hyperlink_style = character.is_some_and(|i| {
                self.styles[i].id.as_deref().unwrap_or("").to_lowercase() == "hyperlink"
                    || self.styles[i]
                        .name
                        .as_deref()
                        .unwrap_or("")
                        .to_lowercase()
                        .contains("hyperlink")
            });
            let inherited = if hyperlink_style {
                None
            } else {
                self.inherited(character, property)?
            };
            values[property] = match inherited {
                Some(value) => value,
                None => self
                    .inherited(self.paragraph_style(paragraph)?, property)?
                    .unwrap_or(false),
            };
        }
        let underline_style = rpr
            .and_then(|p| child(p, "u"))
            .map(|n| attr(n, "val").unwrap_or("single").to_owned())
            .unwrap_or_default();
        let emphasis = rpr
            .and_then(|p| child(p, "em"))
            .and_then(|n| attr(n, "val"))
            .is_some_and(|v| !v.is_empty() && v != "none");
        let script = match rpr.and_then(|p| child(p, "vertAlign")) {
            None => Script::Baseline,
            Some(node) => match attr(node, "val") {
                Some("subscript") => Script::Sub,
                Some("superscript") => Script::Super,
                Some(_) => Script::Baseline,
                None => return Err(Error::malformed("missing Word vertical alignment")),
            },
        };
        Ok(Formatting {
            bold: values[0],
            italic: values[1],
            underline: values[2],
            strikethrough: values[3],
            underline_style,
            emphasis,
            script,
        })
    }
    pub fn chain_ids(&self, paragraph: Node<'_, '_>) -> BTreeSet<String> {
        self.chain(self.paragraph_style(paragraph).ok().flatten())
            .into_iter()
            .filter_map(|i| self.styles[i].id.clone())
            .collect()
    }
    pub fn numbering(&self, paragraph: Node<'_, '_>) -> Result<Option<NumberingProperties>> {
        if let Some(properties) = child(paragraph, "pPr").and_then(|p| child(p, "numPr")) {
            return Ok(Some(NumberingProperties::read(properties)));
        }
        Ok(self
            .chain(self.paragraph_style(paragraph)?)
            .into_iter()
            .find_map(|i| self.styles[i].numbering.clone()))
    }
    pub fn outline(&self, paragraph: Node<'_, '_>) -> Result<Option<i64>> {
        if let Some(properties) = child(paragraph, "pPr").and_then(|p| child(p, "outlineLvl")) {
            return Ok(integer(attr(properties, "val")));
        }
        Ok(self
            .chain(self.paragraph_style(paragraph)?)
            .into_iter()
            .find_map(|i| self.styles[i].outline)
            .flatten())
    }
    pub fn label(&self, paragraph: Node<'_, '_>) -> Result<(String, Option<i64>)> {
        let Some(index) = self.paragraph_style(paragraph)? else {
            return Ok(("Normal".to_owned(), None));
        };
        let Some(label) = &self.styles[index].id else {
            return Ok(("Normal".to_owned(), None));
        };
        for index in self.chain(Some(index)) {
            let style = &self.styles[index];
            if let Some(id) = &style.id {
                let parts = id.split(':').collect::<Vec<_>>();
                if parts.len() == 2 {
                    return Ok((parts[0].to_owned(), integer(Some(parts[1]))));
                }
            }
            for candidate in [style.id.as_deref(), style.name.as_deref()]
                .into_iter()
                .flatten()
            {
                if candidate.to_lowercase().contains("heading") {
                    return Ok(heading(candidate));
                }
            }
        }
        if let Some(outline) = self.outline(paragraph)? {
            return Ok((
                "Heading".to_owned(),
                Some(
                    outline
                        .checked_add(1)
                        .ok_or_else(|| Error::malformed("outline level integer overflow"))?,
                ),
            ));
        }
        Ok((
            self.styles[index]
                .name
                .as_ref()
                .filter(|s| !s.is_empty())
                .unwrap_or(label)
                .clone(),
            None,
        ))
    }
    pub fn toc_level(&self, paragraph: Node<'_, '_>) -> Result<Option<i64>> {
        let Some(index) = self.paragraph_style(paragraph)? else {
            return Ok(None);
        };
        let Some(name) = self.styles[index].name.as_deref() else {
            return Ok(None);
        };
        let tail = if let Some(tail) = name.strip_prefix("目录") {
            tail
        } else if name.get(..3).is_some_and(|s| s.eq_ignore_ascii_case("TOC")) {
            &name[3..]
        } else {
            return Ok(None);
        };
        let tail = tail.trim_start_matches(space);
        let tail = tail.strip_suffix('\n').unwrap_or(tail);
        if tail.is_empty() || !tail.chars().all(|c| super::decimal(c).is_some()) {
            return Ok(None);
        }
        Ok(integer(Some(tail)).map(|i| i - 1))
    }
}
fn heading(value: &str) -> (String, Option<i64>) {
    // Python's \d uses the frozen Unicode decimal set; Rust regex Unicode
    // versions can differ. Preserve both alternatives, including the second
    // alternative's intentionally unanchored trailing text.
    let split = |input: &str, digits: bool| {
        input
            .char_indices()
            .find(|(_, c)| super::decimal(*c).is_some() != digits)
            .map_or(input.len(), |(i, _)| i)
    };
    let first = split(value, false);
    let second = split(&value[first..], true);
    let groups = if first > 0 && second > 0 && matches!(&value[first + second..], "" | "\n") {
        Some((&value[..first], &value[first..first + second]))
    } else {
        let first = split(value, true);
        let second = split(&value[first..], false);
        if first > 0 && second > 0 {
            Some((&value[..first], &value[first..first + second]))
        } else {
            None
        }
    };
    let Some((a, b)) = groups else {
        return (value.to_owned(), None);
    };
    let mut parts = [a, b];
    parts.sort();
    for (index, part) in parts.iter().enumerate() {
        if part.trim_matches(space).to_lowercase() == "heading" {
            return ("Heading".to_owned(), integer(Some(parts[1 - index])));
        }
    }
    (String::new(), Some(0))
}
fn ui_name(name: &str) -> String {
    match name {
        "caption" => "Caption".to_owned(),
        "footer" => "Footer".to_owned(),
        "header" => "Header".to_owned(),
        _ if name.starts_with("heading ")
            && name.len() == 9
            && matches!(name.as_bytes()[8], b'1'..=b'9') =>
        {
            format!("Heading {}", &name[8..])
        }
        _ => name.to_owned(),
    }
}
