//! 原 openpyxl 字体/填充/对齐在投影 IR 中保留的属性。
use super::*;
#[derive(Debug, Clone, Default)]
pub struct Font {
    pub bold: bool,
    pub italic: bool,
    pub strike: bool,
    pub underline: bool,
    pub vertical: Option<String>,
    pub color: Option<String>,
}
#[derive(Debug, Clone, Default)]
pub struct Style {
    pub values: BTreeMap<String, String>,
}
pub struct Styles {
    pub cell: Vec<Style>,
    pub default: Style,
    pub dates: Vec<(bool, bool)>,
}
fn rgb(n: Node<'_, '_>) -> Result<Option<String>> {
    if n.attribute("indexed").is_some()
        || n.attribute("theme").is_some()
        || n.attribute("auto").is_some()
    {
        return Ok(None);
    }
    let s = n.attribute("rgb").unwrap_or("00000000");
    if !matches!(s.len(), 6 | 8) || !s.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err(Error::malformed("invalid spreadsheet RGB color"));
    }
    Ok(Some(format!("#{}", if s.len() == 8 { &s[2..] } else { s })))
}
fn boolean(n: Option<Node<'_, '_>>) -> bool {
    n.is_some_and(|n| !matches!(n.attribute("val"), Some("false" | "f" | "0" | "")))
}
pub fn font(n: Node<'_, '_>) -> Result<Font> {
    let underline = child(n, "u").map(|n| n.attribute("val").unwrap_or("single"));
    if underline.is_some_and(|s| {
        !matches!(
            s,
            "none" | "single" | "double" | "singleAccounting" | "doubleAccounting"
        )
    }) {
        return Err(Error::malformed("invalid spreadsheet underline"));
    }
    let vertical = child(n, "vertAlign")
        .and_then(|n| n.attribute("val"))
        .filter(|s| *s != "none")
        .map(str::to_owned);
    if vertical
        .as_ref()
        .is_some_and(|s| !matches!(s.as_str(), "superscript" | "subscript" | "baseline"))
    {
        return Err(Error::malformed("invalid spreadsheet vertical alignment"));
    }
    Ok(Font {
        bold: boolean(child(n, "b")),
        italic: boolean(child(n, "i")),
        strike: boolean(child(n, "strike")),
        underline: underline.is_some_and(|s| s != "none"),
        vertical,
        color: child(n, "color").map(rgb).transpose()?.flatten(),
    })
}
fn index(s: Option<&str>, len: usize) -> Result<usize> {
    super::index(s.unwrap_or("0"), len)
}
fn style(
    font: Font,
    fill: Option<&BTreeMap<String, String>>,
    alignment: Option<Node<'_, '_>>,
) -> Style {
    let mut values = BTreeMap::new();
    if font.bold {
        values.insert("font-weight".into(), "bold".into());
    }
    if font.italic {
        values.insert("font-style".into(), "italic".into());
    }
    if font.underline {
        values.insert("text-decoration".into(), "underline".into());
    }
    if font.strike {
        values.insert("text-decoration".into(), "line-through".into());
    }
    if let Some(color) = &font.color {
        values.insert("color".into(), color.clone());
    }
    if let Some(fill) = fill {
        values.extend(fill.clone());
    }
    if let Some(n) = alignment {
        for (attr, key) in [("horizontal", "text-align"), ("vertical", "vertical-align")] {
            if let Some(v) = n.attribute(attr).filter(|s| !s.is_empty()) {
                values.insert(key.into(), v.into());
            }
        }
    }
    Style { values }
}
impl Styles {
    pub fn parse(text: Option<&str>) -> Result<Self> {
        // An absent stylesheet keeps openpyxl's Calibri/theme default; its
        // theme color is not an RGB style in the original projection.
        let Some(text) = text else {
            return Ok(Self {
                cell: vec![Style::default()],
                default: Style::default(),
                dates: vec![],
            });
        };
        let tree = xml::parse(text)?;
        let root = tree.root_element();
        let fonts = child(root, "fonts")
            .into_iter()
            .flat_map(|n| n.children())
            .filter(|n| is(*n, "font"))
            .map(font)
            .collect::<Result<Vec<_>>>()?;
        let fills = child(root, "fills")
            .into_iter()
            .flat_map(|n| n.children())
            .filter(|n| is(*n, "fill"))
            .map(|n| {
                let mut values = BTreeMap::new();
                if let Some(fill) =
                    child(n, "patternFill").filter(|n| n.attribute("patternType") == Some("solid"))
                {
                    if let Some(fg) = child(fill, "fgColor") {
                        let rgb = rgb(fg)?;
                        if fg.attribute("indexed").is_none()
                            && fg.attribute("theme").is_none()
                            && fg.attribute("auto").is_none()
                        {
                            if let Some(color) = rgb {
                                values.insert("background-color".into(), color);
                            }
                        }
                    } else {
                        values.insert("background-color".into(), "#000000".into());
                    }
                }
                Ok(values)
            })
            .collect::<Result<Vec<_>>>()?;
        let default = style(
            fonts.first().cloned().unwrap_or_default(),
            fills.first(),
            None,
        );
        let mut cells = vec![];
        for n in child(root, "cellXfs")
            .into_iter()
            .flat_map(|n| n.children())
            .filter(|n| is(*n, "xf"))
        {
            let font = fonts
                .get(index(n.attribute("fontId"), fonts.len())?)
                .cloned()
                .ok_or_else(|| Error::malformed("missing cell font"))?;
            let fill = fills.get(index(n.attribute("fillId"), fills.len())?);
            cells.push(style(font, fill, child(n, "alignment")));
        }
        if cells.is_empty() {
            return Ok(Self {
                cell: vec![Style::default()],
                default: Style::default(),
                dates: vec![],
            });
        }
        Ok(Self {
            cell: cells,
            default,
            dates: crate::chart::workbook::formats(Some(text))?,
        })
    }
}
pub fn rich_html(text: &str, font: &Font) -> String {
    let mut s = super::escape_lines(text);
    if s.is_empty() {
        return s;
    }
    if font.strike || font.underline {
        s = s.replace(' ', "&nbsp;");
    }
    if let Some(v) = &font.vertical {
        if v == "superscript" {
            s = format!("<sup>{s}</sup>");
        } else if v == "subscript" {
            s = format!("<sub>{s}</sub>");
        }
    }
    if font.strike {
        s = format!("<s>{s}</s>");
    }
    if font.underline {
        s = format!("<u>{s}</u>");
    }
    if font.italic {
        s = format!("<em>{s}</em>");
    }
    if font.bold {
        s = format!("<strong>{s}</strong>");
    }
    s
}
