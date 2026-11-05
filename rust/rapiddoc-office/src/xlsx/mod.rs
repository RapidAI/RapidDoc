//! Native XLSX cached cells and Office resources to spreadsheet pages.
//! Package repair/retry and the public host binding are separate delivery work.
use crate::{
    chart::{value, workbook},
    package::{Package, Relationship},
    spreadsheet::{Cell, Merge, Sheet, SourceCell},
    xml, Error, ErrorKind, Format, Result,
};
use roxmltree::Node;
use std::{collections::BTreeMap, sync::Arc};
pub mod normalize;
mod resources;
mod styles;
const NS: &str = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
fn is(n: Node<'_, '_>, name: &str) -> bool {
    n.is_element() && n.tag_name().namespace() == Some(NS) && n.tag_name().name() == name
}
fn child<'a, 'd>(n: Node<'a, 'd>, name: &str) -> Option<Node<'a, 'd>> {
    n.children().find(|n| is(*n, name))
}
fn rid<'a>(n: Node<'a, '_>) -> Option<&'a str> {
    n.attributes()
        .find(|a| a.name() == "id" && a.namespace().is_some_and(|s| s.ends_with("/relationships")))
        .map(|a| a.value())
}
fn integer(s: &str) -> Result<i64> {
    crate::docx::integer_text(Some(s))
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| Error::malformed("invalid worksheet integer"))
}
fn index(s: &str, len: usize) -> Result<usize> {
    let i = integer(s)?;
    let i = if i < 0 { len as i64 + i } else { i };
    usize::try_from(i)
        .ok()
        .filter(|i| *i < len)
        .ok_or_else(|| Error::malformed("spreadsheet index out of range"))
}
fn position(s: &str) -> Result<(u32, u32)> {
    let (r, c) = workbook::coordinate(s)
        .ok_or_else(|| Error::malformed("invalid spreadsheet coordinate"))?;
    Ok((r - 1, c - 1))
}
fn escape_lines(s: &str) -> String {
    crate::document_output::html::escape(s)
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace('\n', "<br>")
}
pub(crate) fn sheet_title(name: &str, existing: &[String]) -> Result<String> {
    let name = if name.is_empty() { "Sheet" } else { name };
    if name
        .chars()
        .any(|c| matches!(c, '\\' | '*' | '?' | ':' | '/' | '[' | ']'))
    {
        return Err(Error::malformed("invalid worksheet title"));
    }
    if existing
        .iter()
        .any(|s| s.to_lowercase() == name.to_lowercase())
    {
        let re = regex::RegexBuilder::new(&format!(r"{}(\d*),?", regex::escape(name)))
            .case_insensitive(true)
            .build()
            .map_err(Error::malformed)?;
        let highest = re
            .captures_iter(&existing.join(","))
            .filter_map(|c| {
                crate::docx::integer_text(Some(&c[1]))
                    .and_then(|s| s.parse::<num_bigint::BigInt>().ok())
            })
            .max()
            .unwrap_or_default();
        Ok(format!("{name}{}", highest + 1))
    } else {
        Ok(name.into())
    }
}
#[derive(Debug, Clone)]
struct Text {
    plain: String,
    rich_html: Option<String>,
}
fn rich(n: Node<'_, '_>) -> Result<Text> {
    if let Some(t) = child(n, "t")
        .and_then(|n| n.text())
        .filter(|s| !s.is_empty())
    {
        return Ok(Text {
            plain: t.replace("x005F_", ""),
            rich_html: None,
        });
    }
    let mut parts = vec![];
    for r in n.children().filter(|n| is(*n, "r")) {
        let text = child(r, "t")
            .and_then(|n| n.text())
            .unwrap_or("")
            .replace("x005F_", "");
        let font = child(r, "rPr").map(styles::font).transpose()?;
        parts.push((text, font));
    }
    let plain = parts.iter().map(|(s, _)| s.as_str()).collect::<String>();
    let rich_html = if parts.is_empty() || parts.len() == 1 && parts[0].1.is_none() {
        None
    } else {
        Some(
            parts
                .iter()
                .map(|(s, f)| {
                    if let Some(f) = f {
                        styles::rich_html(s, f)
                    } else {
                        escape_lines(s)
                    }
                })
                .collect(),
        )
    };
    Ok(Text { plain, rich_html })
}
fn float(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v == f64::INFINITY {
        return "inf".into();
    }
    if v == f64::NEG_INFINITY {
        return "-inf".into();
    }
    let s = format!("{v:?}");
    if let Some((a, b)) = s.split_once('e') {
        let n = b.parse::<i32>().unwrap();
        format!(
            "{a}e{}{n:02}",
            if n < 0 { '-' } else { '+' },
            n = n.unsigned_abs()
        )
    } else {
        s
    }
}
fn value_text(value: value::Value) -> String {
    match value {
        value::Value::Float(f) => float(f),
        v => v.string(false, false),
    }
}
#[derive(Debug)]
pub struct Worksheet {
    pub name: String,
    pub state: String,
    pub part: String,
    pub relationships: Vec<Relationship>,
    pub cells: Sheet,
    pub has_drawings: bool,
    pub has_ole: bool,
    pub has_equations: bool,
    cell_images: BTreeMap<(u32, u32), String>,
}
pub struct Workbook {
    pub package: Package,
    pub sheets: Vec<Worksheet>,
    pub date1904: bool,
}
pub struct Conversion {
    pub pages: Vec<Vec<serde_json::Value>>,
    pub normalized_members: Vec<String>,
}
/// Match the original failed-conversion retry without rewriting an entire ZIP.
pub fn convert(
    data: Arc<[u8]>,
    include_hidden: bool,
    gap: Option<u32>,
    singleton: bool,
) -> Result<Conversion> {
    let package = Package::open(data.clone())?;
    convert_package(package, data, include_hidden, gap, singleton)
}
/// Reuse the caller's already classified package on the normal path.
pub fn convert_package(
    package: Package,
    data: Arc<[u8]>,
    include_hidden: bool,
    gap: Option<u32>,
    singleton: bool,
) -> Result<Conversion> {
    match Workbook::from_package(package).and_then(|w| w.into_pages(include_hidden, gap, singleton))
    {
        Ok(pages) => Ok(Conversion {
            pages,
            normalized_members: vec![],
        }),
        Err(original) => {
            let mut package = Package::open(data)?;
            let normalized_members = normalize::apply(&mut package)?;
            if normalized_members.is_empty() {
                return Err(original);
            }
            let pages =
                Workbook::from_package(package)?.into_pages(include_hidden, gap, singleton)?;
            Ok(Conversion {
                pages,
                normalized_members,
            })
        }
    }
}
#[derive(Debug, Clone)]
struct RawCell {
    value: Option<Text>,
    style: usize,
    merged: bool,
}
impl RawCell {
    fn empty() -> Self {
        Self {
            value: None,
            style: usize::MAX,
            merged: false,
        }
    }
}
impl Workbook {
    pub fn open(bytes: Arc<[u8]>) -> Result<Self> {
        Self::from_package(Package::open(bytes)?)
    }
    pub fn from_package(mut package: Package) -> Result<Self> {
        if package.format()? != Format::Xlsx {
            return Err(Error::malformed("not an XLSX workbook"));
        }
        let main = package.main_part()?;
        let text = package
            .read_xml(&main)?
            .ok_or_else(|| Error::missing(&main))?;
        let tree = xml::parse(&text)?;
        let root = tree.root_element();
        if !is(root, "workbook") {
            return Err(Error::malformed(
                "workbook namespace not recognized by original cell reader",
            ));
        }
        let rels = package.relationships(Some(&main))?;
        let date1904 = child(root, "workbookPr")
            .and_then(|n| n.attribute("date1904"))
            .is_some_and(|v| !matches!(v, "0" | "false" | "f" | ""));
        // The original reader discovers strings by the manifest, while its
        // stylesheet loader uses the conventional fixed archive member.
        let style_text = package.read_xml("xl/styles.xml")?;
        let styles = styles::Styles::parse(style_text.as_deref())?;
        let manifest = package
            .read_xml(crate::package::CONTENT_TYPES)?
            .ok_or_else(|| Error::missing("Content Types"))?;
        let types = xml::parse(&manifest)?;
        let shared_path=types.root_element().children().find(|n|n.is_element()&&n.tag_name().name()=="Override"&&n.attribute("ContentType")==Some("application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml")).and_then(|n|n.attribute("PartName")).map(|s|s.trim_start_matches('/').to_owned());
        let shared = if let Some(path) = shared_path {
            let text = package
                .read_xml(&path)?
                .ok_or_else(|| Error::missing(&path))?;
            let t = xml::parse(&text)?;
            t.root_element()
                .children()
                .filter(|n| is(*n, "si"))
                .map(rich)
                .collect::<Result<Vec<_>>>()?
        } else {
            vec![]
        };
        let mut sheets = vec![];
        let mut titles = vec![];
        for n in child(root, "sheets")
            .into_iter()
            .flat_map(|n| n.children())
            .filter(|n| is(*n, "sheet"))
        {
            let Some(id) = rid(n) else {
                continue;
            };
            if !rels.iter().any(|r| r.id == id) {
                return Err(Error::missing(id));
            }
            let Some(rel) = rels
                .iter()
                .find(|r| r.id == id && !r.external && r.kind_tail() == "worksheet")
            else {
                continue;
            };
            let part = rel.resolved.clone().ok_or_else(|| Error::missing(id))?;
            let Some(text) = package.read_xml(&part)? else {
                continue;
            };
            let relationships = package.relationships(Some(&part))?;
            let (cells, cell_images) =
                read_sheet(&text, &relationships, &styles, &shared, date1904)?;
            let title = sheet_title(
                n.attribute("name")
                    .ok_or_else(|| Error::missing("sheet name"))?,
                &titles,
            )?;
            titles.push(title.clone());
            let tree = xml::parse(&text)?;
            let r = tree.root_element();
            sheets.push(Worksheet {
                name: title,
                state: n.attribute("state").unwrap_or("visible").into(),
                part,
                relationships,
                cells,
                cell_images,
                has_drawings: child(r, "drawing").is_some() || child(r, "legacyDrawing").is_some(),
                has_ole: child(r, "oleObjects").is_some(),
                has_equations: r.descendants().any(|n| {
                    n.is_element() && matches!(n.tag_name().name(), "oMath" | "oMathPara")
                }),
            });
        }
        Ok(Self {
            package,
            sheets,
            date1904,
        })
    }
    /// Consume the workbook once, retaining native resources only for its lifetime.
    /// No second ZIP parse or host-language resource orchestration is involved.
    pub fn into_pages(
        mut self,
        include_hidden: bool,
        gap: Option<u32>,
        singleton_text: bool,
    ) -> Result<Vec<Vec<serde_json::Value>>> {
        let mut context = resources::Context::default();
        let mut pages = Vec::new();
        let mut names = Vec::new();
        for sheet in &mut self.sheets {
            if !include_hidden && sheet.state != "visible" {
                continue;
            }
            pages.push(context.convert(&mut self.package, sheet, gap, singleton_text)?);
            names.push(&sheet.name);
        }
        if pages.iter().filter(|p| !p.is_empty()).count() > 1 {
            for (page, name) in pages.iter_mut().zip(names) {
                if !page.is_empty() {
                    page.insert(0, serde_json::json!({"type":"paragraph_title","level":2,"content":[{"type":"text","content":name}]}));
                }
            }
        }
        Ok(pages)
    }
    /// Cell-only diagnostic path. Resource-bearing workbooks use into_pages.
    pub fn convert_cells(
        &self,
        include_hidden: bool,
        gap: Option<u32>,
        singleton_text: bool,
    ) -> Result<Vec<Vec<serde_json::Value>>> {
        if self
            .package
            .names()
            .iter()
            .any(|p| p.to_ascii_lowercase().ends_with("/cellimages.xml"))
        {
            return Err(Error::new(
                ErrorKind::Unsupported,
                "cell-only conversion excludes XLSX cell images; use into_pages",
            ));
        }
        let mut pages = vec![];
        let mut names = vec![];
        for sheet in &self.sheets {
            if !include_hidden && sheet.state != "visible" {
                continue;
            }
            if sheet.has_drawings
                || sheet.has_ole
                || sheet.has_equations
                || sheet
                    .relationships
                    .iter()
                    .any(|r| matches!(r.kind_tail(), "drawing" | "oleObject"))
            {
                return Err(Error::new(
                    ErrorKind::Unsupported,
                    "cell-only conversion excludes XLSX drawing/OLE/equation resources; use into_pages",
                ));
            }
            let (_, _, tables) = sheet.cells.project(gap);
            pages.push(
                tables
                    .iter()
                    .map(|t| crate::spreadsheet::table::block(t, singleton_text))
                    .collect::<Vec<_>>(),
            );
            names.push(&sheet.name);
        }
        if pages.iter().filter(|p| !p.is_empty()).count() > 1 {
            for (page, name) in pages.iter_mut().zip(names) {
                if !page.is_empty() {
                    page.insert(0,serde_json::json!({"type":"paragraph_title","level":2,"content":[{"type":"text","content":name}]}));
                }
            }
        }
        Ok(pages)
    }
}
fn read_sheet(
    text: &str,
    rels: &[Relationship],
    styles: &styles::Styles,
    shared: &[Text],
    epoch: bool,
) -> Result<(Sheet, BTreeMap<(u32, u32), String>)> {
    let tree = xml::parse(text)?;
    let root = tree.root_element();
    if !is(root, "worksheet") {
        return Err(Error::malformed("invalid worksheet namespace"));
    }
    normalize::validate_filter(root)?;
    let mut cells = BTreeMap::<(u32, u32), RawCell>::new();
    let mut prev_row = 0;
    for row in child(root, "sheetData")
        .into_iter()
        .flat_map(|n| n.children())
        .filter(|n| is(*n, "row"))
    {
        let r = row
            .attribute("r")
            .map(integer)
            .transpose()?
            .unwrap_or(prev_row + 1);
        if r < 1 {
            return Err(Error::malformed("invalid worksheet row"));
        }
        prev_row = r;
        let mut prev_col = 0;
        for c in row.children().filter(|n| is(*n, "c")) {
            let position = if let Some(s) = c.attribute("r") {
                position(s)?
            } else {
                (u32::try_from(r - 1).map_err(Error::malformed)?, prev_col)
            };
            prev_col = position.1 + 1;
            let style = index(c.attribute("s").unwrap_or("0"), styles.cell.len())?;
            let kind = c.attribute("t").unwrap_or("n");
            let raw = child(c, "v")
                .and_then(|n| n.text())
                .filter(|s| !s.is_empty());
            let value = match kind {
                "inlineStr" => child(c, "is").map(rich).transpose()?,
                "s" => raw
                    .map(|s| index(s, shared.len()).map(|i| shared[i].clone()))
                    .transpose()?,
                "b" => raw
                    .map(|s| {
                        crate::docx::integer_text(Some(s))
                            .and_then(|s| s.parse::<num_bigint::BigInt>().ok())
                            .map(|v| Text {
                                plain: if v == num_bigint::BigInt::from(0) {
                                    "False"
                                } else {
                                    "True"
                                }
                                .into(),
                                rich_html: None,
                            })
                            .ok_or_else(|| Error::malformed("invalid boolean cell"))
                    })
                    .transpose()?,
                "d" => raw
                    .map(|s| {
                        value::iso(s)
                            .map(|s| Text {
                                plain: s,
                                rich_html: None,
                            })
                            .ok_or_else(|| Error::malformed("invalid ISO worksheet date"))
                    })
                    .transpose()?,
                "n" => raw
                    .map(|s| {
                        let number = workbook::number(s)?;
                        let date = styles.dates.get(style).copied().unwrap_or_default();
                        let plain = if date.0 {
                            let n = match &number {
                                value::Value::Int(v) => {
                                    v.to_string().parse().unwrap_or(f64::INFINITY)
                                }
                                value::Value::Float(v) => *v,
                                _ => unreachable!(),
                            };
                            if date.1 {
                                value::duration(n, true)
                            } else {
                                value::excel(n, epoch, false)
                            }
                            .unwrap_or_else(|| "#VALUE!".into())
                        } else {
                            value_text(number)
                        };
                        Ok(Text {
                            plain,
                            rich_html: None,
                        })
                    })
                    .transpose()?,
                _ => raw.map(|s| Text {
                    plain: s.into(),
                    rich_html: None,
                }),
            };
            cells.insert(
                position,
                RawCell {
                    value,
                    style,
                    merged: false,
                },
            );
        }
    }
    let mut merges = vec![];
    for n in child(root, "mergeCells")
        .into_iter()
        .flat_map(|n| n.children())
        .filter(|n| is(*n, "mergeCell"))
    {
        let (c0, r0, c1, r1) = n
            .attribute("ref")
            .and_then(workbook::bounds)
            .ok_or_else(|| Error::malformed("invalid worksheet merge"))?;
        if r1 < r0 || c1 < c0 {
            return Err(Error::malformed("inverted worksheet merge"));
        }
        merges.push(Merge {
            min_row: r0 - 1,
            max_row: r1 - 1,
            min_col: c0 - 1,
            max_col: c1 - 1,
        });
        cells.entry((r0 - 1, c0 - 1)).or_insert_with(RawCell::empty);
        for r in r0 - 1..r1 {
            for c in c0 - 1..c1 {
                if (r, c) != (r0 - 1, c0 - 1) {
                    cells.insert(
                        (r, c),
                        RawCell {
                            merged: true,
                            ..RawCell::empty()
                        },
                    );
                }
            }
        }
    }
    let mut targets = BTreeMap::new();
    for link in child(root, "hyperlinks")
        .into_iter()
        .flat_map(|n| n.children())
        .filter(|n| is(*n, "hyperlink"))
    {
        let address = link
            .attribute("ref")
            .ok_or_else(|| Error::missing("hyperlink cell"))?;
        let (c0, r0, c1, r1) = workbook::bounds(address)
            .ok_or_else(|| Error::malformed("invalid hyperlink cell range"))?;
        let target = if let Some(id) = rid(link) {
            Some(
                rels.iter()
                    .find(|r| r.id == id)
                    .ok_or_else(|| Error::missing(id))?
                    .target
                    .clone(),
            )
        } else {
            None
        };
        let location = link.attribute("location");
        let href = target
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .or_else(|| location.filter(|s| !s.is_empty()).map(|s| format!("#{s}")))
            .unwrap_or_default();
        let fallback = target
            .as_deref()
            .filter(|s| !s.is_empty())
            .or(location)
            .unwrap_or("");
        for r in r0 - 1..r1 {
            for c in c0 - 1..c1 {
                let mut p = (r, c);
                if !address.contains(':') {
                    if let Some(m) = merges.iter().find(|m| {
                        r >= m.min_row && r <= m.max_row && c >= m.min_col && c <= m.max_col
                    }) {
                        p = (m.min_row, m.min_col);
                    }
                }
                let cell = cells.entry(p).or_insert_with(RawCell::empty);
                if cell.merged {
                    continue;
                }
                if cell.value.is_none() && !fallback.is_empty() {
                    cell.value = Some(Text {
                        plain: fallback.into(),
                        rich_html: None,
                    });
                }
                targets.insert(p, href.clone());
            }
        }
    }
    let default_cell = Cell {
        styles: styles.default.values.clone(),
        ..Cell::default()
    };
    let mut source = vec![];
    let mut cell_images = BTreeMap::new();
    for ((r, c), cell) in cells {
        let style = if cell.merged || cell.style == usize::MAX {
            &styles.default
        } else {
            &styles.cell[cell.style]
        };
        let has_value = cell.value.is_some();
        let value = cell.value.unwrap_or(Text {
            plain: String::new(),
            rich_html: None,
        });
        let disp = value.plain.contains("DISPIMG");
        let mut text = value
            .rich_html
            .clone()
            .unwrap_or_else(|| value.plain.clone());
        let mut html = value.rich_html.is_some();
        if let Some(href) = targets
            .get(&(r, c))
            .and_then(|s| crate::links::sanitize(Some(s)))
        {
            if !value.plain.is_empty() {
                if !html {
                    text = escape_lines(&text);
                }
                text = format!(
                    "<a href=\"{}\">{text}</a>",
                    crate::document_output::html::escape(href)
                );
                html = true;
            }
        }
        let semantic = !disp && !value.plain.trim_matches(crate::docx::space).is_empty();
        if disp {
            cell_images.insert((r, c), value.plain);
            text.clear();
            html = false;
        }
        source.push(SourceCell {
            row: r,
            col: c,
            has_value,
            semantic,
            cell: Cell {
                row: 0,
                col: 0,
                text,
                row_span: 1,
                col_span: 1,
                styles: style.values.clone(),
                media: vec![],
                equations: vec![],
                text_is_html: html,
                source_row: Some(r),
                source_col: Some(c),
            },
        });
    }
    let mut result = Sheet::new(source, merges)?;
    result.default_cell = default_cell;
    Ok((result, cell_images))
}
