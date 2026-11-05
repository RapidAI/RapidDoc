//! Read-only data_only OOXML cell projection used by original chart formulas.
//! This is not the complete XLSX converter (no recalculation or display formats).
use super::{
    escape,
    value::{self, Value},
};
use crate::{
    package::{Package, Relationship},
    xml, Error, Result,
};
use roxmltree::Node;
use std::{
    collections::BTreeMap,
    sync::{Arc, OnceLock},
};
const NS: &str = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
fn is(n: Node<'_, '_>, name: &str) -> bool {
    n.is_element() && n.tag_name().namespace() == Some(NS) && n.tag_name().name() == name
}
fn child<'a, 'd>(n: Node<'a, 'd>, name: &str) -> Option<Node<'a, 'd>> {
    n.children().find(|n| is(*n, name))
}
fn rich(n: Node<'_, '_>) -> String {
    n.children()
        .filter_map(|n| {
            if is(n, "t") {
                Some(n.text().unwrap_or("").to_owned())
            } else if is(n, "r") {
                Some(
                    child(n, "t")
                        .and_then(|n| n.text())
                        .unwrap_or("")
                        .to_owned(),
                )
            } else {
                None
            }
        })
        .collect()
}
#[derive(Debug)]
struct Sheet {
    cells: BTreeMap<(u32, u32), Value>,
    rows: u32,
    columns: u32,
    read_error: Option<(u32, Error)>,
}
pub struct Workbook {
    sheets: Vec<(String, Sheet)>,
}
fn part(rels: &[Relationship], kind: &str) -> Option<String> {
    rels.iter()
        .find(|r| !r.external && r.kind_tail() == kind)
        .and_then(|r| r.resolved.clone())
}
pub(crate) fn number(s: &str) -> Result<Value> {
    if s.contains(['.', 'e', 'E']) {
        s.parse::<f64>().map(Value::Float).map_err(Error::malformed)
    } else {
        crate::docx::integer_text(Some(s))
            .and_then(|s| s.parse().ok())
            .map(Value::Int)
            .ok_or_else(|| Error::malformed("invalid worksheet numeric value"))
    }
}
fn date_format(s: &str) -> bool {
    let s = s.split(';').next().unwrap_or("");
    static STRIP: OnceLock<regex::Regex> = OnceLock::new();
    let re = STRIP.get_or_init(|| regex::Regex::new(r#""[^"\n]*"|\[[^\]]*\]"#).unwrap());
    let stripped = re.replace_all(s, |c: &regex::Captures<'_>| {
        let token = c.get(0).unwrap().as_str();
        if matches!(token, "[h]" | "[hh]" | "[m]" | "[mm]" | "[s]" | "[ss]") {
            token.to_owned()
        } else {
            String::new()
        }
    });
    let mut previous = None;
    for c in stripped.chars() {
        if matches!(c, 'd' | 'm' | 'h' | 'y' | 's' | 'D' | 'M' | 'H' | 'Y' | 'S')
            && !matches!(previous, Some('_' | '\\'))
        {
            return true;
        }
        previous = Some(c);
    }
    false
}
pub(crate) fn formats(text: Option<&str>) -> Result<Vec<(bool, bool)>> {
    let Some(text) = text else {
        return Ok(vec![]);
    };
    let tree = xml::parse(text)?;
    let root = tree.root_element();
    let mut custom = BTreeMap::new();
    if let Some(formats) = child(root, "numFmts") {
        for n in formats.children().filter(|n| is(*n, "numFmt")) {
            if let Some((id, code)) = n
                .attribute("numFmtId")
                .and_then(|s| s.parse::<u32>().ok())
                .zip(n.attribute("formatCode"))
            {
                static DURATION: OnceLock<regex::Regex> = OnceLock::new();
                let duration = DURATION
                    .get_or_init(|| regex::Regex::new(r"(?i)\[(?:hh?|mm?|ss?)\]").unwrap())
                    .is_match(code.split(';').next().unwrap_or(""));
                custom.insert(id, (date_format(code), duration));
            }
        }
    }
    Ok(child(root, "cellXfs")
        .map(|xfs| {
            xfs.children()
                .filter(|n| is(*n, "xf"))
                .map(|xf| {
                    let id = xf
                        .attribute("numFmtId")
                        .and_then(|s| s.parse::<u32>().ok())
                        .unwrap_or(0);
                    custom
                        .get(&id)
                        .copied()
                        .unwrap_or((matches!(id,14..=22|45..=47), id == 46))
                })
                .collect()
        })
        .unwrap_or_default())
}
impl Workbook {
    pub fn open(bytes: Arc<[u8]>) -> Result<Self> {
        let mut package = Package::open(bytes)?;
        let main = package.main_part()?;
        let text = package
            .read_xml(&main)?
            .ok_or_else(|| Error::missing(&main))?;
        let tree = xml::parse(&text)?;
        let root = tree.root_element();
        if !is(root, "workbook") {
            return Err(Error::malformed(
                "embedded workbook namespace not recognized by original reader",
            ));
        }
        let rels = package.relationships(Some(&main))?;
        let epoch = child(root, "workbookPr")
            .and_then(|n| n.attribute("date1904"))
            .is_some_and(|s| matches!(s, "1" | "true" | "True"));
        let shared = if let Some(path) = part(&rels, "sharedStrings") {
            let text = package
                .read_xml(&path)?
                .ok_or_else(|| Error::missing(&path))?;
            let t = xml::parse(&text)?;
            t.root_element()
                .children()
                .filter(|n| is(*n, "si"))
                .map(rich)
                .collect::<Vec<_>>()
        } else {
            vec![]
        };
        let style_text = part(&rels, "styles")
            .map(|p| package.read_xml(&p))
            .transpose()?
            .flatten();
        let dates = formats(style_text.as_deref())?;
        let mut sheets = vec![];
        for n in child(root, "sheets")
            .into_iter()
            .flat_map(|n| n.children())
            .filter(|n| is(*n, "sheet"))
        {
            let id = n
                .attributes()
                .find(|a| {
                    a.name() == "id" && a.namespace().is_some_and(|s| s.ends_with("/relationships"))
                })
                .map(|a| a.value())
                .ok_or_else(|| Error::missing("worksheet relationship"))?;
            let Some(rel) = rels
                .iter()
                .find(|r| r.id == id && !r.external && r.kind_tail() == "worksheet")
            else {
                continue;
            };
            let path = rel.resolved.clone().ok_or_else(|| Error::missing(id))?;
            let Some(text) = package.read_xml(&path)? else {
                continue;
            };
            let tree = xml::parse(&text)?;
            let root = tree.root_element();
            if !is(root, "worksheet") {
                continue;
            }
            let extent = child(root, "dimension")
                .and_then(|n| n.attribute("ref"))
                .and_then(bounds);
            let mut cells = BTreeMap::new();
            let (mut maxrow, mut maxcol) = (0, 0);
            let mut previous_row = 0;
            let mut read_error = None;
            for row in child(root, "sheetData")
                .into_iter()
                .flat_map(|n| n.children())
                .filter(|n| is(*n, "row"))
            {
                let ri = row
                    .attribute("r")
                    .map(|s| s.parse::<u32>().map_err(Error::malformed))
                    .transpose()?
                    .unwrap_or(previous_row + 1);
                previous_row = ri;
                let mut ci = 0;
                let result: Result<()> = (|| {
                    for c in row.children().filter(|n| is(*n, "c")) {
                        let (r, col) = if let Some(address) = c.attribute("r") {
                            coordinate(address).ok_or_else(|| {
                                Error::malformed("invalid worksheet cell coordinate")
                            })?
                        } else {
                            (ri, ci + 1)
                        };
                        ci = col;
                        maxrow = maxrow.max(r);
                        maxcol = maxcol.max(col);
                        let kind = c.attribute("t").unwrap_or("n");
                        let raw = child(c, "v")
                            .and_then(|n| n.text())
                            .filter(|s| !s.is_empty());
                        let mut value = match kind {
                            "inlineStr" => child(c, "is")
                                .map(|n| Value::Text(rich(n)))
                                .unwrap_or(Value::Empty),
                            "s" => match raw {
                                Some(s) => Value::Text(
                                    shared
                                        .get(s.parse::<usize>().map_err(Error::malformed)?)
                                        .cloned()
                                        .ok_or_else(|| Error::malformed("shared string index"))?,
                                ),
                                None => Value::Empty,
                            },
                            "b" => match raw {
                                Some(s) => {
                                    Value::Bool(s.parse::<i64>().map_err(Error::malformed)? != 0)
                                }
                                None => Value::Empty,
                            },
                            "e" | "str" => {
                                raw.map(|s| Value::Text(s.into())).unwrap_or(Value::Empty)
                            }
                            "d" => match raw {
                                Some(s) => Value::DateTime(value::iso(s).ok_or_else(|| {
                                    Error::malformed("invalid worksheet ISO date")
                                })?),
                                None => Value::Empty,
                            },
                            _ => match raw {
                                Some(s) if !s.is_empty() => number(s)?,
                                _ => Value::Empty,
                            },
                        };
                        let style = c
                            .attribute("s")
                            .and_then(|s| s.parse::<usize>().ok())
                            .unwrap_or(0);
                        if kind == "n" && dates.get(style).is_some_and(|v| v.0) {
                            let n = match &value {
                                Value::Float(f) => Some(*f),
                                Value::Int(i) => i.to_string().parse::<f64>().ok(),
                                _ => None,
                            };
                            if let Some(n) = n {
                                value = (if dates[style].1 {
                                    value::duration(n, true)
                                } else {
                                    value::excel(n, epoch, false)
                                })
                                .map(Value::DateTime)
                                .unwrap_or_else(|| Value::Text("#VALUE!".into()));
                            }
                        }
                        cells.insert((r, col), value);
                    }
                    Ok(())
                })();
                if let Err(error) = result {
                    read_error = Some((ri, error));
                    break;
                }
            }
            let (rows, columns) = extent
                .map(|(_, _, c, r)| (r, c))
                .unwrap_or((maxrow, maxcol));
            sheets.push((
                n.attribute("name").unwrap_or("").into(),
                Sheet {
                    cells,
                    rows,
                    columns,
                    read_error,
                },
            ));
        }
        Ok(Self { sheets })
    }
    pub fn vector(&self, formula: &str) -> Result<Option<Vec<Value>>> {
        let Some((sheet, mincol, minrow, maxcol, maxrow)) = parse_formula(formula) else {
            return Ok(None);
        };
        let Some((_, sheet)) = self.sheets.iter().find(|(name, _)| name == &sheet) else {
            return Ok(None);
        };
        if mincol != maxcol && minrow != maxrow {
            return Ok(None);
        }
        // Read-only openpyxl scans preceding rows to obtain the requested cell.
        // A bad later row must not poison a valid earlier chart range.
        if let Some((row, error)) = &sheet.read_error {
            if *row <= maxrow {
                return Err(error.clone());
            }
        }
        let mut values = vec![];
        if mincol == maxcol {
            for row in minrow..=maxrow {
                values.push(
                    sheet
                        .cells
                        .get(&(row, mincol))
                        .cloned()
                        .unwrap_or(Value::Empty),
                );
            }
        } else {
            for col in mincol..=maxcol {
                values.push(
                    sheet
                        .cells
                        .get(&(minrow, col))
                        .cloned()
                        .unwrap_or(Value::Empty),
                );
            }
        }
        Ok(Some(values))
    }
    pub fn scalar(&self, formula: &str) -> Result<Option<String>> {
        Ok(self
            .vector(formula)?
            .and_then(|v| v.first().map(|v| v.string(false, false)))
            .filter(|s| !s.is_empty()))
    }
    pub fn html(&self) -> Result<String> {
        for (_, sheet) in &self.sheets {
            if let Some((_, error)) = &sheet.read_error {
                return Err(error.clone());
            }
            let mut rows = BTreeMap::<u32, BTreeMap<u32, String>>::new();
            let mut width = 0;
            for (&(r, c), v) in &sheet.cells {
                if r > sheet.rows || c > sheet.columns {
                    continue;
                }
                let text = v.string(false, false);
                if !text.is_empty() {
                    width = width.max(c);
                    rows.entry(r).or_default().insert(c, text);
                }
            }
            let (Some(&first), Some(&last)) = (rows.keys().next(), rows.keys().last()) else {
                continue;
            };
            let mut out = String::from("<table><thead><tr>");
            for c in 1..=width {
                out.push_str(&format!(
                    "<th>{}</th>",
                    escape(
                        rows.get(&first)
                            .and_then(|r| r.get(&c))
                            .map(String::as_str)
                            .unwrap_or("")
                    )
                ));
            }
            out.push_str("</tr></thead><tbody>");
            for r in first + 1..=last {
                out.push_str("<tr>");
                for c in 1..=width {
                    out.push_str(&format!(
                        "<td>{}</td>",
                        escape(
                            rows.get(&r)
                                .and_then(|r| r.get(&c))
                                .map(String::as_str)
                                .unwrap_or("")
                        )
                    ));
                }
                out.push_str("</tr>");
            }
            out.push_str("</tbody></table>");
            return Ok(out);
        }
        Ok(String::new())
    }
}
fn column(s: &str) -> Option<u32> {
    if s.is_empty() || s.len() > 3 || !s.bytes().all(|c| c.is_ascii_alphabetic()) {
        return None;
    }
    let mut col = 0;
    for c in s.bytes() {
        col = col * 26 + (c.to_ascii_uppercase() - b'A' + 1) as u32;
    }
    Some(col)
}
pub(crate) fn coordinate(s: &str) -> Option<(u32, u32)> {
    let s = s.replace('$', "");
    let index = s.find(|c: char| !c.is_ascii_alphabetic())?;
    let col = column(&s[..index])?;
    let row = s[index..].parse::<u32>().ok()?;
    if row == 0 {
        return None;
    }
    Some((row, col))
}
pub(crate) fn bounds(s: &str) -> Option<(u32, u32, u32, u32)> {
    let (a, b) = s.split_once(':').unwrap_or((s, s));
    let (r, c) = coordinate(a)?;
    let (rr, cc) = coordinate(b)?;
    Some((c, r, cc, rr))
}
pub fn parse_formula(formula: &str) -> Option<(String, u32, u32, u32, u32)> {
    // The original range_to_tuple regex is prefix-matching, not a full formula
    // evaluator; whole rows/columns lack integer bounds and return None.
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re=RE.get_or_init(||regex::Regex::new(r"^(?:'((?:[^']|'')*)'|([^'^ !]*))!((?:\$?[A-Za-z]{1,3})?(?:\$?[0-9]+)?(?::(?:\$?[A-Za-z]{1,3})?(?:\$?[0-9]+)?)?)").unwrap());
    let formula = formula.trim().strip_prefix('=').unwrap_or(formula.trim());
    let m = re.captures(formula)?;
    let sheet = m.get(1).or_else(|| m.get(2))?.as_str().replace("''", "'");
    let (a, b, c, d) = bounds(m.get(3)?.as_str())?;
    Some((sheet, a, b, c, d))
}
