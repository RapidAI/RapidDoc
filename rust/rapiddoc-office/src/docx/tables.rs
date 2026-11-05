//! Existing Mammoth table span semantics plus the local DOCX matching signature.
//! Cell content remains a source range; this module does not substitute plain text
//! for the pending rich-text/image/equation HTML rendering path.
use super::*;
use std::{
    collections::{BTreeMap, HashSet},
    ops::Range,
    sync::OnceLock,
};

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct Signature {
    pub row_count: usize,
    pub cell_count: usize,
    pub image_count: usize,
    pub text: String,
}
pub fn signatures_match(xml: &Signature, html: &Signature) -> bool {
    if !xml.text.is_empty() || !html.text.is_empty() {
        !xml.text.is_empty()
            && !html.text.is_empty()
            && (xml.text.starts_with(&html.text) || html.text.starts_with(&xml.text))
            && (xml.cell_count == html.cell_count || xml.row_count == html.row_count)
    } else {
        xml.row_count == html.row_count
            && xml.cell_count == html.cell_count
            && xml.image_count == html.image_count
    }
}
pub(super) fn mc(node: Node<'_, '_>, name: &str) -> bool {
    node.is_element()
        && node.tag_name().name() == name
        && node.tag_name().namespace()
            == Some("http://schemas.openxmlformats.org/markup-compatibility/2006")
}
/// Python int(value, 16), including Unicode decimal digits and separators.
/// Signature matching tolerates invalid symbols; the Mammoth reader raises.
pub(super) fn symbol_number(input: &str) -> Option<num_bigint::BigInt> {
    let mut input: String = input
        .trim_matches(space)
        .chars()
        .map(|c| decimal(c).map(|d| char::from(b'0' + d)).unwrap_or(c))
        .collect();
    let negative = input.starts_with('-');
    if input.starts_with(['+', '-']) {
        input.remove(0);
    }
    let prefixed = input.starts_with("0x") || input.starts_with("0X");
    if prefixed {
        input.drain(..2);
        if input.starts_with('_') {
            input.remove(0);
        }
    }
    let mut digits = String::new();
    let mut previous = false;
    for c in input.chars() {
        if c == '_' && previous {
            previous = false;
            continue;
        }
        if !c.is_ascii_hexdigit() {
            return None;
        }
        digits.push(c);
        previous = true;
    }
    if !previous {
        return None;
    }
    let number = num_bigint::BigInt::parse_bytes(digits.as_bytes(), 16)?;
    Some(if negative { -number } else { number })
}
pub fn char_fragment(node: Node<'_, '_>) -> String {
    if is_word(node, "t") {
        return text(node);
    }
    if is_word(node, "noBreakHyphen") {
        return "\u{2011}".to_owned();
    }
    if is_word(node, "softHyphen") {
        return "\u{ad}".to_owned();
    }
    if !is_word(node, "sym") {
        return String::new();
    }
    static SYMBOLS: OnceLock<BTreeMap<String, BTreeMap<u32, char>>> = OnceLock::new();
    let symbols = SYMBOLS.get_or_init(|| {
        let records: Vec<(String, u32, u32)> =
            serde_json::from_str(include_str!("mammoth_symbols.json"))
                .expect("generated symbol data");
        let mut symbols: BTreeMap<String, BTreeMap<u32, char>> = BTreeMap::new();
        for (font, code, value) in records {
            symbols
                .entry(font)
                .or_default()
                .insert(code, char::from_u32(value).expect("symbol scalar"));
        }
        symbols
    });
    let font = attr(node, "font").unwrap_or("");
    let code = attr(node, "char").unwrap_or("");
    let parse_hex = |s: &str| symbol_number(s).and_then(|v| u32::try_from(v).ok());
    let lookup = |s: &str| {
        parse_hex(s)
            .and_then(|code| symbols.get(font).and_then(|font| font.get(&code)))
            .copied()
    };
    lookup(code)
        .or_else(|| {
            if code.starts_with("F0")
                && code.chars().skip(2).take(2).filter(|c| *c != '\n').count() == 2
            {
                lookup(&code[2..])
            } else {
                None
            }
        })
        .map(|c| c.to_string())
        .unwrap_or_default()
}
pub fn signature(table: Node<'_, '_>) -> Signature {
    let mut ignored = HashSet::new();
    for cell in table.descendants().filter(|n| is_word(*n, "tc")) {
        if child(cell, "tcPr")
            .and_then(|p| child(p, "vMerge"))
            .is_some_and(|n| matches!(attr(n, "val"), None | Some("continue")))
        {
            ignored.extend(cell.descendants().map(|n| n.id()));
        }
    }
    for alternate in table.descendants().filter(|n| mc(*n, "AlternateContent")) {
        for branch in alternate.children().filter(|n| !mc(*n, "Fallback")) {
            ignored.extend(branch.descendants().map(|n| n.id()));
        }
    }
    for field in table.descendants().filter(|n| is_word(*n, "fldSimple")) {
        ignored.extend(field.descendants().map(|n| n.id()));
    }
    for control in table.descendants().filter(|n| is_word(*n, "sdt")) {
        let checkbox = child(control, "sdtPr").is_some_and(|p| {
            p.children().any(|n| {
                n.is_element()
                    && n.tag_name().name() == "checkbox"
                    && n.tag_name().namespace()
                        == Some("http://schemas.microsoft.com/office/word/2010/wordml")
            })
        });
        if checkbox {
            if let Some(content) = child(control, "sdtContent") {
                if let Some(first) = content
                    .descendants()
                    .find(|n| !ignored.contains(&n.id()) && !char_fragment(*n).is_empty())
                {
                    ignored.insert(first.id());
                }
            }
        }
    }
    let mut output = Signature {
        row_count: 0,
        cell_count: 0,
        image_count: 0,
        text: String::new(),
    };
    for node in table.descendants() {
        if node.is_element() {
            match node.tag_name().name() {
                "tr" => output.row_count += 1,
                "tc" => output.cell_count += 1,
                "blip" | "imagedata" => output.image_count += 1,
                _ => {}
            }
        }
        if !ignored.contains(&node.id()) {
            output
                .text
                .extend(char_fragment(node).chars().filter(|c| !space(*c)));
        }
    }
    output
}

#[derive(Debug, Serialize)]
pub struct CellLayout {
    pub column: i64,
    pub colspan: i64,
    pub rowspan: usize,
    /// Original XML bytes stay with Context/Source, including all nested objects.
    #[serde(skip)]
    pub source: Range<usize>,
}
#[derive(Debug, Serialize)]
pub struct RowLayout {
    pub header: bool,
    pub cells: Vec<CellLayout>,
    #[serde(skip)]
    pub source: Range<usize>,
}
#[derive(Debug, Serialize)]
pub struct TableLayout {
    pub rows: Vec<RowLayout>,
    pub merge_disabled: bool,
    #[serde(skip)]
    pub unplaced: Vec<Range<usize>>,
}
pub(super) fn structural_children<'a, 'input>(node: Node<'a, 'input>) -> Vec<Node<'a, 'input>> {
    let mut stack: Vec<_> = node.children().rev().collect();
    let mut output = Vec::new();
    while let Some(node) = stack.pop() {
        if !node.is_element() {
            continue;
        }
        if mc(node, "AlternateContent") {
            if let Some(fallback) = node.children().find(|n| mc(*n, "Fallback")) {
                stack.extend(fallback.children().rev());
            }
        } else if is_word(node, "sdt") {
            if let Some(content) = child(node, "sdtContent") {
                stack.extend(content.children().rev());
            }
        } else if ["ins", "object", "smartTag", "drawing", "txbxContent"]
            .iter()
            .any(|name| is_word(node, name))
            || (node.tag_name().namespace() == Some("urn:schemas-microsoft-com:vml")
                && matches!(
                    node.tag_name().name(),
                    "group" | "rect" | "roundrect" | "shape" | "textbox"
                ))
        {
            stack.extend(node.children().rev());
        } else if [
            "tblPr",
            "tblGrid",
            "trPr",
            "tcPr",
            "pPr",
            "rPr",
            "sectPr",
            "proofErr",
            "bookmarkEnd",
            "lastRenderedPageBreak",
            "commentRangeStart",
            "commentRangeEnd",
            "del",
            "annotationRef",
            "footnoteRef",
            "endnoteRef",
        ]
        .iter()
        .any(|name| is_word(node, name))
        {
        } else {
            output.push(node);
        }
    }
    output
}
/// Port of Mammoth's calculate_row_spans. No rectangular grid is allocated.
pub fn layout(table: Node<'_, '_>) -> Result<TableLayout> {
    let mut output = TableLayout {
        rows: Vec::new(),
        merge_disabled: false,
        unplaced: Vec::new(),
    };
    let mut merging: Vec<Vec<bool>> = Vec::new();
    for row in structural_children(table) {
        if !is_word(row, "tr") {
            // Unknown XML is retained for the full reader. Recognized non-row
            // body objects disable merging, just as Mammoth's fallback does.
            output.merge_disabled |= matches!(
                row.tag_name().name(),
                "p" | "tbl" | "r" | "hyperlink" | "t" | "br" | "bookmarkStart"
            );
            output.unplaced.push(row.range());
            continue;
        }
        let props = child(row, "trPr");
        if props.is_some_and(|p| child(p, "del").is_some()) {
            continue;
        }
        let mut row_output = RowLayout {
            source: row.range(),
            header: props.is_some_and(|p| child(p, "tblHeader").is_some()),
            cells: Vec::new(),
        };
        let mut flags = Vec::new();
        let mut column = 0i64;
        for cell in structural_children(row) {
            if !is_word(cell, "tc") {
                output.merge_disabled |= matches!(
                    cell.tag_name().name(),
                    "p" | "tbl" | "r" | "hyperlink" | "t" | "br" | "bookmarkStart"
                );
                output.unplaced.push(cell.range());
                continue;
            }
            let props = child(cell, "tcPr");
            let span = props
                .and_then(|p| child(p, "gridSpan"))
                .and_then(|n| attr(n, "val"));
            let colspan = match span {
                None => 1,
                Some(s) => {
                    integer(Some(s)).ok_or_else(|| Error::malformed("invalid table gridSpan"))?
                }
            };
            let merge = props
                .and_then(|p| child(p, "vMerge"))
                .is_some_and(|n| matches!(attr(n, "val"), None | Some("") | Some("continue")));
            row_output.cells.push(CellLayout {
                column,
                colspan,
                rowspan: 1,
                source: cell.range(),
            });
            flags.push(merge);
            column = column
                .checked_add(colspan)
                .ok_or_else(|| Error::malformed("table column integer overflow"))?;
        }
        output.rows.push(row_output);
        merging.push(flags);
    }
    if output.merge_disabled {
        return Ok(output);
    }
    let mut owners: BTreeMap<i64, (usize, usize)> = BTreeMap::new();
    for row in 0..output.rows.len() {
        for cell in 0..output.rows[row].cells.len() {
            let column = output.rows[row].cells[cell].column;
            if merging[row][cell] {
                if let Some(&(owner_row, owner_cell)) = owners.get(&column) {
                    output.rows[owner_row].cells[owner_cell].rowspan += 1;
                    continue;
                }
            }
            owners.insert(column, (row, cell));
            merging[row][cell] = false;
        }
    }
    for (row, flags) in output.rows.iter_mut().zip(merging) {
        let cells = std::mem::take(&mut row.cells);
        row.cells = cells
            .into_iter()
            .zip(flags)
            .filter_map(|(cell, merge)| if merge { None } else { Some(cell) })
            .collect();
    }
    Ok(output)
}
