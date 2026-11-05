//! One Word story reader for main text, headers and notes; fields reset per story.
use super::{
    auxiliary,
    fields::Fields,
    formatting::{sprms, Formatting},
    paragraph,
    pieces::{Piece, TextStream},
    resources::{Resources, Visual},
    styles::Stylesheet,
};
use crate::{docx::spans::Span, Result};
use std::collections::{BTreeMap, BTreeSet};
pub struct Paragraph {
    pub start: u32,
    pub end: u32,
    pub terminator: char,
    pub content: Vec<Span>,
    pub visuals: Vec<Visual>,
    pub style_id: u16,
    pub properties: paragraph::Paragraph,
    pub toc: bool,
    pub caption: bool,
}
pub struct Reader<'a> {
    pub text: &'a TextStream,
    pub pieces: &'a [Piece],
    pub formatting: &'a Formatting,
    pub styles: &'a Stylesheet,
    pub data: &'a [u8],
    pub equations: &'a BTreeMap<String, String>,
    pub charts: &'a BTreeMap<String, String>,
}
impl Reader<'_> {
    fn finish(
        &self,
        index: usize,
        start: u32,
        end: u32,
        terminator: char,
        content: Vec<Span>,
        visuals: Vec<Visual>,
        toc: bool,
        caption: bool,
    ) -> Option<Paragraph> {
        let fc = self.text.fcs[index];
        let run = Formatting::at(&self.formatting.paragraphs, fc);
        let style_id = run.map(|r| r.style_id).unwrap_or(0);
        let properties = paragraph::Paragraph::resolve(
            self.styles.get(style_id),
            run.map(|r| r.grpprl.as_slice()).unwrap_or(&[]),
            &self.pieces[self.text.piece_indexes[index] as usize].prm,
            self.data,
        );
        let in_table = properties.in_table.unwrap_or(false)
            || terminator == '\u{7}'
            || properties.inner_cell.unwrap_or(false)
            || properties.row_mark.unwrap_or(false)
            || properties.inner_row.unwrap_or(false);
        if content.is_empty() && visuals.is_empty() && !in_table {
            return None;
        }
        Some(Paragraph {
            start,
            end,
            terminator,
            content,
            visuals,
            style_id,
            properties,
            toc,
            caption,
        })
    }
    pub fn paragraphs(
        &self,
        start: u32,
        end: u32,
        refs: &BTreeMap<u32, String>,
        resources: &mut Resources,
    ) -> Result<Vec<Paragraph>> {
        let mut out = vec![];
        self.visit(start, end, refs, resources, |p| out.push(p))?;
        Ok(out)
    }
    pub fn visit(
        &self,
        start: u32,
        end: u32,
        refs: &BTreeMap<u32, String>,
        resources: &mut Resources,
        mut emit: impl FnMut(Paragraph),
    ) -> Result<()> {
        let a = self.text.cps.partition_point(|cp| *cp < start);
        let b = self.text.cps.partition_point(|cp| *cp < end);
        let mut fields = Fields::default();
        let mut content = vec![];
        let mut visuals = vec![];
        let mut images = BTreeSet::new();
        let mut caption = false;
        let mut paragraph_start = start;
        for i in a..b {
            let c = self.text.chars[i];
            let cp = self.text.cps[i];
            let chpx = Formatting::at(&self.formatting.characters, self.text.fcs[i])
                .map(|r| r.grpprl.as_slice())
                .unwrap_or(&[]);
            match c {
                '\u{13}' => {
                    fields.begin();
                    continue;
                }
                '\u{14}' => {
                    fields.separator();
                    caption |= fields.keyword().as_deref() == Some("SEQ");
                    let id = picture_location(chpx).map(|n| n.to_string());
                    fields.bind_object(
                        &mut content,
                        id.as_ref()
                            .and_then(|id| self.equations.get(id))
                            .map(String::as_str),
                        id.as_ref()
                            .and_then(|id| self.charts.get(id))
                            .map(String::as_str),
                    );
                    continue;
                }
                '\u{15}' => {
                    caption |= fields.keyword().as_deref() == Some("SEQ");
                    if let Some(chart) = fields.end_with_visual(&mut content) {
                        visuals.push(chart);
                    }
                    continue;
                }
                _ => {}
            }
            if let Some(label) = refs.get(&cp) {
                fields.push(
                    &mut content,
                    Span::Text {
                        content: format!("[{label}]"),
                        styles: vec!["superscript"],
                    },
                );
                continue;
            }
            match c {
                '\r' | '\u{7}' | '\u{c}' | '\u{e}' => {
                    if let Some(p) = self.finish(
                        i,
                        paragraph_start,
                        cp + 1,
                        c,
                        std::mem::take(&mut content),
                        std::mem::take(&mut visuals),
                        fields.toc_active(),
                        caption,
                    ) {
                        emit(p);
                    }
                    images.clear();
                    caption = false;
                    paragraph_start = cp + 1;
                }
                '\u{1}' => {
                    if fields.native_formula() {
                        continue;
                    }
                    let offset = picture_location(chpx);
                    let visual = if let Some(offset) = offset {
                        resources.inline(self.data, offset)?
                    } else {
                        None
                    };
                    if fields.has_chart() {
                        let preview = if let Some(offset) = offset {
                            resources.preview(offset)?
                        } else {
                            None
                        };
                        if let Some(chart) = fields.chart_picture(preview) {
                            visuals.push(chart);
                        }
                    } else if let (Some(offset), Some(v)) = (offset, visual) {
                        if resources
                            .identity(offset)
                            .is_some_and(|key| images.insert(key))
                        {
                            visuals.push(v);
                        }
                    }
                }
                _ if c == '\t' || c == '\u{b}' || c == '\u{1e}' || c >= ' ' => {
                    let style = auxiliary::character(
                        self.text,
                        i,
                        self.pieces,
                        self.formatting,
                        self.styles,
                    );
                    if fields.in_instruction() || !style.hidden && !style.deleted {
                        fields.push(
                            &mut content,
                            Span::Text {
                                content: match c {
                                    '\u{b}' => "\n".into(),
                                    '\u{1e}' => "-".into(),
                                    _ => c.to_string(),
                                },
                                styles: style.names(),
                            },
                        );
                    }
                }
                _ => {}
            }
        }
        if b > a && (!content.is_empty() || !visuals.is_empty()) {
            if let Some(p) = self.finish(
                b - 1,
                paragraph_start,
                end,
                '\0',
                content,
                visuals,
                fields.toc_active(),
                caption,
            ) {
                emit(p);
            }
        }
        Ok(())
    }
}
fn picture_location(chpx: &[u8]) -> Option<u32> {
    sprms(chpx)
        .into_iter()
        .filter(|(op, _)| *op == 0x6a03)
        .map(|(_, v)| u32::from_le_bytes(v.try_into().unwrap()))
        .last()
}
