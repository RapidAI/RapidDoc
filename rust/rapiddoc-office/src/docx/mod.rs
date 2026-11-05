//! DOCX context, body/list/TOC/section orchestration and shared resources,
//! preserving current Python rules, including table notes and embedded styles.
//! Unified export and public Office API adapters are separate.
pub mod body;
mod decimal_data;
pub mod embedded;
mod fields;
pub mod formulas;
pub mod normalize;
mod numbering;
pub mod spans;
pub mod style_map;
mod styles;
pub mod table_html;
pub mod table_images;
mod table_markup;
mod table_resources;
pub mod tables;

use crate::{
    package::{Package, Relationship},
    xml, Error, Format, Result,
};
pub use numbering::{Numbering, NumberingProperties};
use roxmltree::Node;
use serde::Serialize;
use std::{collections::BTreeMap, sync::Arc};
pub use styles::{Formatting, Script, Styles};

pub(super) fn is_word(node: Node<'_, '_>, local: &str) -> bool {
    node.is_element()
        && node.tag_name().name() == local
        && crate::package::word_namespace(node.tag_name().namespace())
}
pub(super) fn child<'a, 'input>(node: Node<'a, 'input>, local: &str) -> Option<Node<'a, 'input>> {
    node.children().find(|n| is_word(*n, local))
}
pub(super) fn attr<'a>(node: Node<'a, '_>, local: &str) -> Option<&'a str> {
    node.attributes()
        .find(|a| a.name() == local && crate::package::word_namespace(a.namespace()))
        .map(|a| a.value())
}
pub(super) fn space(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{1c}'..='\u{1f}')
}
pub(super) fn integer(text: Option<&str>) -> Option<i64> {
    integer_text(text)?.parse().ok()
}
pub(crate) fn integer_text(text: Option<&str>) -> Option<String> {
    let text = text?.trim_matches(space);
    let mut normalized = String::with_capacity(text.len());
    let mut digit = false;
    for (index, c) in text.chars().enumerate() {
        if index == 0 && matches!(c, '+' | '-') {
            normalized.push(c);
            continue;
        }
        if c == '_' && digit {
            digit = false;
            continue;
        }
        normalized.push(char::from(b'0' + decimal(c)?));
        digit = true;
    }
    if !digit {
        return None;
    }
    Some(normalized)
}
pub(super) fn decimal(c: char) -> Option<u8> {
    let zeroes = decimal_data::DECIMAL_ZEROES;
    let code = c as u32;
    let index = zeroes
        .partition_point(|zero| *zero <= code)
        .checked_sub(1)?;
    let value = code - zeroes[index];
    if value < 10 {
        Some(value as u8)
    } else {
        None
    }
}
pub(super) fn text(node: Node<'_, '_>) -> String {
    node.text().unwrap_or("").to_owned()
}

/// XML buffers have one owner; tree views borrow them instead of copying nodes.
pub struct Source {
    pub main_part: String,
    pub document: String,
    pub styles: Option<String>,
    pub numbering: Option<String>,
    pub relationships: Vec<Relationship>,
    pub package: Package,
    pub normalization: normalize::Report,
}
impl Source {
    pub fn open(bytes: Arc<[u8]>) -> Result<Self> {
        Self::from_package(Package::open(bytes)?)
    }
    pub fn from_package(mut package: Package) -> Result<Self> {
        let normalization = normalize::apply(&mut package)?;
        if package.format()? != Format::Docx {
            return Err(Error::malformed("not a DOCX main document"));
        }
        let main_part = package.main_part()?;
        let document = package
            .read_xml(&main_part)?
            .ok_or_else(|| Error::missing(&main_part))?;
        let relationships = package.relationships(Some(&main_part))?;
        let mut parts = BTreeMap::new();
        for relationship in &relationships {
            if !relationship.external && matches!(relationship.kind_tail(), "styles" | "numbering")
            {
                if let Some(part) = &relationship.resolved {
                    // Keep missing optional parts distinct from malformed present XML.
                    parts.insert(relationship.kind_tail(), package.read_xml(part)?);
                }
            }
        }
        Ok(Self {
            main_part,
            document,
            styles: parts.remove("styles").flatten(),
            numbering: parts.remove("numbering").flatten(),
            relationships,
            package,
            normalization,
        })
    }
    pub fn context(&self) -> Result<Context<'_>> {
        Context::from_parts(
            &self.document,
            self.styles.as_deref(),
            self.numbering.as_deref(),
            &self.relationships,
        )
    }
}

pub struct Context<'a> {
    document: roxmltree::Document<'a>,
    pub styles: Styles,
    pub numbering: Numbering,
    relationships: BTreeMap<String, Relationship>,
}
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Element {
    pub text: String,
    pub format: Option<Formatting>,
    pub hyperlink: Option<String>,
}
#[derive(Debug, Serialize)]
pub struct RunFacts {
    pub text: String,
    pub format: Option<Formatting>,
    pub hidden: bool,
}
#[derive(Debug, Serialize)]
pub struct ParagraphFacts {
    pub text: String,
    pub label: String,
    pub heading_level: Option<i64>,
    pub num_id: Option<i64>,
    pub level: Option<i64>,
    pub outline_level: Option<i64>,
    pub toc_level: Option<i64>,
    pub numbered: bool,
    pub start: Option<String>,
    pub elements: Vec<Element>,
    pub runs: Vec<RunFacts>,
    pub field_instructions: Vec<String>,
}
impl<'a> Context<'a> {
    pub fn from_parts(
        document: &'a str,
        styles: Option<&str>,
        numbering: Option<&str>,
        relationships: &[Relationship],
    ) -> Result<Self> {
        Ok(Self {
            document: xml::parse(document)?,
            styles: Styles::parse(Some(styles.unwrap_or(include_str!("default-styles.xml"))))?,
            numbering: Numbering::parse(numbering)?,
            relationships: relationships
                .iter()
                .map(|r| (r.id.clone(), r.clone()))
                .collect(),
        })
    }
    /// Includes table/text-box paragraphs; final body ordering is a separate pass.
    pub fn paragraphs(&self) -> impl Iterator<Item = Node<'_, 'a>> {
        self.document.descendants().filter(|n| is_word(*n, "p"))
    }
    pub fn tables(&self) -> impl Iterator<Item = Node<'_, 'a>> {
        self.document.descendants().filter(|n| is_word(*n, "tbl"))
    }
    pub fn paragraph_facts(&self, paragraph: Node<'_, 'a>) -> Result<ParagraphFacts> {
        let contents = fields::contents(paragraph);
        let text = fields::plain_text(&contents);
        let (label, heading_level) = self.styles.label(paragraph)?;
        let properties = self.styles.numbering(paragraph)?;
        let num_id = properties.as_ref().and_then(|p| p.num_id);
        let level = properties.as_ref().and_then(|p| p.level).or_else(|| {
            if num_id == Some(0) {
                None
            } else {
                num_id.and_then(|id| {
                    self.numbering
                        .infer_level(id, &self.styles.chain_ids(paragraph))
                })
            }
        });
        let toc_level = self.styles.toc_level(paragraph)?;
        let numbered = num_id
            .zip(level)
            .is_some_and(|(id, level)| self.numbering.is_numbered(id, level));
        let start = num_id
            .zip(level)
            .and_then(|(id, level)| self.numbering.start(id, level))
            .map(|v| v.to_string());
        let mut runs = Vec::new();
        for content in &contents {
            match content {
                fields::Content::Run(run) => runs.push(RunFacts {
                    text: fields::run_text(*run),
                    format: if fields::hidden(*run) {
                        None
                    } else {
                        Some(self.styles.format(*run, paragraph)?)
                    },
                    hidden: fields::hidden(*run),
                }),
                fields::Content::Hyperlink(link) => {
                    for run in link.children().filter(|n| is_word(*n, "r")) {
                        runs.push(RunFacts {
                            text: fields::run_text(run),
                            format: if fields::hidden(run) {
                                None
                            } else {
                                Some(self.styles.format(run, paragraph)?)
                            },
                            hidden: fields::hidden(run),
                        });
                    }
                }
            }
        }
        let elements = fields::elements(
            &contents,
            paragraph,
            &self.styles,
            &self.relationships,
            toc_level.is_some(),
        )?;
        Ok(ParagraphFacts {
            text,
            label,
            heading_level,
            num_id,
            level,
            outline_level: self.styles.outline(paragraph)?,
            toc_level,
            numbered,
            start,
            elements,
            runs,
            field_instructions: fields::instructions(paragraph),
        })
    }
}
