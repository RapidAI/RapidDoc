//! DOCX VML equationxml, following the local Word 2003 XML decoder.
use super::omml;
use crate::{xml, Error, ErrorKind, Result};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

const WORD: &str = "http://schemas.microsoft.com/office/word/2003/wordml";
const MATH: &str = "http://schemas.openxmlformats.org/officeDocument/2006/math";
// These are the existing Python Office asset limits, not new PDF budgets.
const MAX_ENTRY: usize = 128 * 1024 * 1024;
const MAX_TOTAL: usize = 128 * 1024 * 1024;
#[derive(Default)]
pub struct Decoder {
    pub total_bytes: usize,
    cache: HashMap<[u8; 32], Option<String>>,
}
impl Decoder {
    pub fn decode(&mut self, input: Option<&str>) -> Result<Option<String>> {
        let Some(input) = input.filter(|s| !s.trim_matches(super::omml::whitespace).is_empty())
        else {
            return Ok(None);
        };
        if input.len() > MAX_ENTRY {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "DOCX equationxml exceeds Python max_entry_bytes",
            ));
        }
        let digest: [u8; 32] = Sha256::digest(input.as_bytes()).into();
        if let Some(cached) = self.cache.get(&digest) {
            return Ok(cached.clone());
        }
        if input.len() > MAX_TOTAL.saturating_sub(self.total_bytes) {
            return Err(Error::new(
                ErrorKind::ResourceLimit,
                "DOCX equationxml exceeds Python max_asset_total_bytes",
            ));
        }
        self.total_bytes += input.len();
        let result = decode(input);
        self.cache.insert(digest, result.clone());
        Ok(result)
    }
}
fn matches(node: roxmltree::Node<'_, '_>, ns: &str, name: &str) -> bool {
    node.is_element() && node.tag_name().namespace() == Some(ns) && node.tag_name().name() == name
}
fn decode(input: &str) -> Option<String> {
    // The shared parser disallows DTD/entity loading. Comments are rejected by
    // the original equationxml contract, even when outside the root element.
    let decoded = xml::decode(input.as_bytes()).ok()?;
    let tree = xml::parse(&decoded).ok()?;
    let root = tree.root_element();
    if !matches(root, WORD, "wordDocument") || tree.descendants().any(|n| n.is_comment()) {
        return None;
    }
    if root.descendants().any(|n| {
        n.is_element()
            && matches!(
                n.tag_name().name(),
                "annotation"
                    | "endnote"
                    | "footnote"
                    | "ftr"
                    | "hdr"
                    | "pict"
                    | "textbox"
                    | "txbxContent"
            )
    }) {
        return None;
    }
    let paragraphs: Vec<_> = root
        .descendants()
        .skip(1)
        .filter(|n| matches(*n, WORD, "p"))
        .collect();
    if paragraphs.len() != 1 {
        return None;
    }
    let maths: Vec<_> = paragraphs[0]
        .children()
        .filter(|n| matches(*n, MATH, "oMathPara"))
        .collect();
    if maths.len() != 1 {
        return None;
    }
    let equations: Vec<_> = maths[0]
        .children()
        .filter(|n| matches(*n, MATH, "oMath"))
        .collect();
    if equations.len() != 1
        || root
            .descendants()
            .skip(1)
            .filter(|n| matches(*n, MATH, "oMath"))
            .count()
            != 1
    {
        return None;
    }
    let latex = omml::convert(equations[0]).ok()?;
    let latex = latex.trim_matches(super::omml::whitespace);
    if latex.is_empty() {
        None
    } else {
        Some(latex.to_owned())
    }
}
