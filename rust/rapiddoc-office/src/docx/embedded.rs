//! Per-part shared resources. Formula detection and image output reuse the same
//! ZIP reader and Arc payloads; no second package parse or document JSON bridge.
use super::{
    formulas::{Embedded, Kind, NativeEmbedded, Token, Tokens},
    table_html::{Html, Resources},
    table_images::Images,
    *,
};
use crate::equation::mtef;
use std::{cell::RefCell, collections::BTreeSet};

pub struct Shared<'a> {
    inner: RefCell<Store<'a>>,
}
struct Store<'a> {
    images: Images<'a>,
    native: NativeEmbedded,
    ole: mtef::OoxmlDecoder,
    image_cache: BTreeMap<String, Option<String>>,
    ole_cache: BTreeMap<String, Option<String>>,
    warnings: BTreeSet<String>,
}
impl<'a> Shared<'a> {
    pub fn new(package: &'a mut Package, relationships: &[Relationship]) -> Result<Self> {
        Ok(Self {
            inner: RefCell::new(Store {
                images: Images::new(package, relationships)?,
                native: NativeEmbedded::default(),
                ole: mtef::OoxmlDecoder::default(),
                image_cache: BTreeMap::new(),
                ole_cache: BTreeMap::new(),
                warnings: BTreeSet::new(),
            }),
        })
    }
    pub fn embedded(&self) -> FormulaHandler<'_, 'a> {
        FormulaHandler(self)
    }
    pub fn images(&self) -> ImageHandler<'_, 'a> {
        ImageHandler(self)
    }
    pub fn vector_reports(&self) -> Vec<super::table_images::VectorReport> {
        self.inner.borrow().images.vector_reports().to_vec()
    }
    pub fn image_fallbacks(&self) -> Vec<super::table_images::ImageFallback> {
        self.inner.borrow().images.fallbacks().to_vec()
    }
    pub(super) fn image_payload(
        &self,
        id: &str,
    ) -> Result<Option<(String, Option<String>, Arc<[u8]>)>> {
        self.inner.borrow_mut().images.payload(Some(id))
    }
    pub(super) fn serialize_body_image(
        &self,
        path: &str,
        mime: Option<&str>,
        data: &[u8],
    ) -> Result<Option<Arc<str>>> {
        self.inner.borrow_mut().images.body_source(path, mime, data)
    }
    /// Change the part-local relationship namespace while retaining document
    /// caches and cumulative OLE/image equation budgets.
    pub(super) fn switch_relationships(&self, relationships: &[Relationship]) {
        let mut state = self.inner.borrow_mut();
        state.images.switch_relationships(relationships);
        state.image_cache.clear();
        state.ole_cache.clear();
    }
    pub(super) fn read_xml(&self, path: &str) -> Result<Option<String>> {
        self.inner.borrow_mut().images.read_xml(path)
    }
    pub(super) fn read_part(&self, path: &str) -> Result<Option<Arc<[u8]>>> {
        self.inner.borrow_mut().images.part(path)
    }
    pub(super) fn part_relationships(&self, path: &str) -> Result<Vec<Relationship>> {
        self.inner.borrow_mut().images.part_relationships(path)
    }
    pub fn candidate_total_bytes(&self) -> usize {
        self.inner.borrow().images.image_probe.total_bytes
    }
    pub fn ole_total_bytes(&self) -> usize {
        self.inner.borrow().ole.total_bytes
    }
    pub fn warnings(&self) -> Vec<String> {
        let state = self.inner.borrow();
        state
            .warnings
            .iter()
            .cloned()
            .chain(
                state
                    .images
                    .image_probe
                    .warnings
                    .iter()
                    .map(|(format, digest)| {
                        format!(
                            "image MTEF fallback {format:?} sha256={}",
                            digest[..8]
                                .iter()
                                .map(|v| format!("{v:02x}"))
                                .collect::<String>()
                        )
                    }),
            )
            .collect()
    }
    /// Same preview arbitration as Python's _picture_is_equation_preview. The
    /// final body image assembler must call this before publishing an image.
    pub fn is_equation_preview(&self, image: Node<'_, '_>) -> Result<bool> {
        if self.embedded().decode(image)?.is_some() {
            return Ok(true);
        }
        for ancestor in image.ancestors().skip(1).filter(|n| n.is_element()) {
            match ancestor.tag_name().name() {
                "shape" => {
                    if self.embedded().decode(ancestor)?.is_some() {
                        return Ok(true);
                    }
                }
                "object" | "AlternateContent" => {
                    if Tokens::new(self.embedded())
                        .extract(ancestor)?
                        .iter()
                        .any(|t| matches!(t, Token::Formula(..)))
                    {
                        return Ok(true);
                    }
                }
                _ => (),
            }
        }
        Ok(false)
    }
}
pub struct FormulaHandler<'s, 'p>(&'s Shared<'p>);
pub struct ImageHandler<'s, 'p>(&'s Shared<'p>);
fn relationship<'a>(node: Node<'a, '_>, key: &str) -> Option<&'a str> {
    node.attributes()
        .find(|a| a.name() == key && a.namespace().is_some_and(|s| s.ends_with("/relationships")))
        .map(|a| a.value())
}
impl Embedded for FormulaHandler<'_, '_> {
    fn decode(&mut self, node: Node<'_, '_>) -> Result<Option<(Kind, String)>> {
        let mut state = self.0.inner.borrow_mut();
        let tag = node.tag_name().name();
        if tag == "shape" {
            return state.native.decode(node);
        }
        if tag == "OLEObject" {
            let prog_id = node
                .attribute("ProgID")
                .or_else(|| node.attribute("ProgId"));
            if !mtef::equation_prog_id(prog_id)
                || !node
                    .attribute("Type")
                    .unwrap_or("Embed")
                    .trim_matches(space)
                    .eq_ignore_ascii_case("embed")
                || node
                    .attribute("DrawAspect")
                    .unwrap_or("Content")
                    .trim_matches(space)
                    .eq_ignore_ascii_case("icon")
            {
                return Ok(None);
            }
            let Some(id) = relationship(node, "id") else {
                return Ok(None);
            };
            if let Some(value) = state.ole_cache.get(id) {
                return Ok(value.clone().map(|s| (Kind::Mtef, s)));
            }
            let Some(rel) = state.images.relationships.get(id).filter(|r| {
                !r.external
                    && crate::ole::casefold(r.kind.trim_end_matches('/')).ends_with("/oleobject")
            }) else {
                return Ok(None);
            };
            let _ = rel;
            let payload = state.images.payload(Some(id))?.map(|(_, _, data)| data);
            let value = state.ole.decode(payload, prog_id, false)?;
            if value.is_none() {
                state
                    .warnings
                    .insert(format!("OLE MTEF fallback relationship={id}"));
            }
            state.ole_cache.insert(id.into(), value.clone());
            return Ok(value.map(|s| (Kind::Mtef, s)));
        }
        if matches!(tag, "blip" | "imagedata") {
            let id = relationship(node, "embed")
                .filter(|id| !id.is_empty())
                .or_else(|| relationship(node, "id"));
            let Some(id) = id.filter(|id| !id.is_empty()) else {
                return Ok(None);
            };
            if let Some(value) = state.image_cache.get(id) {
                return Ok(value.clone().map(|s| (Kind::ImageMtef, s)));
            }
            let Some((path, mime, data)) = state.images.payload(Some(id))? else {
                return Ok(None);
            };
            let value = state
                .images
                .image_probe
                .probe(&data, Some(&path), mime.as_deref())?;
            state.image_cache.insert(id.into(), value.clone());
            return Ok(value.map(|s| (Kind::ImageMtef, s)));
        }
        Ok(None)
    }
}
impl Resources for ImageHandler<'_, '_> {
    fn render(&mut self, node: Node<'_, '_>) -> Result<Vec<Html>> {
        if node.tag_name().name() == "OLEObject" {
            return Ok(vec![]);
        } // Mammoth ignores OLE before equation injection.
        self.0.inner.borrow_mut().images.render(node)
    }
}
