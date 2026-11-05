//! Lazy table image extraction. Raster/SVG bytes are preserved; no PNG roundtrip.
use super::{
    table_html::{Html, Resources},
    *,
};
use crate::{package::ContentTypes, ErrorKind};
#[derive(Clone, Debug, serde::Serialize)]
pub struct VectorReport {
    pub part: String,
    pub width: u64,
    pub height: u64,
    pub partial: bool,
    pub diagnostics: Vec<crate::metafile::model::Diagnostic>,
}
#[derive(Clone, Debug, serde::Serialize)]
pub struct ImageFallback {
    pub part: String,
    pub kind: String,
    pub message: String,
}

pub struct Images<'a> {
    package: &'a mut Package,
    types: ContentTypes,
    pub(super) relationships: BTreeMap<String, Relationship>,
    parts: BTreeMap<String, Arc<[u8]>>,
    rendered_vectors: BTreeMap<String, Arc<str>>,
    vector_reports: Vec<VectorReport>,
    fallbacks: Vec<ImageFallback>,
    pub(super) image_probe: crate::equation::image_mtef::Decoder,
}
impl<'a> Images<'a> {
    pub fn new(package: &'a mut Package, relationships: &[Relationship]) -> Result<Self> {
        Ok(Self {
            types: package.content_types()?,
            package,
            parts: BTreeMap::new(),
            rendered_vectors: BTreeMap::new(),
            vector_reports: vec![],
            fallbacks: vec![],
            image_probe: Default::default(),
            relationships: relationships
                .iter()
                .map(|r| (r.id.clone(), r.clone()))
                .collect(),
        })
    }
    pub fn vector_reports(&self) -> &[VectorReport] {
        &self.vector_reports
    }
    pub fn fallbacks(&self) -> &[ImageFallback] {
        &self.fallbacks
    }
    pub(super) fn body_source(
        &mut self,
        path: &str,
        mime: Option<&str>,
        bytes: &[u8],
    ) -> Result<Option<Arc<str>>> {
        if !crate::image::is_vector(bytes, Some(path), mime) {
            return crate::image::serialize_with_decoder(
                bytes,
                Some(path),
                mime,
                &mut self.image_probe,
            );
        }
        if let Some(source) = self.rendered_vectors.get(path) {
            return Ok(Some(Arc::clone(source)));
        }
        let result = crate::image::vector_with_decoder(bytes, None, &mut self.image_probe)?;
        if let Some(rendered) = result.rendered {
            self.vector_reports.push(VectorReport {
                part: path.into(),
                width: rendered.width,
                height: rendered.height,
                partial: rendered.partial,
                diagnostics: rendered.diagnostics,
            });
        }
        if let Some(error) = result.failure {
            self.fallbacks.push(ImageFallback {
                part: path.into(),
                kind: error.kind.code().into(),
                message: error.message,
            });
        }
        self.rendered_vectors
            .insert(path.into(), Arc::clone(&result.source));
        Ok(Some(result.source))
    }
    pub(super) fn switch_relationships(&mut self, relationships: &[Relationship]) {
        self.relationships = relationships
            .iter()
            .map(|r| (r.id.clone(), r.clone()))
            .collect();
    }
    pub(super) fn read_xml(&mut self, path: &str) -> Result<Option<String>> {
        self.part(path)?
            .map(|bytes| crate::xml::decode(&bytes).map(|s| s.into_owned()))
            .transpose()
    }
    pub(super) fn part_relationships(&mut self, path: &str) -> Result<Vec<Relationship>> {
        self.package.relationships(Some(path))
    }
    fn image(&mut self, id: &str, alt: Option<&str>) -> Result<Html> {
        let rel = self
            .relationships
            .get(id)
            .ok_or_else(|| Error::missing(id))?;
        if rel.external {
            // External file access policy belongs to the public document loader.
            return Err(Error::new(
                ErrorKind::Unsupported,
                "external image loader not connected",
            ));
        }
        let path = rel.resolved.clone().ok_or_else(|| Error::missing(id))?;
        let bytes = self.part(&path)?.ok_or_else(|| Error::missing(&path))?;
        let fallback = match path
            .rsplit('.')
            .next()
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str()
        {
            "png" => "image/png",
            "gif" => "image/gif",
            "jpg" | "jpeg" => "image/jpeg",
            "tif" | "tiff" => "image/tiff",
            "bmp" => "image/bmp",
            _ => "None",
        };
        let mime = self.types.get(&path).unwrap_or(fallback);
        self.image_probe.probe(&bytes, Some(&path), Some(mime))?;
        let vector = crate::image::is_vector(&bytes, None, Some(mime));
        let source = if self
            .image_probe
            .limit_reason(&bytes, Some(&path), Some(mime))
            .is_some()
        {
            crate::image::image_limit_placeholder()?.to_string()
        } else if vector {
            if let Some(cached) = self.rendered_vectors.get(&path) {
                cached.to_string()
            } else {
                let result =
                    crate::image::vector_with_decoder(&bytes, None, &mut self.image_probe)?;
                let source = result.source.to_string();
                if let Some(result) = result.rendered {
                    self.vector_reports.push(VectorReport {
                        part: path.clone(),
                        width: result.width,
                        height: result.height,
                        partial: result.partial,
                        diagnostics: result.diagnostics,
                    });
                }
                if let Some(error) = result.failure {
                    self.fallbacks.push(ImageFallback {
                        part: path.clone(),
                        kind: error.kind.code().into(),
                        message: error.message,
                    });
                }
                self.rendered_vectors
                    .insert(path, Arc::from(source.as_str()));
                source
            }
        } else {
            format!("data:{mime};base64,{}", base64::encode(&bytes))
        };
        let mut html = Html::element("img", false, vec![]).attribute("src", source);
        if let Some(alt) = alt.filter(|s| !s.is_empty()) {
            html = html.attribute("alt", alt);
        }
        Ok(html)
    }
    pub(super) fn part(&mut self, path: &str) -> Result<Option<Arc<[u8]>>> {
        if let Some(bytes) = self.parts.get(path) {
            return Ok(Some(Arc::clone(bytes)));
        }
        let bytes = self.package.read(path)?;
        if let Some(bytes) = &bytes {
            self.parts.insert(path.into(), Arc::clone(bytes));
        }
        Ok(bytes)
    }
    pub(super) fn payload(
        &mut self,
        id: Option<&str>,
    ) -> Result<Option<(String, Option<String>, Arc<[u8]>)>> {
        let Some(rel) = id
            .and_then(|id| self.relationships.get(id))
            .filter(|r| !r.external)
        else {
            return Ok(None);
        };
        let Some(path) = rel.resolved.clone() else {
            return Ok(None);
        };
        let Some(bytes) = self.part(&path)? else {
            return Ok(None);
        };
        let mime = self.types.get(&path).map(str::to_owned);
        Ok(Some((path, mime, bytes)))
    }
}
impl Resources for Images<'_> {
    fn render(&mut self, node: Node<'_, '_>) -> Result<Vec<Html>> {
        let namespace = node.tag_name().namespace().unwrap_or("");
        if namespace == "urn:schemas-microsoft-com:vml" && node.tag_name().name() == "imagedata" {
            let Some(id) = relationship_attr(node, "id") else {
                return Ok(vec![]);
            };
            // Mammoth's Office XML namespace map has no `o` prefix, so its
            // v:imagedata reader does not resolve namespaced o:title as alt.
            return Ok(vec![self.image(id, None)?]);
        }
        if crate::package::namespace_family(Some(namespace), "drawingml/wordprocessingDrawing")
            && matches!(node.tag_name().name(), "inline" | "anchor")
        {
            let props = node
                .children()
                .find(|n| drawing(*n, "docPr", "wordprocessingDrawing"));
            let alt = props
                .and_then(|n| n.attribute("descr"))
                .filter(|s| !s.trim_matches(space).is_empty())
                .or_else(|| props.and_then(|n| n.attribute("title")));
            let href = props
                .and_then(|n| n.children().find(|n| drawing(*n, "hlinkClick", "main")))
                .and_then(|n| relationship_attr(n, "id"))
                .filter(|id| !id.is_empty())
                .map(|id| {
                    self.relationships
                        .get(id)
                        .map(|r| r.target.clone())
                        .ok_or_else(|| Error::missing(id))
                })
                .transpose()?;
            let mut output = Vec::new();
            // Restrict to Mammoth's graphic/graphicData/pic/blipFill path;
            // alternate representations elsewhere must not duplicate the image.
            for graphic in node.children().filter(|n| drawing(*n, "graphic", "main")) {
                for data in graphic
                    .children()
                    .filter(|n| drawing(*n, "graphicData", "main"))
                {
                    for pic in data.children().filter(|n| drawing(*n, "pic", "picture")) {
                        for fill in pic
                            .children()
                            .filter(|n| drawing(*n, "blipFill", "picture"))
                        {
                            for blip in fill.children().filter(|n| drawing(*n, "blip", "main")) {
                                let Some(id) = relationship_attr(blip, "embed")
                                    .or_else(|| relationship_attr(blip, "link"))
                                else {
                                    continue;
                                };
                                let image = self.image(id, alt)?;
                                output.push(if let Some(href) = &href {
                                    Html::element("a", true, vec![image]).attribute("href", href)
                                } else {
                                    image
                                });
                            }
                        }
                    }
                }
            }
            return Ok(output);
        }
        Err(Error::new(
            ErrorKind::Unsupported,
            format!(
                "DOCX resource renderer not connected: {}",
                node.tag_name().name()
            ),
        ))
    }
}
fn drawing(node: Node<'_, '_>, name: &str, family: &str) -> bool {
    node.is_element()
        && node.tag_name().name() == name
        && crate::package::namespace_family(
            node.tag_name().namespace(),
            &format!("drawingml/{family}"),
        )
}
fn relationship_attr<'a>(node: Node<'a, '_>, name: &str) -> Option<&'a str> {
    node.attributes()
        .find(|a| {
            a.name() == name
                && a.namespace()
                    .is_some_and(|ns| ns.ends_with("/relationships"))
        })
        .map(|a| a.value())
}
