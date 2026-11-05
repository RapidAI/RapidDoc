//! PICF inline pictures and MathType image comments, cached by Data offset.
use crate::{
    docx::body::{Block, Content, Kind},
    equation::image_mtef::Decoder,
    officeart::{self, Payload},
    xls::records::Budget,
    ErrorKind, Result,
};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};
#[derive(Clone)]
pub enum Visual {
    Image(Arc<str>),
    Equation(String),
    Chart {
        html: String,
        preview: Option<Arc<str>>,
    },
}
impl Visual {
    pub fn block(&self) -> Block {
        match self {
            Self::Image(source) => {
                let mut b = Block::new(Kind::Image, Content::None);
                b.image_base64 = Some(source.clone());
                b
            }
            Self::Equation(latex) => Block::new(Kind::Equation, Content::Text(latex.clone())),
            Self::Chart { html, preview } => {
                let mut block = Block::new(Kind::Chart, Content::Text(html.clone()));
                block.image_base64 = preview.clone();
                block
            }
        }
    }
    pub fn html(&self) -> String {
        match self {
            Self::Image(source) => format!("<img src=\"{source}\"/>"),
            Self::Equation(latex) => format!(
                "<eq>{}</eq>",
                latex
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
                    .replace('>', "&gt;")
            ),
            Self::Chart { html, preview } => format!(
                "{}{html}",
                preview
                    .as_ref()
                    .map(|p| format!("<img src=\"{p}\"/>"))
                    .unwrap_or_default()
            ),
        }
    }
}
#[derive(Default)]
pub struct Resources {
    cache: BTreeMap<u32, Option<Visual>>,
    identities: BTreeMap<u32, ImageKey>,
    payload_cache: BTreeMap<ImageKey, Option<Visual>>,
    payloads: BTreeMap<ImageKey, Payload>,
    previews: BTreeMap<ImageKey, Option<Arc<str>>>,
    decoder: Decoder,
    budget: Budget,
    digests: BTreeSet<[u8; 32]>,
    total: usize,
    pub diagnostics: Vec<String>,
}
pub type ImageKey = ([u8; 32], Option<[i64; 2]>);
impl Resources {
    pub fn identity(&self, offset: u32) -> Option<ImageKey> {
        self.identities.get(&offset).copied()
    }
    /// Chart previews remain pictures even when the image contains MathType data.
    pub fn preview(&mut self, offset: u32) -> Result<Option<Arc<str>>> {
        let Some(key) = self.identity(offset) else {
            return Ok(None);
        };
        if let Some(value) = self.previews.get(&key) {
            return Ok(value.clone());
        }
        let Some(payload) = self.payloads.get(&key) else {
            return Ok(None);
        };
        let value = match payload.serialize_with_decoder(&mut self.decoder) {
            Ok(v) => v,
            Err(e) if e.kind == ErrorKind::ResourceLimit => return Err(e),
            Err(e) => {
                self.diagnostics
                    .push(format!("DOC_IMAGE_FALLBACK: chart preview: {e}"));
                None
            }
        };
        self.previews.insert(key, value.clone());
        Ok(value)
    }
    pub fn inline(&mut self, data: &[u8], offset: u32) -> Result<Option<Visual>> {
        if let Some(v) = self.cache.get(&offset) {
            return Ok(v.clone());
        }
        let p = offset as usize;
        let decoded = (|| -> Result<Option<Visual>> {
            let Some(h) = data.get(p..p + 6) else {
                return Ok(None);
            };
            let size = u32::from_le_bytes(h[..4].try_into().unwrap()) as usize;
            let header = u16::from_le_bytes(h[4..6].try_into().unwrap()) as usize;
            if size < header {
                return Ok(None);
            }
            let Some(picf) = data.get(p..p + size) else {
                self.diagnostics.push(format!(
                    "DOC picture at Data offset {offset} is out of bounds"
                ));
                return Ok(None);
            };
            let art = &picf[header..];
            let mut payload = None;
            let mut image_limit = false;
            officeart::visit(art, &mut self.budget, |r, budget| {
                if payload.is_some() || image_limit {
                    return Ok(());
                }
                let inner = if r.kind == 0xf007 && r.payload.len() >= 36 {
                    officeart::at(
                        r.payload,
                        36 + r.payload[33] as usize,
                        r.payload.len(),
                        budget,
                    )?
                } else {
                    None
                };
                match officeart::blip(inner.unwrap_or(r)) {
                    Ok(v) => payload = v,
                    Err(_) => image_limit = true,
                }
                Ok(())
            })?;
            if image_limit {
                self.diagnostics.push(format!(
                    "DOC_IMAGE_LIMIT: placeholder at Data offset {offset}"
                ));
                return self.convert_payload(Payload::placeholder()?);
            }
            if payload.is_none() {
                for (magic, extension, content_type) in [
                    (b"\xd7\xcd\xc6\x9a" as &[u8], "wmf", "image/wmf"),
                    (b"\x89PNG\r\n\x1a\n", "png", "image/png"),
                    (b"\xff\xd8\xff", "jpg", "image/jpeg"),
                    (b"GIF8", "gif", "image/gif"),
                    (b"BM", "bmp", "image/bmp"),
                    (b"II*\0", "tiff", "image/tiff"),
                    (b"MM\0*", "tiff", "image/tiff"),
                ] {
                    if let Some(i) = art.windows(magic.len()).position(|w| w == magic) {
                        payload = Some(Payload {
                            data: Arc::from(&art[i..]),
                            extension,
                            content_type,
                            render_size_emu: None,
                        });
                        break;
                    }
                }
            }
            let Some(payload) = payload else {
                return Ok(None);
            };
            self.identities.insert(
                offset,
                (
                    Sha256::digest(&payload.data).into(),
                    payload.render_size_emu,
                ),
            );
            self.convert_payload(payload)
        })()?;
        self.cache.insert(offset, decoded.clone());
        Ok(decoded)
    }
    fn convert_payload(&mut self, payload: Payload) -> Result<Option<Visual>> {
        let digest: [u8; 32] = Sha256::digest(&payload.data).into();
        let key = (digest, payload.render_size_emu);
        if let Some(cached) = self.payload_cache.get(&key) {
            return Ok(cached.clone());
        }
        if self.digests.insert(digest) {
            self.total += payload.data.len();
            if self.total > 128 * 1024 * 1024 {
                self.total -= payload.data.len();
                self.diagnostics
                    .push("OFFICE_IMAGE_LIMIT: aggregate image budget; using placeholder".into());
                return Ok(Some(
                    Visual::Image(crate::image::image_limit_placeholder()?),
                ));
            }
        }
        if let Some(latex) = self.decoder.probe(
            &payload.data,
            Some(&format!("image.{}", payload.extension)),
            Some(payload.content_type),
        )? {
            self.payloads.insert(key, payload);
            let visual = Some(Visual::Equation(latex));
            self.payload_cache.insert(key, visual.clone());
            return Ok(visual);
        }
        if self
            .decoder
            .limit_reason(
                &payload.data,
                Some(&format!("image.{}", payload.extension)),
                Some(payload.content_type),
            )
            .is_some()
        {
            let visual = Some(Visual::Image(crate::image::image_limit_placeholder()?));
            self.payload_cache.insert(key, visual.clone());
            return Ok(visual);
        }
        let visual = match payload.serialize_with_decoder(&mut self.decoder) {
            Ok(value) => {
                self.previews.insert(key, value.clone());
                Ok(value.map(Visual::Image))
            }
            Err(e) if e.kind == ErrorKind::ResourceLimit => Err(e),
            Err(e) => {
                self.diagnostics
                    .push(format!("DOC_IMAGE_FALLBACK: image serialization: {e}"));
                Ok(None)
            }
        }?;
        self.payload_cache.insert(key, visual.clone());
        Ok(visual)
    }
    pub fn floating(
        &mut self,
        table: &[u8],
        word: &[u8],
        shape: super::fib::Pair,
        drawing: super::fib::Pair,
    ) -> Result<Vec<(u32, Visual)>> {
        let range = |pair: super::fib::Pair| {
            table.get(pair.fc as usize..(pair.fc as usize).checked_add(pair.lcb as usize)?)
        };
        let (Some(plc), Some(drawing)) = (range(shape), range(drawing)) else {
            return Ok(vec![]);
        };
        if plc.len() < 4 || (plc.len() - 4) % 30 != 0 {
            return Ok(vec![]);
        }
        let count = (plc.len() - 4) / 30;
        if count == 0 {
            return Ok(vec![]);
        }
        let read = |d: &[u8], p: usize| u32::from_le_bytes(d[p..p + 4].try_into().unwrap());
        let anchors: Vec<_> = (0..count)
            .map(|i| (read(plc, (count + 1) * 4 + i * 26), read(plc, i * 4)))
            .collect();
        let assets =
            officeart::bstore(drawing, Some(word), &mut self.budget, &mut self.diagnostics)?;
        let first = officeart::at(drawing, 0, drawing.len(), &mut self.budget)?;
        let mut cursor = first.map(|r| 8 + r.payload.len()).unwrap_or(drawing.len());
        let mut shapes = BTreeMap::new();
        while cursor < drawing.len() {
            cursor += 1;
            let Some(container) = officeart::at(drawing, cursor, drawing.len(), &mut self.budget)?
            else {
                break;
            };
            for shape in officeart::excel_shapes(container.payload, &mut self.budget)? {
                if let Some(id) = shape.shape_id {
                    shapes.entry(id).or_insert(shape);
                }
            }
            cursor += 8 + container.payload.len();
        }
        let mut out = vec![];
        for (id, shape) in shapes {
            if shape.hidden {
                continue;
            }
            let Some(payload) = shape.pib.and_then(|p| assets.get(&p)) else {
                continue;
            };
            let cp = anchors
                .iter()
                .find(|(key, _)| *key == id)
                .or_else(|| {
                    anchors
                        .iter()
                        .filter(|(key, _)| *key <= id)
                        .max_by_key(|(key, cp)| (*key, *cp))
                })
                .map(|(_, cp)| *cp);
            if let Some(cp) = cp {
                if let Some(v) = self.convert_payload(payload.clone())? {
                    out.push((cp, v));
                }
            }
        }
        Ok(out)
    }
}
