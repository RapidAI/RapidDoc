// Copyright (c) 2026 Xiaomeng Zhao (myhloli). SPDX-License-Identifier: MIT
//! Per-render decode cache. Decoded images remain immutable across placements.
use super::{
    dib_decode::{self, RgbaImage},
    mask::Mask,
    model::*,
    sampling,
};
use crate::Result;
use std::{
    collections::HashMap,
    hash::{Hash, Hasher},
    sync::Arc,
};
#[derive(Clone, PartialEq)]
struct ClipKey {
    clip: Arc<[Clip]>,
    matrix: Matrix,
    size: [usize; 2],
}
impl Eq for ClipKey {}
fn float_hash<H: Hasher>(v: f64, h: &mut H) {
    if v == 0. {
        0u64.hash(h);
    } else {
        v.to_bits().hash(h);
    }
}
impl Hash for ClipKey {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.size.hash(h);
        for v in [
            self.matrix.a,
            self.matrix.b,
            self.matrix.c,
            self.matrix.d,
            self.matrix.e,
            self.matrix.f,
        ] {
            float_hash(v, h);
        }
        self.clip.len().hash(h);
        for c in self.clip.iter() {
            c.mode.hash(h);
            c.fill_rule.hash(h);
            c.path.segments.len().hash(h);
            for s in c.path.segments.iter() {
                (s.verb as u8).hash(h);
                s.points.len().hash(h);
                for p in &s.points {
                    for &v in p {
                        float_hash(v, h);
                    }
                }
            }
        }
    }
}
#[derive(Clone, PartialEq, Eq, Hash)]
enum Key {
    Image(ImagePayload),
    Clip(ClipKey),
}
#[derive(Clone)]
enum Cached {
    Image(Arc<RgbaImage>),
    Mask(Arc<Mask>),
}
impl Cached {
    fn bytes(&self) -> usize {
        match self {
            Self::Image(i) => i.pixels().len(),
            Self::Mask(m) => m.pixels().len(),
        }
    }
}
struct Entry {
    value: Cached,
    stamp: u64,
}
pub struct ImageSession {
    entries: HashMap<Key, Entry>,
    limit: usize,
    used: usize,
    clock: u64,
    hits: usize,
    misses: usize,
    reported_fonts: std::collections::BTreeSet<(String, String)>,
    pub font_diagnostics: Vec<Diagnostic>,
    pub font_diagnostics_omitted: usize,
}
impl Default for ImageSession {
    fn default() -> Self {
        Self::new(64 * 1024 * 1024)
    }
}
impl ImageSession {
    pub fn new(limit: usize) -> Self {
        Self {
            entries: HashMap::new(),
            limit,
            used: 0,
            clock: 0,
            hits: 0,
            misses: 0,
            reported_fonts: Default::default(),
            font_diagnostics: vec![],
            font_diagnostics_omitted: 0,
        }
    }
    pub fn used(&self) -> usize {
        self.used
    }
    pub fn font(&mut self, c: &TextCommand, size: u32) -> Result<std::rc::Rc<super::fonts::Face>> {
        let face = super::fonts::resolve(&c.font, size)?;
        if let Some(reason) = &face.reason {
            if self.reported_fonts.insert((
                c.font.face_name.clone(),
                format!("{}{}", face.family, face.style),
            )) {
                if self.font_diagnostics.len() < 256 {
                    self.font_diagnostics.push(Diagnostic {
                        code: "font_substituted".into(),
                        level: "info".into(),
                        message: format!(
                            "{} -> {} {}: {reason}",
                            c.font.face_name, face.family, face.style
                        ),
                        record_type: c.location.record_type,
                        record_index: c.location.record_index,
                        offset: c.location.offset,
                    });
                } else {
                    self.font_diagnostics_omitted += 1;
                }
            }
        }
        Ok(face)
    }
    pub fn hits(&self) -> usize {
        self.hits
    }
    pub fn misses(&self) -> usize {
        self.misses
    }
    pub fn clear(&mut self) {
        self.entries.clear();
        self.used = 0;
    }
    pub fn decode<P: Clone + Into<ImagePayload>>(&mut self, payload: &P) -> Result<Arc<RgbaImage>> {
        let payload = payload.clone().into();
        let key = Key::Image(payload.clone());
        if let Some(Cached::Image(image)) = self.get(&key) {
            return Ok(image);
        }
        let image = Arc::new(match &payload {
            ImagePayload::Dib(p) => dib_decode::decode(p)?,
            ImagePayload::Encoded { png } => super::emfplus::image::png(png)?,
        });
        self.put(key, Cached::Image(Arc::clone(&image)));
        Ok(image)
    }
    pub(crate) fn get_mask(
        &mut self,
        clip: &Arc<[Clip]>,
        matrix: Matrix,
        size: [usize; 2],
    ) -> Option<Arc<Mask>> {
        match self.get(&Key::Clip(ClipKey {
            clip: Arc::clone(clip),
            matrix,
            size,
        })) {
            Some(Cached::Mask(m)) => Some(m),
            _ => None,
        }
    }
    pub(crate) fn put_mask(
        &mut self,
        clip: &Arc<[Clip]>,
        matrix: Matrix,
        size: [usize; 2],
        mask: Arc<Mask>,
    ) {
        self.put(
            Key::Clip(ClipKey {
                clip: Arc::clone(clip),
                matrix,
                size,
            }),
            Cached::Mask(mask),
        );
    }
    fn get(&mut self, key: &Key) -> Option<Cached> {
        self.clock += 1;
        if let Some(entry) = self.entries.get_mut(key) {
            self.hits += 1;
            entry.stamp = self.clock;
            return Some(entry.value.clone());
        }
        self.misses += 1;
        None
    }
    fn put(&mut self, key: Key, value: Cached) {
        let bytes = value.bytes();
        if bytes <= self.limit {
            if let Some(old) = self.entries.remove(&key) {
                self.used -= old.value.bytes();
            }
            while self.used + bytes > self.limit {
                let oldest = self
                    .entries
                    .iter()
                    .min_by_key(|(_, v)| v.stamp)
                    .map(|(k, _)| k.clone())
                    .unwrap();
                let entry = self.entries.remove(&oldest).unwrap();
                self.used -= entry.value.bytes();
            }
            self.entries.insert(
                key,
                Entry {
                    value,
                    stamp: self.clock,
                },
            );
            self.used += bytes;
        }
    }
    pub fn unclipped_layer(
        &mut self,
        command: &ImageCommand,
        matrix: Matrix,
        size: [usize; 2],
    ) -> Result<RgbaImage> {
        let decoded = self.decode(&command.image)?;
        sampling::unclipped_layer(&decoded, command, matrix, size)
    }
}

/// Native bitmap-only document backend, until shared path/text/clip rendering
/// is implemented. Reject those documents explicitly instead of dropping them.
pub fn render_bitmaps(
    document: &super::playback::GeometryDocument,
    session: &mut ImageSession,
) -> Result<RgbaImage> {
    use crate::{Error, ErrorKind};
    if document.commands.is_empty()
        || document.commands.iter().any(|c| match c {
            Command::Path(_) => true,
            Command::Image(c) => !c.clip.is_empty(),
            Command::Text(_) => true,
            Command::Clear(_) => true,
        })
    {
        return Err(Error::new(
            ErrorKind::Unsupported,
            "path, text and clip raster rendering not connected",
        ));
    }
    super::header::check_render_size(
        document.width,
        document.height,
        document.commands.len() as u64,
    )?;
    let size = [document.width as usize, document.height as usize];
    let bounds = document.bounds.normalized();
    let (sx, sy) = (
        document.width as f64 / bounds.width(),
        document.height as f64 / bounds.height(),
    );
    let matrix = Matrix {
        a: sx,
        d: sy,
        e: -bounds.left * sx,
        f: -bounds.top * sy,
        ..Matrix::default()
    };
    let mut canvas = RgbaImage::blank(size)?;
    for p in canvas.pixels.chunks_exact_mut(4) {
        p[..3].fill(255);
    }
    for command in &document.commands {
        let Command::Image(command) = command else {
            unreachable!()
        };
        let layer = session.unclipped_layer(command, matrix, size)?;
        super::composite::image(&mut canvas.pixels, layer.pixels(), command.rop)?;
    }
    Ok(canvas)
}
