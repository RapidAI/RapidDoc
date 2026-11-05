// Copyright (c) 2026 Xiaomeng Zhao (myhloli). SPDX-License-Identifier: MIT
//! Original mixed path/bitmap/text backend and adaptive supersampling.
//! EMF+ command replay is still a separate pending stage.
use super::{
    composite, dib_decode::RgbaImage, geometry::FlattenBudget, header, image_session::ImageSession,
    model::*, paths, playback::GeometryDocument,
};
use crate::Result;
pub fn choose_scale(document: &GeometryDocument, maximum: u32) -> u32 {
    if document
        .commands
        .iter()
        .any(|c| matches!(c, Command::Image(_)))
    {
        return 1;
    }
    let work = work(document);
    for factor in [4, 2] {
        if factor <= maximum
            && document
                .width
                .checked_mul(factor as u64)
                .zip(document.height.checked_mul(factor as u64))
                .is_some_and(|(w, h)| header::check_render_size(w, h, work).is_ok())
        {
            return factor;
        }
    }
    1
}
fn work(document: &GeometryDocument) -> u64 {
    document
        .commands
        .iter()
        .map(|c| {
            1 + match c {
                Command::Path(c) => c.clip.len(),
                Command::Image(c) => c.clip.len(),
                Command::Text(c) => c.clip.len(),
                Command::Clear(c) => c.clip.len(),
            } as u64
        })
        .sum::<u64>()
        .max(1)
}
pub fn render(
    document: &GeometryDocument,
    budget: &mut FlattenBudget,
    session: &mut ImageSession,
) -> Result<RgbaImage> {
    let scale = choose_scale(document, 4);
    let canvas = render_once(document, scale, budget, session)?;
    if scale == 1 {
        return Ok(canvas);
    }
    Ok(super::filters::unsharp(&super::filters::lanczos(
        &canvas,
        [document.width as usize, document.height as usize],
    )?))
}
pub fn render_once(
    document: &GeometryDocument,
    scale: u32,
    budget: &mut FlattenBudget,
    session: &mut ImageSession,
) -> Result<RgbaImage> {
    let width = document.width * scale as u64;
    let height = document.height * scale as u64;
    let work = document
        .commands
        .iter()
        .map(|c| {
            1 + match c {
                Command::Path(c) => c.clip.len(),
                Command::Image(c) => c.clip.len(),
                Command::Text(c) => c.clip.len(),
                Command::Clear(c) => c.clip.len(),
            } as u64
        })
        .sum::<u64>()
        .max(1);
    header::check_render_size(width, height, work)?;
    let size = [width as usize, height as usize];
    let bounds = document.bounds.normalized();
    let (sx, sy) = (
        width as f64 / bounds.width(),
        height as f64 / bounds.height(),
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
        match command {
            Command::Clear(c) => {
                let mask = paths::clip_mask(&c.clip, matrix, size, budget, session)?;
                let ink = [c.color.red, c.color.green, c.color.blue, c.color.alpha];
                for (i, p) in canvas.pixels.chunks_exact_mut(4).enumerate() {
                    let m = mask.as_ref().map_or(255, |m| m.pixels()[i]) as u32;
                    for k in 0..4 {
                        let v = p[k] as u32 * (255 - m) + ink[k] as u32 * m + 128;
                        p[k] = (((v >> 8) + v) >> 8) as u8;
                    }
                }
            }
            Command::Text(c) => {
                let layer = super::text_render::render(c, matrix, size, budget, session)?;
                composite::source_over(&mut canvas.pixels, layer.pixels())?;
            }
            Command::Path(c) => {
                let layer = paths::render_path(c, matrix, size, scale, budget, session)?;
                composite::path(&mut canvas.pixels, layer.pixels(), c.rop2)?;
            }
            Command::Image(c) => {
                let decoded = session.decode(&c.image)?;
                let visible = super::bitmap::crop_source(decoded.size(), c.source).is_some();
                let mut layer = super::sampling::unclipped_layer(&decoded, c, matrix, size)?;
                if visible {
                    paths::apply_clip(&mut layer, &c.clip, matrix, budget, session)?;
                }
                composite::image(&mut canvas.pixels, layer.pixels(), c.rop)?;
            }
        }
    }
    Ok(canvas)
}
