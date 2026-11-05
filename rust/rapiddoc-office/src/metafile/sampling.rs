// Adapted from metafile-render 0.3.0 backends/bitmap.py (MIT, Xiaomeng Zhao)
// and Pillow 12.2.0 Geometry.c / Convert.c (MIT-CMU, licenses/PILLOW.txt).
use super::{dib_decode::RgbaImage, model::*};
use crate::Result;

fn cubic(v: [f64; 4], d: f64) -> f64 {
    let p1 = v[1];
    let p2 = -v[0] + v[2];
    let p3 = 2. * (v[0] - v[1]) + v[2] - v[3];
    let p4 = -v[0] + v[1] - v[2] + v[3];
    p1 + d * (p2 + d * (p3 + d * p4))
}
fn premultiply(image: &RgbaImage) -> RgbaImage {
    let mut out = image.clone();
    for p in out.pixels.chunks_exact_mut(4) {
        for c in 0..3 {
            let v = u32::from(p[c]) * u32::from(p[3]) + 128;
            p[c] = (((v >> 8) + v) >> 8) as u8;
        }
    }
    out
}
fn unpremultiply(p: &mut [u8]) {
    if p[3] != 0 && p[3] != 255 {
        for c in 0..3 {
            p[c] = (u32::from(p[c]) * 255 / u32::from(p[3])).min(255) as u8;
        }
    }
}
fn bicubic(image: &RgbaImage, x: f64, y: f64, out: &mut [u8]) {
    if x < 0. || x >= image.width as f64 || y < 0. || y >= image.height as f64 {
        return;
    }
    let (x, y) = (x - 0.5, y - 0.5);
    let (ix, iy) = (x.floor() as i64 - 1, y.floor() as i64 - 1);
    let (dx, dy) = (x - x.floor(), y - y.floor());
    for c in 0..4 {
        let mut rows = [0.; 4];
        for row in 0..4 {
            let sy = iy + row as i64;
            if row > 0 && (sy < 0 || sy >= image.height as i64) {
                rows[row] = rows[row - 1];
                continue;
            }
            let sy = sy.clamp(0, image.height as i64 - 1) as usize;
            let mut values = [0.; 4];
            for (col, v) in values.iter_mut().enumerate() {
                let sx = (ix + col as i64).clamp(0, image.width as i64 - 1) as usize;
                *v = f64::from(image.pixels[(sy * image.width + sx) * 4 + c]);
            }
            rows[row] = cubic(values, dx);
        }
        out[c] = cubic(rows, dy).clamp(0., 255.) as u8;
    }
    unpremultiply(out);
}

pub fn affine(
    image: &RgbaImage,
    destination: [Point; 4],
    matrix: Matrix,
    size: [usize; 2],
    stretch_mode: u32,
) -> Result<RgbaImage> {
    let [first, second, _, fourth] = destination.map(|p| matrix.point(p));
    let axis_x = [second[0] - first[0], second[1] - first[1]];
    let axis_y = [fourth[0] - first[0], fourth[1] - first[1]];
    let det = axis_x[0] * axis_y[1] - axis_x[1] * axis_y[0];
    if det.abs() < 1e-9 {
        return RgbaImage::blank(size);
    }
    let (i00, i01, i10, i11) = (
        axis_y[1] / det,
        -axis_y[0] / det,
        -axis_x[1] / det,
        axis_x[0] / det,
    );
    let (sx, sy) = (image.width as f64, image.height as f64);
    let a = [
        i00 * sx,
        i01 * sx,
        -(i00 * first[0] + i01 * first[1]) * sx,
        i10 * sy,
        i11 * sy,
        -(i10 * first[0] + i11 * first[1]) * sy,
    ];
    transform(image, a, size, stretch_mode)
}
pub fn transform(
    image: &RgbaImage,
    a: [f64; 6],
    size: [usize; 2],
    stretch_mode: u32,
) -> Result<RgbaImage> {
    let mut out = RgbaImage::blank(size)?;
    if stretch_mode == 4 {
        let source = premultiply(image);
        for y in 0..size[1] {
            for x in 0..size[0] {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                bicubic(
                    &source,
                    a[0] * px + a[1] * py + a[2],
                    a[3] * px + a[4] * py + a[5],
                    &mut out.pixels[(y * size[0] + x) * 4..][..4],
                );
            }
        }
        return Ok(out);
    }
    // Pillow dispatches pure scale to repeated floating-point additions, small
    // general affine transforms to 16.16, then large transforms to float.
    let copy = |out: &mut RgbaImage, x: usize, y: usize, px: i64, py: i64| {
        if px >= 0 && py >= 0 && px < image.width as i64 && py < image.height as i64 {
            out.pixels[(y * size[0] + x) * 4..][..4].copy_from_slice(
                &image.pixels[(py as usize * image.width + px as usize) * 4..][..4],
            );
        }
    };
    let coord = |v: f64| if v < 0. { -1 } else { v as i64 };
    if a[1] == 0. && a[3] == 0. {
        let mut yo = a[5] + a[4] * 0.5;
        let mut xs = Vec::with_capacity(size[0]);
        let mut xo = a[2] + a[0] * 0.5;
        for _ in 0..size[0] {
            xs.push(coord(xo));
            xo += a[0];
        }
        for y in 0..size[1] {
            for (x, &px) in xs.iter().enumerate() {
                copy(&mut out, x, y, px, coord(yo));
            }
            yo += a[4];
        }
    } else if [
        [0., 0.],
        [size[0] as f64, size[1] as f64],
        [0., size[1] as f64],
        [size[0] as f64, 0.],
    ]
    .iter()
    .all(|&[x, y]| {
        (x * a[0] + y * a[1] + a[2]).abs() < 32768. && (x * a[3] + y * a[4] + a[5]).abs() < 32768.
    }) {
        let fix = |v: f64| (v * 65536. + 0.5).floor() as i32;
        let (a0, a1, a3, a4) = (fix(a[0]), fix(a[1]), fix(a[3]), fix(a[4]));
        let (mut a2, mut a5) = (
            fix(a[2] + a[0] * 0.5 + a[1] * 0.5),
            fix(a[5] + a[3] * 0.5 + a[4] * 0.5),
        );
        for y in 0..size[1] {
            let (mut xx, mut yy) = (a2, a5);
            for x in 0..size[0] {
                copy(&mut out, x, y, i64::from(xx >> 16), i64::from(yy >> 16));
                xx = xx.wrapping_add(a0);
                yy = yy.wrapping_add(a3);
            }
            a2 = a2.wrapping_add(a1);
            a5 = a5.wrapping_add(a4);
        }
    } else {
        let (mut xo, mut yo) = (
            a[2] + a[1] * 0.5 + a[0] * 0.5,
            a[5] + a[4] * 0.5 + a[3] * 0.5,
        );
        for y in 0..size[1] {
            let (mut xx, mut yy) = (xo, yo);
            for x in 0..size[0] {
                copy(&mut out, x, y, coord(xx), coord(yy));
                xx += a[0];
                yy += a[3];
            }
            xo += a[1];
            yo += a[4];
        }
    }
    Ok(out)
}

/// Image placement before the shared final clip mask. Cropping and constant
/// alpha modify only a private image, never the decoded session cache.
pub fn unclipped_layer(
    image: &RgbaImage,
    command: &ImageCommand,
    matrix: Matrix,
    size: [usize; 2],
) -> Result<RgbaImage> {
    let Some(crop) = super::bitmap::crop_source(image.size(), command.source) else {
        return RgbaImage::blank(size);
    };
    let [l, t, r, b] = crop.bounds;
    let (w, h) = (r - l, b - t);
    let mut pixels = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = (
                if crop.flip_x { r - 1 - x } else { l + x },
                if crop.flip_y { b - 1 - y } else { t + y },
            );
            let mut p: [u8; 4] = image.pixels[(sy * image.width + sx) * 4..][..4]
                .try_into()
                .unwrap();
            p[3] = (u32::from(p[3]) * u32::from(command.constant_alpha) / 255) as u8;
            pixels.extend(p);
        }
    }
    affine(
        &RgbaImage::new(w, h, pixels)?,
        super::bitmap::crop_destination(command.destination, crop.fraction),
        matrix,
        size,
        command.stretch_mode,
    )
}
