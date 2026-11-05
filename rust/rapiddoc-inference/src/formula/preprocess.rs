//! Actual PP-FormulaNet Plus PPPreProcess: crop margin → Pillow resize/thumbnail → black pad → float32 BGR gray.
use crate::pillow_resize::{fail, reduce, resize, size, Rgb, Work};
use rapiddoc_core::{Error, ErrorCode};
use serde::Serialize;

#[derive(Serialize)]
pub struct PreparedFormula {
    #[serde(skip)]
    pub(crate) work: usize,
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
    pub crop_box: [usize; 4],
    pub resized_size: [usize; 2],
    pub rgb: Vec<u8>,
}
pub fn from_png(bytes: &[u8], height: usize, width: usize) -> Result<PreparedFormula, Error> {
    let rgb = decode_png(bytes)?;
    prepare(&rgb.data, rgb.w, rgb.h, height, width)
}
pub(crate) fn decode_png(bytes: &[u8]) -> Result<Rgb, Error> {
    let (mut bgr, w, h) = crate::preprocess::decode_pipeline_rgb(bytes, 4_000_000)?;
    for p in bgr.as_chunks_mut::<3>().0 {
        p.swap(0, 2);
    }
    Ok(Rgb { data: bgr, w, h })
}
fn round_aspect(n: f64, aspect: f64, other: usize, x: bool) -> usize {
    let lo = n.floor();
    let hi = n.ceil();
    let key = |v: f64| {
        if x {
            (aspect - v / other as f64).abs()
        } else if v == 0. {
            0.
        } else {
            (aspect - other as f64 / v).abs()
        }
    };
    (if key(lo) <= key(hi) { lo } else { hi }).max(1.) as usize
}
/// RGB8 only; target sizes may be nonmultiples of16 for independent differential tests.
pub fn prepare(
    rgb: &[u8],
    w: usize,
    h: usize,
    target_h: usize,
    target_w: usize,
) -> Result<PreparedFormula, Error> {
    let pixels = size(w, h)?;
    if pixels > rapiddoc_core::resources::limit(4_000_000) || rgb.len() != pixels * 3 {
        return Err(fail(ErrorCode::InvalidInput));
    }
    if target_h < 2
        || target_w < 2
        || target_h > rapiddoc_core::resources::limit(2048)
        || target_w > rapiddoc_core::resources::limit(2048)
    {
        return Err(fail(ErrorCode::ResourceLimit));
    }
    let gray: Vec<_> = rgb
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| {
            ((p[0] as u32 * 19595 + p[1] as u32 * 38470 + p[2] as u32 * 7471 + 32768) >> 16) as u8
        })
        .collect();
    let (min, max) = (*gray.iter().min().unwrap(), *gray.iter().max().unwrap());
    let mut crop = [0, 0, w, h];
    if min != max {
        crop = [w, h, 0, 0];
        for (i, &v) in gray.iter().enumerate() {
            if (v - min) as f64 / (max - min) as f64 * 255. < 200. {
                let (x, y) = (i % w, i / w);
                crop[0] = crop[0].min(x);
                crop[1] = crop[1].min(y);
                crop[2] = crop[2].max(x + 1);
                crop[3] = crop[3].max(y + 1);
            }
        }
    }
    let (cw, ch) = (crop[2] - crop[0], crop[3] - crop[1]);
    let mut data = Vec::with_capacity(cw * ch * 3);
    for y in crop[1]..crop[3] {
        data.extend_from_slice(&rgb[(y * w + crop[0]) * 3..(y * w + crop[2]) * 3]);
    }
    let short = target_h.min(target_w);
    let (nw, nh) = if cw <= ch {
        (short, (short as f64 * ch as f64 / cw as f64) as usize)
    } else {
        ((short as f64 * cw as f64 / ch as f64) as usize, short)
    };
    size(nw, nh)?;
    let mut work = Work(0);
    let mut im = resize(
        Rgb { data, w: cw, h: ch },
        nw,
        nh,
        (cw as f32, ch as f32),
        false,
        &mut work,
    )?;
    if im.w > target_w || im.h > target_h {
        let aspect = im.w as f64 / im.h as f64;
        let (tw, th) = if target_w as f64 / target_h as f64 >= aspect {
            (
                round_aspect(target_h as f64 * aspect, aspect, target_h, true),
                target_h,
            )
        } else {
            (
                target_w,
                round_aspect(target_w as f64 / aspect, aspect, target_w, false),
            )
        };
        let (fx, fy) = (
            ((im.w as f64 / tw as f64 / 2.) as usize).max(1),
            ((im.h as f64 / th as f64 / 2.) as usize).max(1),
        );
        let end = (im.w as f64 / fx as f64, im.h as f64 / fy as f64);
        if fx > 1 || fy > 1 {
            im = reduce(im, fx, fy, &mut work)?;
        }
        im = resize(im, tw, th, (end.0 as f32, end.1 as f32), true, &mut work)?;
    }
    let resized_size = [im.w, im.h];
    let (left, top) = ((target_w - im.w) / 2, (target_h - im.h) / 2);
    let mut padded = vec![0; target_w * target_h * 3];
    for y in 0..im.h {
        let dst = ((y + top) * target_w + left) * 3;
        padded[dst..dst + im.w * 3].copy_from_slice(&im.data[y * im.w * 3..(y + 1) * im.w * 3]);
    }
    let (oh, ow) = (target_h.div_ceil(16) * 16, target_w.div_ceil(16) * 16);
    let mut values = vec![1.; oh * ow];
    let norm = |v: u8| (v as f32 * (1. / 255.) as f32 - 0.7931f32) / 0.1738f32;
    for y in 0..target_h {
        for x in 0..target_w {
            let p = &padded[(y * target_w + x) * 3..][..3];
            // OpenCV公开SIMD表达式；IPP/CPU派发可有少量末位舍入差异。
            values[y * ow + x] = norm(p[2]).mul_add(
                0.299f32,
                norm(p[1]).mul_add(0.587f32, norm(p[0]) * 0.114f32),
            );
        }
    }
    Ok(PreparedFormula {
        work: work.0,
        shape: vec![1, 1, oh, ow],
        values,
        crop_box: crop,
        resized_size,
        rgb: padded,
    })
}
