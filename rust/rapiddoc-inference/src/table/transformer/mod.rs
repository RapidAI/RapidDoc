//! TATR structure-only adapter; Microsoft table-transformer postprocessing.
//! The original MIT notices are bundled in TATR-NOTICE.txt.
mod structure;
use crate::{
    error,
    named::{Input, OutputRef},
    Model,
};
use rapiddoc_core::{
    pillow_resize::{self, Rgb, Work},
    Error, ErrorCode,
};
use std::collections::BTreeMap;
pub use structure::{decode_objects, Cell, Object, Token};

pub fn prepare(rgb: Vec<u8>, width: usize, height: usize) -> Result<Input, Error> {
    let size = pillow_resize::size(width, height)?;
    if rgb.len() != size * 3 {
        return Err(error(ErrorCode::InvalidInput, "Invalid TATR RGB image"));
    }
    let scale = 1000. / width.max(height) as f64;
    let w = ((width as f64 * scale).round_ties_even() as usize).max(1);
    let h = ((height as f64 * scale).round_ties_even() as usize).max(1);
    let image = pillow_resize::resize(
        Rgb {
            data: rgb,
            w: width,
            h: height,
        },
        w,
        h,
        (width as f32, height as f32),
        true,
        &mut Work(0),
    )?;
    let mut values = vec![0f32; w * h * 3];
    for (i, pixel) in image.data.chunks_exact(3).enumerate() {
        for c in 0..3 {
            values[c * w * h + i] = ((pixel[c] as f32 / 255.) - [0.485f32, 0.456, 0.406][c])
                / [0.229f32, 0.224, 0.225][c];
        }
    }
    Ok(Input {
        shape: vec![1, 3, h, w],
        values,
    })
}

pub fn objects(
    logits: &[f32],
    boxes: &[f32],
    width: usize,
    height: usize,
) -> Result<Vec<Object>, Error> {
    if logits.len() != 125 * 7
        || boxes.len() != 125 * 4
        || logits.iter().chain(boxes).any(|x| !x.is_finite())
    {
        return Err(error(ErrorCode::InvalidInput, "Invalid TATR logits/boxes"));
    }
    let mut result = vec![];
    for (row, b) in logits.chunks_exact(7).zip(boxes.chunks_exact(4)) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp = row.iter().map(|&x| (x - max).exp()).collect::<Vec<_>>();
        let sum = exp.iter().sum::<f32>();
        let label = (0..7)
            .max_by(|&a, &b| row[a].total_cmp(&row[b]).then(b.cmp(&a)))
            .unwrap();
        if label == 6 {
            continue;
        }
        let bbox = [
            (b[0] - b[2] / 2.) * width as f32,
            (b[1] - b[3] / 2.) * height as f32,
            (b[0] + b[2] / 2.) * width as f32,
            (b[1] + b[3] / 2.) * height as f32,
        ]
        .map(f64::from);
        result.push(Object::new(label, (exp[label] / sum) as f64, bbox));
    }
    Ok(result)
}

impl Model {
    pub fn transformer_cells_png(
        &mut self,
        png: &[u8],
        tokens: &[Token],
    ) -> Result<Vec<Cell>, Error> {
        let (mut rgb, w, h) = crate::preprocess::decode_png(png, 16_000_000)?;
        for p in rgb.chunks_exact_mut(3) {
            p.swap(0, 2);
        }
        let input = prepare(rgb, w, h)?;
        self.run_named_with(BTreeMap::from([("images".into(), input)]), |mut out| {
            let logits = match out.remove("logits") {
                Some(OutputRef::Float32 { shape, values }) if shape == [1, 125, 7] => values,
                _ => {
                    return Err(error(
                        ErrorCode::InvalidInput,
                        "TATR logits must be 1x125x7",
                    ))
                }
            };
            let boxes = match out.remove("pred_boxes") {
                Some(OutputRef::Float32 { shape, values }) if shape == [1, 125, 4] => values,
                _ => return Err(error(ErrorCode::InvalidInput, "TATR boxes must be 1x125x4")),
            };
            Ok(decode_objects(&objects(logits, boxes, w, h)?, tokens))
        })
    }
}
