//! Aspect-ratio sorting, width buckets, shared padding and order restoration.
use crate::preprocess::{recognition_bgr_padded, PreparedTensor, RecognitionConfig};
use rapiddoc_core::{Error, ErrorCode};
use serde::Serialize;
#[derive(Serialize)]
pub struct Batch {
    pub indices: Vec<usize>,
    pub width: usize,
}
pub struct Image {
    pub bgr: Vec<u8>,
    pub width: usize,
    pub height: usize,
}
fn error(code: ErrorCode, message: &str) -> Error {
    Error {
        code,
        message: message.into(),
    }
}

pub fn plan(
    sizes: &[[usize; 2]],
    batch_size: usize,
    max_elements: usize,
) -> Result<Vec<Batch>, Error> {
    if batch_size == 0 || batch_size > rapiddoc_core::resources::limit(64) {
        return Err(error(
            ErrorCode::InvalidInput,
            "识别 batch_size 必须在 1..64 内",
        ));
    }
    if sizes.len() > rapiddoc_core::resources::limit(1000) {
        return Err(error(ErrorCode::ResourceLimit, "识别图片数量超过预算"));
    }
    let mut total = 0usize;
    let mut ratios = Vec::new();
    for &[w, h] in sizes {
        let pixels = crate::preprocess::check_size(w, h, 16_000_000)?;
        total = total
            .checked_add(pixels)
            .filter(|n| *n <= rapiddoc_core::resources::limit(32_000_000))
            .ok_or_else(|| error(ErrorCode::ResourceLimit, "识别累计像素超过预算"))?;
        ratios.push(w as f64 / h as f64);
    }
    let indices = crate::numpy_argsort::indices(&ratios)?;
    let mut batches = Vec::new();
    let mut start = 0;
    while start < indices.len() {
        let bucket = |i: usize| ((48. * ratios[i] / 320.).ceil() as usize).max(1);
        let mut end = start + 1;
        while end < (start + batch_size).min(indices.len())
            && bucket(indices[end]) == bucket(indices[start])
        {
            end += 1;
        }
        let batch = &indices[start..end];
        let ratio = batch.iter().map(|&i| ratios[i]).fold(320. / 48., f64::max);
        let width = (48. * ratio) as usize;
        if width == 0
            || width > rapiddoc_core::resources::limit(4096)
            || batch
                .len()
                .checked_mul(3 * 48 * width)
                .is_none_or(|n| n > max_elements)
        {
            return Err(error(ErrorCode::ResourceLimit, "识别批张量超过预算"));
        }
        batches.push(Batch {
            indices: batch.to_vec(),
            width,
        });
        start = end;
    }
    Ok(batches)
}
pub fn prepare(
    images: &[Image],
    batch: &Batch,
    max_elements: usize,
) -> Result<PreparedTensor, Error> {
    let config = RecognitionConfig {
        max_elements,
        ..Default::default()
    };
    let elements = batch
        .indices
        .len()
        .checked_mul(3 * 48 * batch.width)
        .filter(|n| *n <= max_elements)
        .ok_or_else(|| error(ErrorCode::ResourceLimit, "识别批张量超过预算"))?;
    let mut values = Vec::with_capacity(elements);
    for &i in &batch.indices {
        let image = images
            .get(i)
            .ok_or_else(|| error(ErrorCode::InvalidInput, "识别批索引无效"))?;
        let tensor =
            recognition_bgr_padded(&image.bgr, image.width, image.height, &config, batch.width)?;
        values.extend(tensor.values);
    }
    Ok(PreparedTensor {
        shape: vec![batch.indices.len(), 3, 48, batch.width],
        values,
        resized_width: batch.width,
    })
}
