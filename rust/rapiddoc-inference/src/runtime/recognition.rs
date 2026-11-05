//! Request-local OCR preparation/decoding; only forward execution needs the model lock.
use crate::{ctc, error, named, recognition_batch};
use rapiddoc_core::{Error, ErrorCode};
use std::sync::Arc;

pub(crate) enum Pending {
    Full {
        alphabet: Arc<Vec<String>>,
        value: ort::value::DynValue,
    },
    // Compact outputs contain only per-timestep indices/scores, not full logits.
    Compact {
        alphabet: Arc<Vec<String>>,
        outputs: Vec<named::Output>,
    },
}
impl Pending {
    pub(crate) fn decode(self) -> Result<Vec<ctc::Recognition>, Error> {
        match self {
            Self::Full { alphabet, value } => {
                let (shape, values) = value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| error(ErrorCode::Unsupported, e))?;
                if shape.len() != 3 {
                    return Err(error(
                        ErrorCode::Unsupported,
                        "CTC 输出必须为 batch/time/classes",
                    ));
                }
                let sizes = shape
                    .iter()
                    .map(|v| {
                        usize::try_from(*v)
                            .map_err(|_| error(ErrorCode::InvalidInput, "CTC 尺寸无效"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                ctc::decode_finite(sizes[0], sizes[1], sizes[2], values, &alphabet)
            }
            Self::Compact { alphabet, outputs } => {
                if outputs.len() != 2 {
                    return Err(error(
                        ErrorCode::Unsupported,
                        "Compact CTC requires two outputs",
                    ));
                }
                let indices = outputs
                    .iter()
                    .find_map(|v| match v {
                        named::Output::Int64 { shape, values } => Some((shape, values)),
                        _ => None,
                    })
                    .ok_or_else(|| error(ErrorCode::Unsupported, "Missing compact CTC indices"))?;
                let scores = outputs
                    .iter()
                    .find_map(|v| match v {
                        named::Output::Float32 { shape, values } => Some((shape, values)),
                        _ => None,
                    })
                    .ok_or_else(|| error(ErrorCode::Unsupported, "Missing compact CTC scores"))?;
                if indices.0 != scores.0 || indices.0.len() != 2 {
                    return Err(error(
                        ErrorCode::InvalidInput,
                        "Compact CTC shapes disagree",
                    ));
                }
                let batch =
                    usize::try_from(indices.0[0]).map_err(|e| error(ErrorCode::InvalidInput, e))?;
                let steps =
                    usize::try_from(indices.0[1]).map_err(|e| error(ErrorCode::InvalidInput, e))?;
                ctc::decode_compact(batch, steps, indices.1, scores.1, &alphabet)
            }
        }
    }
}
pub(crate) fn recognize(
    images: &[recognition_batch::Image],
    batch_size: usize,
    max_elements: usize,
    progress: &mut dyn FnMut(usize, usize) -> Result<(), Error>,
    run: &mut dyn FnMut(Vec<usize>, Vec<f32>) -> Result<Pending, Error>,
) -> Result<Vec<ctc::Recognition>, Error> {
    recognition_batch::plan(&[], batch_size, max_elements)?;
    if images.len() > rapiddoc_core::resources::limit(1000) {
        return Err(error(ErrorCode::ResourceLimit, "识别图片数量超过预算"));
    }
    let sizes: Vec<_> = images.iter().map(|i| [i.width, i.height]).collect();
    let batches = recognition_batch::plan(&sizes, batch_size, max_elements)?;
    // Opt-in diagnostics: preserve the actual crop order and batch padding.
    // No serialization or file IO is performed during normal parsing.
    if let Some(root) = std::env::var_os("RAPIDDOC_RECOGNITION_TRACE") {
        static CALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let folder = std::path::PathBuf::from(root).join(format!(
            "call-{}",
            CALL.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let save = || -> Result<(), Box<dyn std::error::Error>> {
            std::fs::create_dir_all(&folder)?;
            std::fs::write(
                folder.join("plan.json"),
                serde_json::to_vec(&serde_json::json!({"sizes":sizes,"batches":batches}))?,
            )?;
            for (i, image) in images.iter().enumerate() {
                std::fs::write(folder.join(format!("crop-{i}.bgr")), &image.bgr)?;
            }
            Ok(())
        };
        save().map_err(|e| error(ErrorCode::InvalidInput, &format!("Recognition trace: {e}")))?;
    }
    for image in images {
        if image.bgr.len() != image.width * image.height * 3 {
            return Err(error(
                ErrorCode::InvalidInput,
                "Recognition BGR byte length mismatch",
            ));
        }
    }
    progress(0, images.len())?;
    let mut completed = 0;
    let mut results: Vec<Option<ctc::Recognition>> = (0..images.len()).map(|_| None).collect();
    for batch in batches {
        let tensor = recognition_batch::prepare(images, &batch, max_elements)?;
        let recognition = run(tensor.shape, tensor.values)?.decode()?;
        if recognition.len() != batch.indices.len() {
            return Err(error(
                ErrorCode::Unsupported,
                "识别模型 batch 输出长度不一致",
            ));
        }
        for (index, result) in batch.indices.into_iter().zip(recognition) {
            results[index] = Some(result);
            completed += 1;
        }
        progress(completed, images.len())?;
    }
    results
        .into_iter()
        .map(|r| r.ok_or_else(|| error(ErrorCode::InvalidInput, "识别结果缺失")))
        .collect()
}
