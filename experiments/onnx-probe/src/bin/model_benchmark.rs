//! Same-tensor ORT benchmark. File IO, model load and JSON stay outside timing.
use ort::{
    session::Session,
    value::{Tensor, TensorElementType, ValueType},
};
use serde::Deserialize;
use serde_json::json;
use std::{path::PathBuf, time::Instant};

#[derive(Deserialize)]
struct Input {
    name: String,
    shape: Vec<usize>,
    file: PathBuf,
}
#[derive(Deserialize)]
struct Case {
    runtime: PathBuf,
    model: PathBuf,
    sha256: String,
    #[serde(default)]
    operator_libraries: Vec<PathBuf>,
    inputs: Vec<Input>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("usage: model_benchmark CASE_JSON OUTPUT_DIRECTORY".into());
    }
    let case: Case = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let destination = PathBuf::from(&args[2]);
    if destination.exists() {
        return Err("output directory already exists".into());
    }
    std::fs::create_dir_all(&destination)?;
    rapiddoc_inference::verify_model(&case.model, &case.sha256)?;
    let environment = ort::init_from(&case.runtime)?
        .with_telemetry(false)
        .build()?;
    let mut builder = Session::builder(&environment)?
        .with_intra_threads(1)?
        .with_inter_threads(1)?;
    #[cfg(feature = "cuda")]
    {
        builder = builder
            .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])?;
    }
    #[cfg(not(feature = "cuda"))]
    return Err("this benchmark requires --features cuda".into());
    for library in &case.operator_libraries {
        builder = builder.with_operator_library(library)?;
    }
    let load = Instant::now();
    let mut session = builder.commit_from_file(&case.model)?;
    let load_seconds = load.elapsed().as_secs_f64();
    let mut buffers = Vec::new();
    for input in &case.inputs {
        let bytes = std::fs::read(&input.file)?;
        let count = input
            .shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or("shape overflow")?;
        if count.checked_mul(4) != Some(bytes.len()) {
            return Err("input shape/byte count mismatch".into());
        }
        let values = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        if values.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite model input".into());
        }
        buffers.push((
            input.name.clone(),
            Tensor::from_array((input.shape.clone(), values))?,
        ));
    }
    let mut timings = Vec::new();
    let mut descriptions = Vec::new();
    for iteration in 0..13 {
        let inputs = buffers
            .iter()
            .map(|(name, value)| (name.as_str(), value))
            .collect::<Vec<_>>();
        let start = Instant::now();
        let outputs = session.run(inputs)?;
        let seconds = start.elapsed().as_secs_f64();
        if iteration >= 3 {
            timings.push(seconds);
        }
        if iteration != 12 {
            continue;
        }
        for (index, (name, value)) in outputs.iter().enumerate() {
            let ValueType::Tensor { ty, shape, .. } = value.dtype() else {
                return Err("non-tensor output".into());
            };
            let filename = format!("output-{index}.bin");
            macro_rules! save {
                ($type:ty) => {{
                    let (_, values) = value.try_extract_tensor::<$type>()?;
                    let bytes = values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>();
                    std::fs::write(destination.join(&filename), bytes)?;
                }};
            }
            let dtype = match ty {
                TensorElementType::Float32 => {
                    save!(f32);
                    "float32"
                }
                TensorElementType::Int64 => {
                    save!(i64);
                    "int64"
                }
                TensorElementType::Int32 => {
                    save!(i32);
                    "int32"
                }
                TensorElementType::Uint8 => {
                    save!(u8);
                    "uint8"
                }
                _ => return Err(format!("unsupported output: {ty:?}").into()),
            };
            descriptions.push(json!({"name":name,"dtype":dtype,"shape":shape.iter().copied().collect::<Vec<_>>(),"file":filename}));
        }
    }
    let result = json!({"engine":"rust","load_seconds":load_seconds,"warmup":3,"seconds":timings,"outputs":descriptions,
        "scope":"ORT run with prebuilt CPU tensors, CPU input transfer and CPU outputs; excludes preprocessing, decoding, JSON and file IO",
        "threads":1});
    std::fs::write(
        destination.join("report.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(())
}
