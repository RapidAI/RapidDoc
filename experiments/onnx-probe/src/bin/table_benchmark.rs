//! Raw partitioned table run: fixed captured input, no preprocessing/IO timing.
use rapiddoc_inference::registry::{cached, Descriptor};
use serde::Deserialize;
use std::{path::PathBuf, time::Instant};
#[derive(Deserialize)]
struct Input {
    shape: Vec<usize>,
    file: PathBuf,
}
#[derive(Deserialize)]
struct Case {
    inputs: Vec<Input>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: table_benchmark DESCRIPTOR CASE OUTPUT".into());
    }
    let descriptor: Descriptor = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let case: Case = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    if case.inputs.len() != 1 {
        return Err("one captured input required".into());
    }
    let bytes = std::fs::read(&case.inputs[0].file)?;
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let shared = cached(&descriptor)?;
    let mut model = shared.lock().unwrap();
    let out = PathBuf::from(&args[3]);
    if out.exists() {
        return Err("output exists".into());
    }
    std::fs::create_dir_all(&out)?;
    let mut seconds = vec![];
    let mut descriptions = vec![];
    for i in 0..13 {
        let input = values.clone();
        let start = Instant::now();
        let outputs = model.run_table(case.inputs[0].shape.clone(), input)?;
        let elapsed = start.elapsed().as_secs_f64();
        if i >= 3 {
            seconds.push(elapsed)
        }
        if i == 12 {
            for (index, (name, value)) in outputs.into_iter().enumerate() {
                let file = format!("output-{index}.bin");
                let raw: Vec<u8> = value.values.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(out.join(&file), raw)?;
                descriptions.push(serde_json::json!({"name":name,"shape":value.shape,"file":file}));
            }
        }
    }
    std::fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(&serde_json::json!({"seconds":seconds,"outputs":descriptions}))?,
    )?;
    Ok(())
}
