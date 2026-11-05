//! Native same-input UNET inference over changing production shapes.
use rapiddoc_inference::{
    named::{Input, Output},
    registry::{cached, Descriptor},
};
use serde::Deserialize;
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf, time::Instant};
#[derive(Deserialize)]
struct Case {
    tensor_shape: Vec<usize>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: unet_model_benchmark DESCRIPTOR CASES OUTPUT".into());
    }
    let descriptor: Descriptor = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let shared = cached(&descriptor)?;
    let mut model = shared.lock().unwrap();
    let out = PathBuf::from(&args[3]);
    std::fs::create_dir(&out)?;
    let mut folders: Vec<_> = std::fs::read_dir(&args[2])?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.join("case.json").exists())
        .collect();
    folders.sort();
    let mut cases = vec![];
    for folder in folders {
        let case: Case = serde_json::from_slice(&std::fs::read(folder.join("case.json"))?)?;
        let values: Vec<f32> = std::fs::read(folder.join("input.f32"))?
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let expected = std::fs::read(folder.join("mask.u8"))?;
        cases.push((
            folder.file_name().unwrap().to_string_lossy().to_string(),
            case.tensor_shape,
            values,
            expected,
        ));
    }
    let mut records = vec![];
    for pass in 0..4 {
        let mut total = 0.;
        let mut differences = 0;
        for (name, shape, values, expected) in &cases {
            let input = BTreeMap::from([(
                "input".into(),
                Input {
                    shape: shape.clone(),
                    values: values.clone(),
                },
            )]);
            let start = Instant::now();
            let outputs = model.run_named(input)?;
            let elapsed = start.elapsed().as_secs_f64();
            total += elapsed;
            let Some(Output::Int64 { values, .. }) = outputs.get("output") else {
                return Err("UNET int64 output required".into());
            };
            let actual: Vec<u8> = values.iter().map(|v| *v as u8).collect();
            if actual.len() != expected.len() {
                return Err("Mask length mismatch".into());
            }
            let count = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
            differences += count;
            if pass == 3 {
                std::fs::write(out.join(format!("{name}.u8")), &actual)?;
            }
            records.push(
                json!({"pass":pass,"case":name,"seconds":elapsed,"different_mask_values":count}),
            );
        }
        println!("pass={pass} seconds={total} differences={differences}");
    }
    std::fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(&records)?,
    )?;
    Ok(())
}
