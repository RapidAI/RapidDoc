//! Raw memory-optimized native output audit against captured Python production.
use rapiddoc_inference::{
    named::{Input, Output},
    registry::{cached, Descriptor},
};
use serde::Deserialize;
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf, time::Instant};
#[derive(Deserialize)]
struct In {
    name: String,
    shape: Vec<usize>,
    file: PathBuf,
}
#[derive(Deserialize)]
struct Case {
    inputs: Vec<In>,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("usage: model_memory_benchmark DESCRIPTOR CASE OUTPUT".into());
    }
    let d: Descriptor = serde_json::from_slice(&std::fs::read(&args[1])?)?;
    let c: Case = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    let out = PathBuf::from(&args[3]);
    std::fs::create_dir(&out)?;
    let start = Instant::now();
    let shared = cached(&d)?;
    let load_seconds = start.elapsed().as_secs_f64();
    assert!(std::sync::Arc::ptr_eq(&shared, &cached(&d)?));
    let mut model = shared.lock().unwrap();
    let inputs: Vec<_> = c
        .inputs
        .into_iter()
        .map(|i| {
            let bytes = std::fs::read(&i.file).unwrap();
            let values: Vec<_> = bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            (i.name, i.shape, values)
        })
        .collect();
    let mut times = vec![];
    let mut descriptions = vec![];
    for iteration in 0..5 {
        let payload = inputs
            .iter()
            .map(|(n, s, v)| {
                (
                    n.clone(),
                    Input {
                        shape: s.clone(),
                        values: v.clone(),
                    },
                )
            })
            .collect::<BTreeMap<_, _>>();
        let start = Instant::now();
        let output = if d.optimization == Some(rapiddoc_inference::onnx::Optimization::Slanet) {
            model
                .run_table(inputs[0].1.clone(), inputs[0].2.clone())?
                .into_iter()
                .map(|(k, v)| {
                    (
                        k,
                        Output::Float32 {
                            shape: v.shape,
                            values: v.values,
                        },
                    )
                })
                .collect()
        } else {
            model.run_named(payload)?
        };
        if iteration > 0 {
            times.push(start.elapsed().as_secs_f64());
        }
        if iteration == 4 {
            for (index, (name, value)) in output.into_iter().enumerate() {
                let file = format!("output-{index}.bin");
                let (shape, dtype, bytes) = match value {
                    Output::Float32 { shape, values } => (
                        shape,
                        "float32",
                        values
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    ),
                    Output::Int32 { shape, values } => (
                        shape,
                        "int32",
                        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    ),
                    Output::Int64 { shape, values } => (
                        shape,
                        "int64",
                        values.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    ),
                    Output::UInt8 { shape, values } => (shape, "uint8", values),
                };
                std::fs::write(out.join(&file), bytes)?;
                descriptions.push(json!({"name":name,"shape":shape,"dtype":dtype,"file":file}));
            }
        }
    }
    std::fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(
            &json!({"load_seconds":load_seconds,"seconds":times,"outputs":descriptions}),
        )?,
    )?;
    println!("LOAD_COMPLETE {:.3}s", load_seconds);
    Ok(())
}
