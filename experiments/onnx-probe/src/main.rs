//! 显式 ORT 动态库和已有 ONNX 模型的 CPU 张量探针；不下载运行库或模型。

use rapiddoc_inference::{Device, Model};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Input {
    shape: Vec<usize>,
    values: Vec<f32>,
}

/// 与 Python 使用同一输入张量、单线程 CPU 和输出名称进行独立进程差分。
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 && args.len() != 6 {
        return Err(
            "usage: rapiddoc-onnx-probe ORT_LIBRARY MODEL INPUT_JSON MODEL_SHA256 [cpu|cuda]"
                .into(),
        );
    }
    let input: Input = serde_json::from_slice(&std::fs::read(&args[3])?)?;
    let device = match args.get(5).map(String::as_str).unwrap_or("cpu") {
        "cpu" => Device::Cpu,
        "cuda" => Device::Cuda,
        _ => return Err("invalid device".into()),
    };
    let mut model = Model::load(
        Path::new(&args[1]),
        Path::new(&args[2]),
        &args[4],
        device,
        1,
        16_000_000,
    )?;
    let result = model.run(input.shape, input.values)?;
    println!("{}", serde_json::to_string(&result)?);
    Ok(())
}
