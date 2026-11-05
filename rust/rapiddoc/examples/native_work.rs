//! Development-only native rule timing/output capture, without Python/model loads.
use std::{path::PathBuf, time::Instant};
fn main() {
    let args: Vec<_> = std::env::args().collect();
    let library = PathBuf::from(&args[1]);
    let data = std::fs::read(&args[2]).unwrap();
    let count: usize = args[3].parse().unwrap();
    let start = Instant::now();
    let output = rapiddoc::processing::native_pdf_document::predict_range(
        &library,
        &data,
        None,
        16_000_000,
        rapiddoc_core::PageRange {
            start: 0,
            end: Some(count),
        },
    )
    .unwrap();
    eprintln!("NATIVE_WORK seconds={:.6}", start.elapsed().as_secs_f64());
    std::fs::write(&args[4], serde_json::to_vec(&output).unwrap()).unwrap();
}
