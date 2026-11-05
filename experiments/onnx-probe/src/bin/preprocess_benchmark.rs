//! CPU-only same-pixel preprocessing probe; model/runtime loading is excluded.
use rapiddoc_inference::layout::prepare_rgb;
use std::{hint::black_box, path::PathBuf, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let destination = PathBuf::from(std::env::args().nth(1).ok_or("output directory required")?);
    if destination.exists() {
        return Err("output directory already exists".into());
    }
    std::fs::create_dir_all(&destination)?;
    let mut rows = Vec::new();
    for [width, height] in [[1600, 2200], [800, 800], [400, 600]] {
        let pixels: Vec<u8> = (0..width * height * 3)
            .map(|i| ((i * 37 + i / 13) % 256) as u8)
            .collect();
        let mut seconds = Vec::new();
        for iteration in 0..13 {
            let start = Instant::now();
            let result = prepare_rgb(black_box(&pixels), width, height, 800, 800, 16_000_000)?;
            let elapsed = start.elapsed().as_secs_f64();
            black_box(&result.values);
            if iteration >= 3 {
                seconds.push(elapsed);
            }
            if iteration == 12 {
                let bytes: Vec<u8> = result.values.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(destination.join(format!("{width}x{height}.f32")), bytes)?;
            }
        }
        rows.push(
            serde_json::json!({"source":[width,height],"target":[800,800],"seconds":seconds}),
        );
    }
    std::fs::write(
        destination.join("report.json"),
        serde_json::to_vec_pretty(&rows)?,
    )?;
    Ok(())
}
