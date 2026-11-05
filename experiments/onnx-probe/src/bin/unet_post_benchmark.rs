//! Real same-input UNET CPU stages; no model/session/IO inside measured spans.
use rapiddoc_inference::{
    table_components, table_lines, table_recover, table_regions, unet, unet_lines,
};
use serde::Deserialize;
use serde_json::json;
use std::{hint::black_box, path::PathBuf, time::Instant};
#[derive(Deserialize)]
struct Case {
    width: usize,
    height: usize,
    mask_width: usize,
    mask_height: usize,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: unet_post_benchmark CASES OUTPUT".into());
    }
    let output = PathBuf::from(&args[2]);
    std::fs::create_dir(&output)?;
    let mut folders: Vec<_> = std::fs::read_dir(&args[1])?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.join("case.json").exists())
        .collect();
    folders.sort();
    let mut reports = vec![];
    for folder in folders {
        let case: Case = serde_json::from_slice(&std::fs::read(folder.join("case.json"))?)?;
        let pixels = std::fs::read(folder.join("pixels.bgr"))?;
        let mask = unet::Mask {
            width: case.mask_width,
            height: case.mask_height,
            classes: std::fs::read(folder.join("mask.u8"))?,
        };
        let mut durations = vec![];
        let mut canonical = Vec::new();
        let mut tensor_bytes = Vec::new();
        let mut classifier_bytes = Vec::new();
        for i in 0..6 {
            let start = Instant::now();
            let classification = rapiddoc_inference::table_classification::prepare_bgr(
                &pixels,
                case.width,
                case.height,
                16_000_000,
            )?;
            let classify = start.elapsed().as_secs_f64();
            black_box(&classification.values);
            let start = Instant::now();
            let tensor = unet::prepare(black_box(&pixels), case.width, case.height, 16_000_000)?;
            let prep = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let masks = unet_lines::prepare(&mask, case.width, case.height, true)?;
            let morph = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let rows =
                table_components::analyze(&masks.horizontal, case.width, case.height, Some(0), 50)?
                    .lines;
            let columns =
                table_components::analyze(&masks.vertical, case.width, case.height, Some(1), 30)?
                    .lines;
            let components = start.elapsed().as_secs_f64();
            let start = Instant::now();
            let lines = table_lines::connect(rows, columns, &table_lines::Config::default())?;
            let connect = start.elapsed().as_secs_f64();
            let lines: Vec<_> = lines.rows.into_iter().chain(lines.columns).collect();
            let start = Instant::now();
            let mut cells =
                table_regions::regions_from_lines(&lines, case.width, case.height, true)?;
            let regions = start.elapsed().as_secs_f64();
            let start = Instant::now();
            if !cells.rotated_polygons.is_empty() {
                cells.topology = Some(table_recover::recover_legacy(
                    &cells
                        .rotated_polygons
                        .iter()
                        .map(|p| p.map(|xy| xy.map(f64::from)))
                        .collect::<Vec<_>>(),
                    10.,
                    15.,
                )?);
            }
            let topology = start.elapsed().as_secs_f64();
            black_box(&cells);
            black_box(&tensor.values);
            if i > 0 {
                durations.push(json!({"classification":classify,"preprocess":prep,"resize_morphology":morph,"components":components,"connect":connect,"regions":regions,"topology":topology}));
            }
            if i == 5 {
                classifier_bytes = classification
                    .values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                canonical = serde_json::to_vec(&cells)?;
                tensor_bytes = tensor.values.iter().flat_map(|v| v.to_le_bytes()).collect();
            }
        }
        let name = folder.file_name().unwrap().to_string_lossy().to_string();
        let dest = output.join(&name);
        std::fs::create_dir(&dest)?;
        std::fs::write(dest.join("cells.json"), canonical)?;
        std::fs::write(dest.join("classifier.f32"), classifier_bytes)?;
        std::fs::write(dest.join("input.f32"), &tensor_bytes)?;
        let expected = std::fs::read(folder.join("input.f32"))?;
        let row = json!({"case":name,"width":case.width,"height":case.height,"durations":durations,"python_tensor_exact":expected==tensor_bytes});
        println!("{}", row);
        reports.push(row);
    }
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&reports)?,
    )?;
    Ok(())
}
