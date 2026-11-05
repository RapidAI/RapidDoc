use rapiddoc_core::ErrorCode;
use rapiddoc_inference::{crop, recognition_batch, Device, Model};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
fn root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn fixture(name: &str) -> Value {
    serde_json::from_slice(
        &std::fs::read(root().join(format!("build/migration/{name}.json"))).unwrap(),
    )
    .unwrap()
}
fn decode(png: &[u8]) -> recognition_batch::Image {
    let rgb = image::load_from_memory(png).unwrap().to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let mut bgr = rgb.into_raw();
    for p in bgr.as_chunks_mut::<3>().0 {
        p.swap(0, 2);
    }
    recognition_batch::Image {
        bgr,
        width: w,
        height: h,
    }
}
#[test]
fn identical_crop_transfer_timing() {
    use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder};
    let (w, h) = (768, 512);
    let data: Vec<_> = (0..w * h * 3)
        .map(|i| ((i * 17 + i / 23) % 256) as u8)
        .collect();
    let mut png = vec![];
    PngEncoder::new(&mut png)
        .write_image(&data, w as u32, h as u32, ExtendedColorType::Rgb8)
        .unwrap();
    let source = crop::PipelineImage::decode(&png, 4_000_000).unwrap();
    let regions: Vec<_> = (0..24)
        .map(|i| {
            let x = 8. + (i % 4) as f32 * 8.;
            let y = 8. + (i % 6) as f32 * 72.;
            let width = 320. + (i % 4) as f32 * 64.;
            let height = 48. + (i % 3) as f32 * 8.;
            [
                [x, y],
                [x + width, y + 0.25],
                [x + width - 0.5, y + height],
                [x + 0.25, y + height + 0.5],
            ]
        })
        .collect();
    let run = |encoded: bool| {
        let started = std::time::Instant::now();
        let mut digest = Sha256::new();
        for &points in &regions {
            let image = if encoded {
                decode(&source.crop(points, 4_000_000).unwrap().2)
            } else {
                let c = source.crop_pixels(points, 4_000_000).unwrap();
                recognition_batch::Image {
                    bgr: c.bgr,
                    width: c.width,
                    height: c.height,
                }
            };
            digest.update(&image.bgr);
        }
        (
            started.elapsed().as_secs_f64(),
            format!("{:x}", digest.finalize()),
        )
    };
    let expected = run(false).1;
    assert_eq!(run(true).1, expected);
    let (mut encoded, mut raw) = (vec![], vec![]);
    for i in 0..6 {
        for flag in if i % 2 == 0 {
            [true, false]
        } else {
            [false, true]
        } {
            let (time, hash) = run(flag);
            assert_eq!(hash, expected);
            if flag {
                encoded.push(time)
            } else {
                raw.push(time)
            }
        }
    }
    encoded.sort_by(f64::total_cmp);
    raw.sort_by(f64::total_cmp);
    let result = serde_json::json!({"encoded_seconds":encoded,"raw_seconds":raw,"encoded_median_seconds":(encoded[2]+encoded[3])/2.,"raw_median_seconds":(raw[2]+raw[3])/2.,"sha256":expected,"crops_per_iteration":24,"scope":"Current identical resampling kernel; RGB/BGR PNG round-trip vs raw memory, checksum included; no model execution"});
    let path = root().join("build/recognition-crop-transfer-timing.json");
    assert!(!path.exists());
    std::fs::write(path, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
    println!("Crop transfer timing: {result}");
}
#[test]
fn old_body_and_table_crop_pixels_are_preserved() {
    let f = fixture("model-ocr-execution");
    let mut body = 0;
    for c in f["preparations"].as_array().unwrap() {
        let images = c["images"].as_array().unwrap();
        if images.is_empty() {
            continue;
        }
        let png = std::fs::read(root().join(c["source"].as_str().unwrap())).unwrap();
        let source = crop::PipelineImage::decode(&png, 4_000_000).unwrap();
        for (p, e) in c["expected"]["processed"][0]["boxes"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|p| p[2][0].as_f64().unwrap() - p[0][0].as_f64().unwrap() >= 3.)
            .zip(images)
        {
            let points = serde_json::from_value(p.clone()).unwrap();
            let raw = source.crop_pixels(points, 4_000_000).unwrap();
            let (w, h, png) = source.crop(points, 4_000_000).unwrap();
            let decoded = decode(&png);
            assert_eq!((raw.width, raw.height), (w, h));
            assert_eq!(raw.bgr, decoded.bgr);
            let mut rgb = raw.bgr;
            for p in rgb.as_chunks_mut::<3>().0 {
                p.swap(0, 2);
            }
            assert_eq!(
                format!("{:x}", Sha256::digest(&rgb)),
                e["pixels_sha256"].as_str().unwrap()
            );
            body += 1;
        }
    }
    assert_eq!(body, 16);
    let mut tables = 0;
    for c in fixture("table-text")["crops"].as_array().unwrap() {
        let png = std::fs::read(root().join(c["source"].as_str().unwrap())).unwrap();
        let boxes = serde_json::from_value::<Vec<[[f32; 2]; 4]>>(c["boxes"].clone()).unwrap();
        let raw = crop::pipeline_pixels_batch(&png, &boxes).unwrap();
        let encoded = crop::pipeline_png_batch(&png, &boxes).unwrap();
        assert_eq!(raw.len(), encoded.len());
        for ((a, b), expected) in raw
            .iter()
            .zip(&encoded)
            .zip(c["expected"].as_array().unwrap())
        {
            assert_eq!(a.bgr, decode(&b.png).bgr);
            assert_eq!((a.width, a.height), (b.width, b.height));
            let mut rgb = a.bgr.clone();
            for p in rgb.as_chunks_mut::<3>().0 {
                p.swap(0, 2);
            }
            assert_eq!(
                format!("{:x}", Sha256::digest(&rgb)),
                expected["pixels_sha256"].as_str().unwrap()
            );
            tables += 1;
        }
    }
    println!("Exact old crop pixel references: {body} body / {tables} table");
}
#[test]
fn original_batch_tensor_bits_and_real_ctc_results_are_preserved() {
    let f = fixture("recognition-batch");
    let mut model = Model::load(
        &root().join(".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"),
        &root().join(f["model"].as_str().unwrap()),
        f["model_sha256"].as_str().unwrap(),
        Device::Cpu,
        1,
        16_000_000,
    )
    .unwrap();
    for c in f["cases"].as_array().unwrap() {
        let pngs: Vec<_> = c["sources"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| std::fs::read(root().join(p.as_str().unwrap())).unwrap())
            .collect();
        let images: Vec<_> = pngs.iter().map(|p| decode(p)).collect();
        let batch_size = c["batch_size"].as_u64().unwrap() as usize;
        let sizes: Vec<_> = images.iter().map(|i| [i.width, i.height]).collect();
        for batch in recognition_batch::plan(&sizes, batch_size, 16_000_000).unwrap() {
            let tensor = recognition_batch::prepare(&images, &batch, 16_000_000).unwrap();
            // Public owned-output API is the prior CTC path; compare all fields
            // against decoding the borrowed ORT output before its owner is dropped.
            let copied = model
                .run(tensor.shape.clone(), tensor.values.clone())
                .unwrap();
            let output = copied.values().next().unwrap();
            let dictionary = model.character_dictionary().unwrap();
            let mut alphabet = vec!["blank".to_owned()];
            alphabet.extend(dictionary.lines().map(str::to_owned));
            alphabet.push(" ".to_owned());
            let owned = rapiddoc_inference::ctc::decode(
                output.shape[0] as usize,
                output.shape[1] as usize,
                output.shape[2] as usize,
                &output.values,
                &alphabet,
            )
            .unwrap();
            let borrowed = model
                .run_ctc(tensor.shape.clone(), tensor.values.clone())
                .unwrap();
            assert_eq!(
                serde_json::to_value(owned).unwrap(),
                serde_json::to_value(borrowed).unwrap()
            );
            let stride = tensor.values.len() / batch.indices.len();
            for (row, &index) in batch.indices.iter().enumerate() {
                let bytes: Vec<_> = tensor.values[row * stride..(row + 1) * stride]
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                assert_eq!(
                    format!("{:x}", Sha256::digest(bytes)),
                    c["tensors"][index]["sha256"].as_str().unwrap(),
                    "{}",
                    c["name"]
                );
            }
        }
        let png = model.recognize_batch_png(&pngs, batch_size).unwrap();
        let mut progress = vec![];
        let raw = model
            .recognize_images_with_progress(&images, batch_size, &mut |done, total| {
                progress.push((done, total));
                Ok(())
            })
            .unwrap();
        assert_eq!(
            serde_json::to_value(&raw).unwrap(),
            serde_json::to_value(&png).unwrap()
        );
        assert_eq!(progress.first(), Some(&(0, images.len())));
        assert_eq!(progress.last(), Some(&(images.len(), images.len())));
        for (a, e) in raw.iter().zip(c["results"].as_array().unwrap()) {
            assert_eq!(a.text, e[0].as_str().unwrap());
            assert!((a.confidence as f64 - e[1].as_f64().unwrap()).abs() <= 1e-5);
        }
    }
    let invalid = recognition_batch::Image {
        bgr: vec![],
        width: 1,
        height: 1,
    };
    assert_eq!(
        model.recognize_images(&[invalid], 6).unwrap_err().code,
        ErrorCode::InvalidInput
    );
    assert!(model.recognize_images(&[], 0).is_err());
    assert!(model
        .recognize_images_with_progress(&[], 6, &mut |_, _| Err(rapiddoc_core::Error {
            code: ErrorCode::InvalidInput,
            message: "cancel".into()
        }))
        .is_err());
}
