use rapiddoc_inference::{named::Output, Device, Model};
use serde_json::Value;
use std::path::Path;

#[test]
fn rendered_pixel_layout_matches_png_models_and_errors() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let f: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/layout-post.json")).unwrap(),
    )
    .unwrap();
    let mut model = Model::load(
        &root.join(".venv/Lib/site-packages/onnxruntime/capi/onnxruntime.dll"),
        &root.join(f["model"].as_str().unwrap()),
        f["sha256"].as_str().unwrap(),
        Device::Cpu,
        1,
        32_000_000,
    )
    .unwrap();
    let geos = std::fs::read_dir(root.join("bindings/python/python/rapid_doc_rust/runtime/geos"))
        .unwrap()
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("geos_c")
        })
        .unwrap();
    let mut sources = f["png_cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|case| std::fs::read(root.join(case["input"].as_str().unwrap())).unwrap())
        .collect::<Vec<_>>();
    let mut colored = image::load_from_memory(&sources[0]).unwrap().to_rgb8();
    for p in colored.pixels_mut() {
        p[0] = p[0].saturating_sub(17);
        p[2] /= 2;
    }
    let mut png = Vec::new();
    use image::ImageEncoder;
    image::codecs::png::PngEncoder::new(&mut png)
        .write_image(
            colored.as_raw(),
            colored.width(),
            colored.height(),
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();
    sources.push(png);
    for (i, bytes) in sources.iter().enumerate() {
        let rgb = image::load_from_memory(&bytes).unwrap().to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let pixels = rgb.into_raw();
        let old = model.layout_raw_png(&bytes).unwrap();
        let new = model.layout_raw_rgb(&pixels, w, h).unwrap();
        assert_eq!(
            old.keys().collect::<Vec<_>>(),
            new.keys().collect::<Vec<_>>()
        );
        for (key, a) in &old {
            match (a, &new[key]) {
                (
                    Output::Float32 {
                        shape: a,
                        values: x,
                    },
                    Output::Float32 {
                        shape: b,
                        values: y,
                    },
                ) => {
                    assert_eq!(a, b);
                    assert_eq!(x.len(), y.len());
                    assert!(x.iter().zip(y).all(|(a, b)| a.to_bits() == b.to_bits()));
                }
                (
                    Output::Int32 {
                        shape: a,
                        values: x,
                    },
                    Output::Int32 {
                        shape: b,
                        values: y,
                    },
                ) => {
                    assert_eq!(a, b);
                    assert_eq!(x, y);
                }
                (
                    Output::Int64 {
                        shape: a,
                        values: x,
                    },
                    Output::Int64 {
                        shape: b,
                        values: y,
                    },
                ) => {
                    assert_eq!(a, b);
                    assert_eq!(x, y);
                }
                _ => panic!("Layout output type changed"),
            }
        }
        assert_eq!(
            serde_json::to_value(model.layout_png(&bytes, 0.3).unwrap()).unwrap(),
            serde_json::to_value(model.layout_rgb(&pixels, w, h, 0.3).unwrap()).unwrap()
        );
        if i == 0 {
            for mode in ["auto", "poly", "quad"] {
                assert_eq!(
                    model
                        .layout_polygons_png(&bytes, 0.3, mode, Some(&geos))
                        .unwrap(),
                    model
                        .layout_polygons_rgb(&pixels, w, h, 0.3, mode, Some(&geos))
                        .unwrap()
                );
            }
        }
    }
    assert!(model.layout_raw_rgb(&[], 1, 1).is_err());
    assert!(model.layout_raw_rgb(&[], 0, 1).is_err());
    assert!(model.layout_raw_rgb(&[], 16_000_001, 1).is_err());
    assert!(model.layout_rgb(&[0, 0, 0], 1, 1, f32::NAN).is_err());
}
