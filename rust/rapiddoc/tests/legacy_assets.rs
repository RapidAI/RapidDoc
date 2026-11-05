use rapiddoc::legacy_page::assets::{assemble, OriginalImage};
use serde_json::Value;
use sha2::{Digest, Sha256};

#[test]
fn color_profiles_exif_and_transparent_mode_keep_old_metadata_and_names() {
    use image::{ImageDecoder, ImageFormat, ImageReader};
    use std::io::Cursor;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/legacy-assets-metadata.json")).unwrap(),
    )
    .unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let originals: Vec<_> = c["originals"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| OriginalImage {
                bbox: std::array::from_fn(|i| o["bbox"][i].as_f64().unwrap()),
                png: std::fs::read(root.join(o["source"].as_str().unwrap())).unwrap(),
            })
            .collect();
        let png = std::fs::read(root.join(c["source"].as_str().unwrap())).unwrap();
        let output = assemble(
            &c["request"],
            &png,
            &originals,
            c["extract_original"].as_bool().unwrap(),
            0.9,
        )
        .unwrap();
        assert_eq!(output.page, c["expected"], "{}", c["name"]);
        for (asset, expected) in output.assets.iter().zip(c["calls"].as_array().unwrap()) {
            assert_eq!(asset.name, expected["name"].as_str().unwrap());
            let mode = match asset.png[25] {
                0 => "L",
                2 => "RGB",
                4 => "LA",
                6 => "RGBA",
                _ => panic!("unexpected mode"),
            };
            assert_eq!(mode, expected["mode"]);
            let mut decoder = ImageReader::with_format(Cursor::new(&asset.png), ImageFormat::Png)
                .into_decoder()
                .unwrap();
            assert_eq!(
                decoder.exif_metadata().unwrap().is_some(),
                expected.get("exif_sha256").is_some()
            );
            if let Some(hash) = expected["icc_profile_sha256"].as_str() {
                assert_eq!(
                    format!(
                        "{:x}",
                        Sha256::digest(decoder.icc_profile().unwrap().unwrap())
                    ),
                    hash
                );
            }
            if let Some(hash) = expected["exif_sha256"].as_str() {
                let mut exif = b"Exif\0\0".to_vec();
                exif.extend(decoder.exif_metadata().unwrap().unwrap());
                assert_eq!(format!("{:x}", Sha256::digest(exif)), hash);
            }
        }
    }
}

#[test]
fn invalid_png_and_configuration_fail_before_asset_output() {
    use rapiddoc_core::ErrorCode;
    let request = serde_json::json!({});
    let error = assemble(&request, b"not PNG", &[], false, 0.9)
        .err()
        .unwrap();
    assert_eq!(error.code, ErrorCode::InvalidInput);
    let error = assemble(&request, b"not PNG", &[], false, f64::NAN)
        .err()
        .unwrap();
    assert_eq!(error.code, ErrorCode::InvalidInput);
}

#[test]
fn huge_crops_are_rejected_before_pixel_allocation() {
    use rapiddoc_core::ErrorCode;
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/legacy-assets.json")).unwrap(),
    )
    .unwrap();
    let c = &fixture["cases"][0];
    let mut request = c["request"].clone();
    request["model"]["layout_dets"][0]["poly"] =
        serde_json::json!([0, 0, 3000, 0, 3000, 3000, 0, 3000]);
    let png = std::fs::read(root.join(c["source"].as_str().unwrap())).unwrap();
    assert_eq!(
        assemble(&request, &png, &[], false, 0.9)
            .err()
            .unwrap()
            .code,
        ErrorCode::ResourceLimit
    );
}

#[test]
fn writer_names_order_pixels_and_complete_pages_match_old_python() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(root.join("build/migration/legacy-assets.json")).unwrap(),
    )
    .unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let originals: Vec<_> = c["originals"]
            .as_array()
            .unwrap()
            .iter()
            .map(|o| OriginalImage {
                bbox: std::array::from_fn(|i| o["bbox"][i].as_f64().unwrap()),
                png: std::fs::read(root.join(o["source"].as_str().unwrap())).unwrap(),
            })
            .collect();
        let png = std::fs::read(root.join(c["source"].as_str().unwrap())).unwrap();
        let output = assemble(
            &c["request"],
            &png,
            &originals,
            c["extract_original"].as_bool().unwrap(),
            c["threshold"].as_f64().unwrap(),
        )
        .unwrap();
        assert_eq!(output.page, c["expected"], "{}", c["name"]);
        assert_eq!(
            output.assets.len(),
            c["calls"].as_array().unwrap().len(),
            "{}",
            c["name"]
        );
        for (actual, expected) in output.assets.iter().zip(c["calls"].as_array().unwrap()) {
            assert_eq!(
                actual.name,
                expected["name"].as_str().unwrap(),
                "{}",
                c["name"]
            );
            let image = image::load_from_memory(&actual.png).unwrap();
            assert_eq!(image.width() as u64, expected["width"].as_u64().unwrap());
            assert_eq!(image.height() as u64, expected["height"].as_u64().unwrap());
            assert_eq!(
                format!("{:x}", Sha256::digest(image.into_bytes())),
                expected["pixels_sha256"].as_str().unwrap(),
                "{}",
                c["name"]
            );
        }
    }
}
