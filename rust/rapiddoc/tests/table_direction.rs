use rapiddoc::legacy_model::table_direction;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn fixtures() -> Vec<Value> {
    ["table-direction.json", "table-direction-portrait.json"]
        .iter()
        .map(|p| {
            serde_json::from_slice(&std::fs::read(root().join("build/migration").join(p)).unwrap())
                .unwrap()
        })
        .collect()
}
#[test]
fn native_angles_and_portrait_classifier_gate() {
    for f in fixtures() {
        for c in f["plans"].as_array().unwrap() {
            let mut expected = c["expected"].clone();
            expected["needs_classifier"] = json!(
                expected["angles"].as_array().unwrap().is_empty()
                    && c["request"]["height"].as_f64().unwrap()
                        / c["request"]["width"].as_f64().unwrap()
                        > 1.2
            );
            assert_eq!(
                table_direction::plan(&c["request"]).unwrap(),
                expected,
                "{}",
                c["request"]
            );
        }
    }
}
#[test]
fn ambiguous_table_model_predictions_do_not_rotate() {
    for (angle, score, margin, wanted) in [
        (270, 0.904, 0.86, "270"),
        (90, 0.95, 0.9, "90"),
        (90, 0.879, 0.8, "0"),
        (270, 0.91, 0.1, "0"),
        (0, 0.99, 0.98, "0"),
        (180, 0.99, 0.98, "0"),
    ] {
        assert_eq!(
            table_direction::select_model_rotation(
                &json!({"rotation_degrees":angle,"score":score,"margin":margin})
            )
            .unwrap(),
            wanted
        );
    }
}
#[test]
fn old_table_rotations_and_bgr_model_pixels() {
    for f in fixtures() {
        for c in f["pixels"].as_array().unwrap() {
            let source = std::fs::read(root().join(c["source"].as_str().unwrap())).unwrap();
            let out = table_direction::image(
                &source,
                c["angle"].as_str().unwrap(),
                c["classifier_bgr"].as_bool().unwrap(),
            )
            .unwrap();
            let p = image::load_from_memory(&out.png).unwrap().to_rgb8();
            assert_eq!(
                json!({"width":p.width(),"height":p.height(),"pixels_sha256":format!("{:x}",Sha256::digest(p.as_raw()))}),
                c["expected"]
            );
        }
    }
}
#[test]
fn invalid_direction_inputs_and_limits() {
    let req = json!({"page":{"blocks":[]},"table":{"poly":[0,0,10,0,10,10,0,10]},"width":10,"height":10,"boxes":[],"scale":1});
    for (key, value) in [
        ("scale", json!(0)),
        ("width", json!(0)),
        ("height", json!(4_000_001)),
        ("boxes", json!([[[0, 0]]])),
    ] {
        let mut bad = req.clone();
        bad[key] = value;
        assert!(table_direction::plan(&bad).is_err());
    }
    let mut bad = req.clone();
    bad["page"]["rotation"] = json!("１");
    assert!(table_direction::plan(&bad).is_err());
    bad["page"]["rotation"] = json!("1_0");
    assert!(table_direction::plan(&bad).is_err());
    assert!(table_direction::image(b"bad", "90", false).is_err());
}

#[test]
fn borrowed_table_images_keep_old_pixels_and_validation() {
    for f in fixtures() {
        for c in f["pixels"].as_array().unwrap() {
            let source = std::fs::read(root().join(c["source"].as_str().unwrap())).unwrap();
            let angle = c["angle"].as_str().unwrap();
            let old = table_direction::image(&source, angle, false).unwrap();
            let new = table_direction::oriented_png(&source, angle).unwrap();
            let dims = table_direction::dimensions(&source).unwrap();
            let raw = image::load_from_memory(&source).unwrap();
            assert_eq!(dims, (raw.width() as usize, raw.height() as usize));
            assert_eq!(
                image::load_from_memory(&old.png).unwrap().to_rgb8(),
                image::load_from_memory(&new).unwrap().to_rgb8()
            );
            if !matches!(angle, "90" | "270") {
                assert!(matches!(new, std::borrow::Cow::Borrowed(_)));
                assert_eq!(new.as_ptr(), source.as_ptr());
            }
            for end in [0, 8, 32, source.len() / 2] {
                let bytes = &source[..end];
                assert!(table_direction::image(bytes, angle, false).is_err());
                assert!(table_direction::oriented_png(bytes, angle).is_err());
                assert!(table_direction::dimensions(bytes).is_err());
            }
        }
    }
}
