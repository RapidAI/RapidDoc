use rapiddoc_office::metafile::{bitmap, model::*};
use serde_json::Value;
use std::io::Read;
fn fixture() -> Value {
    let mut data = Vec::new();
    flate2::read::GzDecoder::new(
        &include_bytes!(
            "../../../tests/fixtures/office_rust_reference_068/metafile_bitmap.json.gz"
        )[..],
    )
    .read_to_end(&mut data)
    .unwrap();
    serde_json::from_slice(&data).unwrap()
}
fn bytes(value: &Value) -> Vec<u8> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect()
}
fn check<T: serde::Serialize>(result: rapiddoc_office::Result<T>, case: &Value, key: &str) {
    let name = case["name"].as_str().unwrap();
    if let Some(error) = case.get(format!("{key}_error")) {
        match result {
            Err(e) => assert_eq!(e.kind.code(), error.as_str().unwrap(), "{name} {key}"),
            Ok(_) => panic!("{name} {key} expected error"),
        }
    } else {
        assert_eq!(
            serde_json::to_value(result.unwrap_or_else(|e| panic!("{name} {key}: {e}"))).unwrap(),
            case[key],
            "{name} {key}"
        );
    }
}
#[test]
fn original_dib_validation_and_premultiplied_alpha() {
    for case in fixture()["cases"].as_array().unwrap() {
        let image = DibPayload {
            header: bytes(&case["header"]).into(),
            bits: bytes(&case["bits"]).into(),
            use_source_alpha: true,
        };
        check(bitmap::validate(&image), case, "validate");
        check(bitmap::bits_offset(&image.header), case, "offset");
        check(bitmap::raw_alpha(&image), case, "alpha");
    }
}
#[test]
fn original_bitmap_crop_and_destination() {
    let destination = [[10., 5.], [40., 15.], [35., 45.], [5., 35.]];
    for case in fixture()["crops"].as_array().unwrap() {
        let source = serde_json::from_value(case["source"].clone()).unwrap();
        let crop = bitmap::crop_source([4, 3], source);
        let (pixels, fraction) = if let Some(c) = crop {
            let [l, t, r, b] = c.bounds;
            let mut pixels = Vec::new();
            let xs: Vec<_> = if c.flip_x {
                (l..r).rev().collect()
            } else {
                (l..r).collect()
            };
            let ys: Vec<_> = if c.flip_y {
                (t..b).rev().collect()
            } else {
                (t..b).collect()
            };
            for y in ys {
                for &x in &xs {
                    pixels.extend([x as u8, y as u8, 17, 255]);
                }
            }
            (serde_json::to_value(pixels).unwrap(), c.fraction)
        } else {
            (
                Value::Null,
                Rect {
                    left: 0.,
                    top: 0.,
                    right: 0.,
                    bottom: 0.,
                },
            )
        };
        assert_eq!(pixels, case["pixels"], "{:?}", source);
        assert_eq!(serde_json::to_value(fraction).unwrap(), case["fraction"]);
        assert_eq!(
            serde_json::to_value(bitmap::crop_destination(destination, fraction)).unwrap(),
            case["destination"]
        );
    }
}
#[test]
fn original_bitwise_compositing() {
    let fixture = fixture();
    let layer = bytes(&fixture["layer"]);
    for case in fixture["composites"].as_array().unwrap() {
        let mut canvas = bytes(&fixture["canvas"]);
        let operation = serde_json::from_value(case["operation"].clone()).unwrap();
        rapiddoc_office::metafile::composite::paste_bitwise(&mut canvas, &layer, operation)
            .unwrap();
        assert_eq!(canvas, bytes(&case["pixels"]), "{}", case["operation"]);
    }
}
#[test]
fn original_general_dib_decode() {
    for case in fixture()["decodes"].as_array().unwrap() {
        let image = DibPayload {
            header: bytes(&case["header"]).into(),
            bits: bytes(&case["bits"]).into(),
            use_source_alpha: case["use_source_alpha"].as_bool().unwrap(),
        };
        let result = rapiddoc_office::metafile::dib_decode::decode(&image);
        let name = case["name"].as_str().unwrap();
        if let Some(error) = case.get("error") {
            match result {
                Err(e) => assert_eq!(e.kind.code(), error.as_str().unwrap(), "{name}"),
                Ok(_) => panic!("{name}: expected {error}"),
            }
        } else {
            let decoded = result.unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(
                serde_json::to_value(decoded.size()).unwrap(),
                case["size"],
                "{name}"
            );
            assert_eq!(decoded.pixels(), bytes(&case["pixels"]), "{name}");
        }
    }
}
