use rapiddoc_office::image;
use serde_json::{json, Value};
use std::io::Read;
fn fixture() -> Value {
    let mut v = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!(
            "../../../tests/fixtures/office_rust_reference_068/office_image_policy.json.gz"
        )[..],
    )
    .read_to_end(&mut v)
    .unwrap();
    serde_json::from_slice(&v).unwrap()
}
#[test]
fn original_placeholder_pixels_and_jpeg() {
    let f = fixture();
    for c in f["placeholders"].as_array().unwrap() {
        let size = serde_json::from_value(c["size"].clone()).unwrap();
        let lines = c["lines"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect::<Vec<_>>();
        let image = image::text_placeholder(size, &lines).unwrap();
        assert_eq!(
            json!(image.pixels()),
            c["rgba"],
            "{} {}",
            c["size"],
            c["lines"]
        );
    }
    let source = image::standard_vector_placeholder().unwrap();
    let same = image::standard_vector_placeholder().unwrap();
    assert!(std::sync::Arc::ptr_eq(&source, &same));
    let raw = base64::decode(source.split_once(',').unwrap().1).unwrap();
    let decoded = turbojpeg::decompress(&raw, turbojpeg::PixelFormat::RGBA).unwrap();
    assert_eq!(json!(decoded.pixels), f["standard"]["rgba"]);
}
#[test]
fn original_vector_fallback_and_emu_size() {
    let f = fixture();
    for c in f["vectors"].as_array().unwrap() {
        let data = c["hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        let result = image::vector(&data, Some([109728, 91440])).unwrap();
        assert_eq!(
            result.failure.is_some(),
            c["placeholder"].as_bool().unwrap(),
            "{}",
            c["name"]
        );
        assert_eq!(
            result.source.starts_with("data:image/jpeg;"),
            result.failure.is_some()
        );
    }
    for c in f["sizing"].as_array().unwrap() {
        let size = serde_json::from_value(c["size"].clone()).unwrap();
        assert_eq!(
            json!(image::size_from_emu(size, 200)),
            c["result"],
            "{}",
            c["size"]
        );
    }
}
