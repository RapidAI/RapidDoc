use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Read;
#[test]
fn original_svg_sizing_pixels_and_rejection() {
    let mut raw = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/office_svg.json.gz")[..],
    )
    .read_to_end(&mut raw)
    .unwrap();
    let f: Value = serde_json::from_slice(&raw).unwrap();
    for c in f["cases"].as_array().unwrap() {
        let data = c["hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        let result = rapiddoc_office::image::svg::serialize(&data).unwrap();
        let actual = result.map(|s| {
            let bytes = base64::decode(s.split_once(',').unwrap().1).unwrap();
            let image = rapiddoc_office::metafile::emfplus::image::png(&bytes).unwrap();
            json!({"size":image.size(),"sha256":format!("{:x}",Sha256::digest(image.pixels()))})
        });
        assert_eq!(json!(actual), c["expected"], "{}", c["name"]);
    }
}
