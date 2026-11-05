use rapiddoc_office::metafile::{
    composite,
    dib_decode::{self, RgbaImage},
    model::*,
    sampling,
};
use serde_json::Value;
use std::{io::Read, sync::Arc};
fn fixture() -> Value {
    let mut data = Vec::new();
    flate2::read::GzDecoder::new(
        &include_bytes!(
            "../../../tests/fixtures/office_rust_reference_068/metafile_bitmap_pixels.json.gz"
        )[..],
    )
    .read_to_end(&mut data)
    .unwrap();
    serde_json::from_slice(&data).unwrap()
}
fn bytes(v: &Value) -> Vec<u8> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u8)
        .collect()
}
fn equal(actual: &[u8], expected: &Value, label: &str) {
    let expected = bytes(expected);
    assert_eq!(actual.len(), expected.len(), "{label}");
    if let Some((i, (a, b))) = actual
        .iter()
        .zip(&expected)
        .enumerate()
        .find(|(_, (a, b))| a != b)
    {
        panic!("{label}: byte {i} {a} != {b}");
    }
}
#[test]
fn original_affine_sampling_and_transparent_edges() {
    let f = fixture();
    let source = RgbaImage::new(5, 3, bytes(&f["source"])).unwrap();
    for (i, c) in f["affines"].as_array().unwrap().iter().enumerate() {
        let destination = serde_json::from_value(c["destination"].clone()).unwrap();
        let matrix = serde_json::from_value(c["matrix"].clone()).unwrap();
        let result = sampling::affine(
            &source,
            destination,
            matrix,
            [11, 9],
            c["mode"].as_u64().unwrap() as u32,
        )
        .unwrap();
        equal(result.pixels(), &c["pixels"], &format!("affine {i}"));
    }
}
#[test]
fn original_decode_crop_alpha_and_placement() {
    for (i, c) in fixture()["layers"].as_array().unwrap().iter().enumerate() {
        let payload = DibPayload {
            header: bytes(&c["header"]).into(),
            bits: bytes(&c["bits"]).into(),
            use_source_alpha: c["use_source_alpha"].as_bool().unwrap(),
        };
        let command = ImageCommand {
            image: payload.clone().into(),
            destination: [[1., 2.], [10., 3.], [9., 8.], [0., 7.]],
            source: serde_json::from_value(c["source"].clone()).unwrap(),
            rop: 0xcc0020,
            stretch_mode: c["mode"].as_u64().unwrap() as u32,
            constant_alpha: c["alpha"].as_u64().unwrap() as u8,
            clip: Arc::from([]),
            location: Location::default(),
        };
        let decoded = dib_decode::decode(&payload).unwrap();
        let result =
            sampling::unclipped_layer(&decoded, &command, Matrix::default(), [11, 9]).unwrap();
        equal(result.pixels(), &c["pixels"], &format!("layer {i}"));
    }
}
#[test]
fn original_source_over_all_alpha_pairs_and_rop_dispatch() {
    let f = fixture();
    let mut canvas = bytes(&f["canvas"]);
    let layer = bytes(&f["layer"]);
    composite::source_over(&mut canvas, &layer).unwrap();
    equal(&canvas, &f["source_over"], "source over");
    for c in f["dispatch"].as_array().unwrap() {
        let mut canvas = bytes(&f["canvas"])[..256].to_vec();
        let rop = c["rop"].as_u64().unwrap() as u32;
        if c["path"].as_bool().unwrap() {
            composite::path(&mut canvas, &layer[..256], rop).unwrap();
        } else {
            composite::image(&mut canvas, &layer[..256], rop).unwrap();
        }
        equal(
            &canvas,
            &c["pixels"],
            &format!("ROP {rop} path {}", c["path"]),
        );
    }
}
#[test]
fn original_bitmap_documents_and_cache() {
    use rapiddoc_office::metafile::{
        self,
        header::{self, Source},
        image_session::{self, ImageSession},
    };
    let f = fixture();
    for c in f["documents"].as_array().unwrap() {
        let bytes: Vec<_> = c["hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
            .collect();
        let header = header::prepare(&bytes, 200, Some([16, 12])).unwrap();
        let doc = match header.source_format {
            Source::Wmf => metafile::wmf::replay(&bytes, header, Some([16, 12])),
            Source::Emf => metafile::emf::replay(&bytes, header, Some([16, 12])),
        }
        .unwrap();
        let output = image_session::render_bitmaps(&doc, &mut ImageSession::default()).unwrap();
        equal(output.pixels(), &c["pixels"], c["name"].as_str().unwrap());
    }
    let mut cache = ImageSession::new(32);
    for c in f["cache"].as_array().unwrap() {
        let mut header = vec![0u8; 40];
        header[..4].copy_from_slice(&40u32.to_le_bytes());
        header[4..8].copy_from_slice(&2i32.to_le_bytes());
        header[8..12].copy_from_slice(&2i32.to_le_bytes());
        header[12..14].copy_from_slice(&1u16.to_le_bytes());
        header[14..16].copy_from_slice(&32u16.to_le_bytes());
        let payload = DibPayload {
            header: header.into(),
            bits: vec![(c["index"].as_u64().unwrap() as u8 + 1) * 20; 16].into(),
            use_source_alpha: false,
        };
        let image = cache.decode(&payload).unwrap();
        equal(image.pixels(), &c["pixels"], "cache");
        assert_eq!(cache.hits(), c["hits"].as_u64().unwrap() as usize);
        assert_eq!(cache.misses(), c["misses"].as_u64().unwrap() as usize);
        assert_eq!(cache.used(), c["used"].as_u64().unwrap() as usize);
    }
    cache.clear();
    assert_eq!(cache.used(), 0);
}
