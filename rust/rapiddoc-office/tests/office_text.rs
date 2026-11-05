use rapiddoc_office::metafile::{
    fonts, geometry::FlattenBudget, image_session::ImageSession, model::*, text, text_render,
};
use serde_json::Value;
use std::io::Read;
fn fixture() -> Value {
    let mut b = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/metafile_text.json.gz")
            [..],
    )
    .read_to_end(&mut b)
    .unwrap();
    serde_json::from_slice(&b).unwrap()
}
fn bytes(v: &Value) -> Vec<u8> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u8)
        .collect()
}
fn equal(a: &[u8], e: &Value, label: &str) {
    let e = bytes(e);
    assert_eq!(a.len(), e.len(), "{label}");
    if let Some((i, (a, e))) = a.iter().zip(e).enumerate().find(|(_, (a, e))| **a != *e) {
        panic!("{label} byte {i}: {a} != {e}");
    }
}
#[test]
fn original_font_selection_metrics_and_glyph_pixels() {
    for (i, c) in fixture()["fonts"].as_array().unwrap().iter().enumerate() {
        let font = Font {
            face_name: c["face"].as_str().unwrap().into(),
            weight: c["weight"].as_i64().unwrap() as i32,
            italic: c["italic"].as_bool().unwrap(),
            ..Font::default()
        };
        let face = fonts::resolve(&font, c["size"].as_u64().unwrap() as u32).unwrap();
        let label = format!("font {i} {}", c["face"]);
        assert_eq!(face.family, c["family"].as_str().unwrap(), "{label}");
        assert_eq!(face.style, c["style"].as_str().unwrap(), "{label}");
        assert_eq!(
            serde_json::to_value(&face.reason).unwrap(),
            c["reason"],
            "{label}"
        );
        let text = c["text"].as_str().unwrap();
        assert_eq!(
            face.length(text).unwrap(),
            c["length"].as_f64().unwrap(),
            "{label}"
        );
        let anchor = c["anchor"].as_str().unwrap();
        assert_eq!(
            serde_json::to_value(face.bounds(text, anchor).unwrap()).unwrap(),
            c["bounds"],
            "{label}"
        );
        let (offset, size, mask) = face
            .mask(
                text,
                anchor,
                serde_json::from_value(c["start"].clone()).unwrap(),
            )
            .unwrap();
        assert_eq!(
            serde_json::to_value(offset).unwrap(),
            c["offset"],
            "{label}"
        );
        assert_eq!(
            serde_json::to_value(size).unwrap(),
            c["mask_size"],
            "{label}"
        );
        equal(&mask, &c["mask"], &label);
    }
}
#[test]
fn original_text_layer_alignment_rotation_and_decorations() {
    for (i, c) in fixture()["layers"].as_array().unwrap().iter().enumerate() {
        let command = serde_json::from_value(c["command"].clone()).unwrap();
        let mut budget = FlattenBudget::default();
        let mut session = ImageSession::default();
        let layer = text_render::render(
            &command,
            Matrix::default(),
            [80, 64],
            &mut budget,
            &mut session,
        )
        .unwrap();
        equal(layer.pixels(), &c["pixels"], &format!("text layer {i}"));
        assert_eq!(budget.used, c["used"].as_u64().unwrap() as usize);
        assert_eq!(
            text_render::svg(&command, Matrix::default(), &mut session).unwrap(),
            c["svg"].as_str().unwrap(),
            "layer {i} SVG"
        );
    }
}
#[test]
fn original_font_rules_and_ansi_codecs() {
    let f = fixture();
    for c in f["aliases"].as_array().unwrap() {
        let face = c["face"].as_str().unwrap();
        assert_eq!(
            serde_json::to_value(fonts::aliases(face, c["charset"].as_u64().unwrap() as u8))
                .unwrap(),
            c["aliases"]
        );
        for s in c["styles"].as_array().unwrap() {
            assert_eq!(
                serde_json::to_value(fonts::style_candidates(
                    face,
                    s["weight"].as_i64().unwrap() as i32,
                    s["italic"].as_bool().unwrap()
                ))
                .unwrap(),
                s["candidates"]
            );
        }
    }
    for (i, c) in f["codecs"].as_array().unwrap().iter().enumerate() {
        assert_eq!(
            text::ansi(&bytes(&c["raw"]), c["charset"].as_u64().unwrap() as u8),
            c["text"].as_str().unwrap(),
            "codec {i}"
        );
    }
}
#[test]
fn original_all_dbcs_byte_pairs() {
    use sha2::{Digest, Sha256};
    for c in fixture()["codec_checks"].as_array().unwrap() {
        let mut digest = Sha256::new();
        for n in 0u32..65536 {
            let decoded = text::ansi(
                &(n as u16).to_be_bytes(),
                c["charset"].as_u64().unwrap() as u8,
            );
            digest.update((decoded.len() as u32).to_le_bytes());
            digest.update(decoded.as_bytes());
        }
        assert_eq!(
            format!("{:x}", digest.finalize()),
            c["pairs_sha256"].as_str().unwrap(),
            "charset {}",
            c["charset"]
        );
    }
}
#[test]
fn public_png_resolution_and_font_diagnostics() {
    use rapiddoc_office::metafile::{self, OutputFormat};
    let mut input = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!(
            "../../../tests/fixtures/office_rust_reference_068/metafile_geometry.json.gz"
        )[..],
    )
    .read_to_end(&mut input)
    .unwrap();
    let f: Value = serde_json::from_slice(&input).unwrap();
    for c in f["cases"].as_array().unwrap().iter().filter(|c| {
        c["name"]
            .as_str()
            .unwrap()
            .starts_with("EMF font substitution ")
    }) {
        let bytes: Vec<_> = c["hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
            .collect();
        let svg = metafile::render(&bytes, OutputFormat::Svg, 200, Some([24, 20])).unwrap();
        let png = metafile::render(&bytes, OutputFormat::Png, 200, Some([24, 20])).unwrap();
        assert_eq!(
            serde_json::to_value(&svg.diagnostics).unwrap(),
            serde_json::to_value(&png.diagnostics).unwrap()
        );
        assert_eq!(
            svg.diagnostics.len(),
            usize::from(c["name"] == "EMF font substitution not-a-real-font")
        );
        let reader = png::Decoder::new(std::io::Cursor::new(png.data))
            .read_info()
            .unwrap();
        assert_eq!(reader.info().pixel_dims.unwrap().xppu, 7874);
    }
}
