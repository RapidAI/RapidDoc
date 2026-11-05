use rapiddoc_office::docx::{body, Source};
use serde_json::Value;
use std::io::Read;
#[test]
fn original_complete_docx_body_outputs() {
    let mut raw = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/docx_body.json.gz")[..],
    )
    .read_to_end(&mut raw)
    .unwrap();
    let f: Value = serde_json::from_slice(&raw).unwrap();
    for c in f["cases"].as_array().unwrap() {
        let bytes = c["package_hex"]
            .as_str()
            .unwrap()
            .as_bytes()
            .chunks_exact(2)
            .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
            .collect::<Vec<_>>();
        let actual = Source::open(bytes.into()).and_then(body::convert);
        if c.get("error").is_some() {
            assert!(actual.is_err(), "{} expected error", c["name"]);
            continue;
        }
        let actual = actual.unwrap_or_else(|e| panic!("{}: {e}", c["name"]));
        let serialized = serde_json::to_string(&actual.pages).unwrap();
        let mut starts = regex::Regex::new(r#""start":(-?[0-9]+)"#)
            .unwrap()
            .captures_iter(&serialized)
            .map(|c| c[1].to_string())
            .collect::<Vec<_>>();
        let mut expected_starts = c["starts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        starts.sort();
        expected_starts.sort();
        assert_eq!(starts, expected_starts, "{} exact starts", c["name"]);
        let mut json = serde_json::to_value(actual.pages).unwrap();
        normalize_images(&mut json);
        if json != c["pages"] {
            let build = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../build");
            std::fs::create_dir_all(&build).unwrap();
            std::fs::write(
                build.join("docx-body-actual.json"),
                serde_json::to_vec_pretty(&json).unwrap(),
            )
            .unwrap();
            std::fs::write(
                build.join("docx-body-expected.json"),
                serde_json::to_vec_pretty(&c["pages"]).unwrap(),
            )
            .unwrap();
            panic!(
                "{} output differs; see build/docx-body-actual.json and expected",
                c["name"]
            );
        }
    }
}

fn normalize_images(value: &mut Value) {
    match value {
        Value::Object(map) => {
            for (key, value) in map.iter_mut() {
                if key == "image_base64" {
                    let source = value.as_str().unwrap();
                    let (mime, data) = source.split_once(',').unwrap();
                    let bytes = base64::decode(data).unwrap();
                    if mime.starts_with("data:image/svg+xml;") {
                        *value = serde_json::json!({"svg":normalized_svg(bytes)});
                    } else {
                        let (size, rgba) = if mime.starts_with("data:image/png;") {
                            let i = rapiddoc_office::metafile::emfplus::image::png(&bytes).unwrap();
                            (i.size(), i.pixels().to_vec())
                        } else {
                            let i = turbojpeg::decompress(&bytes, turbojpeg::PixelFormat::RGBA)
                                .unwrap();
                            ([i.width, i.height], i.pixels)
                        };
                        *value = serde_json::json!({"mime":mime.trim_start_matches("data:").split(';').next().unwrap(),"size":size,"rgba":rgba});
                    }
                } else {
                    normalize_images(value);
                }
            }
        }
        Value::Array(items) => {
            for value in items {
                normalize_images(value)
            }
        }
        _ => {}
    }
}
fn normalized_svg(raw: Vec<u8>) -> Value {
    use sha2::{Digest, Sha256};
    let patterns = [
        regex::Regex::new(r"data:image/png;base64,([A-Za-z0-9+/=]+)").unwrap(),
        regex::Regex::new(r#"data-mime="image/png">([A-Za-z0-9+/=]+)"#).unwrap(),
    ];
    let mut markup = String::from_utf8(raw).unwrap();
    let mut images = vec![];
    for pattern in &patterns {
        markup=pattern.replace_all(&markup,|caps:&regex::Captures<'_>|{
                let encoded=&caps[1];let raw=base64::decode(encoded).unwrap();
                let mut decoder=png::Decoder::new(std::io::Cursor::new(raw));decoder.set_transformations(png::Transformations::EXPAND|png::Transformations::STRIP_16);
                let mut reader=decoder.read_info().unwrap();let mut pixels=vec![0;reader.output_buffer_size().unwrap()];let info=reader.next_frame(&mut pixels).unwrap();pixels.truncate(info.buffer_size());assert_eq!(info.color_type,png::ColorType::Rgba);
                let ppm=reader.info().pixel_dims.map(|d|vec![d.xppu,d.yppu]);
                images.push(serde_json::json!({"size":[info.width,info.height],"pixels_sha256":format!("{:x}",Sha256::digest(&pixels)),"ppm":ppm}));
                let prefix=&caps[0][..caps[0].len()-encoded.len()];format!("{prefix}PNG[{}]",images.len()-1)
            }).into_owned();
    }
    serde_json::json!({"markup":markup,"images":images})
}
