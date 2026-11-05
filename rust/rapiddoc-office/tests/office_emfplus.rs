use rapiddoc_office::metafile;
use serde_json::{json, Value};
use std::io::Read;
fn bytes(s: &str) -> Vec<u8> {
    s.as_bytes()
        .chunks_exact(2)
        .map(|v| u8::from_str_radix(std::str::from_utf8(v).unwrap(), 16).unwrap())
        .collect()
}
fn equal(a: &Value, e: &Value, path: &str) {
    match (a, e) {
        (Value::Number(a), Value::Number(e)) => {
            let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
            assert!(
                (a - e).abs() <= 1e-11 * e.abs().max(1.),
                "{path}: {a} != {e}"
            );
        }
        (Value::Array(a), Value::Array(e)) => {
            assert_eq!(a.len(), e.len(), "{path} length");
            for (i, (a, e)) in a.iter().zip(e).enumerate() {
                equal(a, e, &format!("{path}[{i}]"));
            }
        }
        (Value::Object(a), Value::Object(e)) => {
            assert_eq!(a.len(), e.len(), "{path} keys");
            for (k, e) in e {
                equal(a.get(k).unwrap_or(&Value::Null), e, &format!("{path}.{k}"));
            }
        }
        _ => assert_eq!(a, e, "{path}"),
    }
}
fn normalize(v: &mut Value) {
    match v {
        Value::Object(m) if m.len() == 1 && m.contains_key("png") => {
            let bytes = m["png"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u8)
                .collect::<Vec<_>>();
            let image = metafile::emfplus::image::png(&bytes).unwrap();
            *v = json!({"size":image.size(),"rgba":image.pixels()});
        }
        Value::Object(m) => {
            for v in m.values_mut() {
                normalize(v)
            }
        }
        Value::Array(a) => {
            for v in a {
                normalize(v)
            }
        }
        _ => {}
    }
}
#[test]
fn original_emfplus_replay_and_outputs() {
    let mut raw = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/emfplus.json.gz")[..],
    )
    .read_to_end(&mut raw)
    .unwrap();
    let fixture: Value = serde_json::from_slice(&raw).unwrap();
    for c in fixture["cases"].as_array().unwrap() {
        let name = c["name"].as_str().unwrap();
        let data = bytes(c["hex"].as_str().unwrap());
        let size = Some([24, 20]);
        let doc = metafile::header::prepare(&data, 200, size)
            .and_then(|h| metafile::emf::replay(&data, h, size));
        if let Some(error) = c.get("error") {
            assert_eq!(doc.err().map(|e| e.kind.code()), error.as_str(), "{name}");
            continue;
        }
        let doc = doc.unwrap_or_else(|e| panic!("{name}: {e}"));
        let mut actual = serde_json::to_value(&doc).unwrap();
        normalize(&mut actual);
        equal(&actual, &c["document"], name);
        // Inspect input accounting independently of the output/cache passes.
        let h = metafile::header::prepare(&data, 200, size).unwrap();
        let mut p = metafile::playback::Playback::new(h.clone());
        let mut plus = metafile::emfplus::playback::Plus::new(
            h.bounds.unwrap(),
            h.device_pixels_per_mm.map(|v| v * 25.4),
        );
        for r in
            metafile::binary::Records::new(&data, h.record_start, h.record_end, h.source_format)
                .unwrap()
        {
            let r = r.unwrap();
            p.set_record(r.kind, r.index, r.offset);
            if r.kind == 70 {
                plus.comment(r.data, &mut p).unwrap();
            } else if plus.gdi_active && !plus.halted && !matches!(r.kind, 1 | 14) {
                metafile::emf::handle(r.kind, r.data, &mut p).unwrap();
            }
        }
        let mut budget = serde_json::to_value(&p.budget).unwrap();
        let mut reference = c["budget"].clone();
        if reference["decoded_bytes"].as_u64().unwrap() > 0 {
            // Encoded PNG lengths depend on the encoder. Charge actual native
            // bytes; repeated draws must not charge the object again.
            let mut images = std::collections::HashSet::new();
            let mut charged = 0;
            for command in &doc.commands {
                if let metafile::model::Command::Image(command) = command {
                    if let metafile::model::ImagePayload::Encoded { png } = &command.image {
                        if images.insert(png.clone()) {
                            let image = metafile::emfplus::image::png(png).unwrap();
                            charged += image.size()[0] * image.size()[1] * 4 + png.len();
                        }
                    }
                }
            }
            assert_eq!(
                p.budget.decoded_bytes, charged,
                "{name} image definitions, not draw count"
            );
            budget.as_object_mut().unwrap().remove("decoded_bytes");
            reference.as_object_mut().unwrap().remove("decoded_bytes");
        }
        equal(&budget, &reference, &format!("{name} budget"));
        for (kind, format) in [
            ("png", metafile::OutputFormat::Png),
            ("svg", metafile::OutputFormat::Svg),
        ] {
            let expected = &c["outputs"][kind];
            let result = metafile::render(&data, format, 200, size);
            if let Some(error) = expected.get("error") {
                assert_eq!(
                    result.err().map(|e| e.kind.code()),
                    error.as_str(),
                    "{name} {kind}"
                );
                continue;
            }
            let result = result.unwrap_or_else(|e| panic!("{name} {kind}: {e}"));
            equal(
                &json!({"width":result.width,"height":result.height,"partial":result.partial,"diagnostics":result.diagnostics}),
                &json!({"width":expected["width"],"height":expected["height"],"partial":expected["partial"],"diagnostics":expected["diagnostics"]}),
                &format!("{name} {kind}"),
            );
            if kind == "png" {
                let image = metafile::emfplus::image::png(&result.data).unwrap();
                assert_eq!(json!(image.pixels()), expected["rgba"], "{name} pixels");
            } else {
                assert_eq!(normalized_svg(result.data), expected["svg"], "{name} SVG");
            }
        }
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
