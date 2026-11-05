use rapiddoc_office::document_output as output;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
pub fn verify(parsed: output::Output, c: &Value) {
    let mut aliases = BTreeMap::new();
    let mut used = BTreeSet::new();
    assert_eq!(
        parsed.assets.len(),
        c["assets"].as_object().unwrap().len(),
        "{} asset count",
        c["name"]
    );
    for (name, data) in &parsed.assets {
        let summary = if name.ends_with(".svg") {
            json!({"svg":normalized_svg(data.to_vec())})
        } else {
            let (size, pixels) = if name.ends_with(".jpg") {
                let image = turbojpeg::decompress(data, turbojpeg::PixelFormat::RGBA).unwrap();
                ([image.width, image.height], image.pixels)
            } else {
                let image = image_codecs::load_from_memory(data).unwrap().into_rgba8();
                (
                    [image.width() as usize, image.height() as usize],
                    image.into_raw(),
                )
            };
            json!({"size":size,"sha256":format!("{:x}",Sha256::digest(&pixels))})
        };
        let expected = c["assets"]
            .as_object()
            .unwrap()
            .iter()
            .find(|(name, value)| !used.contains(*name) && **value == summary)
            .unwrap_or_else(|| panic!("{} asset {name}: {summary}", c["name"]));
        used.insert(expected.0.clone());
        aliases.insert(name.clone(), expected.0.clone());
    }
    let serialized = serde_json::to_string(&parsed.middle).unwrap();
    let mut starts = regex::Regex::new(r#""start":(-?[0-9]+)"#)
        .unwrap()
        .captures_iter(&serialized)
        .map(|c| c[1].to_owned())
        .collect::<Vec<_>>();
    let mut expected = c["starts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    starts.sort();
    expected.sort();
    assert_eq!(starts, expected, "{} exact starts", c["name"]);
    compare(
        serde_json::from_str(&serialized).unwrap(),
        &c["middle"],
        &aliases,
        c,
        "middle",
    );
    for mode in [
        "mm_markdown",
        "nlp_markdown",
        "content_list",
        "content_list_v2",
    ] {
        compare(
            output::render::export(&parsed.middle, mode, "images", &Default::default()),
            &c["exports"][mode],
            &aliases,
            c,
            mode,
        );
    }
}
fn compare(
    mut actual: Value,
    expected: &Value,
    aliases: &BTreeMap<String, String>,
    case: &Value,
    stage: &str,
) {
    fn rename(v: &mut Value, aliases: &BTreeMap<String, String>) {
        match v {
            Value::String(s) => {
                for (a, b) in aliases {
                    *s = s.replace(a, b);
                }
            }
            Value::Array(v) => {
                for v in v {
                    rename(v, aliases)
                }
            }
            Value::Object(m) => {
                for v in m.values_mut() {
                    rename(v, aliases)
                }
            }
            _ => (),
        }
    }
    rename(&mut actual, aliases);
    if &actual != expected {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/office-output-diff.json");
        std::fs::write(
            path,
            serde_json::to_vec_pretty(
                &json!({"name":case["name"],"stage":stage,"actual":actual,"expected":expected}),
            )
            .unwrap(),
        )
        .unwrap();
        panic!(
            "{} {stage} differs; see build/office-output-diff.json",
            case["name"]
        );
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
