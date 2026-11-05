use serde_json::Value;
use sha2::{Digest, Sha256};
fn hash(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
pub fn normalize(value: &mut Value) {
    match value {
        Value::Array(a) => a.iter_mut().for_each(normalize),
        Value::Object(o) => o.values_mut().for_each(normalize),
        Value::String(s) => {
            let re =
                regex::Regex::new(r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=]+").unwrap();
            *s = re
                .replace_all(s, |c: &regex::Captures| {
                    let (header, encoded) = c[0].split_once(',').unwrap();
                    let bytes = base64::decode(encoded).unwrap();
                    if header == "data:image/svg+xml;base64" {
                        return format!("verified-svg:{}", hash(&bytes));
                    }
                    let (width, height, pixels) = if header == "data:image/jpeg;base64" {
                        let i =
                            turbojpeg::decompress(&bytes, turbojpeg::PixelFormat::RGBA).unwrap();
                        (i.width as u32, i.height as u32, i.pixels)
                    } else {
                        let i = image_codecs::load_from_memory(&bytes).unwrap().to_rgba8();
                        (i.width(), i.height(), i.into_raw())
                    };
                    format!("verified-image:{header}:{width}x{height}:{}", hash(&pixels))
                })
                .into_owned();
        }
        _ => {}
    }
}
