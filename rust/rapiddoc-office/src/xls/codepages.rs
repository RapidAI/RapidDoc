//! Frozen CPython codec semantics, including undefined-byte replacement.
use serde::Deserialize;
use std::{collections::BTreeMap, io::Read, sync::OnceLock};
#[derive(Deserialize)]
pub(crate) struct Table {
    symbols: Vec<String>,
    transitions: Vec<(usize, usize)>,
    r#final: Vec<usize>,
}
#[derive(Deserialize)]
struct Korean {
    single: Vec<Option<String>>,
    pairs: Vec<(usize, String)>,
    hangul: BTreeMap<String, String>,
}
pub(crate) fn load<T: serde::de::DeserializeOwned>(raw: &[u8]) -> T {
    let mut bytes = vec![];
    flate2::read::GzDecoder::new(raw)
        .read_to_end(&mut bytes)
        .expect("generated codec gzip");
    serde_json::from_slice(&bytes).expect("generated codec table")
}
pub fn decode(bytes: &[u8], cp: u16) -> String {
    if cp == 949 {
        static K: OnceLock<Korean> = OnceLock::new();
        let k = K.get_or_init(|| load(include_bytes!("codepages/949.json.gz")));
        let mut out = String::new();
        let mut i = 0;
        while i < bytes.len() {
            if let Some(s) = &k.single[bytes[i] as usize] {
                out.push_str(s);
                i += 1;
                continue;
            }
            if bytes.get(i..i + 2) == Some(&[0xa4, 0xd4]) {
                if bytes.len() - i < 8 {
                    out.push('\u{fffd}');
                    break;
                }
                let key = bytes[i..i + 8]
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>();
                if let Some(s) = k.hangul.get(&key) {
                    out.push_str(s);
                    i += 8;
                    continue;
                }
                out.push('\u{fffd}');
                i += 1;
                continue;
            }
            if i + 1 == bytes.len() {
                out.push('\u{fffd}');
                break;
            }
            let (count, text) = &k.pairs[bytes[i] as usize * 256 + bytes[i + 1] as usize];
            out.push_str(text);
            i += count;
        }
        return out;
    }
    macro_rules! table {
        ($file:literal) => {{
            static T: OnceLock<Table> = OnceLock::new();
            T.get_or_init(|| load(include_bytes!($file)))
        }};
    }
    let t = match cp {
        874 => table!("codepages/874.json.gz"),
        932 => table!("codepages/932.json.gz"),
        936 => table!("codepages/936.json.gz"),
        950 => table!("codepages/950.json.gz"),
        1250 => table!("codepages/1250.json.gz"),
        1251 => table!("codepages/1251.json.gz"),
        1253 => table!("codepages/1253.json.gz"),
        1254 => table!("codepages/1254.json.gz"),
        1255 => table!("codepages/1255.json.gz"),
        1256 => table!("codepages/1256.json.gz"),
        1257 => table!("codepages/1257.json.gz"),
        1258 => table!("codepages/1258.json.gz"),
        _ => table!("codepages/1252.json.gz"),
    };
    decode_table(bytes, t)
}
pub(crate) fn decode_table(bytes: &[u8], t: &Table) -> String {
    let mut state = 0;
    let mut out = String::new();
    for byte in bytes {
        let (next, symbol) = t.transitions[state * 256 + *byte as usize];
        out.push_str(&t.symbols[symbol]);
        state = next;
    }
    out.push_str(&t.symbols[t.r#final[state]]);
    out
}
