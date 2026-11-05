//! DOC's Windows CP932/949/950 are distinct from XLS Shift-JIS/EUC-KR/Big5.
//! Reuse the native interpreter, with tables frozen from the DOC Python codecs.
use crate::xls::codepages::{self, Table};
use std::sync::OnceLock;
pub fn decode(bytes: &[u8], cp: u16) -> String {
    macro_rules! table {
        ($file:literal) => {{
            static T: OnceLock<Table> = OnceLock::new();
            T.get_or_init(|| codepages::load(include_bytes!($file)))
        }};
    }
    let table = match cp {
        932 => table!("codepages/932.json.gz"),
        949 => table!("codepages/949.json.gz"),
        950 => table!("codepages/950.json.gz"),
        _ => return codepages::decode(bytes, cp),
    };
    codepages::decode_table(bytes, table)
}
