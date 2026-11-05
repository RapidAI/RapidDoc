use rapiddoc_office::xls::{chart, number_format as nf, records as rec, strings as st};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::Read;
fn bytes(s: &str) -> Vec<u8> {
    s.as_bytes()
        .chunks_exact(2)
        .map(|b| u8::from_str_radix(std::str::from_utf8(b).unwrap(), 16).unwrap())
        .collect()
}
#[test]
fn original_xls_number_formats_strings_records_and_codepages() {
    let mut raw = vec![];
    flate2::read::GzDecoder::new(
        &include_bytes!("../../../tests/fixtures/office_rust_reference_068/xls_primitives.json.gz")
            [..],
    )
    .read_to_end(&mut raw)
    .unwrap();
    let f: Value = serde_json::from_slice(&raw).unwrap();
    for c in f["numbers"].as_array().unwrap() {
        let v = f64::from_le_bytes(bytes(c["bits"].as_str().unwrap()).try_into().unwrap());
        let actual = nf::number(v, c["code"].as_str(), c["date1904"].as_bool().unwrap());
        if c.get("error").is_some() {
            assert!(actual.is_err(), "{c}");
        } else {
            assert_eq!(actual.unwrap(), c["output"], "value={v:?} {c}");
        }
    }
    for c in f["texts"].as_array().unwrap() {
        assert_eq!(
            nf::text(c["text"].as_str().unwrap(), c["code"].as_str()),
            c["output"],
            "{c}"
        );
    }
    let fonts: Vec<st::Font> = serde_json::from_value(f["fonts"].clone()).unwrap();
    for c in f["strings"].as_array().unwrap() {
        let parts = c["parts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| bytes(s.as_str().unwrap()))
            .collect::<Vec<_>>();
        let mut r = rec::Segments::new(parts.iter().map(Vec::as_slice).collect());
        let d = st::read(
            &mut r,
            c["biff8"].as_bool().unwrap(),
            c["codepage"].as_u64().unwrap() as u16,
            c["short"].as_bool().unwrap(),
            c["rich"].as_bool().unwrap(),
        );
        assert_eq!(json!(d), c["decoded"], "{c}");
        assert_eq!(
            json!(d.as_ref().map(|d| st::rich(d, &fonts))),
            c["rich_text"],
            "{c}"
        );
        assert_eq!(
            json!([r.index, r.offset]),
            json!([c["index"], c["offset"]]),
            "{c}"
        );
    }
    for c in f["codepages"].as_array().unwrap() {
        assert_eq!(
            st::decode_bytes(
                &bytes(c["hex"].as_str().unwrap()),
                c["codepage"].as_u64().unwrap() as u16
            ),
            c["output"],
            "{c}"
        );
    }
    for c in f["codec_pair_hashes"].as_array().unwrap() {
        let cp = c["codepage"].as_u64().unwrap() as u16;
        let mut hash = Sha256::new();
        for a in 0..256 {
            for b in 0..256 {
                let text = st::decode_bytes(&[a as u8, b as u8], cp);
                hash.update((text.len() as u32).to_le_bytes());
                hash.update(text.as_bytes());
            }
        }
        assert_eq!(
            format!("{:x}", hash.finalize()),
            c["sha256"],
            "exhaustive two-byte codepage {cp}"
        );
    }
    for c in f["txo"].as_array().unwrap() {
        let base = bytes(c["base"].as_str().unwrap());
        let parts = c["parts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| bytes(s.as_str().unwrap()))
            .collect::<Vec<_>>();
        let refs = parts.iter().map(Vec::as_slice).collect::<Vec<_>>();
        assert_eq!(json!(st::txo(&base, &refs, &fonts)), c["output"], "TXO {c}");
    }
    for c in f["charts"].as_array().unwrap() {
        let payloads = c["payloads"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| bytes(s.as_str().unwrap()))
            .collect::<Vec<_>>();
        let records = payloads
            .iter()
            .map(|p| rec::Record {
                offset: 0,
                kind: 0x1051,
                payload: p,
                next: p.len() + 4,
            })
            .collect::<Vec<_>>();
        let external: Vec<Option<usize>> = serde_json::from_value(c["external"].clone()).unwrap();
        let current = c["current"].as_u64().unwrap() as usize;
        let s = chart::selection(records, current, &external);
        assert_eq!(json!(s), c["selection"], "chart {c}");
        let axes = s
            .filter(|s| s.sheet_index == current)
            .map(|s| (s.rows, s.cols));
        assert_eq!(json!(axes), c["axes"], "chart axes {c}");
    }
    for c in f["records"].as_array().unwrap() {
        let data = bytes(c["hex"].as_str().unwrap());
        let mut cursor = 0;
        let mut records = vec![];
        while cursor < data.len() {
            if data[cursor..].iter().all(|b| *b == 0) {
                break;
            }
            let Some(r) = rec::at(&data, cursor) else {
                break;
            };
            records.push(json!({"offset":r.offset,"kind":r.kind,"payload":r.payload.iter().map(|b|format!("{b:02x}")).collect::<String>(),"next":r.next}));
            cursor = r.next;
        }
        assert_eq!(json!(records), c["records"]);
    }
}
#[test]
fn record_views_and_original_shared_budget_preserve_error_boundaries() {
    let data = [
        1, 0, 1, 0, 0x42, 0x3c, 0, 1, 0, 0x43, 0x3c, 0, 0, 0, 0x0a, 0, 0, 0,
    ];
    let mut budget = rec::Budget { count: 15_999_999 };
    let base = rec::charged(&data, 0, &mut budget).unwrap().unwrap();
    assert_eq!(budget.count, 16_000_000);
    assert!(rec::continues(&data, base, &mut budget).is_err());
    let mut budget = rec::Budget::default();
    let base = rec::charged(&data, 0, &mut budget).unwrap().unwrap();
    let (parts, next) = rec::continues(&data, base, &mut budget).unwrap();
    assert_eq!(budget.count, 3);
    assert_eq!(parts, vec![&[0x42][..], &[0x43][..], &[][..]]);
    assert_eq!(next, 14);
    assert_eq!(base.payload.as_ptr(), data[4..].as_ptr());
    assert!(rec::at(&data, usize::MAX).is_none());
    let mut missing = rec::Budget::default();
    assert!(rec::charged(&data, data.len() - 2, &mut missing)
        .unwrap()
        .is_none());
    assert_eq!(missing.count, 0);
}
