use rapiddoc_office::{equation::image_mtef::Decoder, image, ErrorKind};
fn wmf(count: usize) -> Vec<u8> {
    let mut b = vec![];
    b.extend(1u16.to_le_bytes());
    b.extend(9u16.to_le_bytes());
    b.extend(0x300u16.to_le_bytes());
    b.extend((((18 + count * 8 + 6) / 2) as u32).to_le_bytes());
    b.extend(0u16.to_le_bytes());
    b.extend(4u32.to_le_bytes());
    b.extend(0u16.to_le_bytes());
    for _ in 0..count {
        b.extend(4u32.to_le_bytes());
        b.extend(0x102u16.to_le_bytes());
        b.extend(1u16.to_le_bytes());
    }
    b.extend(3u32.to_le_bytes());
    b.extend(0u16.to_le_bytes());
    b
}
#[test]
fn strict_decoder_and_production_placeholder_are_separate() {
    let data = wmf(100000);
    let mut decoder = Decoder::default();
    assert_eq!(
        decoder.decode(&data, None, None).unwrap_err().kind,
        ErrorKind::ResourceLimit
    );
    assert!(decoder.probe(&data, None, None).unwrap().is_none());
    assert!(decoder.limit_reason(&data, None, None).is_some());
    let expected = image::image_limit_placeholder().unwrap();
    assert_eq!(
        image::serialize_with_decoder(&data, None, None, &mut decoder)
            .unwrap()
            .unwrap(),
        expected
    );
    assert_eq!(image::vector(&data, None).unwrap().source, expected);
    assert_eq!(decoder.total_bytes, 0);
}
#[test]
fn ordinary_wmf_at_exact_record_boundary_remains_allowed() {
    let data = wmf(99999);
    let mut decoder = Decoder::default();
    assert!(decoder.probe(&data, None, None).unwrap().is_none());
    assert!(decoder.limit_reason(&data, None, None).is_none());
}
