use rapiddoc_office::doc::fib;

#[test]
fn parses_variable_word97_fib() {
    let mut d = vec![0u8; 128];
    d[0..2].copy_from_slice(&0xa5ecu16.to_le_bytes());
    d[2..4].copy_from_slice(&0x00c1u16.to_le_bytes());
    d[6..8].copy_from_slice(&0x0409u16.to_le_bytes());
    d[10..12].copy_from_slice(&0x0200u16.to_le_bytes());
    d[24..28].copy_from_slice(&0x100u32.to_le_bytes());
    d[28..32].copy_from_slice(&0x200u32.to_le_bytes());
    d[32..34].copy_from_slice(&1u16.to_le_bytes());
    d[34..36].copy_from_slice(&7u16.to_le_bytes());
    d[36..38].copy_from_slice(&1u16.to_le_bytes());
    d[38..42].copy_from_slice(&10u32.to_le_bytes());
    d[42..44].copy_from_slice(&1u16.to_le_bytes());
    d[44..48].copy_from_slice(&55u32.to_le_bytes());
    d[48..52].copy_from_slice(&9u32.to_le_bytes());
    d[52..54].copy_from_slice(&1u16.to_le_bytes());
    d[54..56].copy_from_slice(&2u16.to_le_bytes());
    let f = fib::parse(&d).unwrap();
    assert_eq!(f.base.n_fib, 0x00c1);
    assert!(f.base.uses_1table());
    assert_eq!(f.rgw, vec![7]);
    assert_eq!(f.pairs[0].fc, 55);
    assert_eq!(f.csw_new, vec![2]);
}
