use rapiddoc_office::{
    doc::resources::{Resources, Visual},
    ole::OleReader,
};
#[test]
fn identical_payloads_share_conversion_but_keep_offset_identity() {
    let mut png = std::io::Cursor::new(vec![]);
    image_codecs::DynamicImage::new_rgb8(1, 1)
        .write_to(&mut png, image_codecs::ImageFormat::Png)
        .unwrap();
    let mut data = vec![];
    let mut offsets = vec![];
    for _ in 0..2 {
        offsets.push(data.len() as u32);
        data.extend_from_slice(&((6 + png.get_ref().len()) as u32).to_le_bytes());
        data.extend_from_slice(&6u16.to_le_bytes());
        data.extend_from_slice(png.get_ref());
    }
    let mut resources = Resources::default();
    let Some(Visual::Image(a)) = resources.inline(&data, offsets[0]).unwrap() else {
        panic!()
    };
    let Some(Visual::Image(b)) = resources.inline(&data, offsets[1]).unwrap() else {
        panic!()
    };
    assert_eq!(
        resources.identity(offsets[0]),
        resources.identity(offsets[1])
    );
    assert!(std::sync::Arc::ptr_eq(&a, &b));
    assert!(std::sync::Arc::ptr_eq(
        &a,
        &resources.preview(offsets[1]).unwrap().unwrap()
    ));
}
#[test]
fn original_inline_picture_candidates() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut ole = OleReader::open(
        std::fs::read(root.join("demo/ms_office_docs/docx_01.doc"))
            .unwrap()
            .into(),
    )
    .unwrap();
    let data = ole.read_stream("Data", true).unwrap();
    let cases: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("tests/fixtures/office_rust_reference_068/doc_images.json"))
            .unwrap(),
    )
    .unwrap();
    let mut r = Resources::default();
    for c in cases.as_array().unwrap() {
        let offset = c["offset"].as_u64().unwrap() as u32;
        let v = r.inline(&data, offset).unwrap();
        assert_eq!(
            v.is_some(),
            c["exists"].as_bool().unwrap(),
            "offset {offset}"
        );
        if let Some(Visual::Equation(latex)) = v {
            assert_eq!(latex, c["equation"].as_str().unwrap());
        }
        let _ = r.inline(&data, offset).unwrap();
    }
}
