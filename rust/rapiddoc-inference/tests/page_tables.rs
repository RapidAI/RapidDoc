use image::{ImageFormat, RgbImage};
use rapiddoc_core::ErrorCode;
use rapiddoc_inference::page_tables::{LayoutRegion, PageImage};
use std::io::Cursor;
fn image() -> PageImage {
    let image = RgbImage::from_fn(31, 23, |x, y| image::Rgb([x as u8, y as u8, 99]));
    let mut output = Cursor::new(Vec::new());
    image.write_to(&mut output, ImageFormat::Png).unwrap();
    PageImage::decode(output.get_ref(), 16000000).unwrap()
}
fn region() -> LayoutRegion {
    LayoutRegion {
        cls_id: 4,
        label: "table".into(),
        score: 0.9,
        coordinate: [-2.2, 2.4, 10.1, 20.1],
        order: 9,
    }
}
#[test]
fn crop_grid_clipping_and_pixel_ownership() {
    let page = image();
    let plan = page.plan(&[region()], 5, 1, 345).unwrap();
    assert_eq!(plan.tables[0].bounds, [0, 0, 15, 23]);
    assert_eq!(plan.crop_pixels, 345);
    let rgb = page.crop_rgb([3, 4, 5, 6]).unwrap();
    assert_eq!(rgb, [3, 4, 99, 4, 4, 99, 3, 5, 99, 4, 5, 99]);
    assert_eq!(
        page.plan(&[region()], 5, 0, 345).err().unwrap().code,
        ErrorCode::ResourceLimit
    );
    assert_eq!(
        page.plan(&[region()], 5, 1, 344).err().unwrap().code,
        ErrorCode::ResourceLimit
    );
    assert!(page.plan(&[region()], 0, 1, 345).is_err());
    assert!(page.crop_png([0, 0, 32, 23]).is_err());
}
#[test]
fn project_both_routes_and_preserve_source_indices() {
    let page = image();
    let local = r#"{"route":"wired","ocr_lines":[{"points":[[0,0],[2,0],[2,3],[0,3]],"text":"A","confidence":0.9}],"cells":{"polygons":[[[0,0],[0,3],[2,3],[2,0]]]},"matches":{"cells":[[{"ocr_index":0,"points":[[0,0],[2,0],[2,3],[0,3]],"text":"A","confidence":0.9,"split":false}]]}}"#;
    let projected = page.project(local, [5, 10, 25, 20]).unwrap();
    assert_eq!(projected.cell_polygons[0][2], [7., 13.]);
    assert_eq!(projected.ocr_lines[0].points[0], [5., 10.]);
    assert_eq!(projected.matched_cells.unwrap()[0][0].ocr_index, Some(0));
    let wireless = r#"{"route":"wireless","ocr_lines":[],"structure":{"cell_bboxes":[[0,0,2,3]]},"matches":{"cells":[[]]}}"#;
    let projected = page.project(wireless, [5, 10, 25, 20]).unwrap();
    assert_eq!(projected.cell_polygons[0][2], [7., 13.]);
    assert!(projected.matched_cells.is_none());
    let eight = wireless.replace("[0,0,2,3]", "[0,0,2,0,2,3,0,3]");
    assert_eq!(
        page.project(&eight, [5, 10, 25, 20]).unwrap().cell_polygons[0],
        [[5., 10.], [7., 10.], [7., 13.], [5., 13.]]
    );
    assert!(page
        .project(&wireless.replace("[[]]", "[[9]]"), [5, 10, 25, 20])
        .is_err());
}
#[test]
fn text_and_table_share_preflight_budget_and_project_ocr() {
    let page = image();
    let mut text = region();
    text.label = "paragraph_title".into();
    text.coordinate = [1., 1., 9., 9.];
    let mut unsupported = region();
    unsupported.label = "display_formula".into();
    let regions = [text, unsupported, region()];
    let plan = page.plan_blocks(&regions, 5, 2, 445).unwrap();
    assert_eq!(
        plan.regions
            .iter()
            .map(|v| v.layout_index)
            .collect::<Vec<_>>(),
        [0, 2]
    );
    assert_eq!(plan.crop_pixels, 445);
    assert_eq!(
        page.plan_blocks(&regions, 5, 2, 444).err().unwrap().code,
        ErrorCode::ResourceLimit
    );
    assert!(page.plan_blocks(&regions, 5, 1, 1000).is_err());
    let lines = r#"[{"points":[[0,0],[2,0],[2,3],[0,3]],"text":"正文<&>","confidence":0.75}]"#;
    let actual = page.project_ocr(lines, [5, 10, 25, 20]).unwrap();
    assert_eq!(actual[0].points[2], [7., 13.]);
    assert_eq!(actual[0].text, "正文<&>");
    assert_eq!(actual[0].confidence, 0.75);
    assert!(page
        .project_ocr(&lines.replace("0.75", "1.5"), [5, 10, 25, 20])
        .is_err());
    assert!(page
        .project_ocr(&lines.replace("[0,0]", "[1000001,0]"), [5, 10, 25, 20])
        .is_err());
}
