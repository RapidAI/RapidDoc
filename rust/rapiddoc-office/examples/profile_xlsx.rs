//! Release-only native stage diagnostic; does not change conversion settings.
use rapiddoc_office::{document_output as output, spreadsheet, xlsx};
use std::{sync::Arc, time::Instant};
fn main() {
    let path = std::env::args().nth(1).expect("XLSX path");
    let data: Arc<[u8]> = std::fs::read(path).unwrap().into();
    let mut rows = vec![];
    for _ in 0..5 {
        let start = Instant::now();
        let w = xlsx::Workbook::open(data.clone()).unwrap();
        let read = start.elapsed();
        let next = Instant::now();
        let pages = w.into_pages(false, None, true).unwrap();
        let resources = next.elapsed();
        let next = Instant::now();
        let body = spreadsheet::into_body_pages(pages).unwrap();
        let typed = next.elapsed();
        let next = Instant::now();
        let mut html_bytes = 0;
        for b in body.iter().flatten() {
            if matches!(b.kind, rapiddoc_office::docx::body::Kind::Table) {
                if let rapiddoc_office::docx::body::Content::Text(s) = &b.content {
                    html_bytes += s.len();
                    std::hint::black_box(output::html::clean(s).unwrap());
                }
            }
        }
        let clean = next.elapsed();
        let next = Instant::now();
        let middle = output::middle(body, "0.9.11").unwrap();
        let assembly = next.elapsed();
        let next = Instant::now();
        let md =
            output::render::export(&middle.middle, "mm_markdown", "images", &Default::default());
        let content = output::render::export(
            &middle.middle,
            "content_list",
            "images",
            &Default::default(),
        );
        let export = next.elapsed();
        std::hint::black_box((md, content));
        rows.push(serde_json::json!({"read_ms":read.as_secs_f64()*1000.,"resources_ms":resources.as_secs_f64()*1000.,"typed_ms":typed.as_secs_f64()*1000.,"clean_ms":clean.as_secs_f64()*1000.,"html_bytes":html_bytes,"middle_ms":assembly.as_secs_f64()*1000.,"export_ms":export.as_secs_f64()*1000.}));
    }
    println!("{}", serde_json::to_string_pretty(&rows).unwrap());
}
