//! cargo run --release -p rapiddoc --example parse_document -- input.docx
//! For PDFs, put release runtime libraries in a directory and pass it second.
use rapiddoc::{ParseOptions, RapidDoc, RapidDocConfig, RuntimeResources};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let input = args
        .next()
        .ok_or("Usage: parse_document INPUT [RUNTIME_DIRECTORY]")?;
    let mut resources = RuntimeResources::default();
    if let Some(directory) = args.next() {
        resources.bundle_dirs.insert(0, directory.into());
    }
    let mut engine = RapidDoc::new(RapidDocConfig::default(), resources);
    let result = engine.parse(input, &ParseOptions::default())?;
    println!("{}", result.markdown.as_str().unwrap_or(""));
    eprintln!(
        "Completed {} pages, {} images",
        result.source_page_count,
        result.images.len()
    );
    Ok(())
}
