//! Container/XML/cell-reader capability probe, never a RapidDoc converter.
use calamine::Reader;
use quick_xml::{events::Event, Reader as XmlReader};
use serde_json::{json, Value};
use std::{error::Error, fs::File, io::Read, path::Path};

fn inspect(path: &Path) -> Result<Value, Box<dyn Error>> {
    let extension = path.extension().and_then(|v| v.to_str()).unwrap_or("");
    let mut output = json!({"source":path.display().to_string(),"extension":extension});
    match extension {
        "docx" | "xlsx" | "pptx" => {
            let mut archive = zip::ZipArchive::new(File::open(path)?)?;
            let names: Vec<_> = archive
                .file_names()
                .map(|name| name.map(|value| value.into_owned()))
                .collect::<Result<_, _>>()?;
            // The root relationships are independent of conventional main-part paths.
            let mut rels = String::new();
            archive.by_name("_rels/.rels")?.read_to_string(&mut rels)?;
            let relationships = roxmltree::Document::parse(&rels)?;
            let main = relationships
                .descendants()
                .find(|node| {
                    node.is_element()
                        && node
                            .attribute("Type")
                            .is_some_and(|value| value.ends_with("/officeDocument"))
                })
                .and_then(|node| node.attribute("Target"))
                .ok_or("missing main relationship")?;
            let main = main.trim_start_matches('/');
            let mut bytes = Vec::new();
            archive.by_name(main)?.read_to_end(&mut bytes)?;
            let mut reader = XmlReader::from_reader(bytes.as_slice());
            let mut elements = 0;
            loop {
                match reader.read_event()? {
                    Event::Start(_) | Event::Empty(_) => elements += 1,
                    Event::Eof => break,
                    _ => {}
                }
            }
            output["package"] = json!({"entries":names.len(),"main_part":main,"main_elements":elements,
                "media_parts":names.iter().filter(|name| name.contains("/media/")).count()});
        }
        "doc" | "xls" | "ppt" => {
            let compound = cfb::open(path)?;
            let streams: Vec<_> = compound
                .walk()
                .filter(|entry| entry.is_stream())
                .map(|entry| entry.path().display().to_string())
                .collect();
            if streams.is_empty() {
                return Err("compound file has no streams".into());
            }
            output["ole_streams"] = json!(streams);
        }
        _ => return Err("unsupported probe extension".into()),
    }
    if matches!(extension, "xlsx" | "xls") {
        let mut workbook = calamine::open_workbook_auto(path)?;
        let mut sheets = Vec::new();
        for name in workbook.sheet_names().to_vec() {
            let cells = workbook.worksheet_range(&name)?;
            sheets.push(json!({"name":name,"nonempty_cells":cells.used_cells().count()}));
        }
        output["cell_reader"] = json!({"sheets":sheets,
            "warning":"Only cell-read capability; no formatting, chart, equation, or RapidDoc parity claim"});
    }
    Ok(output)
}

fn main() -> Result<(), Box<dyn Error>> {
    let paths: Vec<_> = std::env::args_os().skip(1).collect();
    if paths.is_empty() {
        return Err("provide Office fixture paths".into());
    }
    let results: Result<Vec<_>, _> = paths.iter().map(|path| inspect(Path::new(path))).collect();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({"container_probe_only":true,"files":results?}))?
    );
    Ok(())
}
