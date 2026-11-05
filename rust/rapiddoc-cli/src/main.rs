//! 无 Python 依赖的原生 PDF 文字命令行入口。

#[cfg(feature = "mimalloc-allocator")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use rapiddoc_core::{Error, ErrorCode, Limits, PageRange};
use std::{io::Read, path::PathBuf};

/// 生成命令参数错误，不启动运行库。
fn invalid(message: impl Into<String>) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: message.into(),
    }
}

/// 解析显式参数；所有解析错误退出为非零，stdout 仅保存业务结果。
fn run() -> Result<(), Error> {
    let mut args = std::env::args().skip(1);
    let command = args.next().ok_or_else(|| invalid("usage: rapiddoc-cli capabilities | extract-text INPUT --pdfium LIB [--start N --end N --allow-partial]"))?;
    if command == "capabilities" {
        println!("{{\"formats\":[\"pdf_native_text\"],\"ocr\":false,\"layout\":false,\"tables\":false,\"images\":false}}");
        return Ok(());
    }
    if command != "extract-text" {
        return Err(invalid("unknown command"));
    }
    let input = args
        .next()
        .ok_or_else(|| invalid("input file is required"))?;
    let mut library = None;
    let mut range = PageRange::default();
    let mut partial = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--pdfium" => {
                library = Some(PathBuf::from(
                    args.next()
                        .ok_or_else(|| invalid("--pdfium requires a path"))?,
                ))
            }
            "--start" => {
                range.start = args
                    .next()
                    .ok_or_else(|| invalid("--start requires a number"))?
                    .parse()
                    .map_err(|_| invalid("invalid start page"))?
            }
            "--end" => {
                range.end = Some(
                    args.next()
                        .ok_or_else(|| invalid("--end requires a number"))?
                        .parse()
                        .map_err(|_| invalid("invalid end page"))?,
                )
            }
            "--allow-partial" => partial = true,
            _ => return Err(invalid(format!("unknown argument: {arg}"))),
        }
    }
    let library = library
        .or_else(|| std::env::var_os("RAPIDDOC_PDFIUM_LIBRARY").map(PathBuf::from))
        .ok_or_else(|| invalid("--pdfium or RAPIDDOC_PDFIUM_LIBRARY is required"))?;
    let limits = Limits::default();
    let mut bytes = Vec::new();
    std::fs::File::open(input)
        .map_err(|error| invalid(error.to_string()))?
        .take(limits.max_input_bytes as u64 + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| invalid(error.to_string()))?;
    let output = rapiddoc::parse_pdf_text(&library, &bytes, None, &range, &limits, partial)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&output).map_err(|error| invalid(error.to_string()))?
    );
    Ok(())
}

/// 将错误码和文案单独输出至 stderr，便于脚本读取。
fn main() {
    if let Err(error) = run() {
        eprintln!(
            "{}",
            serde_json::json!({"code": error.code.as_str(), "message": error.message})
        );
        std::process::exit(2);
    }
}
