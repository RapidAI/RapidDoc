//! Shared local/HTTP input loading. Downloads stay in memory, like local reads.
use super::{invalid, Error, Progress};
use std::{
    io::Read,
    path::Path,
    time::{Duration, Instant},
};

pub fn is_url(source: &str) -> bool {
    source
        .get(..7)
        .is_some_and(|s| s.eq_ignore_ascii_case("http://"))
        || source
            .get(..8)
            .is_some_and(|s| s.eq_ignore_ascii_case("https://"))
}

fn decode(value: &str) -> String {
    percent_encoding::percent_decode_str(value)
        .decode_utf8_lossy()
        .into_owned()
}

/// A filename only; URL credentials, query tokens and directory components never
/// become output paths. Query-based download endpoints may supply `filename`.
pub fn source_name(source: &str) -> String {
    let raw = if is_url(source) {
        source
            .split('#')
            .next()
            .unwrap_or(source)
            .parse::<ureq::http::Uri>()
            .ok()
            .map(|uri| {
                uri.query()
                    .and_then(|query| {
                        query.split('&').find_map(|part| {
                            let (key, value) = part.split_once('=')?;
                            (decode(key) == "filename").then(|| decode(value))
                        })
                    })
                    .unwrap_or_else(|| decode(uri.path().rsplit('/').next().unwrap_or("")))
            })
            .unwrap_or_default()
    } else {
        Path::new(source)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .into()
    };
    let raw = raw.rsplit(['/', '\\']).next().unwrap_or("");
    let safe: String = raw
        .chars()
        .map(|c| {
            if c.is_control() || "<>:\"|?*".contains(c) {
                '_'
            } else {
                c
            }
        })
        .collect();
    let safe = safe.trim_end_matches(['.', ' ']);
    if safe.is_empty() || safe == "." || safe == ".." {
        "document".into()
    } else {
        safe.into()
    }
}

pub fn read_url(
    source: &str,
    progress: &mut dyn FnMut(Progress<'_>) -> Result<(), Error>,
) -> Result<(String, Vec<u8>), Error> {
    if !is_url(source) {
        return Err(invalid("Document URL must use HTTP or HTTPS"));
    }
    let name = source_name(source);
    let started = Instant::now();
    let mut notify = |stage: &'static str, bytes: usize, total: Option<usize>| {
        progress(Progress {
            stage,
            detail: Some(name.clone()),
            completed: Some(bytes),
            total,
            page_index: None,
            elapsed_seconds: Some(started.elapsed().as_secs_f64()),
        })
    };
    notify("source_download_started", 0, None)?;
    let agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(60)))
        .timeout_recv_response(Some(Duration::from_secs(60)))
        .timeout_per_read(Some(Duration::from_secs(60)))
        .build()
        .new_agent();
    let mut response = agent
        .get(source.split('#').next().unwrap_or(source))
        // Each document owns its client. Avoid reusing a redirect connection
        // that a server has already closed after its empty response.
        .header("Connection", "close")
        .call()
        .map_err(|e| invalid(format!("Document download failed: {e}")))?;
    let total = response
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse().ok());
    let mut data = Vec::new();
    let mut buffer = [0u8; 65536];
    let mut reader = response.body_mut().as_reader();
    let mut last_notice = Instant::now();
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|e| invalid(format!("Document download failed: {e}")))?;
        if count == 0 {
            break;
        }
        data.extend_from_slice(&buffer[..count]);
        if last_notice.elapsed() >= Duration::from_millis(100) {
            notify("source_download_progress", data.len(), total)?;
            last_notice = Instant::now();
        }
    }
    notify("source_download_completed", data.len(), total)?;
    Ok((name, data))
}
