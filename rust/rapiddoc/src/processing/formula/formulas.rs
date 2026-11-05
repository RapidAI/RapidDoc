//! BatchAnalyze 公式任务跨页展开与非空识别结果回填；不实现公式模型。
use crate::legacy_page::{invalid, limit, validate_tree};
use rapiddoc_core::Error;
use serde_json::{json, Value};

/// 准备阶段生成的原 RGB PNG；预检实际解码，损坏图不送入模型适配器。
pub fn validate_png(bytes: &[u8]) -> Result<usize, Error> {
    let pixels = crate::legacy_page::assets::Pixels::decode(bytes)?;
    if pixels.channels != 3 {
        return Err(invalid());
    }
    let count = pixels.width.checked_mul(pixels.height).ok_or_else(limit)?;
    if count > rapiddoc_core::resources::limit(4_000_000) {
        return Err(limit());
    }
    Ok(count)
}

/// 验证 prepared crop plans，并保持页序及页内公式任务顺序。
pub fn plan(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let pages = request["plans"].as_array().ok_or_else(invalid)?;
    if pages.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let mut jobs = Vec::new();
    let mut layouts = 0usize;
    for (page, p) in pages.iter().enumerate() {
        let dets = p["model"]["layout_dets"].as_array().ok_or_else(invalid)?;
        layouts = layouts.checked_add(dets.len()).ok_or_else(limit)?;
        if layouts > rapiddoc_core::resources::limit(20_000) {
            return Err(limit());
        }
        let tasks = p["formulas"].as_array().ok_or_else(invalid)?;
        let expected: Vec<_> = dets
            .iter()
            .enumerate()
            .filter_map(|(i, d)| {
                matches!(d["category_id"].as_i64(), Some(8 | 13 | 14)).then_some(i)
            })
            .collect();
        if tasks.len() != expected.len() {
            return Err(invalid());
        }
        for (t, index) in tasks.iter().zip(expected) {
            if t["layout_index"].as_u64() != Some(index as u64) {
                return Err(invalid());
            }
            let source = t["source"]
                .as_str()
                .filter(|s| !s.is_empty() && s.len() <= 256)
                .ok_or_else(invalid)?;
            jobs.push(json!({"page":page,"layout_index":index,"source":source}));
            if jobs.len() > rapiddoc_core::resources::limit(1000) {
                return Err(limit());
            }
        }
    }
    Ok(json!({"jobs":jobs}))
}

/// 旧 zip 的非空回填规则；结果数不符显式失败，避免静默截断。
pub fn apply(request: &Value) -> Result<Value, Error> {
    let planned = plan(request)?;
    let jobs = planned["jobs"].as_array().unwrap();
    let results = request["results"].as_array().ok_or_else(invalid)?;
    if jobs.len() != results.len() {
        return Err(invalid());
    }
    for result in results {
        if !result.is_null() && !result.is_string() {
            return Err(invalid());
        }
    }
    let mut pages = request["plans"].clone();
    for (job, result) in jobs.iter().zip(results) {
        if result.as_str().is_some_and(|s| !s.is_empty()) {
            let page = job["page"].as_u64().unwrap() as usize;
            let index = job["layout_index"].as_u64().unwrap() as usize;
            pages[page]["model"]["layout_dets"][index]["latex"] = result.clone();
        }
    }
    Ok(pages)
}
