//! 旧 _post_process_ocr 的收集顺序与回填；图片负载由宿主单独提供。
use crate::legacy_page::{invalid, limit, validate_tree};
use rapiddoc_core::Error;
use serde::Serialize;
use serde_json::{json, Value};

#[derive(Serialize)]
pub struct Target {
    pub path: String,
    pub image: Value,
}

pub fn plan(pages: &Value) -> Result<Vec<Target>, Error> {
    validate_tree(pages)?;
    let pages = pages.as_array().ok_or_else(invalid)?;
    if pages.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let mut targets = Vec::new();
    let mut blocks = 0;
    let mut lines = 0;
    let mut spans = 0;
    let mut collect = |block: &Value, path: String| -> Result<(), Error> {
        blocks += 1;
        if blocks > rapiddoc_core::resources::limit(1000) {
            return Err(limit());
        }
        for (l, line) in block["lines"]
            .as_array()
            .ok_or_else(invalid)?
            .iter()
            .enumerate()
        {
            lines += 1;
            if lines > rapiddoc_core::resources::limit(10_000) {
                return Err(limit());
            }
            for (s, span) in line["spans"]
                .as_array()
                .ok_or_else(invalid)?
                .iter()
                .enumerate()
            {
                spans += 1;
                if spans > rapiddoc_core::resources::limit(20_000) || !span.is_object() {
                    return Err(if spans > rapiddoc_core::resources::limit(20_000) {
                        limit()
                    } else {
                        invalid()
                    });
                }
                if let Some(image) = span.get("np_img") {
                    if targets.len() >= rapiddoc_core::resources::limit(1000) {
                        return Err(limit());
                    }
                    targets.push(Target {
                        path: format!("{path}/lines/{l}/spans/{s}"),
                        image: image.clone(),
                    });
                }
            }
        }
        Ok(())
    };
    for (p, page) in pages.iter().enumerate() {
        for (b, block) in page["preproc_blocks"]
            .as_array()
            .ok_or_else(invalid)?
            .iter()
            .enumerate()
        {
            let path = format!("/{p}/preproc_blocks/{b}");
            match block["type"].as_str().ok_or_else(invalid)? {
                "table" | "image" => {
                    for (j, sub) in block["blocks"]
                        .as_array()
                        .ok_or_else(invalid)?
                        .iter()
                        .enumerate()
                    {
                        if matches!(
                            sub["type"].as_str(),
                            Some(
                                "image_caption"
                                    | "image_footnote"
                                    | "table_caption"
                                    | "table_footnote"
                            )
                        ) {
                            collect(sub, format!("{path}/blocks/{j}"))?;
                        }
                    }
                }
                "text" | "title" | "formula_number" => collect(block, path)?,
                _ => {}
            }
        }
        for (b, block) in page["discarded_blocks"]
            .as_array()
            .ok_or_else(invalid)?
            .iter()
            .enumerate()
        {
            collect(block, format!("/{p}/discarded_blocks/{b}"))?;
        }
    }
    Ok(targets)
}

pub fn apply(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let targets = plan(&request["pages"])?;
    let results = request["results"].as_array().ok_or_else(invalid)?;
    if targets.len() != results.len() {
        return Err(invalid());
    }
    let mut checked = Vec::new();
    for result in results {
        let pair = result
            .as_array()
            .filter(|v| v.len() == 2)
            .ok_or_else(invalid)?;
        let text = pair[0].as_str().ok_or_else(invalid)?;
        let score = pair[1]
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(invalid)?;
        let (text, score) = if score > 0.5 {
            (
                text,
                format!("{score:.3}")
                    .parse::<f64>()
                    .map_err(|_| invalid())?,
            )
        } else {
            ("", 0.)
        };
        checked.push((text, score));
    }
    let mut pages = request["pages"].clone();
    for (target, (text, score)) in targets.iter().zip(checked) {
        let span = pages
            .pointer_mut(&target.path)
            .and_then(Value::as_object_mut)
            .ok_or_else(invalid)?;
        span.remove("np_img");
        span.insert("content".into(), json!(text));
        span.insert("score".into(), json!(score));
    }
    Ok(pages)
}
