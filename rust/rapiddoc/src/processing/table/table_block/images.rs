//! 旧materialize_table_media：页内部分成功保留，其余占位符转义文字回退。
use super::*;
use crate::legacy_page::{assets::Pixels, truth};
use rapiddoc_core::{ErrorCode, Limits};
use std::path::Path;

struct Budget {
    work: Work,
    pixels: usize,
    encoded: usize,
    output: usize,
}
impl Default for Budget {
    fn default() -> Self {
        Self {
            work: Work(0),
            pixels: 0,
            encoded: 0,
            output: 0,
        }
    }
}
fn escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}
fn table_indices(blocks: &Value) -> Result<Vec<usize>, Error> {
    let mut indices = Vec::new();
    for (i, b) in array(blocks, 4000)?.iter().enumerate() {
        if !truth(&b["_table_images"]) {
            continue;
        }
        if !b["content"].is_string() {
            return Err(invalid());
        }
        for item in array(&b["_table_images"], 5000)? {
            if !item.is_object() || !item["text"].is_string() {
                return Err(invalid());
            }
        }
        indices.push(i);
    }
    Ok(indices)
}
fn normalize_box(b: [f64; 4], page: Option<[f64; 2]>) -> [f64; 4] {
    let Some(p) = page else { return b };
    let mut ticks = std::array::from_fn::<_, 4, _>(|i| {
        (b[i] / p[i % 2] * 1000.).round_ties_even().clamp(0., 1000.) as i64
    });
    for i in 0..2 {
        if ticks[i + 2] <= ticks[i] {
            if ticks[i] < 1000 {
                ticks[i + 2] = ticks[i] + 1;
            } else {
                ticks[i] = (ticks[i + 2] - 1).max(0);
            }
        }
    }
    ticks.map(|n| n as f64 / 1000.)
}
fn page_size(v: &Value) -> Result<Option<[f64; 2]>, Error> {
    if v.is_null() {
        return Ok(None);
    }
    let a = array(v, 2)?;
    if a.len() != 2 {
        return Err(invalid());
    }
    let p = [number(&a[0])?, number(&a[1])?];
    if p.iter().any(|n| *n <= 0.) {
        return Err(invalid());
    }
    Ok(Some(p))
}
fn replace(
    content: &mut Value,
    needle: &str,
    value: &str,
    budget: &mut Budget,
) -> Result<(), Error> {
    let old = content.as_str().ok_or_else(invalid)?;
    let count = old.matches(needle).count();
    let size = old
        .len()
        .saturating_sub(count * needle.len())
        .saturating_add(count.saturating_mul(value.len()));
    budget.work.charge(old.len().saturating_add(size))?;
    if size > rapiddoc_core::resources::limit(64_000_000) {
        return Err(limit());
    }
    if count > 0 {
        *content = json!(old.replace(needle, value));
    }
    Ok(())
}
fn resolve(
    blocks: &Value,
    png: Option<&[u8]>,
    size: Option<[f64; 2]>,
    budget: &mut Budget,
) -> Result<Value, Error> {
    let tables = table_indices(blocks)?;
    let mut output = blocks.clone();
    if tables.is_empty() {
        return Ok(output);
    }
    let pixels = match png {
        None => None,
        Some(bytes) => match Pixels::decode(bytes) {
            Ok(p) => Some(p),
            Err(e) if matches!(e.code, ErrorCode::ResourceLimit | ErrorCode::Unsupported) => {
                return Err(e)
            }
            Err(_) => None,
        },
    };
    let mut failed = pixels.is_none();
    for &index in &tables {
        if failed {
            break;
        }
        let table = &mut output[index];
        let angle = table.get("angle").cloned().unwrap_or(json!(0));
        let angle = if let Some(n) = angle.as_i64() {
            n
        } else if let Some(n) = angle
            .as_f64()
            .filter(|n| n.is_finite() && matches!(*n, 90. | 180. | 270.))
        {
            n as i64
        } else {
            0
        };
        let image = pixels.as_ref().unwrap();
        for i in 0..table["_table_images"].as_array().unwrap().len() {
            let encoded = (|| -> Result<Vec<u8>, Error> {
                // 不提前拒绝后续坏bbox：旧实现保留同页此前已生成的PNG。
                let b = candidate_rect(&table["_table_images"][i]["bbox"])?;
                let b = normalize_box(b, size);
                let crop = [
                    (b[0] * image.width as f64).floor().max(0.) as i64,
                    (b[1] * image.height as f64).floor().max(0.) as i64,
                    (b[2] * image.width as f64).ceil().min(image.width as f64) as i64,
                    (b[3] * image.height as f64).ceil().min(image.height as f64) as i64,
                ];
                if crop[2] <= crop[0] || crop[3] <= crop[1] {
                    return Err(invalid());
                }
                let count = (crop[2] - crop[0]) as usize * (crop[3] - crop[1]) as usize;
                budget.pixels = budget.pixels.saturating_add(count);
                if budget.pixels > rapiddoc_core::resources::limit(64_000_000) {
                    return Err(limit());
                }
                image.crop(crop)?.rotate_ccw(angle)?.png()
            })();
            match encoded {
                Ok(bytes) => {
                    budget.encoded = budget.encoded.saturating_add(bytes.len());
                    if budget.encoded > rapiddoc_core::resources::limit(64_000_000) {
                        return Err(limit());
                    }
                    let uri = format!("data:image/png;base64,{}\"", base64::encode(bytes));
                    replace(
                        &mut table["content"],
                        &format!("docvortex-table-image:{i}\""),
                        &uri,
                        budget,
                    )?;
                }
                Err(e) if matches!(e.code, ErrorCode::ResourceLimit | ErrorCode::Unsupported) => {
                    return Err(e)
                }
                Err(_) => {
                    failed = true;
                    break;
                }
            }
        }
    }
    for index in tables {
        let table = &mut output[index];
        let items = table
            .as_object_mut()
            .unwrap()
            .remove("_table_images")
            .unwrap();
        for (i, item) in items.as_array().unwrap().iter().enumerate() {
            let text = item["text"].as_str().unwrap();
            budget.work.charge(text.len().saturating_mul(6))?;
            replace(
                &mut table["content"],
                &format!("<img src=\"docvortex-table-image:{i}\" alt=\"Image\" />"),
                &escape(text),
                budget,
            )?;
        }
        budget.output = budget.output.saturating_add(table.to_string().len());
        if budget.output > rapiddoc_core::resources::limit(64_000_000) {
            return Err(limit());
        }
    }
    Ok(output)
}
pub fn page(request: &Value, png: Option<&[u8]>) -> Result<Value, Error> {
    validate_tree(request)?;
    resolve(
        &request["blocks"],
        png,
        page_size(&request["page_size"])?,
        &mut Budget::default(),
    )
}
pub fn pdf(
    request: &Value,
    library: &Path,
    data: &[u8],
    password: Option<&str>,
    max_pixels: usize,
) -> Result<Value, Error> {
    let max_pixels = rapiddoc_core::resources::limit(max_pixels);
    validate_tree(request)?;
    if max_pixels == 0 {
        return Err(invalid());
    }
    let pages = array(&request["pages"], 10000)?;
    let sizes = if request["page_sizes"].is_null() {
        None
    } else {
        Some(array(&request["page_sizes"], 10000)?)
    };
    if sizes.is_some_and(|s| s.len() != pages.len()) {
        return Err(invalid());
    }
    let mut budget = Budget::default();
    let mut output = Vec::new();
    for (i, blocks) in pages.iter().enumerate() {
        let source_index = request
            .get("source_page_start")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize
            + i;
        let size = page_size(sizes.map_or(&Value::Null, |s| &s[i]))?;
        if table_indices(blocks)?.is_empty() {
            output.push(blocks.clone());
            continue;
        }
        let rendered = rapiddoc_pdf::render::render_table_media_page(
            library,
            data,
            source_index,
            max_pixels,
            password,
            &Limits::default(),
        );
        match rendered {
            Ok(page) => output.push(resolve(blocks, Some(&page.png), size, &mut budget)?),
            Err(e) if e.code == ErrorCode::ResourceLimit => return Err(e),
            Err(_) => output.push(resolve(blocks, None, size, &mut budget)?),
        }
    }
    Ok(json!(output))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    fn canonical(pages: Value) -> Value {
        let mut pages = pages;
        let mut images = Vec::new();
        for page in pages.as_array_mut().unwrap() {
            for b in page.as_array_mut().unwrap() {
                let Some(content) = b["content"].as_str() else {
                    continue;
                };
                let mut result = String::new();
                let mut offset = 0;
                let prefix = "data:image/png;base64,";
                while let Some(start) = content[offset..].find(prefix).map(|i| i + offset) {
                    let begin = start + prefix.len();
                    let mut end = begin;
                    while end < content.len()
                        && (content.as_bytes()[end].is_ascii_alphanumeric()
                            || b"+/=".contains(&content.as_bytes()[end]))
                    {
                        end += 1;
                    }
                    let png = base64::decode(&content[begin..end]).unwrap();
                    let p = Pixels::decode(&png).unwrap();
                    images.push(json!({"mode":match p.channels{1=>"L",2=>"LA",3=>"RGB",_=>"RGBA"},"width":p.width,"height":p.height,"pixels_sha256":format!("{:x}",Sha256::digest(&p.data))}));
                    result.push_str(&content[offset..start]);
                    result.push_str(&format!("@PNG{}@", images.len() - 1));
                    offset = end;
                }
                result.push_str(&content[offset..]);
                b["content"] = json!(result);
            }
        }
        json!({"pages":pages,"images":images})
    }
    #[test]
    fn old_png_crops_rotations_partial_fallback_and_point_ticks() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let f: Value = serde_json::from_slice(
            &std::fs::read(root.join("build/migration/native-table-media-png.json")).unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            let png = std::fs::read(root.join(c["source"].as_str().unwrap())).unwrap();
            let actual = page(
                &json!({"blocks":c["blocks"],"page_size":c["page_size"]}),
                Some(&png),
            )
            .unwrap();
            let mut expected = c["expected"].clone();
            for i in expected["images"].as_array_mut().unwrap() {
                i.as_object_mut().unwrap().remove("icc_profile_sha256");
                i.as_object_mut().unwrap().remove("transparency");
            }
            assert_eq!(canonical(json!([actual])), expected, "{}", c["label"]);
        }
    }
}
