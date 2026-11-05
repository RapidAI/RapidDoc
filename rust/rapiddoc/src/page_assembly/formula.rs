//! 逐规则迁移 _optimize_formula_number_blocks；先邻块，再几何同排候选。
use super::*;
use rapiddoc_native::text_boundary::python_space;

fn array_or_empty<'a>(value: &'a Value, key: &str) -> Result<&'a [Value], Error> {
    match value.get(key) {
        None => Ok(&[]),
        Some(Value::Array(a)) => Ok(a),
        _ => Err(invalid()),
    }
}
fn extract_text(block: &Value, work: &mut Work) -> Result<String, Error> {
    let mut text = String::new();
    for line in array_or_empty(block, "lines")? {
        for span in array_or_empty(line, "spans")? {
            if span["type"] == "text" {
                let content = match span.get("content") {
                    None => "",
                    Some(Value::String(s)) => s,
                    _ => return Err(invalid()),
                };
                work.charge(content.len())?;
                text.push_str(content);
            }
        }
    }
    Ok(text.trim_matches(python_space).to_owned())
}
fn tag(text: &str) -> String {
    let mut text: String = text
        .trim_matches(python_space)
        .chars()
        .map(|c| {
            let n = c as u32;
            if (0xff01..=0xff5e).contains(&n) {
                char::from_u32(n - 0xfee0).unwrap()
            } else {
                c
            }
        })
        .collect();
    if let Some(stripped) = text.strip_prefix('(') {
        text = stripped.trim_matches(python_space).into();
    }
    if let Some(stripped) = text.strip_suffix(')') {
        text = stripped.trim_matches(python_space).into();
    }
    text
}
fn append(block: &mut Value, tag: &str, work: &mut Work) -> Result<bool, Error> {
    if tag.is_empty() {
        return Ok(false);
    }
    let Some(lines) = block.as_object_mut().ok_or_else(invalid)?.get_mut("lines") else {
        return Ok(false);
    };
    for line in lines.as_array_mut().ok_or_else(invalid)? {
        let Some(spans) = line.as_object_mut().ok_or_else(invalid)?.get_mut("spans") else {
            continue;
        };
        for span in spans.as_array_mut().ok_or_else(invalid)? {
            if span["type"] != "interline_equation" {
                continue;
            }
            if !truth(&span["content"]) {
                return Ok(false);
            }
            let formula = span["content"].as_str().ok_or_else(invalid)?;
            work.charge(formula.len() + tag.len())?;
            let suffix = format!("\\tag{{{tag}}}");
            if formula.contains("\\tag{") {
                return Ok(formula.contains(&suffix));
            }
            span["content"] = json!(format!("{formula}{suffix}"));
            return Ok(true);
        }
    }
    Ok(false)
}
fn validate(pages: &Value) -> Result<(), Error> {
    let mut nodes = 0;
    let mut bytes = 0;
    let mut stack = vec![(pages, 0usize)];
    while let Some((v, depth)) = stack.pop() {
        nodes += 1;
        if nodes > rapiddoc_core::resources::limit(200_000)
            || depth > rapiddoc_core::resources::limit(64)
        {
            return Err(limit());
        }
        match v {
            Value::Array(a) => {
                if nodes + stack.len() + a.len() > rapiddoc_core::resources::limit(200_000) {
                    return Err(limit());
                }
                stack.extend(a.iter().map(|v| (v, depth + 1)));
            }
            Value::Object(o) => {
                bytes += o.keys().map(String::len).sum::<usize>();
                stack.extend(o.values().map(|v| (v, depth + 1)));
            }
            Value::String(s) => bytes += s.len(),
            _ => {}
        }
        if bytes > rapiddoc_core::resources::limit(2_000_000) {
            return Err(limit());
        }
    }
    let pages = pages.as_array().ok_or_else(invalid)?;
    if pages.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let (mut blocks, mut lines) = (0usize, 0usize);
    for page in pages {
        if !page.is_object() {
            return Err(invalid());
        }
        let page_blocks = array_or_empty(page, "preproc_blocks")?;
        blocks += page_blocks.len();
        if blocks > rapiddoc_core::resources::limit(1000) {
            return Err(limit());
        }
        for block in page_blocks {
            if !block.is_object() {
                return Err(invalid());
            }
            let block_lines = array_or_empty(block, "lines")?;
            lines += block_lines.len();
            if lines > rapiddoc_core::resources::limit(10_000) {
                return Err(limit());
            }
            for line in block_lines {
                if !line.is_object() {
                    return Err(invalid());
                }
                for span in array_or_empty(line, "spans")? {
                    if !span.is_object() {
                        return Err(invalid());
                    }
                }
            }
        }
    }
    Ok(())
}

/// 返回页列表副本，仅修改 preproc_blocks；应在 para_split 之前执行。
/// 保留未知字段和旧 para_blocks；不执行 OCR、数学模型或跨页表格处理。
pub fn optimize(pages: &Value) -> Result<Value, Error> {
    validate(pages)?;
    let mut pages = pages.clone();
    let mut work = Work(0);
    for page in pages.as_array_mut().unwrap() {
        let mut blocks = array_or_empty(page, "preproc_blocks")?.to_vec();
        let mut kept = Vec::new();
        for index in 0..blocks.len() {
            if blocks[index]["type"] != "formula_number" {
                kept.push(index);
                continue;
            }
            let text = extract_text(&blocks[index], &mut work)?;
            let tag = tag(&text);
            if index > 0
                && blocks[index - 1]["type"] == "interline_equation"
                && append(&mut blocks[index - 1], &tag, &mut work)?
            {
                continue;
            }
            if index + 1 < blocks.len()
                && blocks[index + 1]["type"] == "interline_equation"
                && (index + 2 >= blocks.len() || blocks[index + 2]["type"] != "formula_number")
                && append(&mut blocks[index + 1], &tag, &mut work)?
            {
                continue;
            }
            coord(&blocks[index]["bbox"], false)?;
            let number = bb(&blocks[index]["bbox"]);
            let width = match page.get("page_size") {
                None => 0.,
                Some(Value::Array(size)) => size
                    .first()
                    .and_then(Value::as_f64)
                    .filter(|n| n.is_finite() && n.abs() <= 1_000_000.)
                    .ok_or_else(invalid)?,
                _ => return Err(invalid()),
            };
            let mut closest = None;
            let mut delta = f64::INFINITY;
            for (i, candidate) in blocks.iter().enumerate() {
                if candidate["type"] != "interline_equation" {
                    continue;
                }
                work.charge(1)?;
                coord(&candidate["bbox"], false)?;
                let eq = bb(&candidate["bbox"]);
                let overlap = (number[3].min(eq[3]) - number[1].max(eq[1])).max(0.);
                let height = (number[3] - number[1]).min(eq[3] - eq[1]);
                let gap = (number[0] - eq[2]).max(0.);
                if height > 0.
                    && overlap / height >= 0.5
                    && number[0] >= eq[0]
                    && gap <= 0.4 * width
                {
                    let distance = ((eq[1] + eq[3]) - (number[1] + number[3])).abs();
                    if distance < delta {
                        closest = Some(i);
                        delta = distance;
                    }
                }
            }
            if let Some(i) = closest {
                if append(&mut blocks[i], &tag, &mut work)? {
                    continue;
                }
            }
            if text.is_empty() {
                continue;
            }
            blocks[index]["type"] = json!("text");
            kept.push(index);
        }
        // 已放入结果的公式仍会被后续编号修改，保持旧对象引用所产生的最终内容。
        page["preproc_blocks"] = json!(kept
            .into_iter()
            .map(|i| blocks[i].clone())
            .collect::<Vec<_>>());
    }
    Ok(pages)
}
