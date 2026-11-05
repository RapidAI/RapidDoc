//! Original Markdown/content-list rendering, including Magika code blocks.
use super::*;
use rapiddoc_native::text_boundary::{python_space, resolve};
use std::path::PathBuf;

fn trim(s: &str) -> &str {
    s.trim_matches(python_space)
}
fn full_half(s: &str) -> String {
    s.chars()
        .map(|c| {
            let n = c as u32;
            if (0xff21..=0xff3a).contains(&n)
                || (0xff41..=0xff5a).contains(&n)
                || (0xff10..=0xff19).contains(&n)
            {
                char::from_u32(n - 0xfee0).unwrap()
            } else {
                c
            }
        })
        .collect()
}
fn content(s: &Value) -> Result<String, Error> {
    match s.get("content") {
        None => Ok(String::new()),
        Some(Value::Null) => Ok("None".into()),
        Some(Value::String(s)) => Ok(s.clone()),
        _ => Err(invalid()),
    }
}
struct Budget(usize, Option<PathBuf>);
pub(crate) fn merge_caption(block: &Value) -> Result<String, Error> {
    merge(block, &mut Budget(0, None))
}
impl Budget {
    fn scan(&mut self, n: usize) -> Result<(), Error> {
        self.0 = self
            .0
            .checked_add(n)
            .filter(|n| *n <= rapiddoc_core::resources::limit(32_000_000))
            .ok_or_else(limit)?;
        Ok(())
    }
}

fn merge(block: &Value, budget: &mut Budget) -> Result<String, Error> {
    let lines = block["lines"].as_array().ok_or_else(invalid)?;
    if block["original_label"] == "algorithm"
        || lines.iter().any(|l| {
            l["spans"]
                .as_array()
                .unwrap()
                .iter()
                .any(|s| s["original_label"] == "algorithm")
        })
    {
        let mut code_lines = Vec::with_capacity(lines.len());
        for line in lines {
            let mut text = String::new();
            for span in line["spans"].as_array().unwrap() {
                match span["type"].as_str() {
                    Some("text") => text.push_str(&full_half(&content(span)?)),
                    Some("inline_equation" | "interline_equation" | "checkbox") => {
                        text.push_str(&content(span)?);
                    }
                    _ => {}
                }
            }
            budget.scan(text.len())?;
            code_lines.push(text.trim_end_matches(python_space).to_owned());
        }
        let joined = code_lines.join("\n");
        let code = joined.trim_matches('\n');
        if code.is_empty() {
            return Ok(String::new());
        }
        // Original guess_language_by_text catches classifier errors and uses txt.
        let language = rapiddoc_inference::magika::language(code, budget.1.as_deref())
            .unwrap_or_else(|_| "txt".into());
        return Ok(format!("```{language}\n{code}\n```"));
    }
    let mut rendered = String::new();
    let mut raw = String::new();
    let mut previous_tail_text = false;
    for line in lines {
        let mut lr = String::new();
        let mut lt = String::new();
        let mut previous: Option<(Value, String)> = None;
        let mut first_text = false;
        for span in line["spans"].as_array().unwrap() {
            let kind = span["type"].as_str().unwrap();
            if !matches!(
                kind,
                "text" | "inline_equation" | "interline_equation" | "checkbox"
            ) {
                continue;
            }
            let mut value = content(span)?;
            let out = match kind {
                "text" => {
                    value = full_half(&value);
                    let mut escaped = value.clone();
                    for c in ['*', '`', '~'] {
                        escaped = escaped.replace(c, &format!("\\{c}"));
                    }
                    trim(&escaped).to_owned()
                }
                "inline_equation" => {
                    if value.is_empty() {
                        String::new()
                    } else {
                        format!("${value}$")
                    }
                }
                "interline_equation" => {
                    if value.is_empty() {
                        String::new()
                    } else {
                        format!("\n$$\n{value}\n$$\n")
                    }
                }
                _ => trim(&value).to_owned(),
            };
            if out.is_empty() {
                continue;
            }
            let separator = if let Some((prev, pvalue)) = &previous {
                if prev["type"] != "text"
                    || kind != "text"
                    || pvalue.ends_with([' ', '\t'])
                    || value.starts_with([' ', '\t'])
                {
                    " "
                } else if prev.get("bbox").is_some_and(truth) && span.get("bbox").is_some_and(truth)
                {
                    let a = bb(&prev["bbox"]);
                    let b = bb(&span["bbox"]);
                    let h = (a[3] - a[1]).min(b[3] - b[1]);
                    if h > 0. {
                        if b[0] - a[2] > 0.15 * h {
                            " "
                        } else {
                            ""
                        }
                    } else {
                        budget.scan(pvalue.len() + value.len())?;
                        resolve(pvalue, &value).1
                    }
                } else {
                    budget.scan(pvalue.len() + value.len())?;
                    resolve(pvalue, &value).1
                }
            } else {
                first_text = kind == "text";
                ""
            };
            lr.push_str(separator);
            lr.push_str(&out);
            lt.push_str(separator);
            lt.push_str(trim(&value));
            previous = Some((span.clone(), value));
        }
        if lr.is_empty() {
            continue;
        }
        if !rendered.is_empty() {
            let separator = if truth(&line["is_list_start_line"]) {
                raw.push('\n');
                "  \n"
            } else if previous_tail_text && first_text {
                budget.scan(raw.len() + lt.len())?;
                let (adjusted, sep) = resolve(&raw, &lt);
                if adjusted.len() < raw.len()
                    && rendered
                        .chars()
                        .last()
                        .is_some_and(|c| "-\u{ad}\u{2010}\u{2011}\u{2043}".contains(c))
                {
                    rendered.pop();
                }
                raw = adjusted;
                sep
            } else {
                " "
            };
            rendered.push_str(separator);
            raw.push_str(separator);
        }
        rendered.push_str(&lr);
        raw.push_str(&lt);
        previous_tail_text = previous.unwrap().0["type"] == "text";
        budget.scan(lr.len() + lt.len())?;
    }
    Ok(trim(&rendered).to_owned())
}
fn level(block: &Value) -> Result<usize, Error> {
    let n = match block.get("level") {
        None => 1,
        Some(v) => v.as_i64().ok_or_else(invalid)?,
    };
    Ok(n.clamp(0, 4) as usize)
}
fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, Error> {
    match value.get(key) {
        None => Ok(""),
        Some(Value::String(s)) => Ok(s),
        _ => Err(invalid()),
    }
}
fn seal(s: &Value) -> Result<String, Error> {
    match s.get("content") {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(s)) => Ok(trim(s).into()),
        Some(Value::Array(a)) => Ok(a
            .iter()
            .map(|v| v.as_str().ok_or_else(invalid))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|s| !trim(s).is_empty())
            .collect::<Vec<_>>()
            .join(" ")),
        _ => Err(invalid()),
    }
}
fn spans(block: &Value) -> impl Iterator<Item = &Value> {
    block["lines"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|l| l["spans"].as_array().unwrap().iter())
}
fn fallback_prose(html: &str) -> String {
    html.replace("<p>", "")
        .replace("</p>", "\n")
        .replace('\u{2003}', "  ")
        .trim()
        .to_owned()
}
fn fallback_markdown(html: &str) -> String {
    let text = fallback_prose(html);
    if text.contains('<') || text.contains('$') || text.contains("\\(") || text.contains("\\[") {
        return text;
    }
    let text = crate::processing::inline_output::unescape_html(&text);
    if text.is_empty() {
        return text;
    }
    let mut longest = 2usize;
    let mut run = 0usize;
    for c in text.chars() {
        if c == '`' {
            run += 1;
            longest = longest.max(run);
        } else {
            run = 0;
        }
    }
    let fence = "`".repeat(longest + 1);
    format!("{fence}text\n{text}\n{fence}")
}
fn markdown(b: &Value, mode: &str, prefix: &str, budget: &mut Budget) -> Result<String, Error> {
    let kind = b["type"].as_str().ok_or_else(invalid)?;
    let mut text = String::new();
    match kind {
        "text" | "list" | "index" => text = merge(b, budget)?,
        "title" => {
            text = format!("{} {}", "#".repeat(level(b)?), merge(b, budget)?)
                .replace("-\n", "")
                .replace('\n', " ")
        }
        "interline_equation" => {
            let Some(s) = b["lines"]
                .as_array()
                .unwrap()
                .first()
                .and_then(|l| l["spans"].as_array().unwrap().first())
            else {
                return Ok(text);
            };
            if truth(&s["content"]) {
                text = merge(b, budget)?;
            } else {
                text = format!("![]({prefix}/{})", string(s, "image_path")?);
            }
        }
        "image" | "table" => {
            let recovered_text = kind == "table"
                && b["blocks"].as_array().is_some_and(|children| {
                    children
                        .iter()
                        .any(|c| spans(c).any(|s| s["text_fallback"] == true))
                });
            if mode == "nlp_markdown" && !recovered_text {
                return Ok(text);
            }
            let children = b["blocks"].as_array().ok_or_else(invalid)?;
            if kind == "image" {
                let footnote = children.iter().any(|b| b["type"] == "image_footnote");
                if footnote {
                    for c in children.iter().filter(|c| c["type"] == "image_caption") {
                        text.push_str(&merge(c, budget)?);
                        text.push_str("  \n");
                    }
                }
                for c in children.iter().filter(|c| c["type"] == "image_body") {
                    for s in spans(c) {
                        if s["type"] == "image" && !string(s, "image_path")?.is_empty() {
                            text.push_str(&format!("![]({prefix}/{})", string(s, "image_path")?));
                        }
                        if !footnote && s["original_label"] == "seal" {
                            let seal = seal(s)?;
                            if !seal.is_empty() {
                                text.push_str("  \n");
                                text.push_str(&seal);
                            }
                        }
                    }
                }
                let caption = if footnote {
                    "image_footnote"
                } else {
                    "image_caption"
                };
                for c in children.iter().filter(|c| c["type"] == caption) {
                    text.push_str("  \n");
                    text.push_str(&merge(c, budget)?);
                }
            } else {
                for c in children.iter().filter(|c| c["type"] == "table_caption") {
                    text.push_str(&merge(c, budget)?);
                    text.push_str("  \n");
                }
                for c in children.iter().filter(|c| c["type"] == "table_body") {
                    for s in spans(c).filter(|s| s["type"] == "table") {
                        let html = string(s, "html")?;
                        let path = string(s, "image_path")?;
                        if !html.is_empty() {
                            if s["text_fallback"] == true {
                                text.push_str(&fallback_markdown(&html));
                                text.push('\n');
                            } else {
                                text.push_str(&format!("\n{html}\n"));
                            }
                        } else if !path.is_empty() {
                            text.push_str(&format!("![]({prefix}/{path})"));
                        }
                    }
                }
                for c in children.iter().filter(|c| c["type"] == "table_footnote") {
                    text.push('\n');
                    text.push_str(&merge(c, budget)?);
                    text.push_str("  ");
                }
            }
        }
        _ => {}
    }
    Ok(trim(&text).to_owned())
}
fn content_list(
    b: &Value,
    prefix: &str,
    page: &Value,
    budget: &mut Budget,
) -> Result<Option<Value>, Error> {
    let kind = b["type"].as_str().ok_or_else(invalid)?;
    let mut out = json!({});
    match kind {
        "text" | "list" | "index" | "title" | "discarded" => {
            out = json!({"type":if kind=="discarded"{"discarded"}else{"text"},"text":merge(b,budget)?});
            if kind == "title" {
                let n = level(b)?;
                if n != 0 {
                    out["text_level"] = json!(n);
                }
            }
        }
        "interline_equation" => {
            let Some(s) = b["lines"]
                .as_array()
                .unwrap()
                .first()
                .and_then(|l| l["spans"].as_array().unwrap().first())
            else {
                return Ok(None);
            };
            out = json!({"type":"equation","img_path":format!("{prefix}/{}",string(s,"image_path")?)});
            if truth(&s["content"]) {
                out["text"] = json!(merge(b, budget)?);
                out["text_format"] = json!("latex");
            }
        }
        "image" | "table" => {
            let cap = format!("{kind}_caption");
            let foot = format!("{kind}_footnote");
            let body = format!("{kind}_body");
            out = json!({"type":kind,"img_path":""});
            out[&cap] = json!([]);
            out[&foot] = json!([]);
            for c in b["blocks"].as_array().ok_or_else(invalid)? {
                if c["type"] == body {
                    for s in spans(c) {
                        if kind == "image" && s["original_label"] == "seal" {
                            out["text"] = json!(seal(s)?);
                        }
                        if s["type"] == kind {
                            if kind == "table" && !string(s, "html")?.is_empty() {
                                if s["text_fallback"] == true {
                                    out["type"] = json!("text");
                                    out["text"] =
                                        json!(crate::processing::inline_output::unescape_html(
                                            &fallback_prose(&string(s, "html")?)
                                        )
                                        .trim());
                                } else {
                                    out[&body] = s["html"].clone();
                                }
                            }
                            if !string(s, "image_path")?.is_empty() {
                                out["img_path"] =
                                    json!(format!("{prefix}/{}", string(s, "image_path")?));
                            }
                        }
                    }
                }
                if c["type"] == cap {
                    out[&cap]
                        .as_array_mut()
                        .unwrap()
                        .push(json!(merge(c, budget)?));
                }
                if c["type"] == foot {
                    out[&foot]
                        .as_array_mut()
                        .unwrap()
                        .push(json!(merge(c, budget)?));
                }
            }
        }
        _ => {}
    }
    if truth(&b["bbox"]) {
        let box_ = bb(&b["bbox"]);
        let size = page["page_size"].as_array().unwrap();
        out["bbox"] = json!([
            (box_[0] * 1000. / num(&size[0])) as i64,
            (box_[1] * 1000. / num(&size[1])) as i64,
            (box_[2] * 1000. / num(&size[0])) as i64,
            (box_[3] * 1000. / num(&size[1])) as i64
        ]);
    }
    out["page_idx"] = page["page_idx"].clone();
    Ok(Some(out))
}
fn validate(request: &Value) -> Result<(), Error> {
    let policy = super::budget::current();
    let max_nodes = if policy.host {
        policy.nodes
    } else {
        rapiddoc_core::resources::limit(200_000)
    };
    let max_bytes = if policy.host {
        policy.bytes
    } else {
        rapiddoc_core::resources::limit(2_000_000)
    };
    let mut stack = vec![(request, 0usize)];
    let (mut nodes, mut bytes) = (0, 0);
    while let Some((v, depth)) = stack.pop() {
        nodes += 1;
        if nodes > max_nodes || depth > rapiddoc_core::resources::limit(64) {
            return Err(limit());
        }
        match v {
            Value::Array(a) => {
                if nodes + stack.len() + a.len() > max_nodes {
                    return Err(limit());
                }
                stack.extend(a.iter().map(|v| (v, depth + 1)));
            }
            Value::Object(o) => {
                bytes += o.keys().map(String::len).sum::<usize>();
                for key in [
                    "lines",
                    "spans",
                    "blocks",
                    "para_blocks",
                    "discarded_blocks",
                ] {
                    if let Some(a) = o.get(key) {
                        for item in a.as_array().ok_or_else(invalid)? {
                            if !item.is_object() {
                                return Err(invalid());
                            }
                            if key == "lines" {
                                if !item["spans"].is_array() {
                                    return Err(invalid());
                                }
                            } else if !item["type"].is_string() {
                                return Err(invalid());
                            } else if key != "spans" {
                                let kind = item["type"].as_str().unwrap();
                                if matches!(kind, "image" | "table") {
                                    if !item["blocks"].is_array() {
                                        return Err(invalid());
                                    }
                                } else if matches!(
                                    kind,
                                    "text"
                                        | "list"
                                        | "index"
                                        | "title"
                                        | "discarded"
                                        | "interline_equation"
                                        | "image_body"
                                        | "image_caption"
                                        | "image_footnote"
                                        | "table_body"
                                        | "table_caption"
                                        | "table_footnote"
                                ) && !item["lines"].is_array()
                                {
                                    return Err(invalid());
                                }
                            }
                        }
                    }
                }
                if let Some(b) = o.get("bbox") {
                    if b.as_array().is_none_or(|a| !a.is_empty()) {
                        coord(b, false)?;
                    }
                }
                stack.extend(o.values().map(|v| (v, depth + 1)));
            }
            Value::String(s) => bytes += s.len(),
            _ => {}
        }
        if bytes > max_bytes {
            return Err(limit());
        }
    }
    Ok(())
}

/// 默认 $/$$ 分隔符；返回 Markdown 字符串或旧 content_list JSON。
pub fn render(request: &Value) -> Result<Value, Error> {
    validate(request)?;
    let mode = request["make_mode"].as_str().ok_or_else(invalid)?;
    if !matches!(mode, "mm_markdown" | "nlp_markdown" | "content_list") {
        return Err(unsupported("尚未迁移此输出模式"));
    }
    let prefix = request["image_path_prefix"].as_str().ok_or_else(invalid)?;
    let pages = request["pages"].as_array().ok_or_else(invalid)?;
    if pages.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let mut markdown_parts = Vec::new();
    let mut content_parts = Vec::new();
    let mut budget = Budget(0, request["magika_runtime"].as_str().map(PathBuf::from));
    for page in pages {
        let size = page["page_size"]
            .as_array()
            .filter(|s| s.len() == 2)
            .ok_or_else(invalid)?;
        for v in size {
            if !v.as_f64().is_some_and(|n| n > 0. && n <= 1_000_000.) {
                return Err(invalid());
            }
        }
        if page["page_idx"].as_u64().is_none() {
            return Err(invalid());
        }
        let blocks = page["para_blocks"].as_array().ok_or_else(invalid)?;
        // 旧 union_make 即使有 discarded_blocks，也跳过 para_blocks 为空的页。
        if blocks.is_empty() {
            continue;
        }
        for b in blocks {
            if mode == "content_list" {
                if let Some(item) = content_list(b, prefix, page, &mut budget)? {
                    content_parts.push(item);
                }
            } else {
                let text = markdown(b, mode, prefix, &mut budget)?;
                if !text.is_empty() {
                    markdown_parts.push(text);
                }
            }
        }
        if mode == "content_list" {
            if let Some(discarded) = page.get("discarded_blocks") {
                for b in discarded.as_array().ok_or_else(invalid)? {
                    if let Some(item) = content_list(b, prefix, page, &mut budget)? {
                        content_parts.push(item);
                    }
                }
            }
        }
    }
    Ok(if mode == "content_list" {
        Value::Array(content_parts)
    } else {
        json!(markdown_parts.join("\n\n"))
    })
}

#[cfg(test)]
mod scope_tests {
    use super::*;
    #[test]
    fn recovered_table_text_is_visible_in_all_outputs() {
        let block = json!({"type":"table","bbox":[0,0,100,100],"blocks":[{"type":"table_body","lines":[{"spans":[{"type":"table","html":"<p>A &amp; B</p>\n<p>$x$</p>\n","text_fallback":true}]}]}]});
        let pages = json!([{"page_idx":0,"page_size":[100,100],"para_blocks":[block]}]);
        for mode in ["mm_markdown", "nlp_markdown"] {
            let text =
                render(&json!({"pages":pages,"make_mode":mode,"image_path_prefix":"images"}))
                    .unwrap();
            assert!(text.as_str().unwrap().contains("A &amp; B"));
            assert!(text.as_str().unwrap().contains("$x$"));
            assert!(!text.as_str().unwrap().contains("<table"));
        }
        let content =
            render(&json!({"pages":pages,"make_mode":"content_list","image_path_prefix":"images"}))
                .unwrap();
        assert_eq!(content[0]["type"], "text");
        assert!(content[0]["text"].as_str().unwrap().contains("A & B"));
    }
    #[test]
    fn host_render_retains_explicit_capacity_and_restores_diagnostics() {
        let q = json!({"pages":[],"make_mode":"mm_markdown","image_path_prefix":""});
        assert_eq!(render(&q).unwrap(), json!(""));
        super::super::budget::with_resources(
            super::super::budget::HostResources {
                max_json_nodes: Some(1),
                ..Default::default()
            },
            || {
                super::super::with_host_page_budget(|| assert!(render(&q).is_err()));
            },
        );
        assert_eq!(render(&q).unwrap(), json!(""));
    }
}
