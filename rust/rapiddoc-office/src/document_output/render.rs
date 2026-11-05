//! Native legacy Markdown, content_list and structured content_list_v2 exports.
use super::{
    inline::{self, Delimiters},
    *,
};
use serde_json::{json, Value};
fn captions(b: &Block, kind: &str, d: &Delimiters) -> Vec<String> {
    b.children()
        .iter()
        .filter(|b| b.kind == kind)
        .map(|b| inline::merge(b, d, true))
        .collect()
}
fn body(b: &Block) -> (&str, &str) {
    for s in b.spans() {
        match s.kind.as_str() {
            "table" => {
                return (
                    s.image_path.as_deref().unwrap_or(""),
                    s.html.as_deref().unwrap_or(""),
                )
            }
            "chart" | "interline_equation" => {
                return (s.image_path.as_deref().unwrap_or(""), inline::content(s))
            }
            "image" => return (s.image_path.as_deref().unwrap_or(""), ""),
            _ => (),
        }
    }
    for c in b.children() {
        if matches!(c.kind.as_str(), "image_body" | "table_body" | "chart_body") {
            let data = body(c);
            if data != ("", "") || c.kind == "chart_body" {
                return data;
            }
        }
    }
    ("", "")
}
fn media_path(prefix: &str, path: &str) -> String {
    if path.is_empty() {
        String::new()
    } else if prefix.is_empty() {
        path.into()
    } else {
        format!("{prefix}/{path}")
    }
}
fn embedded(text: &str, prefix: &str, d: &Delimiters) -> String {
    html::format_embedded(text, prefix, &d.inline_left, &d.inline_right)
}
fn list(b: &Block, root: i64, d: &Delimiters, v2: bool) -> Vec<Value> {
    let mut out = vec![];
    let level = (b.ilevel.unwrap_or(0) - root).max(0);
    let indent = "    ".repeat(level as usize);
    let ordered = b.attribute.as_deref() == Some("ordered");
    let mut number = b
        .start
        .as_ref()
        .map(|n| n.0.clone())
        .unwrap_or_else(|| BigInt::from(1));
    for c in b.children() {
        if matches!(c.kind.as_str(), "list" | "index") {
            out.extend(list(c, root, d, v2));
        } else {
            let content = if v2 {
                spans_v2(c)
            } else {
                json!(inline::merge(c, d, false))
            };
            let nonempty = if v2 {
                content.as_array().is_some_and(|v| !v.is_empty())
            } else {
                !content.as_str().unwrap().trim().is_empty()
            };
            if nonempty {
                let prefix = if ordered {
                    let value = if let Some(label) = c
                        .prefix
                        .as_ref()
                        .map(|s| s.trim_matches(crate::docx::space))
                        .filter(|s| !s.is_empty())
                    {
                        format!("{indent}{label}")
                    } else {
                        format!("{indent}{number}.")
                    };
                    number += 1;
                    value
                } else {
                    format!("{indent}-")
                };
                if v2 {
                    let mut item = json!({"item_type":"text","ilevel":level,"prefix":prefix,"item_content":content});
                    if let Some(a) = &c.anchor {
                        item["anchor"] = json!(a);
                    }
                    out.push(item);
                } else {
                    out.push(json!(format!("{prefix} {}", content.as_str().unwrap())));
                }
            }
        }
    }
    out
}
fn index(b: &Block, d: &Delimiters) -> Vec<String> {
    let mut out = vec![];
    let indent = "    ".repeat(b.ilevel.unwrap_or(0).max(0) as usize);
    for c in b.children() {
        if c.kind == "index" {
            out.extend(index(c, d));
        } else if c.kind == "text" {
            if let Some(s) = inline::index_leaf(c, &indent, d) {
                out.push(s);
            }
        }
    }
    out
}
fn spans_v2(b: &Block) -> Value {
    fn convert(s: &Span) -> Value {
        let mut out = serde_json::to_value(s).expect("typed output span");
        if s.kind == "inline_equation" {
            out["type"] = json!("equation_inline");
        }
        if let Some(children) = &s.children {
            out["children"] = Value::Array(children.iter().map(convert).collect());
        }
        out
    }
    fn keep(s: &Span) -> bool {
        let text = inline::content(s);
        !text.trim_matches(crate::docx::space).is_empty()
            || !text.is_empty()
                && s.style
                    .iter()
                    .any(|s| matches!(s.as_str(), "underline" | "strikethrough"))
            || s.children.iter().flatten().any(keep)
    }
    let mut out = vec![];
    if b.kind == "title" {
        if let Some(s) = &b.section_number {
            out.push(json!({"type":"text","content":format!("{s} ")}));
        }
    }
    out.extend(b.spans().filter(|s| keep(s)).map(convert));
    Value::Array(out)
}
fn captions_v2(b: &Block, kind: &str) -> Value {
    Value::Array(
        b.children()
            .iter()
            .filter(|b| b.kind == kind)
            .flat_map(|b| spans_v2(b).as_array().unwrap().clone())
            .collect(),
    )
}
fn code_text(b: &Block) -> String {
    b.lines
        .iter()
        .flatten()
        .map(|l| l.spans.iter().map(inline::content).collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}
fn content_list(b: &Block, prefix: &str, page: usize, d: &Delimiters) -> Value {
    let mut out = match b.kind.as_str() {
        "text" | "header" | "footer" | "page_footnote" => {
            json!({"type":b.kind,"text":inline::merge(b,d,true)})
        }
        "code" => json!({"type":"code","text":code_text(b),"text_format":"code"}),
        "title" => {
            let mut v = json!({"type":"text","text":inline::merge(b,d,true)});
            let level = b.level.unwrap_or(2);
            if level != 0 {
                v["text_level"] = json!(level);
            }
            v
        }
        "interline_equation" => {
            json!({"type":"equation","text":inline::merge(b,d,true),"text_format":"latex"})
        }
        "list" => json!({"type":"list","list_items":list(b,b.ilevel.unwrap_or(0),d,false)}),
        "index" => json!({"type":"index","list_items":index(b,d)}),
        "image" => {
            let (path, _) = body(b);
            json!({"type":"image","img_path":if path.is_empty(){String::new()}else{format!("{prefix}/{path}")},"image_caption":captions(b,"image_caption",d)})
        }
        "table" => {
            let (_, text) = body(b);
            let mut v = json!({"type":"table","table_caption":captions(b,"table_caption",d)});
            if !text.is_empty() {
                v["table_body"] = json!(embedded(text, prefix, d));
            }
            v
        }
        "chart" => {
            let (path, text) = body(b);
            json!({"type":"chart","img_path":media_path(prefix,path),"content":embedded(text,prefix,d),"chart_caption":captions(b,"chart_caption",d)})
        }
        _ => json!({}),
    };
    out["page_idx"] = json!(page);
    if let Some(a) = &b.anchor {
        out["anchor"] = json!(a);
    }
    out
}
fn content_v2(b: &Block, prefix: &str, d: &Delimiters) -> Value {
    let (path, text) = body(b);
    let mut out = match b.kind.as_str() {
        "text" => json!({"type":"paragraph","content":{"paragraph_content":spans_v2(b)}}),
        "code" => json!({"type":"code","content":{"code_content":code_text(b)}}),
        "header" | "footer" | "page_footnote" => {
            let kind = match b.kind.as_str() {
                "header" => "page_header",
                "footer" => "page_footer",
                _ => "page_footnote",
            };
            json!({"type":kind,"content":{format!("{kind}_content"):spans_v2(b)}})
        }
        "title" => {
            let level = b.level.unwrap_or(2);
            if level == 0 {
                json!({"type":"paragraph","content":{"paragraph_content":spans_v2(b)}})
            } else {
                json!({"type":"title","content":{"title_content":spans_v2(b),"level":level}})
            }
        }
        "interline_equation" => {
            json!({"type":"equation_interline","content":{"math_content":text,"math_type":"latex"}})
        }
        "image" => {
            json!({"type":"image","content":{"image_source":{"path":format!("{prefix}/{path}")},"image_caption":captions_v2(b,"image_caption")}})
        }
        "table" => {
            let nested = if text.matches("<table").count() > 1 {
                2
            } else {
                1
            };
            let kind = if nested > 1 || text.contains("colspan") || text.contains("rowspan") {
                "complex_table"
            } else {
                "simple_table"
            };
            json!({"type":"table","content":{"table_caption":captions_v2(b,"table_caption"),"html":embedded(text,prefix,d),"table_type":kind,"table_nest_level":nested}})
        }
        "chart" => {
            json!({"type":"chart","content":{"image_source":{"path":media_path(prefix,path)},"content":embedded(text,prefix,d),"chart_caption":captions_v2(b,"chart_caption")}})
        }
        "list" => {
            json!({"type":"list","content":{"list_type":"text_list","attribute":b.attribute.as_deref().unwrap_or("unordered"),"list_items":list(b,b.ilevel.unwrap_or(0),d,true)}})
        }
        "index" => {
            json!({"type":"index","content":{"list_type":"text_list","list_items":list(b,b.ilevel.unwrap_or(0),d,true)}})
        }
        _ => json!({}),
    };
    if let Some(a) = &b.anchor {
        out["anchor"] = json!(a);
    }
    out
}
pub fn export(middle: &Middle, mode: &str, prefix: &str, d: &Delimiters) -> Value {
    let mut output = vec![];
    let mut markdown = vec![];
    for page in &middle.pdf_info {
        match mode {
            "content_list" => {
                for b in page.para_blocks.iter().chain(&page.discarded_blocks) {
                    output.push(content_list(b, prefix, page.page_idx, d));
                }
            }
            "content_list_v2" => output.push(Value::Array(
                page.para_blocks
                    .iter()
                    .chain(&page.discarded_blocks)
                    .map(|b| content_v2(b, prefix, d))
                    .collect(),
            )),
            "mm_markdown" | "nlp_markdown" => {
                for b in &page.para_blocks {
                    let s = match b.kind.as_str() {
                        "text" | "interline_equation" => {
                            let text = inline::merge(b, d, true);
                            if b.kind == "text"
                                && b.anchor.as_ref().is_some_and(|a| a.starts_with("_Toc"))
                            {
                                format!("<a id=\"{}\"></a>\n{text}", b.anchor.as_ref().unwrap())
                            } else {
                                text
                            }
                        }
                        "title" => {
                            let text = format!(
                                "{} {}",
                                "#".repeat(b.level.unwrap_or(2).max(0) as usize),
                                inline::merge(b, d, true)
                            );
                            if let Some(a) = &b.anchor {
                                format!("<a id=\"{a}\"></a>\n{text}")
                            } else {
                                text
                            }
                        }
                        "code" => {
                            let text = code_text(b);
                            let longest = text.split(|c| c != '`').map(str::len).max().unwrap_or(0);
                            let fence = "`".repeat(3.max(longest + 1));
                            format!("{fence}\n{text}\n{fence}")
                        }
                        "list" => {
                            list(b, b.ilevel.unwrap_or(0), d, false)
                                .iter()
                                .map(|v| v.as_str().unwrap())
                                .collect::<Vec<_>>()
                                .join("\n")
                                + "\n"
                        }
                        "index" => index(b, d).join("\n") + "\n",
                        "image" | "table" | "chart" if mode == "mm_markdown" => {
                            let (path, text) = body(b);
                            let mut text = match b.kind.as_str() {
                                "image" => {
                                    if path.is_empty() {
                                        String::new()
                                    } else {
                                        format!("![]({prefix}/{path})")
                                    }
                                }
                                "table" => format!("\n{}\n", embedded(text, prefix, d)),
                                _ => {
                                    if !text.is_empty() {
                                        format!("\n{}\n", embedded(text, prefix, d))
                                    } else if !path.is_empty() {
                                        format!("![]({})", media_path(prefix, path))
                                    } else {
                                        continue;
                                    }
                                }
                            };
                            for caption in captions(b, &format!("{}_caption", b.kind), d) {
                                text.push_str("  \n");
                                text.push_str(&caption);
                            }
                            text
                        }
                        _ => String::new(),
                    };
                    if !s.trim_matches(crate::docx::space).is_empty() {
                        markdown.push(s.trim_matches(['\r', '\n']).to_owned());
                    }
                }
            }
            _ => return Value::Null,
        }
    }
    if matches!(mode, "mm_markdown" | "nlp_markdown") {
        json!(markdown.join("\n\n"))
    } else {
        Value::Array(output)
    }
}
