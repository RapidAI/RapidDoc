//! 有线/无线候选选优：移植 RapidDoc select_best_table_model 的规则与优先级。
//! 输入限本 crate 输出的闭合表格 HTML；不尝试修复任意浏览器 HTML。
use crate::escape_html;
use rapiddoc_core::{Error, ErrorCode};

#[derive(Debug, PartialEq, Eq)]
pub struct Metrics {
    pub cells: usize,
    pub blank: usize,
    pub ocr_hits: usize,
}
impl Metrics {
    pub fn non_blank(&self) -> usize {
        self.cells - self.blank
    }
}
pub struct Selection {
    pub route: &'static str,
    pub reason: &'static str,
    pub wired: Metrics,
    pub wireless: Metrics,
}
fn invalid(message: &str) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: message.into(),
    }
}
fn limited(message: &str) -> Error {
    Error {
        code: ErrorCode::ResourceLimit,
        message: message.into(),
    }
}
fn whitespace(c: char) -> bool {
    c.is_whitespace() || matches!(c, '\u{001c}'..='\u{001f}')
}

fn visible(mut text: &str) -> Result<bool, Error> {
    let mut result = false;
    while let Some((_, after)) = text.split_once('&') {
        let entity = ["amp;", "lt;", "gt;", "quot;", "#39;"]
            .iter()
            .find(|e| after.starts_with(**e))
            .ok_or_else(|| invalid("候选 HTML 含非规范实体；请使用 Rust 表格输出器生成候选"))?;
        result = true;
        text = &after[entity.len()..];
    }
    Ok(result || text.chars().any(|c| !whitespace(c)))
}
fn attributes(attrs: &str) -> Result<(), Error> {
    let mut seen = [false; 2];
    for attr in attrs.split_ascii_whitespace() {
        let (key, raw) = attr
            .split_once('=')
            .ok_or_else(|| invalid("候选单元格属性无效"))?;
        let slot = match key {
            "rowspan" => 0,
            "colspan" => 1,
            _ => return Err(invalid("候选单元格含未知属性")),
        };
        if std::mem::replace(&mut seen[slot], true) {
            return Err(invalid("候选单元格属性重复"));
        }
        let raw = raw
            .strip_prefix('"')
            .and_then(|v| v.strip_suffix('"'))
            .unwrap_or(raw);
        let value = raw.parse::<usize>().ok().filter(|v| (1..=1000).contains(v));
        if value.is_none() {
            return Err(invalid("候选单元格跨度无效"));
        }
    }
    Ok(())
}

/// 只扫描规范标签、行/格跨度和转义文本，不建立 DOM，不产生外部实体或资源访问。
pub fn metrics(mut html: &str) -> Result<Metrics, Error> {
    if html.len() > rapiddoc_core::resources::limit(16_000_000) {
        return Err(limited("表格候选 HTML 超过预算"));
    }
    let mut result = Metrics {
        cells: 0,
        blank: 0,
        ocr_hits: 0,
    };
    let mut stack: Vec<String> = Vec::new();
    let mut cell: Option<bool> = None;
    let mut tables = 0;
    while !html.is_empty() {
        let (text, rest) = html.split_once('<').unwrap_or((html, ""));
        let has_text = visible(text)?;
        if let Some(v) = &mut cell {
            *v |= has_text;
        } else if has_text {
            return Err(invalid("候选 HTML 的文字必须位于单元格内"));
        }
        if rest.is_empty() {
            if html.ends_with('<') {
                return Err(invalid("候选 HTML 标签未闭合"));
            }
            break;
        }
        let (raw, tail) = rest
            .split_once('>')
            .ok_or_else(|| invalid("候选 HTML 标签未闭合"))?;
        if raw.len() > rapiddoc_core::resources::limit(128) {
            return Err(invalid("候选 HTML 标签过长"));
        }
        let tag = raw.trim_ascii().to_ascii_lowercase();
        html = tail;
        if tag == "br" || tag == "br/" || tag == "br /" {
            if cell.is_none() {
                return Err(invalid("候选换行必须位于单元格内"));
            }
            continue;
        }
        if let Some(close) = tag.strip_prefix('/') {
            if stack.pop().as_deref() != Some(close) {
                return Err(invalid("候选 HTML 标签错位"));
            }
            if (close == "td" || close == "th")
                && !cell.take().ok_or_else(|| invalid("候选单元格未开始"))?
            {
                result.blank += 1;
            }
            continue;
        }
        let (name, attrs) = tag
            .split_once(|c: char| c.is_ascii_whitespace())
            .unwrap_or((&tag, ""));
        let parent = stack.last().map(String::as_str).unwrap_or("");
        let allowed = match name {
            "html" => parent.is_empty(),
            "body" => parent == "html" || parent.is_empty(),
            "table" => matches!(parent, "" | "html" | "body") && tables == 0,
            "thead" | "tbody" | "tfoot" => parent == "table",
            "tr" => matches!(parent, "table" | "thead" | "tbody" | "tfoot"),
            "td" | "th" => parent == "tr" && cell.is_none(),
            _ => false,
        };
        if !allowed {
            return Err(invalid("候选 HTML 含未知或错位标签"));
        }
        if name == "td" || name == "th" {
            attributes(attrs)?;
            result.cells += 1;
            cell = Some(false);
        } else if !attrs.is_empty() {
            return Err(invalid("候选容器不允许属性"));
        }
        // 原实现按 <td/<th 字面前缀计数，包含 thead；保留其选优语义。
        if name == "thead" {
            result.cells += 1;
        }
        if result.cells > rapiddoc_core::resources::limit(100_000) {
            return Err(limited("候选 HTML 单元格超过预算"));
        }
        if name == "table" {
            tables += 1;
        }
        stack.push(name.to_owned());
    }
    if !stack.is_empty() || cell.is_some() {
        return Err(invalid("候选 HTML 结构未闭合"));
    }
    Ok(result)
}

/// 所有文本为未转义 OCR；内部采用与输出器一致的转义再执行旧规则的字面子串命中。
pub fn select(texts: &[String], wired_html: &str, wireless_html: &str) -> Result<Selection, Error> {
    if texts.len() > rapiddoc_core::resources::limit(1000)
        || texts.iter().map(String::len).sum::<usize>() > rapiddoc_core::resources::limit(1_000_000)
        || wired_html.len().saturating_add(wireless_html.len())
            > rapiddoc_core::resources::limit(16_000_000)
    {
        return Err(limited("表格选优输入超过预算"));
    }
    let mut wired = metrics(wired_html)?;
    let mut wireless = metrics(wireless_html)?;
    let mut remaining = rapiddoc_core::resources::limit(50_000_000usize);
    for text in texts {
        let text = escape_html(text);
        for (html, m) in [(wired_html, &mut wired), (wireless_html, &mut wireless)] {
            if !text.is_empty() && text.len() <= html.len() {
                remaining = remaining
                    .checked_sub(html.len())
                    .ok_or_else(|| limited("表格文字命中工作量超过预算"))?;
            }
            m.ocr_hits += usize::from(html.contains(&text));
        }
    }
    Ok(select_metrics(wired, wireless))
}

/// 策略与候选HTML解析分开，供旧版HTML兼容入口复用同一决策顺序。
pub fn select_metrics(wired: Metrics, wireless: Metrics) -> Selection {
    let wn = wired.non_blank();
    let ln = wireless.non_blank();
    let gap = wireless.cells as i64 - wired.cells as i64;
    let scale = (wn as f64).sqrt().round_ties_even() as usize;
    let (route, reason) = if wired.ocr_hits > wireless.ocr_hits && wn >= ln {
        ("wired", "wired_text_coverage")
    } else if ln > wn && ln + 3 >= (wn + scale * 2).max(scale * (scale + 2)) {
        ("wireless", "wireless_scale")
    } else if (0..=5).contains(&gap)
        && wired.cells <= (wireless.cells as f64 * 0.75).round_ties_even() as usize
    {
        ("wireless", "wireless_cell_gap")
    } else if gap == 0 && wired.cells <= 4 && wireless.ocr_hits >= wired.ocr_hits {
        ("wireless", "wireless_small_equal")
    } else if wired.ocr_hits as f64 <= wireless.ocr_hits as f64 * 0.6 && wireless.ocr_hits >= 10 {
        ("wireless", "wireless_text_coverage")
    } else {
        ("wired", "wired_default")
    };
    Selection {
        route,
        reason,
        wired,
        wireless,
    }
}
