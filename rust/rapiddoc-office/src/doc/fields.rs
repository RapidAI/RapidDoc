//! Word field stack: instructions are hidden, cached results survive nesting.
use crate::docx::spans::{self, Span};
#[derive(Default)]
struct Frame {
    instruction: String,
    result: bool,
    transparent: bool,
    spans: Vec<Span>,
    native_formula: bool,
    native_chart: Option<String>,
    chart_emitted: bool,
}
#[derive(Default)]
pub struct Fields {
    frames: Vec<Frame>,
}
impl Fields {
    pub fn keyword(&self) -> Option<String> {
        tokens(&self.frames.last()?.instruction)
            .first()
            .map(|s| unquote(s).to_uppercase())
    }
    pub fn toc_active(&self) -> bool {
        self.frames.iter().any(|f| f.transparent)
    }
    pub fn native_formula(&self) -> bool {
        self.frames.last().is_some_and(|f| f.native_formula)
    }
    pub fn bind_object(&mut self, out: &mut Vec<Span>, formula: Option<&str>, chart: Option<&str>) {
        let Some(frame) = self.frames.last_mut() else {
            return;
        };
        if chart.is_some() && is_chart_embed(&frame.instruction) {
            frame.native_chart = chart.map(str::to_owned);
        }
        if let Some(latex) = formula {
            frame.native_formula = true;
            self.push(
                out,
                Span::Equation {
                    content: latex.to_owned(),
                },
            );
        }
    }
    pub fn has_chart(&self) -> bool {
        self.frames.iter().any(|f| f.native_chart.is_some())
    }
    pub fn chart_picture(
        &mut self,
        preview: Option<std::sync::Arc<str>>,
    ) -> Option<super::resources::Visual> {
        let frame = self
            .frames
            .iter_mut()
            .rev()
            .find(|f| f.native_chart.is_some())?;
        frame.chart_emitted = true;
        Some(super::resources::Visual::Chart {
            html: frame.native_chart.clone().unwrap(),
            preview,
        })
    }
    pub fn in_instruction(&self) -> bool {
        self.frames.last().is_some_and(|f| !f.result)
    }
    pub fn begin(&mut self) {
        self.frames.push(Frame::default());
    }
    pub fn separator(&mut self) {
        if let Some(f) = self.frames.last_mut() {
            f.result = true;
            f.transparent = tokens(&f.instruction)
                .first()
                .is_some_and(|s| unquote(s).eq_ignore_ascii_case("TOC"));
        }
    }
    pub fn push(&mut self, out: &mut Vec<Span>, span: Span) {
        if let Some(f) = self.frames.last_mut() {
            if !f.result {
                f.instruction.push_str(&spans::visible(&[span]));
                return;
            }
            if !f.transparent {
                spans::append(&mut f.spans, span);
                return;
            }
        }
        spans::append(out, span);
    }
    pub fn end(&mut self, out: &mut Vec<Span>) {
        self.end_with_visual(out);
    }
    pub fn end_with_visual(&mut self, out: &mut Vec<Span>) -> Option<super::resources::Visual> {
        let f = self.frames.pop()?;
        if f.transparent {
            return None;
        }
        if let Some(html) = f.native_chart {
            return (!f.chart_emitted).then_some(super::resources::Visual::Chart {
                html,
                preview: None,
            });
        }
        if let Some(url) = hyperlink_target(&f.instruction) {
            if !f.spans.is_empty() {
                self.push(
                    out,
                    Span::Link {
                        url,
                        content: f.spans,
                    },
                );
            }
        } else {
            for s in f.spans {
                self.push(out, s);
            }
        }
        None
    }
}
pub fn is_chart_embed(instruction: &str) -> bool {
    let tokens = tokens(instruction);
    tokens
        .first()
        .is_some_and(|s| unquote(s).eq_ignore_ascii_case("EMBED"))
        && tokens.get(1).is_some_and(|s| {
            let id = unquote(s).to_ascii_lowercase();
            id.starts_with("excel.chart") || id.starts_with("msgraph.chart")
        })
}
fn unquote(s: &str) -> String {
    s.strip_prefix('"')
        .and_then(|v| v.strip_suffix('"'))
        .unwrap_or(s)
        .replace("\\\"", "\"")
        .replace("\\\\", "\\")
}
fn tokens(instruction: &str) -> Vec<&str> {
    static TOKENS: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    TOKENS
        .get_or_init(|| regex::Regex::new(r#""(?:\\.|[^"\\])*"|\\\S|\S+"#).unwrap())
        .find_iter(instruction.trim())
        .map(|m| m.as_str())
        .collect()
}
pub fn hyperlink_target(instruction: &str) -> Option<String> {
    let tokens = tokens(instruction);
    if !unquote(tokens.first()?).eq_ignore_ascii_case("HYPERLINK") {
        return None;
    }
    let (mut url, mut anchor) = (None, None);
    let mut i = 1;
    while i < tokens.len() {
        if let Some(switch) = tokens[i].strip_prefix('\\') {
            let switch = switch.to_ascii_lowercase();
            if matches!(switch.as_str(), "l" | "o" | "t")
                && i + 1 < tokens.len()
                && !tokens[i + 1].starts_with('\\')
            {
                i += 1;
                let value = unquote(tokens[i]).trim().to_owned();
                if switch == "l" && !value.is_empty() {
                    anchor = Some(value);
                }
            }
        } else if url.is_none() {
            url = Some(unquote(tokens[i]).trim().to_owned());
        }
        i += 1;
    }
    let value = match (url.filter(|v| !v.is_empty()), anchor) {
        (Some(url), Some(a)) => format!("{url}#{a}"),
        (Some(url), None) => url,
        (None, Some(a)) => format!("#{a}"),
        _ => return None,
    };
    crate::links::sanitize(Some(&value)).map(str::to_owned)
}
