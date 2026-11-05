//! Opt-in table text retention checks. Never rerun OCR or alter model predictions.
use crate::processing::inline_output::unescape_html;
use serde_json::Value;
use std::collections::BTreeMap;

fn counts(text: &str) -> BTreeMap<char, usize> {
    let mut out = BTreeMap::new();
    for c in text.chars().filter(|c| !c.is_whitespace()) {
        *out.entry(c).or_default() += 1;
    }
    out
}

#[derive(Debug)]
pub(super) struct Quality {
    pub valid: bool,
    pub retained: usize,
    pub total: usize,
}
impl Quality {
    pub fn needs_fallback(&self, threshold: f64) -> bool {
        self.total > 0
            && (!self.valid
                || self.retained == 0
                || (self.total - self.retained) as f64 > self.total as f64 * threshold)
    }
}

/// Validate balanced markup and a rectangular logical grid, including spans.
/// Intervals avoid allocating a huge grid for untrusted rowspan/colspan values.
fn table_text(html: &str) -> Option<String> {
    let tags = crate::cached_regex!(r"(?s)<[^>]*>");
    let attrs = crate::cached_regex!(r#"(?i)\b(rowspan|colspan)\s*=\s*["']?([0-9]+)"#);
    let mut stack = Vec::<String>::new();
    let (mut row, mut col, mut width, mut cells, mut tables) = (0u64, 0u64, None, 0, 0);
    let mut active = Vec::<(u64, u64, u64)>::new();
    let mut text = String::new();
    let mut last = 0;
    for token in tags.find_iter(html) {
        if stack.iter().any(|s| s == "td" || s == "th") {
            text.push_str(&html[last..token.start()]);
        } else if !unescape_html(&html[last..token.start()]).trim().is_empty() {
            return None;
        }
        last = token.end();
        let inner = token.as_str().trim_matches(['<', '>']).trim();
        let closing = inner.starts_with('/');
        let name = inner
            .trim_start_matches('/')
            .split_whitespace()
            .next()?
            .trim_end_matches('/')
            .to_ascii_lowercase();
        if closing {
            if stack.pop().as_deref() != Some(&name) {
                return None;
            }
            if name == "tr" {
                while let Some(end) = active
                    .iter()
                    .find(|&&(a, b, end)| end > row && a <= col && col < b)
                    .map(|v| v.1)
                {
                    col = end;
                }
                if col == 0 || width.is_some_and(|w| w != col) {
                    return None;
                }
                width = Some(col);
                row = row.checked_add(1)?;
            }
        } else {
            match name.as_str() {
                "table" => {
                    tables += 1;
                    if tables != 1 || stack.iter().any(|s| s == "table") {
                        return None;
                    }
                }
                "tr" => {
                    if !matches!(
                        stack.last().map(String::as_str),
                        Some("table" | "thead" | "tbody" | "tfoot")
                    ) {
                        return None;
                    }
                    col = 0;
                    active.retain(|v| v.2 > row);
                }
                "td" | "th" => {
                    if stack.last().map(String::as_str) != Some("tr") {
                        return None;
                    }
                    loop {
                        let next = active
                            .iter()
                            .find(|&&(a, b, _)| a <= col && col < b)
                            .map(|v| v.1);
                        if let Some(n) = next {
                            col = n;
                        } else {
                            break;
                        }
                    }
                    let (mut rs, mut cs) = (1u64, 1u64);
                    for a in attrs.captures_iter(inner) {
                        let n = a[2].parse::<u64>().ok().filter(|n| *n > 0)?;
                        if a[1].eq_ignore_ascii_case("rowspan") {
                            rs = n;
                        } else {
                            cs = n;
                        }
                    }
                    let end = col.checked_add(cs)?;
                    if active.iter().any(|&(a, b, _)| a < end && col < b) {
                        return None;
                    }
                    active.push((col, end, row.checked_add(rs)?));
                    col = end;
                    cells += 1;
                }
                "thead" | "tbody" | "tfoot" => {
                    if stack.last().map(String::as_str) != Some("table") {
                        return None;
                    }
                }
                "html" | "body" => {}
                _ => {
                    if !stack.iter().any(|s| s == "td" || s == "th") {
                        return None;
                    }
                }
            }
            if !matches!(
                name.as_str(),
                "br" | "img" | "hr" | "wbr" | "meta" | "input"
            ) {
                stack.push(name);
            }
        }
    }
    if !stack.is_empty()
        || tables != 1
        || cells == 0
        || row == 0
        || active.iter().any(|v| v.2 > row)
        || !html[last..].trim().is_empty()
    {
        return None;
    }
    Some(text)
}

pub(super) fn quality(html: &Value, rows: &Value, excluded: &[String]) -> Quality {
    let texts = rows[1].as_array().map(Vec::as_slice).unwrap_or_default();
    let mut expected = BTreeMap::<char, usize>::new();
    for (i, t) in texts.iter().enumerate() {
        let Some(t) = t.as_str() else { continue };
        if rows[2][i].as_f64().unwrap_or(0.) < 0.5 || excluded.iter().any(|e| e == t) {
            continue;
        }
        for (c, n) in counts(&unescape_html(t)) {
            *expected.entry(c).or_default() += n;
        }
    }
    let parsed = html.as_str().and_then(table_text);
    let mut ordinary = parsed.as_deref().map(unescape_html).unwrap_or_default();
    // Formula and image placeholders are not evidence for ordinary OCR retention.
    for e in excluded {
        ordinary = ordinary.replace(&unescape_html(e), "");
    }
    let actual = counts(&ordinary);
    Quality {
        valid: parsed.is_some(),
        total: expected.values().sum(),
        retained: expected
            .iter()
            .map(|(c, n)| n.min(actual.get(c).unwrap_or(&0)))
            .sum(),
    }
}

pub(super) fn excluded(hints: &Value) -> Vec<String> {
    let mut out = vec![];
    if let Some(images) = hints["fill_images"].as_array() {
        out.extend(
            images
                .iter()
                .filter_map(|i| i["uuid"].as_str().map(str::to_owned)),
        );
    }
    if let Some(formulas) = hints["formulas"].as_array() {
        for f in formulas {
            if let Some(latex) = f["latex"].as_str() {
                out.push(crate::processing::table_text::normalize(&format!(
                    "${latex}$"
                )));
            } else if let Some(checkbox) = f["checkbox"].as_str() {
                out.push(crate::processing::table_text::normalize(checkbox));
            }
        }
    }
    out
}

pub(super) fn plain_html(rows: &Value) -> String {
    let count = rows[1].as_array().map_or(0, Vec::len);
    let boxes: Vec<[f64; 4]> = (0..count)
        .map(|i| {
            let mut b = [
                f64::INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NEG_INFINITY,
            ];
            for p in rows[0][i].as_array().into_iter().flatten() {
                let x = p[0].as_f64().unwrap_or(0.);
                let y = p[1].as_f64().unwrap_or(0.);
                b[0] = b[0].min(x);
                b[1] = b[1].min(y);
                b[2] = b[2].max(x);
                b[3] = b[3].max(y);
            }
            b
        })
        .collect();
    let center = |i: usize| (boxes[i][1] + boxes[i][3]) / 2.;
    let mut order: Vec<usize> = (0..count)
        .filter(|&i| rows[1][i].as_str().is_some_and(|s| !s.is_empty()))
        .collect();
    order.sort_by(|&a, &b| {
        center(a)
            .total_cmp(&center(b))
            .then_with(|| boxes[a][0].total_cmp(&boxes[b][0]))
            .then(a.cmp(&b))
    });
    // Keep a fixed seed: neighboring overlaps must not chain across rows.
    let mut lines = Vec::<(usize, Vec<usize>)>::new();
    for i in order {
        let b = boxes[i];
        let nearest = lines
            .iter()
            .enumerate()
            .filter_map(|(k, (seed, _))| {
                let a = boxes[*seed];
                let height = (a[3] - a[1]).min(b[3] - b[1]);
                let distance = (center(i) - center(*seed)).abs();
                let overlap = a[3].min(b[3]) - a[1].max(b[1]);
                (height > 0. && distance <= height * 0.5 && overlap >= height * 0.5)
                    .then_some((distance / height, k))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        if let Some((_, k)) = nearest {
            lines[k].1.push(i);
        } else {
            lines.push((i, vec![i]));
        }
    }
    let mut result = String::new();
    for (_, mut indices) in lines {
        indices.sort_by(|&a, &b| {
            boxes[a][0]
                .total_cmp(&boxes[b][0])
                .then_with(|| center(a).total_cmp(&center(b)))
                .then(a.cmp(&b))
        });
        let texts: Vec<&str> = indices
            .iter()
            .filter_map(|&i| rows[1][i].as_str())
            .collect();
        result.push_str("<p>");
        result.push_str(&texts.join("  "));
        result.push_str("</p>\n");
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn prose_groups_printed_rows_without_chaining() {
        let q = |x: i32, y: i32, h: i32| json!([[x, y], [x + 20, y], [x + 20, y + h], [x, y + h]]);
        let rows = json!([
            [
                q(100, 9, 12),
                q(0, 11, 12),
                q(70, 12, 12),
                q(0, 32, 12),
                q(100, 30, 12),
                q(70, 31, 12)
            ],
            ["35", "Freecell (80)", "71", "GED (20)", "20", "10"],
            [1, 1, 1, 1, 1, 1]
        ]);
        assert_eq!(
            plain_html(&rows),
            "<p>Freecell (80)  71  35</p>\n<p>GED (20)  10  20</p>\n"
        );
        let rows = json!([
            [q(0, 0, 10), q(20, 4, 10), q(40, 8, 10)],
            ["A", "B", "C"],
            [1, 1, 1]
        ]);
        assert_eq!(plain_html(&rows), "<p>A  B</p>\n<p>C</p>\n");
    }
    #[test]
    fn retention_and_grid_guards() {
        let rows = json!([[], ["A &amp; B", "甲甲乙", "noise"], [1., 0.9, 0.1]]);
        let ok=json!("<table><tr><td>A &amp; B</td><td rowspan='2'>甲甲乙</td></tr><tr><td></td></tr></table>");
        let q = quality(&ok, &rows, &[]);
        assert!(q.valid);
        assert_eq!(q.total, q.retained);
        assert!(!q.needs_fallback(0.2));
        let q = quality(
            &json!("<table><tr><td>A &amp; B甲乙</td></tr></table>"),
            &rows,
            &[],
        );
        assert_eq!((q.total, q.retained), (6, 5));
        assert!(!q.needs_fallback(0.2));
        assert!(q.needs_fallback(0.1));
        for bad in [
            "<table></table>",
            "<table><tr><td>A</tr></table>",
            "<table><tr><td rowspan='3'>A</td></tr></table>",
            "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>",
        ] {
            assert!(!quality(&json!(bad), &rows, &[]).valid, "{bad}");
        }
    }
    #[test]
    fn placeholders_do_not_trigger_or_mask_losses() {
        let rows = json!([[], ["AB", "$AAAA$", "image-id"], [1, 1, 1]]);
        let excluded = vec!["$AAAA$".into(), "image-id".into()];
        let q = quality(
            &json!("<table><tr><td>$AAAA$ image-id B</td></tr></table>"),
            &rows,
            &excluded,
        );
        assert_eq!((q.total, q.retained), (2, 1));
        assert!(q.needs_fallback(0.2));
    }
}
