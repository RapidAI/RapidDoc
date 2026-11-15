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
    pub fn can_replace(&self, primary: &Self, threshold: f64, fallback_to_text: bool) -> bool {
        // With prose recovery disabled, keep the better valid table even when
        // neither candidate passes the text-retention threshold.
        self.valid
            && self.retained > primary.retained
            && (!fallback_to_text || !self.needs_fallback(threshold))
    }

    pub fn needs_fallback(&self, threshold: f64) -> bool {
        self.total > 0
            && (!self.valid
                || self.retained == 0
                || (self.total - self.retained) as f64 > self.total as f64 * threshold)
    }
}

/// Validate balanced markup and a rectangular logical grid, including spans.
/// Intervals avoid allocating a huge grid for untrusted rowspan/colspan values.
pub(super) fn table_text(html: &str) -> Option<String> {
    table_text_with_widths(html, None)
}

fn table_text_with_widths(html: &str, mut widths: Option<&mut Vec<u64>>) -> Option<String> {
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
                if col == 0 || (widths.is_none() && width.is_some_and(|w| w != col)) {
                    return None;
                }
                if let Some(widths) = widths.as_deref_mut() {
                    widths.push(col);
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

pub(super) fn repair_local_grid(html: &str) -> Option<String> {
    let mut widths = Vec::new();
    table_text_with_widths(html, Some(&mut widths))?;
    if widths.len() < 3 {
        return None;
    }
    let mut counts = BTreeMap::new();
    for &width in &widths {
        *counts.entry(width).or_insert(0usize) += 1;
    }
    let (width, count) = counts.into_iter().max_by_key(|&(_, count)| count)?;
    if count == widths.len() || count * 10 < widths.len() * 9 {
        return None;
    }
    let rows = crate::cached_regex!(r"(?is)<tr\b[^>]*>.*?</tr\s*>");
    let cells = crate::cached_regex!(r"(?is)<t[dh]\b([^>]*)>(.*?)</t[dh]\s*>");
    let span = crate::cached_regex!(r#"(?i)\b(?:rowspan|colspan)\s*=\s*["']?([0-9]+)"#);
    let breaks = crate::cached_regex!(r"(?is)<(?:br|wbr)\s*/?>");
    let mut removals = Vec::new();
    for (row, actual) in rows.find_iter(html).zip(widths) {
        if actual == width {
            continue;
        }
        if actual != width + 1 {
            return None;
        }
        let mut empty = cells.captures_iter(row.as_str()).filter(|c| {
            span.captures_iter(&c[1]).all(|s| &s[1] == "1")
                && unescape_html(&breaks.replace_all(&c[2], ""))
                    .trim()
                    .is_empty()
        });
        let cell = empty.next()?.get(0)?;
        if empty.next().is_some() {
            return None;
        }
        removals.push(row.start() + cell.start()..row.start() + cell.end());
    }
    let mut result = html.to_owned();
    for range in removals.into_iter().rev() {
        result.replace_range(range, "");
    }
    table_text(&result)?;
    Some(result)
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
    let mut ordinary = parsed
        .as_deref()
        .map(unescape_html)
        .unwrap_or_else(|| html.as_str().map(cell_text).unwrap_or_default());
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
    fn partial_candidate_selection_respects_text_recovery_and_retention() {
        let rows = json!([[], ["ABCDE"], [1.0]]);
        let main = quality(&json!("<table><tr><td>AB</td></tr></table>"), &rows, &[]);
        assert!(main.needs_fallback(0.2));
        for (html, expected_without_text, expected_with_text) in [
            ("<table><tr><td>ABC</td></tr></table>", true, false),
            ("<table><tr><td>ABCDE</td></tr></table>", true, true),
            ("<table><tr><td>AB</td></tr></table>", false, false),
            ("<table><tr><td>A</td></tr></table>", false, false),
            ("<table><tr><td></td></tr></table>", false, false),
            ("<table><tr><td>ABCDE</tr></table>", false, false),
            (
                "<table><tr><td>ABCDE</td><td></td></tr><tr><td></td></tr></table>",
                false,
                false,
            ),
        ] {
            let backup = quality(&json!(html), &rows, &[]);
            assert_eq!(
                backup.can_replace(&main, 0.2, false),
                expected_without_text,
                "{html}"
            );
            assert_eq!(
                backup.can_replace(&main, 0.2, true),
                expected_with_text,
                "{html}"
            );
        }
    }

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
#[derive(Default)]
pub(super) struct Structure {
    pub rows: usize,
    pub empty: usize,
    pub empty_run: usize,
    pub empty_indices: Vec<usize>,
    pub numbers: BTreeMap<String, usize>,
    pub split_lines: usize,
    pub merged_values: usize,
    pub full_width_rows: usize,
    pub row_misalignment: usize,
    pub collapsed_rows: usize,
}

// HTML 的内容统计独立于网格校验，避免无效网格把可读内容计成零。
fn cell_text(html: &str) -> String {
    let mut text = String::new();
    let mut in_cell = false;
    let mut end = 0;
    for tag in crate::cached_regex!(r"(?s)<[^>]*>").find_iter(html) {
        if in_cell {
            text.push_str(&html[end..tag.start()]);
        }
        let name = tag
            .as_str()
            .trim_matches(['<', '>'])
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        match name.as_str() {
            "td" | "th" => in_cell = true,
            "/td" | "/th" | "/tr" | "/table" => in_cell = false,
            _ => {}
        }
        end = tag.end();
    }
    if in_cell {
        text.push_str(&html[end..]);
    }
    unescape_html(&text)
}

pub(super) fn structure(html: &Value, rows: &Value, excluded: &[String]) -> Structure {
    let html = html.as_str().unwrap_or_default();
    let mut info = Structure::default();
    let row_pattern = crate::cached_regex!(r"(?is)<tr\b[^>]*>");
    let cell_pattern = crate::cached_regex!(r"(?is)<t[dh]\b([^>]*)>(.*?)</t[dh]\s*>");
    let rowspan = crate::cached_regex!(r#"(?i)\browspan\s*=\s*["']?([0-9]{1,4})\b"#);
    let tags = crate::cached_regex!(r"(?s)<[^>]*>");
    let mut all_tokens = Vec::new();
    let mut cell_texts = Vec::new();
    let mut cell_rows = Vec::new();
    let mut row_spans = Vec::new();
    if let Some(texts) = rows[1].as_array() {
        for (i, t) in texts.iter().enumerate() {
            let Some(t) = t.as_str() else {
                continue;
            };
            if rows[2][i].as_f64().unwrap_or(0.) < 0.5 || excluded.iter().any(|e| e == t) {
                continue;
            }
            let key: String = unescape_html(t)
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect();
            let numeric_token = rapiddoc_inference::table_transformer::is_numeric(&key);
            if let Some(points) = rows[0][i].as_array().filter(|p| !p.is_empty()) {
                let mut b = [
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ];
                for point in points {
                    if let (Some(x), Some(y)) = (point[0].as_f64(), point[1].as_f64()) {
                        b = [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)];
                    }
                }
                all_tokens.push((b, key.clone(), numeric_token));
            }
        }
    }
    let (mut until, mut run) = (0, 0);
    // 模型可能漏写最后的 </tr>；按行起始标签切分，仍保留该行的内容证据。
    let row_starts: Vec<_> = row_pattern.find_iter(html).collect();
    for (r, row) in row_starts.iter().enumerate() {
        let row = &html[row.end()..row_starts.get(r + 1).map_or(html.len(), |m| m.start())];
        info.rows += 1;
        let mut occupied = r < until;
        let mut has_cells = false;
        let mut spans = Vec::new();
        for cell in cell_pattern.captures_iter(row) {
            has_cells = true;
            let mut text = unescape_html(
                &tags
                    .split(&cell[2])
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join(" "),
            );
            let colspan = crate::cached_regex!(r#"(?i)\bcolspan\s*=\s*["']?([0-9]{1,4})\b"#);
            spans.push(
                colspan
                    .captures(&cell[1])
                    .and_then(|c| c[1].parse::<usize>().ok())
                    .unwrap_or(1),
            );
            if !text.trim().is_empty() || cell[2].to_ascii_lowercase().contains("<img") {
                occupied = true;
                if let Some(span) = rowspan
                    .captures(&cell[1])
                    .and_then(|c| c[1].parse::<usize>().ok())
                {
                    until = until.max(r + span);
                }
            }
            for marker in excluded {
                text = text.replace(&unescape_html(marker), "");
            }
            cell_texts.push(text.clone());
            cell_rows.push(r);
        }
        if has_cells && !occupied {
            info.empty += 1;
            info.empty_indices.push(r);
            run += 1;
        } else {
            run = 0;
        }
        info.empty_run = info.empty_run.max(run);
        row_spans.push(spans);
    }
    let width = row_spans
        .iter()
        .map(|s| s.iter().sum::<usize>())
        .max()
        .unwrap_or(0);
    if width >= 3 {
        info.full_width_rows = row_spans.iter().filter(|s| s.as_slice() == [width]).count();
    }
    structure_geometry(&mut info, &all_tokens, &cell_texts, &cell_rows);
    info
}

impl Structure {
    pub fn suspicious(&self, ruled_rows: &std::collections::BTreeSet<usize>) -> bool {
        self.split_lines >= 2
            || self.merged_values + self.row_misalignment >= 2
            || self.unruled_empty_run(ruled_rows) >= 4
    }
    fn unruled_empty_run(&self, ruled_rows: &std::collections::BTreeSet<usize>) -> usize {
        let (mut longest, mut run, mut previous) = (0, 0, None);
        for &row in &self.empty_indices {
            if ruled_rows.contains(&row) {
                continue;
            }
            run = if previous == row.checked_sub(1) && previous.is_some() {
                run + 1
            } else {
                1
            };
            longest = longest.max(run);
            previous = Some(row);
        }
        longest
    }
    pub fn removes_empty_rows(
        &self,
        primary: &Self,
        ruled_rows: &std::collections::BTreeSet<usize>,
    ) -> bool {
        primary.unruled_empty_run(ruled_rows) >= 4
            && self.empty_run <= 1
            && (self.rows as f64) <= 0.75 * primary.rows as f64
    }
    pub fn improves(&self, p: &Self) -> bool {
        if p.row_misalignment >= 5
            && self.row_misalignment <= 1
            && (self.rows as f64) <= 1.1 * p.rows as f64
            && self.split_lines <= p.split_lines
            && self.merged_values <= p.merged_values
            && self.numbers == p.numbers
        {
            return true;
        }
        if self.rows > p.rows && p.collapsed_rows < self.collapsed_rows + 2 {
            return false;
        }
        if p.row_misalignment >= self.row_misalignment + 2
            && self.merged_values <= p.merged_values
            && self.full_width_rows <= p.full_width_rows + 1
        {
            return true;
        }
        self.split_lines <= p.split_lines
            && self.merged_values <= p.merged_values
            && self.row_misalignment <= p.row_misalignment
            && self.full_width_rows <= p.full_width_rows + 1
            && (p.split_lines >= self.split_lines + 2
                || p.merged_values + p.row_misalignment
                    >= self.merged_values + self.row_misalignment + 2)
    }
}
fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return 0.;
    }
    v.sort_by(f64::total_cmp);
    (v[(v.len() - 1) / 2] + v[v.len() / 2]) / 2.
}
type B = [f64; 4];
fn number_occurrences(text: &str) -> Vec<String> {
    let re = crate::cached_regex!(
        r"\([+−-]?\d+(?:,\d{3})*(?:\.\d+)?%?\)|[+−-]?\d+(?:,\d{3})*(?:\.\d+)?%?"
    );
    let mut result = Vec::new();
    let mut offset = 0;
    while let Some(m) = re.find_at(text, offset) {
        let word = |c: char| c.is_ascii_alphanumeric() || c == '_';
        let before = text[..m.start()].chars().next_back();
        let mut after = text[m.end()..].chars();
        let first = after.next();
        let valid = !before.is_some_and(|c| word(c) || c == '.' || c == ',')
            && !first.is_some_and(word)
            && !(matches!(first, Some('.' | ','))
                && after.next().is_some_and(|c| c.is_ascii_digit()));
        if valid {
            result.push(m.as_str().to_owned());
            offset = m.end();
        } else {
            // 与 Python 的边界断言一致：括号匹配失败后，内部数值仍可能独立匹配。
            offset = m.start() + text[m.start()..].chars().next().unwrap().len_utf8();
        }
    }
    result
}
fn structure_geometry(
    info: &mut Structure,
    tokens: &[(B, String, bool)],
    cells: &[String],
    cell_rows: &[usize],
) {
    let compact = |text: &str| {
        text.chars()
            .filter(|c| !c.is_whitespace())
            .collect::<String>()
    };
    let compact_cells: Vec<_> = cells.iter().map(|s| compact(s)).collect();
    let mut actual = BTreeMap::<String, usize>::new();
    let mut locations = BTreeMap::<String, Vec<usize>>::new();
    for (i, text) in cells.iter().enumerate() {
        let mut unique = std::collections::BTreeSet::new();
        for number in number_occurrences(text) {
            *actual.entry(number.clone()).or_default() += 1;
            unique.insert(number);
        }
        for number in unique {
            locations.entry(number).or_default().push(i);
        }
    }
    let mut frequencies = BTreeMap::<&str, usize>::new();
    let mut expected = BTreeMap::<String, usize>::new();
    for (_, text, numeric) in tokens {
        *frequencies.entry(text).or_default() += 1;
        if *numeric {
            *expected.entry(text.clone()).or_default() += 1;
        }
    }
    info.numbers = expected
        .into_iter()
        .filter_map(|(n, c)| {
            let count = c.min(*actual.get(&n).unwrap_or(&0));
            (count > 0).then_some((n, count))
        })
        .collect();
    let mut repeated = BTreeMap::<&str, Vec<(f64, f64, usize)>>::new();
    for (i, (b, text, numeric)) in tokens.iter().enumerate() {
        if *numeric && frequencies[text.as_str()] > 1 {
            repeated.entry(text).or_default().push((b[1], b[0], i));
        }
    }
    let mut repeated_assignment = BTreeMap::<usize, usize>::new();
    for (n, mut occurrences) in repeated {
        let slots = locations.get(n).map(Vec::as_slice).unwrap_or_default();
        if occurrences.len() != *actual.get(n).unwrap_or(&0) || occurrences.len() != slots.len() {
            continue;
        }
        occurrences.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then(a.1.total_cmp(&b.1))
                .then(a.2.cmp(&b.2))
        });
        for ((_, _, token), &cell) in occurrences.into_iter().zip(slots) {
            repeated_assignment.insert(token, cell);
        }
    }
    let mut by_cell = BTreeMap::<usize, Vec<(B, bool, usize)>>::new();
    let mut located = Vec::new();
    let mut prose = Vec::new();
    for (token, (b, text, numeric)) in tokens.iter().enumerate() {
        let length = text.chars().count();
        if length < 2 && !numeric {
            continue;
        }
        let ids: Vec<_> = if frequencies[text.as_str()] != 1 {
            repeated_assignment
                .get(&token)
                .copied()
                .into_iter()
                .collect()
        } else if *numeric {
            locations.get(text).cloned().unwrap_or_default()
        } else {
            compact_cells
                .iter()
                .enumerate()
                .filter_map(|(i, c)| c.contains(text).then_some(i))
                .collect()
        };
        let [cell] = ids.as_slice() else {
            continue;
        };
        let row = cell_rows[*cell];
        if b[2] > b[0] && b[3] > b[1] {
            located.push((*b, *cell, row));
        }
        by_cell
            .entry(*cell)
            .or_default()
            .push((*b, *numeric, length));
        if !numeric && b[2] - b[0] >= 4. * (b[3] - b[1]) {
            prose.push((*b, row));
        }
    }
    let mut split = std::collections::BTreeSet::new();
    prose.sort_by(|a, b| {
        a.0[1]
            .total_cmp(&b.0[1])
            .then(a.1.cmp(&b.1))
            .then(a.0[0].total_cmp(&b.0[0]))
    });
    for (b, row) in &prose {
        let h = b[3] - b[1];
        let preceding = prose
            .iter()
            .filter(|(p, _)| {
                p[1] < b[1]
                    && b[1] - p[3] >= -0.2 * h
                    && b[1] - p[3] <= 0.4 * h
                    && (p[0] - b[0]).abs() <= 0.5 * h
            })
            .max_by(|a, b| a.0[1].total_cmp(&b.0[1]));
        if preceding.is_some_and(|(_, r)| r != row) {
            split.insert(*row);
        }
    }
    info.split_lines = split.len();
    for mut boxes in by_cell.into_values() {
        boxes.sort_by(|a, b| a.0[1].total_cmp(&b.0[1]));
        let mut conflict = std::collections::BTreeSet::new();
        let mut separated = false;
        for (i, (a, na, la)) in boxes.iter().enumerate() {
            for (j, (b, nb, lb)) in boxes.iter().enumerate().skip(i + 1) {
                if b[1] >= a[3] {
                    break;
                }
                let h = (a[3] - a[1]).min(b[3] - b[1]);
                if h <= 0. {
                    continue;
                }
                let y = a[3].min(b[3]) - a[1].max(b[1]);
                let x = a[2].min(b[2]) - a[0].max(b[0]);
                separated |= na != nb && y >= 0.5 * h && x < -3. * h;
                if (*na && *nb && y >= 0.5 * h && x < 0.)
                    || ((*la).max(*lb) <= 5 && y >= 0.7 * h && x < -0.6 * h)
                {
                    conflict.insert(i);
                    conflict.insert(j);
                }
            }
        }
        info.merged_values += usize::from(separated).max(conflict.len().saturating_sub(1));
    }
    if located.is_empty() {
        return;
    }
    let h = median(located.iter().map(|(b, _, _)| b[3] - b[1]).collect());
    located.sort_by(|a, b| (a.0[1] + a.0[3]).total_cmp(&(b.0[1] + b.0[3])));
    let mut bands: Vec<(f64, Vec<(B, usize, usize)>)> = Vec::new();
    for item in located {
        let c = (item.0[1] + item.0[3]) / 2.;
        if bands.last().is_none_or(|b| c - b.0 > 0.38 * h) {
            bands.push((c, Vec::new()));
        }
        bands.last_mut().unwrap().1.push(item);
    }
    let mut parallel = BTreeMap::<(usize, usize), Vec<f64>>::new();
    let mut split = 0;
    for (center, mut items) in bands {
        items.sort_by(|a, b| a.0[0].total_cmp(&b.0[0]));
        let mut split_band = false;
        for (i, (a, ac, ar)) in items.iter().enumerate() {
            for (b, bc, br) in items.iter().skip(i + 1) {
                if b[0] - a[2] < 0.6 * h {
                    continue;
                }
                if ar != br {
                    split_band = true;
                } else if ac != bc {
                    parallel.entry((*ac, *bc)).or_default().push(center);
                }
            }
        }
        split += usize::from(split_band);
    }
    info.collapsed_rows = parallel
        .values()
        .map(|ys| ys.windows(2).filter(|w| w[1] - w[0] >= 1.15 * h).count())
        .max()
        .unwrap_or(0);
    info.row_misalignment = split + info.collapsed_rows;
}

#[cfg(test)]
mod structure_tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn numeric_boundaries_retry_inside_rejected_parentheses() {
        assert_eq!(
            number_occurrences("648(48%) (10%)add 1,234.50 -0.3% x123 1.2.3"),
            ["648", "48%", "10%", "1,234.50", "-0.3%"]
        );
    }
    #[test]
    fn unfinished_last_row_retains_content_evidence() {
        let rows = json!([[[[0, 0], [40, 0], [40, 10], [0, 10]]], ["53.38%"], [0.99]]);
        let html = json!("<table><tr><td>other</td></tr><tr><td>53.38%</td></table>");
        let info = structure(&html, &rows, &[]);
        assert_eq!(info.rows, 2);
        assert_eq!(info.numbers.get("53.38%"), Some(&1));
        let q = quality(&html, &rows, &[]);
        assert!(!q.valid);
        assert_eq!(q.retained, q.total);
    }
}
