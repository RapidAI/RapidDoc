use super::*;
use rapiddoc_native::text_boundary::python_space;
fn fragment(c: &Value) -> String {
    let raw = super::super::table_native_multiline::spatial::raw_string(c.get("char"), true);
    let mut out = String::new();
    for c in raw.chars() {
        if ['\u{200b}', '\u{2060}', '\u{feff}', '\u{ad}'].contains(&c) {
            continue;
        }
        if c == '\u{2}' {
            out.push('-');
            continue;
        }
        if python_space(c)
            || matches!(c,'\u{0}'..='\u{8}'|'\u{b}'|'\u{c}'|'\u{e}'..='\u{1f}'|'\u{7f}'..='\u{9f}')
        {
            continue;
        }
        out.push_str(match c {
            'ﬀ' => "ff",
            'ﬁ' => "fi",
            'ﬂ' => "fl",
            'ﬃ' => "ffi",
            'ﬄ' => "ffl",
            'ﬅ' => "ft",
            'ﬆ' => "st",
            _ => {
                out.push(c);
                continue;
            }
        })
    }
    out
}
fn ordered(chars: &[Value], positions: &[usize]) -> Vec<usize> {
    let mut indices = positions.to_vec();
    let keys = indices
        .iter()
        .map(|&i| {
            chars[i]["char_idx"]
                .as_i64()
                .or_else(|| chars[i]["char_idx"].as_bool().map(i64::from))
        })
        .collect::<Option<Vec<_>>>();
    if keys
        .as_ref()
        .is_some_and(|k| k.windows(2).any(|p| p[0] > p[1]))
    {
        indices.sort_by_key(|&i| {
            chars[i]["char_idx"]
                .as_i64()
                .or_else(|| chars[i]["char_idx"].as_bool().map(i64::from))
                .unwrap()
        })
    }
    indices
}
fn segments(
    text: &str,
    resplit: &Resplit,
    w: &mut Work,
) -> Result<Option<Vec<(usize, usize, usize, String)>>, Error> {
    let chars = optional(&resplit.source, "chars", 20000)?;
    let positions = (0..chars.len()).collect::<Vec<_>>();
    let mut source_spans = Vec::new();
    let mut spans = BTreeMap::new();
    let mut parts = String::new();
    let mut cursor = 0usize;
    for i in ordered(chars, &positions) {
        w.charge(1)?;
        let text = fragment(&chars[i]);
        if text.is_empty() {
            continue;
        }
        let end = cursor.checked_add(text.chars().count()).ok_or_else(limit)?;
        source_spans.push((cursor, end));
        spans.insert(i, (cursor, end));
        parts.push_str(&text);
        cursor = end
    }
    if parts != text {
        return Ok(None);
    }
    let evidence = text.chars().collect::<Vec<_>>();
    let mut used = Vec::new();
    let mut segments = Vec::new();
    let mut members = (0..resplit.members.len()).collect::<Vec<_>>();
    members.sort_by_key(|&i| {
        (
            integer(&resplit.members[i], "run_index", 0).unwrap_or(0),
            integer(&resplit.members[i], "source_index", 0).unwrap_or(0),
        )
    });
    for i in members {
        let mut member_spans = Vec::new();
        let mut member_text = String::new();
        for j in ordered(chars, &resplit.positions[i]) {
            w.charge(1)?;
            let part = fragment(&chars[j]);
            if part.is_empty() {
                continue;
            }
            let Some(&(start, end)) = spans.get(&j) else {
                return Ok(None);
            };
            if evidence[start..end].iter().collect::<String>() != part {
                return Ok(None);
            }
            member_spans.push((start, end));
            member_text.push_str(&part)
        }
        if member_spans.is_empty() {
            return Ok(None);
        }
        let start = member_spans.iter().map(|s| s.0).min().unwrap();
        let end = member_spans.iter().map(|s| s.1).max().unwrap();
        if member_spans.iter().map(|(a, b)| b - a).sum::<usize>() != end - start
            || evidence[start..end].iter().collect::<String>() != member_text
        {
            return Ok(None);
        }
        used.extend(member_spans);
        segments.push((i, start, end, member_text));
    }
    used.sort();
    if used != source_spans {
        return Ok(None);
    }
    Ok(Some(segments))
}
pub(super) fn realign(
    v: &Value,
    resplits: &BTreeMap<i64, Resplit>,
    w: &mut Work,
) -> Result<(Vec<Value>, Vec<Value>), Error> {
    let mut boxes = BTreeMap::new();
    for entry in optional(v, "line_bboxes", 4096)? {
        bbox(&entry["bbox"])?;
        boxes.insert(integer(entry, "source_index", 0)?, entry["bbox"].clone());
    }
    let mut result = Vec::new();
    for (key, range_key) in [
        ("style_lines", "style_ranges"),
        ("link_lines", "link_ranges"),
    ] {
        let mut aligned = Vec::new();
        for line in optional(v, key, 20000)? {
            w.charge(1)?;
            let id = integer(line, "source_index", 0)?;
            let Some(resplit) = resplits.get(&id) else {
                let mut line = line.clone();
                if let Some(b) = boxes.get(&id) {
                    line["bbox"] = b.clone()
                }
                aligned.push(line);
                continue;
            };
            let text = line["text"].as_str().ok_or_else(invalid)?;
            let Some(segments) = segments(text, resplit, w)? else {
                aligned.push(line.clone());
                continue;
            };
            for (i, start, end, text) in segments {
                let mut ranges = Vec::new();
                for range in optional(line, range_key, 20000)? {
                    w.charge(1)?;
                    let a = range["start"].as_i64().ok_or_else(invalid)?;
                    let b = range["end"].as_i64().ok_or_else(invalid)?;
                    let a = (start as i64).max(a);
                    let b = (end as i64).min(b);
                    if a < b {
                        let mut range = range.clone();
                        range["start"] = json!(a - start as i64);
                        range["end"] = json!(b - start as i64);
                        ranges.push(range)
                    }
                }
                if range_key == "link_ranges" && ranges.is_empty() {
                    continue;
                }
                let member = &resplit.members[i];
                let mut out = line.clone();
                out["bbox"] = member["bbox"].clone();
                out["source_index"] = member["source_index"].clone();
                out["text"] = json!(text);
                out[range_key] = json!(ranges);
                aligned.push(out)
            }
        }
        result.push(aligned)
    }
    Ok((result.remove(0), result.remove(0)))
}
