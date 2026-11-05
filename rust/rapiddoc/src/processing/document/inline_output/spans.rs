use super::*;
pub(super) fn unescape(s: &str) -> String {
    let cs = chars(s);
    let (mut i, mut out) = (0, String::new());
    while i < cs.len() {
        if cs[i] != '&' {
            out.push(cs[i]);
            i += 1;
            continue;
        }
        let start = i;
        i += 1;
        if cs.get(i) == Some(&'#') {
            let mut j = i + 1;
            let radix = if matches!(cs.get(j), Some('x' | 'X')) {
                j += 1;
                16
            } else {
                10
            };
            let begin = j;
            let mut n = 0u32;
            while j < cs.len() {
                let digit = if cs[j].is_ascii() {
                    cs[j].to_digit(radix)
                } else {
                    None
                };
                let Some(d) = digit else { break };
                n = n.saturating_mul(radix).saturating_add(d);
                j += 1
            }
            if j == begin {
                out.push('&');
                continue;
            }
            if cs.get(j) == Some(&';') {
                j += 1
            }
            i = j;
            if let Some(s) = config()["invalid_charrefs"][n.to_string()].as_str() {
                out.push_str(s)
            } else if (0xd800..=0xdfff).contains(&n) || n > 0x10ffff {
                out.push('\u{fffd}')
            } else if !config()["invalid_codepoints"]
                .as_array()
                .unwrap()
                .contains(&json!(n))
            {
                if let Some(c) = char::from_u32(n) {
                    out.push(c)
                }
            }
            continue;
        }
        let begin = i;
        while i < cs.len() && i - begin < 32 && !"\t\n\u{c} <&#;".contains(cs[i]) {
            i += 1
        }
        if i == begin {
            out.push('&');
            continue;
        }
        if cs.get(i) == Some(&';') {
            i += 1
        }
        let name = text(&cs[begin..i]);
        if let Some(value) = config()["entities"][&name].as_str() {
            out.push_str(value);
            continue;
        }
        let mut found = false;
        for n in (2..i - begin).rev() {
            if let Some(value) = config()["entities"][text(&cs[begin..begin + n])].as_str() {
                out.push_str(value);
                out.push_str(&text(&cs[begin + n..i]));
                found = true;
                break;
            }
        }
        if !found {
            out.push_str(&text(&cs[start..i]))
        }
    }
    out
}
fn append(out: &mut Vec<Value>, s: &str, styles: u8) -> Result<(), Error> {
    if s.is_empty() {
        return Ok(());
    }
    if styles & (1 << 4) != 0 && styles & (1 << 5) != 0 {
        return Err(invalid());
    }
    let names = style_names(styles);
    if let Some(last) = out.last_mut() {
        if last["type"] == "text"
            && last
                .get("styles")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
                == names
        {
            last["content"] = json!(format!("{}{}", last["content"].as_str().unwrap(), s));
            return Ok(());
        }
    }
    let mut span = json!({"type":"text","content":s});
    if !names.is_empty() {
        span["styles"] = json!(names)
    }
    out.push(span);
    Ok(())
}
pub(super) fn script_markers(cs: &[char]) -> Option<(Vec<(usize, usize)>, Vec<Range>)> {
    let mut markers = Vec::new();
    let mut ranges = Vec::new();
    let mut active = None;
    let mut i = 0;
    while i < cs.len() {
        let mut found = None;
        for (tag, style, close) in [
            ("<sup>", 1u8 << 4, false),
            ("</sup>", 1 << 4, true),
            ("<sub>", 1 << 5, false),
            ("</sub>", 1 << 5, true),
        ] {
            let tag = chars(tag);
            if cs[i..].starts_with(&tag) {
                found = Some((tag.len(), style, close));
                break;
            }
        }
        let Some((n, style, close)) = found else {
            i += 1;
            continue;
        };
        markers.push((i, i + n));
        if !close {
            if active.is_some() {
                return None;
            }
            active = Some((style, i + n))
        } else {
            let Some((old, start)) = active.take() else {
                return None;
            };
            if old != style {
                return None;
            }
            if start < i {
                ranges.push(Range {
                    start,
                    end: i,
                    styles: style,
                    target: None,
                    source: 0,
                })
            }
        }
        i += n;
    }
    if active.is_some() || markers.is_empty() {
        None
    } else {
        Some((markers, ranges))
    }
}
pub(super) fn materialize(
    block: &mut Value,
    styles: Vec<Range>,
    links: Vec<Range>,
    work: &mut Work,
) -> Result<(), Error> {
    let b = block.as_object_mut().ok_or_else(invalid)?;
    let owned = b.remove("_pdf_native_script_markup") == Some(json!(true));
    if !listed("inline", b.get("type").unwrap_or(&Value::Null)) {
        return Ok(());
    }
    for k in ["_inline_link_intervals", "_inline_style_intervals"] {
        if let Some(v) = b.remove(k) {
            if v != json!([]) {
                return Err(invalid());
            }
        }
    }
    let Some(raw) = b.get("content").and_then(Value::as_str) else {
        return Ok(());
    };
    let content = chars(raw);
    work.charge(content.len())?;
    let (markers, scripts) = if owned {
        script_markers(&content).unwrap_or_default()
    } else {
        (Vec::new(), Vec::new())
    };
    let mut formulas = Vec::new();
    let mut i = 0;
    while i + 1 < content.len() {
        if content[i..].starts_with(&['\\', '(']) {
            if let Some(end) = formula_end(&content, i + 2, work)? {
                let end = end + 2;
                formulas.push((i, end));
                i = end;
                continue;
            }
        }
        i += 1
    }
    let mut bounds = BTreeSet::from([0, content.len()]);
    for r in styles.iter().chain(&links).chain(&scripts) {
        bounds.extend([r.start, r.end])
    }
    for &(a, b) in markers.iter().chain(&formulas) {
        bounds.extend([a, b])
    }
    let bounds: Vec<_> = bounds.into_iter().filter(|n| *n <= content.len()).collect();
    work.charge(bounds.len().saturating_mul(
        styles.len() + links.len() + scripts.len() + markers.len() + formulas.len() + 1,
    ))?;
    let mut out = Vec::new();
    for pair in bounds.windows(2) {
        let (start, end) = (pair[0], pair[1]);
        if start >= end || markers.iter().any(|&(a, b)| a <= start && end <= b) {
            continue;
        }
        if formulas.contains(&(start, end)) {
            let latex = text(&content[start + 2..end - 2]);
            let latex = latex.trim_matches(python_space);
            if !latex.is_empty() {
                out.push(json!({"type":"equation_inline","content":latex}))
            }
            continue;
        }
        let mask = styles
            .iter()
            .filter(|r| r.start <= start && end <= r.end)
            .fold(0, |m, r| m | r.styles);
        let native = scripts
            .iter()
            .filter(|r| r.start <= start && end <= r.end)
            .fold(0, |m, r| m | r.styles);
        let mask = super::conflicts::resolved_mask(mask, native);
        let slice = &content[start..end];
        let label = text(slice);
        if let Some(link) = links.iter().find(|r| r.start <= start && end <= r.end) {
            let url = link.target.as_ref().unwrap().trim_matches(python_space);
            if url.is_empty() {
                append(&mut out, &label, mask & !(1 << 2))?;
                continue;
            }
            // PDF evidence originates from the same four external schemes as the native reader.
            let url =
                rapiddoc_pdf::link_annotations::validate_external_uri(url).ok_or_else(invalid)?;
            let mut children = Vec::new();
            append(&mut children, &label, mask & !(1 << 2))?;
            if let Some(last) = out.last_mut() {
                if last["type"] == "hyperlink" && last["url"] == url {
                    let old = last["content"].as_array_mut().unwrap();
                    for child in children {
                        let st = child.get("styles").map_or(Ok(0), style_mask)?;
                        append(old, child["content"].as_str().unwrap(), st)?
                    }
                    continue;
                }
            }
            out.push(json!({"type":"hyperlink","url":url,"content":children}));
        } else if mask != 0 && slice.iter().any(|c| !python_space(*c)) {
            let left = slice.iter().take_while(|c| python_space(**c)).count();
            let right = slice.len() - slice.iter().rev().take_while(|c| python_space(**c)).count();
            append(&mut out, &text(&slice[..left]), 0)?;
            append(&mut out, &text(&slice[left..right]), mask)?;
            append(&mut out, &text(&slice[right..]), 0)?;
        } else {
            append(&mut out, &label, mask)?
        }
    }
    // Hyperlink children come from the strict typed dump; unlike top-level merged text,
    // an unstyled child retains its explicit default styles array.
    for span in &mut out {
        if span["type"] == "hyperlink" {
            for child in span["content"].as_array_mut().unwrap() {
                if child["type"] == "text" && child.get("styles").is_none() {
                    child["styles"] = json!([])
                }
            }
        }
    }
    b.insert("content".into(), json!(out));
    Ok(())
}
