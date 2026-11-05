//! 旧注释块物化与结构失败后的首页弱表撤销。
use super::*;
use crate::legacy_model::table_text_styles::preparation::{family, fold};
use unicode_normalization::UnicodeNormalization;

fn rotate_value(b: &Value, p: &Value, a: i64) -> Value {
    fn sub(a: &Value, b: &Value) -> Value {
        match (a.as_i64(), b.as_i64()) {
            (Some(a), Some(b)) => json!(a - b),
            _ => json!(a.as_f64().unwrap() - b.as_f64().unwrap()),
        }
    }
    match a {
        90 => json!([b[1], sub(&p[0], &b[2]), b[3], sub(&p[0], &b[0])]),
        180 => json!([
            sub(&p[0], &b[2]),
            sub(&p[1], &b[3]),
            sub(&p[0], &b[0]),
            sub(&p[1], &b[1])
        ]),
        270 => json!([sub(&p[1], &b[3]), b[0], sub(&p[1], &b[1]), b[2]]),
        _ => b.clone(),
    }
}
fn coverage(l: &Line<'_>) -> Result<f64, Error> {
    l.value.get("font_coverage").map_or(Ok(0.), number)
}
fn height_value(l: &Line<'_>, b: &Value) -> Result<Value, Error> {
    let em = l.value.get("em_height").cloned().unwrap_or(json!(0.));
    let h = l
        .value
        .get("effective_height")
        .cloned()
        .unwrap_or(json!(0.));
    let value = if flag(l.value, "style_scale_repaired")? && number(&em)? > 0. {
        em
    } else if number(&h)? != 0. {
        h
    } else {
        match (b[3].as_i64(), b[1].as_i64()) {
            (Some(a), Some(z)) => json!(a - z),
            _ => json!(number(&b[3])? - number(&b[1])?),
        }
    };
    Ok(if number(&value)? > 0.1 {
        value
    } else {
        json!(0.1)
    })
}
fn signature(l: &Line<'_>) -> Option<(String, Value)> {
    let a = l.value["font_signature"].as_array()?;
    Some((a[0].as_str().unwrap().to_owned(), a[1].clone()))
}
fn coerce(v: &Value) -> Result<Option<Box4>, Error> {
    let Some(a) = v.as_array().filter(|a| a.len() == 4) else {
        return Ok(None);
    };
    let mut b = [0.; 4];
    for (i, n) in a.iter().enumerate() {
        let Some(n) = super::super::table_text_styles::numeric(Some(n)) else {
            return Ok(None);
        };
        if n.abs() > 1e9 {
            return Err(limit());
        }
        b[i] = n;
    }
    let b = [
        b[0].min(b[2]),
        b[1].min(b[3]),
        b[0].max(b[2]),
        b[1].max(b[3]),
    ];
    Ok((b[0] < b[2] && b[1] < b[3]).then_some(b))
}
fn normalized(text: &str) -> String {
    let raw = text.replace("\r\n", "\n").replace('\r', "\n");
    let cs: Vec<_> = raw
        .chars()
        .filter_map(|c| match c {
            '\u{200b}' | '\u{2060}' | '\u{feff}' => None,
            '\u{85}' | '\u{2028}' | '\u{2029}' => Some('\n'),
            '\u{a0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200a}'
            | '\u{202f}'
            | '\u{205f}'
            | '\u{3000}' => Some(' '),
            _ => Some(c),
        })
        .collect();
    let mut result = String::new();
    for (i, &c) in cs.iter().enumerate() {
        let c = if matches!(c, '\u{2}' | '\u{ad}') {
            let mut j = i + 1;
            while j < cs.len() && matches!(cs[j], '\t' | ' ') {
                j += 1;
            }
            if i > 0 && cs[i - 1].is_ascii_alphabetic() && (j == cs.len() || cs[j] == '\n') {
                '-'
            } else {
                c
            }
        } else {
            c
        };
        if c == '\u{ad}'
            || c == '\n'
            || matches!(c,'\u{0}'..='\u{8}'|'\u{b}'..='\u{1f}'|'\u{7f}'..='\u{9f}')
        {
            continue;
        }
        let c = if c == '\t' { ' ' } else { c };
        if c != ' ' || !result.ends_with(' ') {
            result.push(c);
        }
    }
    result.trim_matches(python_space).to_owned()
}
fn content(group: &[&Line<'_>], work: &mut Work) -> Result<String, Error> {
    let mut texts = Vec::new();
    let mut scanned = 0;
    for l in group {
        let s = l.value["text"].as_str().unwrap();
        work.charge(s.len())?;
        let s = normalized(s);
        if !s.is_empty() {
            scanned += s.len();
            work.charge(scanned)?;
            texts.push(s);
        }
    }
    rapiddoc_native::text_boundary::merge(&texts)
}
pub(super) fn annotations(
    source: &Value,
    lines: &[Line<'_>],
    candidate: &Value,
    work: &mut Work,
) -> Result<Value, Error> {
    let page = &source["page_size"];
    let p = [number(&page[0])?, number(&page[1])?];
    let angle = integer(&candidate["angle"])?;
    let mut aa = super::annotations(candidate)?.iter().collect::<Vec<_>>();
    aa.sort_by(|a, b| {
        let ab = rotate(rect(&a["bbox"]).unwrap(), p, angle);
        let bb = rotate(rect(&b["bbox"]).unwrap(), p, angle);
        ab[1]
            .partial_cmp(&bb[1])
            .unwrap()
            .then(ab[0].partial_cmp(&bb[0]).unwrap())
            .then(a["kind"].as_str().cmp(&b["kind"].as_str()))
    });
    let mut blocks = Vec::new();
    let mut external = BTreeSet::new();
    let mut failed = Vec::new();
    let mut output_bytes = 0usize;
    for a in aa {
        let members = ids(&a["line_indices"])?;
        work.charge(lines.len() + members.len())?;
        let mut selected = lines
            .iter()
            .filter(|l| members.contains(&l.source))
            .collect::<Vec<_>>();
        selected.sort_by_key(|l| l.source);
        for l in &selected {
            work.charge(
                l.value["text"].as_str().unwrap().len()
                    + l.value["font_signature"][0].as_str().map_or(0, str::len)
                    + 64,
            )?;
        }
        let boxes = selected
            .iter()
            .map(|l| rotate(l.bbox, p, angle))
            .collect::<Vec<_>>();
        let max_width = boxes
            .iter()
            .map(|b| b[2] - b[0])
            .reduce(f64::max)
            .unwrap_or(0.);
        let left = boxes.iter().map(|b| b[0]).reduce(f64::min).unwrap_or(0.);
        let mut groups: Vec<Vec<&Line<'_>>> = Vec::new();
        for (i, &l) in selected.iter().enumerate() {
            work.charge(1)?;
            let mut split = false;
            if i > 0 {
                let prev = selected[i - 1];
                let b = boxes[i];
                let pb = boxes[i - 1];
                let h = height(prev, pb)?.max(height(l, b)?);
                let first = signature(prev);
                let second = signature(l);
                let switch = first.is_some()
                    && second.is_some()
                    && coverage(prev)? >= 0.6
                    && coverage(l)? >= 0.75
                    && family(&first) != family(&second);
                // None/None 两个空字体族不能被当作共享字体族。
                let switch = switch
                    || (first.is_some()
                        && second.is_some()
                        && coverage(prev)? >= 0.6
                        && coverage(l)? >= 0.75
                        && family(&first).is_none());
                let em = |l: &Line<'_>, b: Box4| -> Result<f64, Error> {
                    let em = l.value.get("em_height").map_or(Ok(0.), number)?;
                    if em != 0. {
                        Ok(em)
                    } else {
                        height(l, b)
                    }
                };
                let x = em(prev, pb)?;
                let y = em(l, b)?;
                let ratio = x.max(y) / x.min(y).max(0.1);
                let gap = b[1] - (pb[1] + height(prev, pb)?);
                split = pb[2] - pb[0] <= 0.45 * max_width
                    && b[2] - b[0] >= 0.75 * max_width
                    && b[0] - left <= 3. * h
                    && gap >= -0.25 * h
                    && gap <= 0.75 * h
                    && (switch || ratio >= 1.25);
            }
            if i == 0 || split {
                groups.push(Vec::new());
            }
            groups.last_mut().unwrap().push(l);
        }
        let initial = blocks.len();
        for group in groups {
            let content = content(&group, work)?;
            if content.is_empty() {
                continue;
            }
            let mut local = Vec::new();
            let mut output = Vec::new();
            let mut heights = Vec::new();
            let mut fonts = Vec::new();
            let mut repaired = false;
            let mut bbox = None;
            for l in group {
                work.charge(1)?;
                let b = rotate_value(&l.value["bbox"], page, angle);
                heights.push(height_value(l, &b)?);
                local.push(b.clone());
                let out = if let Some(ink) = coerce(&l.value["ink_bbox"])? {
                    let q = [
                        (ink[0] - 1.).clamp(0., p[0]),
                        (ink[1] - 1.).clamp(0., p[1]),
                        (ink[2] + 1.).clamp(0., p[0]),
                        (ink[3] + 1.).clamp(0., p[1]),
                    ];
                    if q[0] < q[2] && q[1] < q[3] {
                        repaired = true;
                        rotate_value(&json!(q), page, angle)
                    } else {
                        b
                    }
                } else {
                    b
                };
                output.push(out);
                let b = a["line_bboxes"]
                    .get(l.source.to_string())
                    .unwrap_or(&l.value["bbox"]);
                bbox = Some(bbox.map_or_else(|| b.clone(), |v| union_value(&v, b)));
                if !l.value["font_signature"].is_null()
                    && coverage(l)? >= 0.5
                    && !fonts.iter().any(|f| eq(f, &l.value["font_signature"]))
                {
                    fonts.push(l.value["font_signature"].clone());
                }
            }
            fonts.sort_by(|a, b| {
                a[0].as_str()
                    .cmp(&b[0].as_str())
                    .then(integer(&a[1]).unwrap().cmp(&integer(&b[1]).unwrap()))
            });
            let block = json!({"type":a["kind"],"bbox":bbox,"angle":candidate["angle"],"content":content,"_local_line_bboxes":local,"_local_output_line_bboxes":output,"_output_bbox_repaired":repaired,"_line_heights":heights,"_font_signatures":fonts,"_table_annotation_complete":true});
            let bytes = block.to_string().len();
            work.charge(bytes)?;
            output_bytes = output_bytes.saturating_add(bytes);
            if output_bytes > rapiddoc_core::resources::limit(8_000_000)
                || blocks.len() >= rapiddoc_core::resources::limit(4000)
            {
                return Err(limit());
            }
            blocks.push(block);
        }
        if blocks.len() == initial {
            failed.push(a.clone());
        } else {
            external.extend(members);
        }
    }
    Ok(
        json!({"annotation_blocks":blocks,"externalized_line_indices":external,"failed_annotations":failed}),
    )
}
fn icase(s: &str) -> String {
    s.chars()
        .flat_map(|c| match c {
            'İ' | 'ı' => 'i'.to_lowercase(),
            'ſ' => 's'.to_lowercase(),
            'K' => 'k'.to_lowercase(),
            _ => c.to_lowercase(),
        })
        .collect()
}
fn word(c: char) -> bool {
    python_alnum(c) || c == '_'
}
fn role(s: &str) -> Option<&'static str> {
    let s: String = s
        .nfkc()
        .filter(|c| !python_space(*c) && !matches!(c, ':' | '：'))
        .collect();
    let s = fold(&s);
    match s.as_str() {
        "articleinfo" | "articleinformation" | "文章信息" | "论文信息" => Some("metadata"),
        "abstract" | "summary" | "摘要" | "内容摘要" => Some("abstract"),
        "affiliations" | "authoraffiliations" | "作者单位" | "作者机构" | "机构信息" => {
            Some("affiliations")
        }
        _ => None,
    }
}
pub(in crate::processing) fn field(s: &str) -> Option<&'static str> {
    let s = icase(s.trim_matches(python_space));
    for (name, prefixes) in [
        ("received", &["received", "收稿"][..]),
        ("revised", &["revised", "修订", "修回"][..]),
        ("accepted", &["accepted", "录用", "接受日期"][..]),
        ("published", &["published", "出版日期", "发表日期"][..]),
    ] {
        if prefixes.iter().any(|p| s.starts_with(p)) {
            return Some(name);
        }
    }
    if let Some(rest) = s.strip_prefix("available") {
        let t = rest.trim_start_matches(python_space);
        if t.len() < rest.len() && t.starts_with("online") {
            return Some("published");
        }
    }
    if s.starts_with("关键词")
        || s.starts_with("关键字")
        || s.strip_prefix("key")
            .is_some_and(|s| s.trim_start_matches(python_space).starts_with("word"))
    {
        return Some("keywords");
    }
    if s.contains("通讯作者") || s.contains("通信作者") {
        return Some("contact");
    }
    let cs: Vec<_> = s.chars().collect();
    for term in ["corresponding", "correspondence"] {
        for (i, _) in s.match_indices(term) {
            if s[..i].chars().last().is_none_or(|c| !word(c))
                && s[i + term.len()..].chars().next().is_none_or(|c| !word(c))
            {
                return Some("contact");
            }
        }
    }
    for (i, &c) in cs.iter().enumerate() {
        if c != '@' || i == 0 || !(word(cs[i - 1]) || ".+-".contains(cs[i - 1])) {
            continue;
        }
        let tail = &cs[i + 1..];
        for (j, &c) in tail.iter().enumerate() {
            if !(word(c) || ".-".contains(c)) {
                break;
            }
            if c == '.'
                && j > 0
                && tail.get(j + 1).is_some_and(char::is_ascii_lowercase)
                && tail.get(j + 2).is_some_and(char::is_ascii_lowercase)
            {
                return Some("contact");
            }
        }
    }
    None
}
pub fn restore(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    restore_with_work(v, &mut Work(0))
}
pub(super) fn restore_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    let source = &v["source"];
    let lines = parse_lines(source)?;
    let bbox = candidate_rect(&v["bbox"])?;
    let mut result = json!({"restore":false,"lines":source["lines"]});
    if source["page_index"].is_null() || integer(&source["page_index"])? != 0 {
        return Ok(result);
    }
    let mut selected = Vec::new();
    for l in &lines {
        work.charge(l.value["text"].as_str().unwrap().len() + 1)?;
        if l.angle == 0 && inside(center(l.bbox), bbox) {
            selected.push(l);
        }
    }
    let fields = selected
        .iter()
        .filter(|l| field(l.value["text"].as_str().unwrap()).is_some())
        .copied()
        .collect::<Vec<_>>();
    let kinds = fields
        .iter()
        .filter_map(|l| field(l.value["text"].as_str().unwrap()))
        .collect::<BTreeSet<_>>();
    if kinds.len() < 2 {
        return Ok(result);
    }
    work.charge(
        selected.len().saturating_mul(fields.len())
            + selected
                .iter()
                .map(|l| l.value["text"].as_str().unwrap().len().saturating_mul(8))
                .sum::<usize>(),
    )?;
    let em = median(
        selected
            .iter()
            .map(|l| height(l, l.bbox))
            .collect::<Result<Vec<_>, _>>()?,
    );
    let field_box = fields
        .iter()
        .map(|l| l.value["bbox"].clone())
        .reduce(|a, b| union_value(&a, &b))
        .unwrap();
    let f = rect(&field_box)?;
    let prose = selected
        .iter()
        .copied()
        .filter(|l| {
            !fields.iter().any(|f| f.source == l.source)
                && role(l.value["text"].as_str().unwrap()).is_none()
                && {
                    let s = l.value["text"].as_str().unwrap();
                    s.split(python_space).filter(|s| !s.is_empty()).count() >= 8
                        || s.chars()
                            .filter(|c| ('\u{3400}'..='\u{9fff}').contains(c))
                            .count()
                            >= 12
                }
                && (l.bbox[0] >= f[2] + em || l.bbox[2] <= f[0] - em)
        })
        .collect::<Vec<_>>();
    if prose.is_empty() {
        return Ok(result);
    }
    let pb = prose
        .iter()
        .map(|l| l.value["bbox"].clone())
        .reduce(|a, b| union_value(&a, &b))
        .unwrap();
    let p = rect(&pb)?;
    if !prose
        .iter()
        .all(|l| (l.bbox[0] - p[0]).abs() <= em || (l.bbox[2] - p[2]).abs() <= em)
    {
        return Ok(result);
    }
    let abstract_present = selected.iter().any(|l| {
        role(l.value["text"].as_str().unwrap()) == Some("abstract")
            && overlap(l.bbox, p) >= 0.5
            && p[1] - l.bbox[3] >= 0.
            && p[1] - l.bbox[3] <= 3. * em
    });
    if (prose.len() < 3 && !abstract_present) || f[1].max(p[1]) - f[3].min(p[3]) > em {
        return Ok(result);
    }
    let mut vertical = false;
    let mut horizontal = 0;
    for r in array(&source["drawing_lines"], 5000)? {
        work.charge(1)?;
        let b = rect(&r["bbox"])?;
        if !matches!(r["orientation"].as_str(), Some("horizontal" | "vertical")) {
            return Err(invalid());
        }
        if !inside(center(b), bbox) {
            continue;
        }
        if r["orientation"] == "vertical"
            && bbox[0] + em < center(b).0
            && center(b).0 < bbox[2] - em
        {
            vertical = true;
        }
        if r["orientation"] == "horizontal"
            && bbox[1] + em < center(b).1
            && center(b).1 < bbox[3] - em
        {
            horizontal += 1;
        }
    }
    if vertical && horizontal >= 2 {
        return Ok(result);
    }
    let mut ids = BTreeSet::new();
    for l in selected {
        if matches!(
            role(l.value["text"].as_str().unwrap()),
            Some("metadata" | "abstract")
        ) {
            ids.insert(l.source);
        }
    }
    for l in result["lines"].as_array_mut().unwrap() {
        if ids.contains(&integer(&l["source_index"])?) {
            l["semantic_type"] = json!("paragraph_title");
            l["explicit_section_title"] = json!(true);
        }
    }
    result["restore"] = json!(true);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_annotations_and_front_matter() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-table-annotations.json");
        let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                super::super::prepare_annotations(&c["request"])
                    .unwrap_or_else(|e| panic!("{}: {e}", c["label"])),
                c["expected"],
                "{}",
                c["label"]
            );
        }
        for c in f["front"].as_array().unwrap() {
            assert_eq!(
                restore(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
        for c in f["roles"].as_array().unwrap() {
            assert_eq!(json!(role(c["text"].as_str().unwrap())), c["role"]);
            assert_eq!(json!(field(c["text"].as_str().unwrap())), c["field"]);
        }
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-table-annotations-types.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                super::super::prepare_annotations(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
}
