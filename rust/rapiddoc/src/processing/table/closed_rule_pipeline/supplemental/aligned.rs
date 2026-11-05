//! 原生字符→整页数值组→区域边界→区域字符网格确认的完整候选分支。
use super::super::super::{table_native_aligned_numeric as aligned, table_native_text as text};
use super::*;
pub(super) fn detect(v: &Request, work: &mut Work) -> Result<Value, Error> {
    let source = v.source;
    let lines = optional(source, "lines", 4096)?;
    let mut count = 0;
    for l in lines {
        let t = l["text"].as_str().ok_or_else(invalid)?;
        if t.chars().count() > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        work.charge(t.len() + 1)?;
        let t = t.trim_matches(python_space);
        if !t.contains(['%', '‰']) && aligned::numeric_token(t) {
            count += 1;
        }
    }
    if count < 9 {
        return Ok(json!({"candidates":[]}));
    }
    let size = array(&source["page_size"], 2)?;
    if size.len() != 2 {
        return Err(invalid());
    }
    for n in size {
        if num(n)? <= 0. {
            return Err(invalid());
        }
    }
    let chars = optional(source, "chars", 20000)?;
    let mut rules = Vec::new();
    let mut rectangles = Vec::new();
    for r in optional(source, "drawing_lines", 5000)? {
        work.charge(1)?;
        if r["orientation"] != "horizontal" && r["orientation"] != "vertical" {
            continue;
        }
        let b = bbox(&r["bbox"])?;
        let w = r
            .get("width")
            .filter(|v| !v.is_null())
            .map_or(Ok(0.), num)?
            .max(0.);
        rules.push(json!({"bbox":b.raw,"orientation":r["orientation"],"width":w}));
    }
    for r in optional(source, "path_infos", 5000)? {
        work.charge(1)?;
        let b = bbox(&r["bbox"])?;
        rectangles.push(json!({"bbox":b.raw,"segment_count":integer(r,"segment_count",0)?,"form_depth":integer(r,"form_depth",0)?,"fill_visible":flag(r,"fill_visible")?,"stroke_visible":flag(r,"stroke_visible")?}));
    }
    let input = json!({"table_bbox":[0.,0.,size[0],size[1]],"page_size":size,"angle":0,"chars":chars,"drawing_lines":rules,"rectangles":rectangles});
    let Some(text) = text::prepare_with_work(&input, work)? else {
        return Ok(json!({"candidates":[]}));
    };
    let regions = aligned::regions(&input, &text, work)?;
    let excluded = optional(v.options, "excluded_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let mut candidates = Vec::new();
    let mut total = 0;
    for region in regions {
        let b = bbox(&json!(region))?;
        work.charge(excluded.len() + 1)?;
        if excluded.iter().any(|e| overlap_smaller(&b, e) >= 0.5) {
            continue;
        }
        let mut regional = input.clone();
        regional["table_bbox"] = json!(region);
        let Some(t) = text::prepare_with_work(&regional, work)? else {
            continue;
        };
        if !aligned::accepts(&regional, &t, work)? {
            continue;
        }
        work.charge(lines.len() + 1)?;
        let mut members = Vec::new();
        for l in lines {
            if overlap_first(&bbox(&l["bbox"])?, &b) >= 0.8 {
                members.push(l);
            }
        }
        total += members.len();
        if total > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        let mut c = candidate(b, &members, vec![], vec![], 120)?;
        c["preserve_inline_font_styles"] = json!(true);
        candidates.push(c);
    }
    let out = json!({"candidates":candidates});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(out)
}
