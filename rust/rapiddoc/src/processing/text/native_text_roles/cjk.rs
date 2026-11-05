use super::*;
fn chinese(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}
fn visible(l: &Value, b: &Box4) -> Result<Box4, Error> {
    let mut boxes = Vec::new();
    for c in optional(l, "chars", 20000)? {
        if !c["tight_bbox"].is_null() {
            boxes.push(bbox(&c["tight_bbox"])?);
        }
    }
    if !boxes.is_empty() {
        combined(boxes.iter())
    } else if !l["ink_bbox"].is_null() {
        bbox(&l["ink_bbox"])
    } else {
        Ok(b.clone())
    }
}
fn mark_run(run: &[usize], ls: &mut [Value]) -> Result<(), Error> {
    if run.len() < 2
        || run
            .iter()
            .map(|&i| text(&ls[i]).unwrap().chars().count())
            .sum::<usize>()
            < 35
    {
        return Ok(());
    }
    let group = 1_000_000i64
        .checked_add(integer(&ls[run[0]], "source_index", 0)?)
        .ok_or_else(limit)?;
    mark(run.iter().copied(), ls, group);
    Ok(())
}
pub(super) fn restore(
    ls: &mut [Value],
    bs: &[Box4],
    sz: [f64; 2],
    containers: &[Value],
    page: i64,
    w: &mut Work,
) -> Result<(), Error> {
    let mut chars = Vec::new();
    for l in ls.iter() {
        chars.extend_from_slice(optional(l, "chars", 20000)?);
        if chars.len() > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
    }
    if !super::super::diagram_components::inflated(&json!({"chars":chars}), w)? {
        return Ok(());
    }
    let mut widths = Vec::new();
    for c in &chars {
        if c["char"]
            .as_str()
            .is_some_and(|s| s.chars().count() == 1 && s.chars().any(chinese))
        {
            let b = bbox(&c["bbox"])?;
            widths.push(b.n[2] - b.n[0]);
        }
    }
    let h = median(widths);
    let heading = constant_regex!(&format!(
        r"^{SPACE}*(?:[0-9]+[.、]{SPACE}*[\u{{4e00}}-\u{{9fff}}]|[（(][一二三四五六七八九十]+[）)]|[一二三四五六七八九十]+、)"
    ));
    let cs = containers.iter().map(bbox).collect::<Result<Vec<_>, _>>()?;
    let vb = ls
        .iter()
        .zip(bs)
        .map(|(l, b)| visible(l, b))
        .collect::<Result<Vec<_>, _>>()?;
    for i in 0..ls.len() {
        w.charge(ls.len() + cs.len() + 1)?;
        if integer(&ls[i], "angle", 0)? != 0
            || ls[i]["caption_start"] == true
            || ls[i]["title_suppressed"] == true
        {
            continue;
        }
        let b = &vb[i];
        if page > 0
            && ls[i]["semantic_type"] == "doc_title"
            && text(&ls[i])?.chars().count() > 25
            && !bold(&ls[i])?
        {
            ls[i]["semantic_type"] = Value::Null
        }
        if !ls[i]["semantic_type"].is_null()
            || cs
                .iter()
                .any(|c| super::super::path_graphics::overlap_first(b, c) >= 0.8)
        {
            continue;
        }
        let t = text(&ls[i])?;
        let numbered = heading.is_match(&mapped(t))
            && t.chars().count() <= 40
            && !t.contains(['。', '！', '？', ';', '；']);
        let display = bold(&ls[i])? && 1.5 * h <= b.n[3] - b.n[1] && t.chars().count() <= 40;
        if !numbered && !display {
            continue;
        }
        if !(0.07 * sz[1]..=0.92 * sz[1]).contains(&cy(b)) {
            continue;
        }
        let follower = (0..ls.len()).any(|j| {
            j != i
                && ls[j]["angle"] == 0
                && ls[j]["semantic_type"].is_null()
                && text(&ls[j]).unwrap().chars().count() >= 15
                && vb[j].n[1] - b.n[3] > 0.
                && vb[j].n[1] - b.n[3] <= 8. * h
                && (bs[j].n[0] - bs[i].n[0]).abs() <= 3. * h
        });
        if !follower && !display {
            continue;
        }
        ls[i]["semantic_type"] = json!("paragraph_title");
        ls[i]["structural_title"] = json!(true);
        ls[i]["explicit_section_title"] = json!(true);
        ls[i]["title_band_id"] = ls[i]["source_index"].clone();
    }
    let ids = (0..ls.len())
        .filter(|&i| {
            ls[i]["angle"] == 0
                && ls[i]["semantic_type"].is_null()
                && ls[i]["paragraph_group"].is_null()
                && ls[i]["caption_start"] != true
                && (text(&ls[i])
                    .unwrap()
                    .chars()
                    .filter(|&c| chinese(c))
                    .count()
                    >= 2
                    || (trim(text(&ls[i]).unwrap()).chars().count() <= 10
                        && text(&ls[i])
                            .unwrap()
                            .chars()
                            .any(rapiddoc_native::text_boundary::python_alnum)))
        })
        .collect::<Vec<_>>();
    if ids.is_empty() {
        return Ok(());
    }
    let lanes = super::super::table_text_styles::lanes::infer_layout(
        &json!({"line_geometry":ids.iter().map(|&i|json!([ls[i],vb[i].raw])).collect::<Vec<_>>(),"local_page_width":sz[0],"median_height":h}),
        w,
    )?;
    for lane in array(&lanes, 4096)? {
        let indices = array(&lane["line_indices"], 4096)?
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|i| ids.get(i as usize).copied())
                    .ok_or_else(invalid)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let rows = ordered(indices, &vb);
        let gaps = rows
            .windows(2)
            .map(|r| vb[r[1]].n[1] - vb[r[0]].n[1])
            .filter(|&g| h <= g && g <= 3. * h)
            .collect::<Vec<_>>();
        if gaps.len() < 2 {
            continue;
        }
        let pitch = median(gaps);
        let width = num(&lane["right"])? - num(&lane["left"])?;
        let mut run = Vec::<usize>::new();
        for i in rows {
            w.charge(1)?;
            if let Some(&p) = run.last() {
                let b: &Box4 = &vb[p];
                if vb[i].n[1] - b.n[1] > 1.25 * pitch
                    || (vb[i].n[0] - b.n[0]).abs() > 1.2 * h
                    || b.n[2] - b.n[0] < 0.8 * width
                    || !eq(&ls[i]["font_signature"], &ls[p]["font_signature"])
                {
                    mark_run(&run, ls)?;
                    run.clear();
                }
            }
            run.push(i)
        }
        mark_run(&run, ls)?;
    }
    Ok(())
}
