use super::*;
#[derive(Clone)]
struct Script {
    line: Line,
    meta: Vec<Value>,
    angle: Value,
}
fn read(v: &Value) -> Result<Vec<Script>, Error> {
    let mut out = Vec::new();
    for raw in array(v, 4096)? {
        let bounds = bbox(&raw["bbox"])?.ok_or_else(invalid)?;
        let text = chars(raw["text"].as_str().ok_or_else(invalid)?);
        let source = raw["source_index"].as_i64().ok_or_else(invalid)?;
        raw["angle"].as_i64().ok_or_else(invalid)?;
        let mut ranges = Vec::new();
        let mut meta = Vec::new();
        for r in array(&raw["script_ranges"], 5000)? {
            let start = r["start"].as_u64().ok_or_else(invalid)? as usize;
            let end = r["end"].as_u64().ok_or_else(invalid)? as usize;
            if start > text.len() || end > text.len() {
                return Err(invalid());
            }
            let styles = match r["style"].as_str() {
                Some("superscript") => 1 << 4,
                Some("subscript") => 1 << 5,
                _ => return Err(invalid()),
            };
            bbox(&r["bbox"])?.ok_or_else(invalid)?;
            r["stable_body_count"].as_i64().ok_or_else(invalid)?;
            r["formula_region"].as_bool().ok_or_else(invalid)?;
            ranges.push(Range {
                start,
                end,
                styles,
                target: None,
                source,
            });
            meta.push(r.clone());
        }
        out.push(Script {
            line: Line {
                bbox: bounds,
                text,
                source,
                ranges,
            },
            meta,
            angle: raw["angle"].clone(),
        });
    }
    Ok(out)
}
fn target_bbox(v: &Value, p: [f64; 2]) -> Result<Option<[f64; 4]>, Error> {
    let Some(mut b) = bbox(v)? else {
        return Ok(None);
    };
    if b.iter().all(|n| *n >= 0. && *n <= 1.) {
        for i in 0..4 {
            b[i] *= p[i % 2]
        }
    }
    Ok(Some(b))
}
fn score(a: [f64; 4], b: [f64; 4]) -> Option<(bool, f64, f64)> {
    let x = (a[0] + a[2]) / 2.;
    let y = (a[1] + a[3]) / 2.;
    let inside = b[0] <= x && x <= b[2] && b[1] <= y && y <= b[3];
    let ratio = (a[2].min(b[2]) - a[0].max(b[0])).max(0.)
        * (a[3].min(b[3]) - a[1].max(b[1])).max(0.)
        / ((a[2] - a[0]) * (a[3] - a[1])).max(0.01);
    (inside || ratio >= 0.5).then_some((inside, ratio, -((b[2] - b[0]) * (b[3] - b[1]))))
}
fn assign(
    blocks: &[Value],
    lines: &[Script],
    p: [f64; 2],
    work: &mut Work,
) -> Result<(Vec<(usize, Vec<Script>)>, BTreeMap<usize, Vec<Token>>), Error> {
    let mut projected = BTreeMap::new();
    let mut targets = Vec::new();
    for (i, b) in blocks.iter().enumerate() {
        if !listed("natural", &b["type"]) {
            continue;
        }
        let Some(content) = b["content"].as_str() else {
            continue;
        };
        if let Some(bb) = target_bbox(&b["bbox"], p)? {
            targets.push((i, bb));
            projected.insert(i, project(&chars(content), work)?);
        }
    }
    let mut out: Vec<(usize, Vec<Script>)> = Vec::new();
    let mut primary = Vec::new();
    for l in lines {
        let a = assignments(blocks, std::slice::from_ref(&l.line), p, work)?;
        let owner = a.keys().next().copied();
        primary.push(owner);
        if let Some(i) = owner {
            if let Some((_, ls)) = out.iter_mut().find(|(j, _)| *j == i) {
                ls.push(l.clone())
            } else {
                out.push((i, vec![l.clone()]))
            }
        }
    }
    let mut fallback: Vec<(usize, usize, Vec<usize>)> = Vec::new();
    for (index, l) in lines.iter().enumerate() {
        for (ri, r) in l.line.ranges.iter().enumerate() {
            let mut one = l.line.clone();
            one.ranges = vec![r.clone()];
            if let Some(i) = primary[index] {
                if !matching::match_script(&projected[&i], &one, work)?.is_empty() {
                    continue;
                }
            }
            let tight = bbox(&l.meta[ri]["bbox"])?.ok_or_else(invalid)?;
            let mut best = None;
            for &(i, b) in &targets {
                if Some(i) == primary[index] {
                    continue;
                }
                let matches = matching::match_script(&projected[&i], &one, work)?;
                if matches.is_empty() {
                    continue;
                }
                work.charge(1)?;
                if let Some(s) = score(tight, b) {
                    if best.as_ref().is_none_or(|(_, old)| s > *old) {
                        best = Some((i, s))
                    }
                }
            }
            if let Some((i, _)) = best {
                if let Some((_, _, rs)) = fallback
                    .iter_mut()
                    .find(|(bi, li, _)| *bi == i && *li == index)
                {
                    rs.push(ri)
                } else {
                    fallback.push((i, index, vec![ri]))
                }
            }
        }
    }
    for (i, index, indices) in fallback {
        let l = &lines[index];
        let mut new = l.clone();
        new.line.ranges = indices.iter().map(|&i| l.line.ranges[i].clone()).collect();
        new.meta = indices.iter().map(|&i| l.meta[i].clone()).collect();
        if let Some((_, ls)) = out.iter_mut().find(|(j, _)| *j == i) {
            ls.push(new)
        } else {
            out.push((i, vec![new]))
        }
    }
    for (_, ls) in &mut out {
        ls.sort_by(|a, b| {
            a.line
                .source
                .cmp(&b.line.source)
                .then(a.line.bbox[1].total_cmp(&b.line.bbox[1]))
                .then(a.line.bbox[0].total_cmp(&b.line.bbox[0]))
        })
    }
    Ok((out, projected))
}
pub(super) fn apply(
    blocks: &mut [Value],
    p: [f64; 2],
    input: &Value,
    styles: &mut BTreeMap<usize, Vec<Range>>,
    diagnostics: bool,
    work: &mut Work,
) -> Result<Vec<Value>, Error> {
    let lines = read(input)?;
    let (assignments, projected) = assign(blocks, &lines, p, work)?;
    let mut output = Vec::new();
    for (i, ls) in assignments {
        let mut regions = Vec::new();
        if let Some(raw) = blocks[i].get("_inline_math_regions") {
            for raw in array(raw, 5000)? {
                if let Some(b) = target_bbox(raw, p)? {
                    regions.push(b)
                }
            }
        }
        let mut eligible = Vec::new();
        let mut projected_lines = Vec::new();
        for l in &ls {
            work.charge(l.meta.len().saturating_mul(regions.len() + 1))?;
            let mut retained = l.clone();
            let mut indices = Vec::new();
            for (j, m) in l.meta.iter().enumerate() {
                let b = bbox(&m["bbox"])?.ok_or_else(invalid)?;
                let x = (b[0] + b[2]) / 2.;
                let y = (b[1] + b[3]) / 2.;
                if m["formula_region"] == true
                    || !regions
                        .iter()
                        .any(|r| r[0] <= x && x <= r[2] && r[1] <= y && y <= r[3])
                {
                    indices.push(j)
                }
            }
            retained.line.ranges = indices.iter().map(|&j| l.line.ranges[j].clone()).collect();
            retained.meta = indices.iter().map(|&j| l.meta[j].clone()).collect();
            let mut projected = retained.line.clone();
            for r in &mut projected.ranges {
                for s in [4, 5] {
                    if !config()["targets"][config()["styles"][s].as_str().unwrap()]
                        .as_array()
                        .unwrap()
                        .contains(&blocks[i]["type"])
                    {
                        r.styles &= !(1 << s)
                    }
                }
            }
            projected.ranges.retain(|r| r.styles != 0);
            eligible.push(retained);
            projected_lines.push(projected);
        }
        let Some(content) = blocks[i]["content"].as_str() else {
            continue;
        };
        if content.is_empty() {
            continue;
        }
        let content = chars(content);
        let tokens = &projected[&i];
        projected_lines.sort_by(|a, b| {
            a.source
                .cmp(&b.source)
                .then(a.bbox[1].total_cmp(&b.bbox[1]))
                .then(a.bbox[0].total_cmp(&b.bbox[0]))
                .then(a.text.cmp(&b.text))
                .then_with(|| {
                    a.ranges
                        .iter()
                        .map(|r| (r.start, r.end, style_key(r.styles)))
                        .collect::<Vec<_>>()
                        .cmp(
                            &b.ranges
                                .iter()
                                .map(|r| (r.start, r.end, style_key(r.styles)))
                                .collect::<Vec<_>>(),
                        )
                })
        });
        let matches = super::conflicts::collect(tokens, &projected_lines, work)?;
        let combined = merge_styles(&matches.iter().map(|m| m.range.clone()).collect::<Vec<_>>());
        let raw = raw_ranges(&content, tokens, &combined, false);
        if !raw.is_empty() {
            let mut existing = styles.remove(&i).unwrap_or_default();
            existing.extend(raw);
            styles.insert(i, merge_raw_styles(&content, existing));
        }
        if diagnostics {
            let effective = super::conflicts::effective(
                &content,
                styles.get(&i).map_or(&[][..], Vec::as_slice),
                blocks[i]["_pdf_native_script_markup"] == true,
                work,
            )?;
            for l in &eligible {
                for (j, r) in l.line.ranges.iter().enumerate() {
                    for matched in &matches {
                        if matched.line.source != l.line.source
                            || matched.line.bbox != l.line.bbox
                            || matched.line.text != l.line.text
                            || matched.source_range.start >= r.end
                            || r.start >= matched.source_range.end
                            || matched.source_range.styles & r.styles == 0
                        {
                            continue;
                        }
                        for interval in &effective {
                            if interval.styles & r.styles == 0 {
                                continue;
                            }
                            let indices: Vec<_> = (matched.range.start..matched.range.end)
                                .filter(|&index| {
                                    interval.start <= tokens[index].start
                                        && tokens[index].end <= interval.end
                                })
                                .collect();
                            let Some(&first) = indices.first() else {
                                continue;
                            };
                            let last = *indices.last().unwrap();
                            let start = matched.source_range.start + first - matched.range.start;
                            let end = matched.source_range.start + last - matched.range.start + 1;
                            let meta = &l.meta[j];
                            let record = json!({"block_index":i,"raw_start":tokens[first].start,"raw_end":tokens[last].end,"source_index":l.line.source,"range_start":start,"range_end":end,"role":meta["style"],"text":text(&l.line.text[start..end]),"bbox":meta["bbox"],"angle":l.angle,"formula_region":meta["formula_region"],"stable_body_count":meta["stable_body_count"]});
                            if !output.contains(&record) {
                                output.push(record);
                            }
                        }
                    }
                }
            }
        }
    }
    for b in blocks {
        b.as_object_mut()
            .ok_or_else(invalid)?
            .remove("_inline_math_regions");
    }
    Ok(output)
}
