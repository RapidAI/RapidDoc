//! Original canonical structural candidates and legacy/canonical promotion arbitration.
use super::*;
fn page_size(p: &Value) -> Result<[f64; 2], Error> {
    size(p.get("page_size").unwrap_or(&p["source"]["page_size"]))
}
fn containers(p: &Value) -> Result<Vec<Box4>, Error> {
    optional(p, "fixed_blocks", 4000)?
        .iter()
        .filter(|b| {
            !(b["_inline_visual_row_id"].is_i64()
                || b["_inline_visual_row_id"].is_u64()
                || b["_inline_visual_row_id"].is_boolean())
        })
        .map(|b| bbox(&b["bbox"]))
        .collect()
}
fn canonical(l: &Value, b: &Box4) -> Result<f64, Error> {
    let em = l.get("em_height").map_or(Ok(0.), num)?;
    let h = l.get("effective_height").map_or(Ok(0.), num)?;
    Ok((if em > 0. {
        em
    } else if h != 0. {
        h
    } else {
        b.n[3] - b.n[1]
    })
    .max(0.1))
}
fn family_key(l: &Value) -> Result<Value, Error> {
    if l["font_signature"].is_null() {
        Ok(Value::Null)
    } else {
        Ok(json!([
            family(&l["font_signature"])?,
            l["font_signature"][1]
        ]))
    }
}
fn style_key(l: &Value) -> Result<Option<String>, Error> {
    let h = l.get("em_height").map_or(Ok(0.), num)?;
    if l["font_signature"].is_null() || h <= 0. {
        return Ok(None);
    }
    let Some(f) = family(&l["font_signature"])? else {
        return Ok(None);
    };
    Ok(Some(
        json!([
            f,
            l["font_signature"][1],
            (h * 4.).round_ties_even() / 4.,
            integer(l, "angle", 0)?
        ])
        .to_string(),
    ))
}
fn candidates(pages: &mut [Value], body: &Value, w: &mut Work) -> Result<(), Error> {
    let bh = num(&body["body_height"])?.max(0.1);
    let mut strong = Vec::<(usize, usize, Value)>::new();
    let mut start = Vec::<(usize, usize, Value)>::new();
    let mut anchors = Vec::<(usize, i64, f64, f64)>::new();
    for (page, p) in pages.iter().enumerate() {
        let size = page_size(p)?;
        let lines = array(&p["remaining_lines"], 4096)?.to_vec();
        let angles = lines
            .iter()
            .filter(|l| l["semantic_type"].is_null())
            .map(|l| integer(l, "angle", 0))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let containers = containers(p)?;
        for a in angles {
            let s = State {
                boxes: lines
                    .iter()
                    .map(|l| {
                        bbox(if l["source_bbox"].is_null() {
                            &l["bbox"]
                        } else {
                            &l["source_bbox"]
                        })
                        .map(|b| local(&b, size, a))
                    })
                    .collect::<Result<Vec<_>, _>>()?,
                lines: lines.clone(),
            };
            let mut ids = s
                .lines
                .iter()
                .enumerate()
                .filter(|(_, l)| {
                    integer(l, "angle", 0).ok() == Some(a) && l["semantic_type"].is_null()
                })
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            s.order(&mut ids);
            if ids.len() < 4 {
                continue;
            }
            let lp = if a == 90 || a == 270 {
                [size[1], size[0]]
            } else {
                size
            };
            let em = median(
                ids.iter()
                    .map(|&i| canonical(&s.lines[i], &s.boxes[i]))
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let lanes = super::super::table_text_styles::lanes::infer_layout(
                &json!({"line_geometry":ids.iter().map(|&i|json!([lane_line(&s.lines[i]),s.boxes[i].raw])).collect::<Vec<_>>(),"local_page_width":lp[0],"median_height":em}),
                w,
            )?;
            let physical = evidence::physical(&s, &ids, w)?;
            let mut gaps = BTreeMap::new();
            for g in physical {
                gaps.insert(
                    integer(&g, "source_index", 0)?,
                    (
                        if g["above"].is_null() {
                            None
                        } else {
                            Some(num(&g["above"])?)
                        },
                        if g["below"].is_null() {
                            None
                        } else {
                            Some(num(&g["below"])?)
                        },
                    ),
                );
            }
            let local_containers = containers
                .iter()
                .map(|b| local(b, size, a))
                .collect::<Vec<_>>();
            for &i in &ids {
                let l = &s.lines[i];
                let b = &s.boxes[i];
                if page == 0 && cy(b) < 0.64 * lp[1] {
                    continue;
                }
                let mut chosen = None;
                for lane in array(&lanes, 4096)? {
                    w.charge(1)?;
                    let count = array(&lane["line_indices"], 4096)?.len();
                    let left = num(&lane["left"])?;
                    let right = num(&lane["right"])?;
                    if lane["is_span"] == true
                        || count < 3
                        || left - bh > cx(b)
                        || cx(b) > right + bh
                    {
                        continue;
                    }
                    let key = (count, right - left);
                    if chosen.is_none_or(|(_, _, old_count, old_width)| {
                        key.0 > old_count || key.0 == old_count && key.1 > old_width
                    }) {
                        chosen = Some((left, right, count, right - left))
                    }
                }
                let Some((left, right, _, _)) = chosen else {
                    continue;
                };
                let width = (right - left).max(0.1);
                let wr = (b.n[2] - b.n[0]) / width;
                let sr = canonical(l, b)? / bh;
                let offset = (b.n[0] - left) / bh;
                let regular = regular(l, body)?;
                if !(0.75..=1.35).contains(&sr)
                    || wr > 0.8
                    || offset > 0.75
                    || offset < if regular { -0.75 } else { -3. }
                    || inside(b, &local_containers)
                {
                    continue;
                }
                let mut follower = false;
                for &j in &ids {
                    w.charge(1)?;
                    let c = &s.boxes[j];
                    if i != j
                        && b.n[1] < c.n[1]
                        && c.n[1] - b.n[3] <= 3. * bh
                        && c.n[0] >= left - bh
                        && c.n[0] <= left + 3.5 * bh
                        && c.n[2] - c.n[0] >= 0.4 * width
                    {
                        follower = true;
                        break;
                    }
                }
                if !follower {
                    continue;
                }
                let (ga, gb) = gaps
                    .get(&integer(l, "source_index", 0)?)
                    .copied()
                    .unwrap_or((None, None));
                let lr = (b.n[3] - b.n[1]) / bh;
                let coverage = coverage(l)?;
                if regular && lr < 1.3 && coverage < 0.75 {
                    continue;
                }
                let standard = matches!((ga,gb),(Some(ga),Some(gb))if ga>=0.65*bh&&gb>=0.35*bh&&(!regular||gb<=1.5*bh));
                let mut low = matches!((ga,gb),(Some(ga),Some(gb))if ga>=0.5*bh&&gb>=0.45*bh&&gb<=0.7*bh&&wr>=0.75&&coverage<=0.7&&lr>=1.8);
                if low
                    && anchors.iter().any(|&(p, aa, x, y)| {
                        p == page
                            && aa == a
                            && (x - left).abs() <= bh
                            && cy(b) - y > 0.
                            && cy(b) - y <= 12. * bh
                    })
                {
                    low = false
                }
                let compact = matches!((ga,gb),(Some(ga),Some(gb))if ga>=0.65*bh&&gb>=0.15*bh&&wr<=0.45&&coverage>=0.75&&lr<=1.3);
                let key = family_key(l)?;
                if standard
                    || low
                    || compact
                    || ga.is_none() && gb.is_some_and(|g| g >= 0.6 * bh) && !regular
                {
                    strong.push((page, i, key));
                    anchors.push((page, a, left, cy(b)));
                    continue;
                }
                if ga.is_none()
                    && b.n[1] <= 0.18 * lp[1]
                    && wr <= 0.6
                    && (coverage >= 0.75 || !regular && coverage >= 0.5)
                {
                    start.push((page, i, key))
                }
            }
        }
    }
    for &(p, i, _) in &strong {
        pages[p]["remaining_lines"][i]["semantic_type"] = json!("paragraph_title");
        pages[p]["remaining_lines"][i]["structural_title"] = json!(true)
    }
    for (p, i, key) in start {
        if !regular(&pages[p]["remaining_lines"][i], body)?
            || !key.is_null() && strong.iter().any(|(pp, _, k)| *pp == p && eq(k, &key))
        {
            pages[p]["remaining_lines"][i]["semantic_type"] = json!("paragraph_title");
            pages[p]["remaining_lines"][i]["structural_title"] = json!(true)
        }
    }
    Ok(())
}
pub(super) fn classify(
    pages: &mut [Value],
    canonical_body: &Value,
    legacy: &Value,
    titles: &Value,
    w: &mut Work,
) -> Result<(), Error> {
    if canonical_body.is_null() || canonical_body["has_style_scale_repairs"] != true {
        return Ok(());
    }
    let mut probes = pages.to_vec();
    for p in &mut probes {
        for l in p["remaining_lines"].as_array_mut().ok_or_else(invalid)? {
            l["style_scale_repaired"] = json!(true);
            l["structural_title"] = json!(false)
        }
    }
    candidates(&mut probes, canonical_body, w)?;
    let mut canonical = BTreeSet::new();
    for (p, page) in probes.iter().enumerate() {
        for l in array(&page["remaining_lines"], 4096)? {
            if l["structural_title"] == true {
                canonical.insert((p, integer(l, "source_index", 0)?));
            }
        }
    }
    let mut legacy_sources = BTreeSet::new();
    if !legacy.is_null() {
        let mut profile = legacy.clone();
        profile["has_style_scale_repairs"] = json!(false);
        for (p, page) in pages.iter().enumerate() {
            let mut lines = array(&page["remaining_lines"], 4096)?.to_vec();
            for l in &mut lines {
                l["style_scale_repaired"] = json!(false);
                l["structural_title"] = json!(false)
            }
            let captions = optional(page, "fixed_blocks", 4000)?
                .iter()
                .filter(|b| matches!(b["type"].as_str(), Some("image" | "code")))
                .map(|b| b["bbox"].clone())
                .collect::<Vec<_>>();
            let out = super::prepare_with_work(
                &json!({"lines":lines,"page_size":page_size(page)?,"page_index":p,"container_bboxes":containers(page)?.iter().map(|b|b.raw.clone()).collect::<Vec<_>>(),"caption_container_bboxes":captions,"document_body_profile":profile,"document_title_profile":titles,"classify_page":true}),
                w,
            )?;
            for l in array(&out["lines"], 4096)? {
                if l["semantic_type"] == "paragraph_title" {
                    legacy_sources.insert((p, integer(l, "source_index", 0)?));
                }
            }
        }
    }
    let bh = num(&canonical_body["body_height"])?.max(0.1);
    let mut styles = BTreeMap::<String, Vec<usize>>::new();
    for (p, page) in pages.iter_mut().enumerate() {
        let size = page_size(page)?;
        for l in page["remaining_lines"].as_array_mut().ok_or_else(invalid)? {
            w.charge(1)?;
            let key = (p, integer(l, "source_index", 0)?);
            if !canonical.contains(&key) {
                continue;
            }
            let b = local(
                &bbox(if l["source_bbox"].is_null() {
                    &l["bbox"]
                } else {
                    &l["source_bbox"]
                })?,
                size,
                integer(l, "angle", 0)?,
            );
            let ratio = (b.n[3] - b.n[1]) / bh;
            if legacy_sources.contains(&key) || ratio >= 1.8 || !regular(l, canonical_body)? {
                l["semantic_type"] = json!("paragraph_title");
                l["structural_title"] = json!(true);
                if !legacy_sources.contains(&key) && ratio >= 1.8 {
                    if let Some(k) = style_key(l)? {
                        styles.entry(k).or_default().push(p)
                    }
                }
            }
        }
    }
    let styles = styles
        .into_iter()
        .filter(|(_, ps)| {
            let mut counts = BTreeMap::<usize, usize>::new();
            for &p in ps {
                *counts.entry(p).or_default() += 1
            }
            counts.len() >= 2 || counts.values().copied().max().unwrap_or(0) >= 3
        })
        .map(|(k, _)| k)
        .collect::<BTreeSet<_>>();
    if !styles.is_empty() {
        for page in pages {
            page["canonical_formula_geometry"] = json!(true);
            for l in page["remaining_lines"].as_array_mut().ok_or_else(invalid)? {
                if style_key(l)?.is_some_and(|k| styles.contains(&k)) {
                    l["style_scale_repaired"] = json!(true)
                }
            }
        }
    }
    Ok(())
}
