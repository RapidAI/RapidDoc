//! Original _build_text_blocks through component materialization, before block refinements.
use super::*;
pub(super) fn components(v: &Value, w: &mut Work) -> Result<Vec<Value>, Error> {
    let mut s = State {
        lines: array(&v["lines"], 4096)?.to_vec(),
        page: size(&v["page_size"])?,
        tables: optional(v, "table_bboxes", 5000)?
            .iter()
            .map(bbox)
            .collect::<Result<Vec<_>, _>>()?,
        rules: vec![],
    };
    s.validate(w)?;
    let (mut blocks, consumed) = footnotes::build(&s, optional(v, "page_footnote_groups", 4096)?)?;
    s.lines
        .retain(|l| !consumed.contains(&integer(l, "source_index", 0).unwrap()));
    let angles = s
        .lines
        .iter()
        .map(|l| integer(l, "angle", 0))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let page = if v["page_index"].is_null() {
        None
    } else {
        Some(integer(v, "page_index", 0)?)
    };
    for angle in angles {
        let mut rs = s.geometry(angle)?;
        if rs.is_empty() {
            continue;
        }
        s.ordered(&mut rs, true);
        let h = median(rs.iter().map(|r| s.h(r)).collect::<Result<Vec<_>, _>>()?)?;
        let (lp, ph) = if angle == 90 || angle == 270 {
            (s.page[1], s.page[0])
        } else {
            (s.page[0], s.page[1])
        };
        let visuals = optional(v, "visual_bboxes", 5000)?
            .iter()
            .map(|b| Ok(local(&bbox(b)?, s.page, angle)))
            .collect::<Result<Vec<_>, Error>>()?;
        let lanes = s.lanes(&rs, lp, h, w)?;
        let bullets = breaks::bullets(&s, &rs)?;
        s.transform_rules(optional(v, "drawing_lines", 5000)?, angle)?;
        let mut split_rows = BTreeMap::new();
        for r in &rs {
            let l = s.l(r);
            if !l["visual_row_id"].is_null() && l["split_from_row"] == true {
                *split_rows
                    .entry(integer(l, "visual_row_id", 0)?)
                    .or_insert(0usize) += 1;
            }
        }
        for mut lane in lanes {
            s.ordered(&mut lane.rows, true);
            if lane.rows.is_empty() {
                continue;
            }
            w.charge(
                lane.rows
                    .len()
                    .saturating_mul(lane.rows.len())
                    .saturating_mul(4),
            )?;
            let (regular, mad) = s.lane_gap(&mut lane, w)?;
            let (locals, map) = grouping::local_lanes(&s, &lane)?;
            let mut structured = breaks::structured(&s, &lane, regular, mad)?;
            structured.extend(breaks::isolated(&s, &lane, regular, mad)?);
            let visual = breaks::visual(&s, &lane, &visuals, ph)?;
            let typo = prose::typography(&s, &lane, regular, mad)?;
            let formula = breaks::formula(&s, &lane)?;
            structured.extend(&visual);
            structured.extend(&typo);
            structured.extend(&formula);
            let mut protected = bullets.clone();
            protected.extend(visual);
            protected.extend(typo);
            protected.extend(formula);
            protected.extend(breaks::caption(&s, &lane)?);
            protected.extend(breaks::italic(&s, &lane)?);
            protected.extend(breaks::top(&s, &lane, ph)?);
            protected.extend(breaks::front(&s, &lane, ph, page)?);
            let mut explicit = prose::explicit(&s, &lane)?;
            explicit.extend(prose::prose(&s, &lane, regular, mad)?);
            explicit.extend(prose::cjk_prose(&s, &lane, regular, mad)?);
            explicit.extend(prose::cjk_entries(&s, &lane)?);
            protected.extend(&explicit);
            structured.extend(&protected);
            let mut rule = BTreeSet::new();
            for p in lane.rows.windows(2) {
                let (a, b) = (&p[0], &p[1]);
                if !eq(&s.l(a)["semantic_type"], &s.l(b)["semantic_type"]) {
                    continue;
                }
                let ink = |r: &Row| -> Result<Row, Error> {
                    Ok(Row {
                        index: r.index,
                        b: if s.l(r)["ink_bbox"].is_null() {
                            r.b.clone()
                        } else {
                            local(&bbox(&s.l(r)["ink_bbox"])?, s.page, angle)
                        },
                    })
                };
                let cl = map.get(&s.id(b)?).map_or(&lane, |&i| &locals[i]);
                if s.rule_barrier(&ink(a)?, &ink(b)?, cl)? {
                    rule.insert(s.id(b)?);
                }
            }
            structured.extend(&rule);
            protected.extend(&rule);
            let hanging = grouping::hanging(&s, &lane)?;
            let mut components: Vec<Vec<Row>> = vec![vec![lane.rows[0].clone()]];
            for p in lane.rows.windows(2) {
                let (a, b) = (&p[0], &p[1]);
                let (al, bl) = (s.l(a), s.l(b));
                let bid = s.id(b)?;
                let connect = if !al["paragraph_group"].is_null()
                    || !bl["paragraph_group"].is_null()
                {
                    eq(&al["paragraph_group"], &bl["paragraph_group"])
                } else if rule.contains(&bid) || !eq(&al["semantic_type"], &bl["semantic_type"]) {
                    false
                } else if !al["semantic_type"].is_null() {
                    connections::semantic(&s, a, b, &lane, regular)?
                } else {
                    let aid = s.id(a)?;
                    let ag = hanging.get(&aid);
                    let bg = hanging.get(&bid);
                    let a_local = map.get(&aid);
                    let b_local = map.get(&bid);
                    let cl = if b_local.is_some() && a_local == b_local {
                        &locals[*b_local.unwrap()]
                    } else {
                        &lane
                    };
                    let repaired = bl["style_scale_repaired"] == true
                        && bl["split_from_row"] == true
                        && !bl["visual_row_id"].is_null()
                        && split_rows
                            .get(&integer(bl, "visual_row_id", 0)?)
                            .copied()
                            .unwrap_or(0)
                            >= 2;
                    if repaired
                        || structured.contains(&bid)
                        || grouping::starts_reference(&s, a, b)?
                    {
                        false
                    } else if hyphen(s.t(a)?) {
                        connections::text(&s, a, b, cl, regular, mad)?
                    } else if ag.is_some() || bg.is_some() {
                        ag.is_some() && ag == bg
                    } else {
                        connections::text(&s, a, b, cl, regular, mad)?
                    }
                };
                if connect {
                    components.last_mut().unwrap().push(b.clone())
                } else {
                    components.push(vec![b.clone()])
                }
            }
            for rs in components {
                let first = &rs[0];
                let f = s.l(first);
                let source = s.id(first)?;
                let cl = map
                    .get(&source)
                    .filter(|i| rs.iter().all(|r| map.get(&s.id(r).unwrap()) == Some(*i)))
                    .map_or(&lane, |&i| &locals[i]);
                let ts = rs
                    .iter()
                    .map(|r| s.t(r).map(str::to_owned))
                    .collect::<Result<Vec<_>, _>>()?;
                let content = if f["semantic_type"] == "doc_title" {
                    ts.iter()
                        .map(|t| norm(t))
                        .filter(|s| !s.is_empty())
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    join_text(&ts)?
                };
                if content.is_empty() {
                    continue;
                }
                let visual = block_common::numeric_set(
                    rs.iter()
                        .filter(|r| !s.l(r)["visual_row_id"].is_null())
                        .map(|r| s.l(r)["visual_row_id"].clone()),
                )?;
                let single = if rs.len() == 1 && f["split_from_row"] == true {
                    f["visual_row_id"].clone()
                } else {
                    Value::Null
                };
                let mut output_boxes = vec![];
                let mut repaired = false;
                for r in &rs {
                    let b = tight(s.l(r), s.page)?;
                    repaired |= b.is_some();
                    output_boxes.push(local(&b.unwrap_or(s.raw(r)?), s.page, angle).raw)
                }
                let mut fonts = vec![];
                for r in &rs {
                    let f = &s.l(r)["font_signature"];
                    if !f.is_null() && s.coverage(r)? >= 0.5 && !fonts.iter().any(|v| eq(v, f)) {
                        fonts.push(f.clone())
                    }
                }
                block_common::sort_set(&mut fonts);
                let inline = rs
                    .iter()
                    .map(|r| optional(s.l(r), "inline_math_regions", 1000))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>();
                let group = if f["paragraph_group"].is_null()
                    || f["paragraph_group"] == 0
                    || f["paragraph_group"] == false
                {
                    hanging.get(&source).map_or(Value::Null, |n| json!(n))
                } else {
                    f["paragraph_group"].clone()
                };
                let raw = rs.iter().map(|r| s.raw(r)).collect::<Result<Vec<_>, _>>()?;
                blocks.push(json!({"type":if f["semantic_type"].is_null(){json!("text")}else{f["semantic_type"].clone()},"bbox":combined(raw.iter())?.raw,"angle":angle,"content":content,"_text_lines":rs.iter().map(|r|s.l(r).clone()).collect::<Vec<_>>(),"_reference_group":f["paragraph_group"],"_visual_row_ids":visual,"_single_run_row_id":single,"_local_line_bboxes":rs.iter().map(|r|r.b.raw.clone()).collect::<Vec<_>>(),"_local_output_line_bboxes":output_boxes,"_output_bbox_repaired":repaired,"_line_heights":rs.iter().map(|r|s.h(r)).collect::<Result<Vec<_>,_>>()?,"_font_signatures":fonts,"_inline_math_regions":inline,"_paragraph_formula_context":rs.iter().any(|r|s.l(r)["paragraph_formula_context"]==true),"_lane_interval":[cl.left,cl.right],"_lane_is_span":cl.span,"_hard_break_before":structured.contains(&source),"_protected_hard_break_before":protected.contains(&source),"_explicit_break_before":explicit.contains(&source),"_rule_break_before":rule.contains(&source),"_hanging_indent_group":group,"_leading_emphasis_start":grouping::emphasized(&s,&rs)?}));
            }
        }
    }
    Ok(blocks)
}
