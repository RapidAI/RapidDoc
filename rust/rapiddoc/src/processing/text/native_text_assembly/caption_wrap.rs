//! Original mirrored caption obstacle recovery, preserving shallow snapshot identity.
use super::member_blocks as m;
use super::*;
pub(super) fn restore(
    bs: Vec<Value>,
    images: &[Box4],
    p: [f64; 2],
    w: &mut Work,
) -> Result<Vec<Value>, Error> {
    let captions = bs
        .iter()
        .enumerate()
        .filter(|(_, b)| super::captions::marker(m::text(b)))
        .map(|(i, b)| (i, b.clone()))
        .collect::<Vec<_>>();
    let mut serial = bs.len();
    let mut output = bs.into_iter().enumerate().collect::<Vec<_>>();
    for (cid, caption) in captions {
        let cb = bbox(&caption["bbox"])?;
        for (bid, block) in output.clone() {
            let mut ls = m::lines(&block)?.to_vec();
            ls.sort_by(|a, b| {
                let a = m::raw(a).unwrap();
                let b = m::raw(b).unwrap();
                a.n[1].total_cmp(&b.n[1]).then(a.n[0].total_cmp(&b.n[0]))
            });
            if block["type"] != "text" || bid == cid || ls.len() < 3 {
                continue;
            }
            if ls.iter().any(|l| {
                integer(l, "angle", 0).ok() != Some(0)
                    || !l["paragraph_group"].is_null()
                    || !l["semantic_type"].is_null()
            }) {
                continue;
            }
            let h = median(ls.iter().map(m::height).collect::<Result<Vec<_>, _>>()?)?;
            if !images
                .iter()
                .any(|b| (0.0..=3. * h).contains(&(cb.n[1] - b.n[3])) && axis(&cb, b, 0) >= 0.75)
            {
                continue;
            }
            let right = m::raw(&ls[0])?.n[0] >= cb.n[2] - 0.25 * h;
            let mirror = |b: &Box4| -> Result<Box4, Error> {
                if right {
                    bbox(&json!([p[0] - b.n[2], b.n[1], p[0] - b.n[0], b.n[3]]))
                } else {
                    Ok(b.clone())
                }
            };
            let obstacle = mirror(&cb)?;
            let mut split = None;
            for (i, l) in ls.iter().enumerate() {
                let b = m::raw(l)?;
                if b.n[1] >= cb.n[3] - 0.25 * h && mirror(&b)?.n[2] > obstacle.n[0] + 3. * h {
                    split = Some(i);
                    break;
                }
            }
            let Some(split) = split.filter(|&i| i >= 2) else {
                continue;
            };
            let mut narrow = ls[..split].to_vec();
            let tails = ls[split..].to_vec();
            let union = |ls: &[Value]| -> Result<Box4, Error> {
                let boxes = ls
                    .iter()
                    .map(|l| mirror(&m::raw(l)?))
                    .collect::<Result<Vec<_>, _>>()?;
                combined(boxes.iter())
            };
            let mut nb = union(&narrow)?;
            let tb = mirror(&m::raw(&tails[0])?)?;
            if !(nb.n[2] <= obstacle.n[0] + 0.25 * h
                && nb.n[3] > cb.n[1]
                && (cb.n[3] - 0.25 * h..=cb.n[3] + h).contains(&tb.n[1])
                && (tb.n[0] - nb.n[0]).abs() <= 0.5 * h
                && tb.n[2] - tb.n[0] >= 1.5 * (nb.n[2] - nb.n[0]))
            {
                continue;
            }
            let mut consumed = vec![bid];
            let mut first = block.clone();
            while ![
                "_explicit_break_before",
                "_rule_break_before",
                "_geometry_break_before",
            ]
            .iter()
            .any(|k| m::truth(&first[*k]))
            {
                let mut best: Option<(usize, Value, f64)> = None;
                for (id, other) in &output {
                    w.charge(1)?;
                    if consumed.contains(id)
                        || other["type"] != "text"
                        || m::lines(other)?.is_empty()
                    {
                        continue;
                    }
                    let ob = bbox(&other["bbox"])?;
                    let mb = mirror(&ob)?;
                    if (mb.n[0] - nb.n[0]).abs() <= 0.5 * h
                        && mb.n[2] <= obstacle.n[0] + 0.25 * h
                        && (0.0..=0.6 * h).contains(&(nb.n[1] - ob.n[3]))
                        && best.as_ref().is_none_or(|(_, _, bottom)| ob.n[3] > *bottom)
                    {
                        best = Some((*id, other.clone(), ob.n[3]))
                    }
                }
                let Some((id, previous, _)) = best else { break };
                let previous_lines = m::lines(&previous)?;
                let tail = previous_lines.last().unwrap();
                if m::truth(&tail["paragraph_terminal"])
                    || regex(r#"[.!?。！？][\])’"']*$"#)
                        .is_match(m::text(&previous).trim_end_matches(python_space))
                    || !fonts(tail, &narrow[0])?
                {
                    break;
                }
                consumed.push(id);
                let mut merged = previous_lines.to_vec();
                merged.extend(narrow);
                narrow = merged;
                nb = union(&narrow)?;
                first = previous;
            }
            let build = |ls: &[Value], w: &mut Work| -> Result<Vec<Value>, Error> {
                let out = super::build_with_work(
                    &json!({"lines":ls,"page_size":p,"table_bboxes":[]}),
                    w,
                )?;
                Ok(array(&out, 4096)?.to_vec())
            };
            let rebuilt = build(&narrow, w)?;
            let mut tail_blocks = build(&tails, w)?;
            if let Some(b) = tail_blocks.first_mut() {
                b["_geometry_break_before"] = json!(true)
            }
            let mut all = rebuilt;
            all.extend(tail_blocks);
            for b in &mut all {
                b["_geometry_barriers"] = json!([cb.raw]);
            }
            output.retain(|(id, _)| !consumed.contains(id));
            for b in all {
                output.push((serial, b));
                serial += 1;
            }
        }
    }
    Ok(output.into_iter().map(|(_, b)| b).collect())
}
