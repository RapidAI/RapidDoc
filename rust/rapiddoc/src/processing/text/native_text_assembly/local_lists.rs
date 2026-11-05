//! Original local list inference from consecutive lowercase markers or filled vector dots.
use super::member_blocks as m;
use super::*;
fn bullet(l: &Value, paths: &[Value], em: f64) -> Result<Option<Box4>, Error> {
    let lb = m::raw(l)?;
    for path in paths {
        if !m::truth(&path["fill_visible"])
            || m::truth(&path["stroke_visible"])
            || !optional(path, "rectangle_bboxes", 5000)?.is_empty()
            || !(8..=100).contains(&integer(path, "segment_count", 0)?)
        {
            continue;
        }
        let b = bbox(&path["bbox"])?;
        let bw = b.n[2] - b.n[0];
        let bh = b.n[3] - b.n[1];
        if !(0.12 * em..=0.5 * em).contains(&bw) || !(0.12 * em..=0.5 * em).contains(&bh) {
            continue;
        }
        if bh == 0. {
            return Err(invalid());
        }
        if (0.8..=1.2).contains(&(bw / bh))
            && (0.2 * em..=2. * em).contains(&(lb.n[0] - b.n[2]))
            && ((b.n[1] + b.n[3] - lb.n[1] - lb.n[3]) * 0.5).abs() <= 0.4 * em
        {
            return Ok(Some(b));
        }
    }
    Ok(None)
}
pub(super) fn split(bs: Vec<Value>, paths: &[Value], w: &mut Work) -> Result<Vec<Value>, Error> {
    let mut out = vec![];
    for b in bs {
        let mut ls = m::lines(&b)?.to_vec();
        ls.sort_by(|a, b| {
            let a = m::raw(a).unwrap();
            let b = m::raw(b).unwrap();
            a.n[1].total_cmp(&b.n[1]).then(a.n[0].total_cmp(&b.n[0]))
        });
        if b["type"] != "text"
            || ls.len() < 3
            || ls.iter().any(|l| integer(l, "angle", 0).ok() != Some(0))
        {
            out.push(b);
            continue;
        }
        let hs = ls.iter().map(m::height).collect::<Result<Vec<_>, _>>()?;
        let em = median(hs.clone())?;
        if em <= 0.
            || hs.iter().any(|h| !(0.85 * em..=1.15 * em).contains(h))
            || ls.windows(2).any(|p| {
                !(-0.3 * em..=0.7 * em)
                    .contains(&(m::raw(&p[1]).unwrap().n[1] - m::raw(&p[0]).unwrap().n[3]))
            })
        {
            out.push(b);
            continue;
        }
        let re = regex(r"^\s*([a-z])[.)]\s+\S");
        let letters = ls
            .iter()
            .map(|l| {
                re.captures(l["text"].as_str().unwrap())
                    .map(|c| c[1].as_bytes()[0])
            })
            .collect::<Vec<_>>();
        w.charge(paths.len() * ls.len())?;
        let dots = ls
            .iter()
            .map(|l| bullet(l, paths, em))
            .collect::<Result<Vec<_>, _>>()?;
        let mut starts = letters
            .iter()
            .enumerate()
            .filter(|(_, x)| x.is_some())
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        let lettered = starts.len() >= 3
            && starts[0] == 0
            && starts
                .windows(2)
                .all(|p| letters[p[1]].unwrap() == letters[p[0]].unwrap() + 1);
        if !lettered {
            starts = dots
                .iter()
                .enumerate()
                .filter(|(_, x)| x.is_some())
                .map(|(i, _)| i)
                .collect()
        }
        if starts.len() < 3
            || starts[0] != 0
            || starts.iter().any(|&i| {
                (m::raw(&ls[i]).unwrap().n[0] - m::raw(&ls[0]).unwrap().n[0]).abs() > 0.2 * em
            })
            || starts
                .iter()
                .any(|&i| !eq(&ls[i]["font_signature"], &ls[0]["font_signature"]))
        {
            out.push(b);
            continue;
        }
        let mut items = vec![];
        for (k, &start) in starts.iter().enumerate() {
            let end = starts.get(k + 1).copied().unwrap_or(ls.len());
            let members = &ls[start..end];
            let left = m::raw(&ls[start])?.n[0];
            if members
                .iter()
                .any(|l| !(-0.2 * em..=2. * em).contains(&(m::raw(l).unwrap().n[0] - left)))
            {
                break;
            }
            let mut boxes = members.iter().map(m::raw).collect::<Result<Vec<_>, _>>()?;
            if !lettered {
                boxes.push(dots[start].clone().unwrap())
            }
            let mut item = b.clone();
            item["type"] = json!("text");
            item["content"] = json!(join_text(
                &members
                    .iter()
                    .map(|l| l["text"].as_str().map(str::to_owned).ok_or_else(invalid))
                    .collect::<Result<Vec<_>, _>>()?
            )?);
            item["bbox"] = json!(combined(boxes.iter())?.raw);
            item["_text_lines"] = json!(members);
            let raw = members
                .iter()
                .map(m::raw)
                .map(|r| r.map(|b| b.raw))
                .collect::<Result<Vec<_>, _>>()?;
            item["_local_line_bboxes"] = json!(raw);
            item["_local_output_line_bboxes"] = json!(raw);
            item["_line_heights"] = json!(members
                .iter()
                .map(m::height)
                .collect::<Result<Vec<_>, _>>()?);
            item["_output_bbox_repaired"] = json!(false);
            items.push(item)
        }
        if items.len() != starts.len() {
            out.push(b);
            continue;
        }
        let boxes = items
            .iter()
            .map(|i| bbox(&i["bbox"]))
            .collect::<Result<Vec<_>, _>>()?;
        out.push(json!({"type":"list","content":"","bbox":combined(boxes.iter())?.raw,"angle":0}));
        out.extend(items)
    }
    Ok(out)
}
