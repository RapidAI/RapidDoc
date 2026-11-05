//! Original continuity member reconstruction, with source last-value/first-position semantics.
use super::block_common as c;
use super::*;
pub(super) fn lines(b: &Value) -> Result<&[Value], Error> {
    optional(b, "_text_lines", 4096)
}
pub(super) fn raw(l: &Value) -> Result<Box4, Error> {
    bbox(&l["bbox"])
}
pub(super) fn ink(l: &Value) -> Result<Box4, Error> {
    if l["ink_bbox"].is_null() {
        raw(l)
    } else {
        bbox(&l["ink_bbox"])
    }
}
pub(super) fn height(l: &Value) -> Result<f64, Error> {
    reference_height(l, &raw(l)?)
}
pub(super) fn text(b: &Value) -> &str {
    b["content"].as_str().unwrap_or("")
}
pub(super) fn reassemble(ls: &[Value], previous: &str) -> Result<String, Error> {
    let mut unique: Vec<(i64, Value)> = vec![];
    for l in ls {
        let id = integer(l, "source_index", 0)?;
        if let Some((_, v)) = unique.iter_mut().find(|(k, _)| *k == id) {
            *v = l.clone()
        } else {
            unique.push((id, l.clone()))
        }
    }
    let mut ls = unique.into_iter().map(|(_, l)| l).collect::<Vec<_>>();
    ls.sort_by(|a, b| {
        cy(&ink(a).unwrap())
            .total_cmp(&cy(&ink(b).unwrap()))
            .then(raw(a).unwrap().n[0].total_cmp(&raw(b).unwrap().n[0]))
    });
    let mut rows: Vec<Vec<Value>> = vec![];
    for l in ls {
        let b = ink(&l)?;
        let same = if let Some(row) = rows.last() {
            let center = median(
                row.iter()
                    .map(|l| ink(l).map(|b| cy(&b)))
                    .collect::<Result<Vec<_>, _>>()?,
            )?;
            let h = reference_height(&l, &b)?.max(median(
                row.iter().map(height).collect::<Result<Vec<_>, _>>()?,
            )?);
            (cy(&b) - center).abs() <= 0.55 * h
        } else {
            false
        };
        if same {
            rows.last_mut().unwrap().push(l)
        } else {
            rows.push(vec![l])
        }
    }
    let mut contents = vec![];
    for mut row in rows {
        row.sort_by(|a, b| raw(a).unwrap().n[0].total_cmp(&raw(b).unwrap().n[0]));
        let mut content = row[0]["text"].as_str().ok_or_else(invalid)?.to_owned();
        for p in row.windows(2) {
            let a = p[0]["text"].as_str().ok_or_else(invalid)?;
            let b = p[1]["text"].as_str().ok_or_else(invalid)?;
            if regex(r"^\[\d+\]").is_match(b)
                && a.chars().next_back().is_some_and(python_alnum)
                && raw(&p[0])?.n[2] >= raw(&p[1])?.n[0]
            {
                content.push_str(b)
            } else {
                content = join_text(&[content, b.to_owned()])?
            }
        }
        contents.push(content)
    }
    let rebuilt = join_text(&contents)?;
    let strip = |t: &str| t.chars().filter(|&c| !python_space(c)).collect::<String>();
    Ok(
        if !previous.is_empty() && strip(previous) == strip(&rebuilt) {
            previous.to_owned()
        } else {
            rebuilt
        },
    )
}
pub(super) fn merge(mut bs: Vec<Value>, p: [f64; 2], w: &mut Work) -> Result<Vec<Value>, Error> {
    let all = bs
        .iter()
        .map(lines)
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
        .cloned()
        .collect::<Vec<_>>();
    let angles = all
        .iter()
        .map(|l| integer(l, "angle", 0))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let mut layouts = BTreeMap::new();
    for a in angles {
        layouts.insert(a, super::super::native_layout::build(&all, p, a, &[], w)?);
    }
    let mut groups: Vec<(Value, Vec<usize>)> = vec![];
    for (i, b) in bs.iter().enumerate() {
        if b["_reference_group"].is_null() {
            continue;
        }
        if let Some((_, ids)) = groups
            .iter_mut()
            .find(|(k, _)| eq(k, &b["_reference_group"]))
        {
            ids.push(i)
        } else {
            groups.push((b["_reference_group"].clone(), vec![i]))
        }
    }
    let mut used = BTreeSet::new();
    for (_, ids) in groups {
        let mut b = c::merge(&bs, &ids, false)?;
        b["content"] = json!(reassemble(lines(&b)?, text(&b))?);
        bs[ids[0]] = b;
        used.extend(ids.into_iter().skip(1));
    }
    bs = bs
        .into_iter()
        .enumerate()
        .filter_map(|(i, b)| (!used.contains(&i)).then_some(b))
        .collect();
    loop {
        w.charge(bs.len() * bs.len())?;
        let mut found = None;
        for i in 0..bs.len() {
            let first = &bs[i];
            let fl = lines(first)?;
            if first["type"] != "text" || fl.is_empty() {
                continue;
            }
            for j in i + 1..bs.len() {
                let second = &bs[j];
                let sl = lines(second)?;
                if second["type"] != "text" || sl.is_empty() {
                    continue;
                }
                let fb = bbox(&first["bbox"])?;
                let sb = bbox(&second["bbox"])?;
                let later = if sb.n[1] >= fb.n[1] { second } else { first };
                if truth(&later["_geometry_break_before"]) {
                    continue;
                }
                if truth(&later["_explicit_break_before"]) || truth(&later["_rule_break_before"]) {
                    let a = optional(first, "_visual_row_ids", 4096)?;
                    let b = optional(second, "_visual_row_ids", 4096)?;
                    if !a.iter().any(|a| b.iter().any(|b| eq(a, b))) {
                        continue;
                    }
                }
                let ft = text(first).trim_end_matches(python_space);
                let st = text(second);
                let lower = st.chars().next().is_some_and(char::is_lowercase);
                let continuation = ft.ends_with('-') && lower
                    || regex(r"^[•●▪]").is_match(ft) && ft.ends_with([':', '：']);
                let h = median(fl.iter().map(height).collect::<Result<Vec<_>, _>>()?)?;
                let gap = sb.n[1] - fb.n[3];
                let mut adjacent = continuation && (0.0..=1.5 * h).contains(&gap);
                adjacent |= regex(r"^\d{1,2}[.]\s").is_match(ft)
                    && (0.5 * h..=4. * h).contains(&(sb.n[0] - fb.n[0]))
                    && (0.0..=h).contains(&gap)
                    && second["_protected_hard_break_before"] != true;
                adjacent |= (!optional(first, "_inline_math_regions", 20000)?.is_empty()
                    || truth(&first["_paragraph_formula_context"])
                    || fl.iter().any(|l| truth(&l["paragraph_formula_context"])))
                    && lower
                    && !regex(r"[.!?。！？]$").is_match(ft)
                    && (fb.n[0] - sb.n[0]).abs() <= h
                    && (-h..=h).contains(&gap)
                    && second["_protected_hard_break_before"] != true;
                let (narrow, host, nb, hb) = if fb.n[2] - fb.n[0] <= sb.n[2] - sb.n[0] {
                    (first, second, &fb, &sb)
                } else {
                    (second, first, &sb, &fb)
                };
                let nt = text(narrow).trim_matches(python_space);
                let prefix = lines(narrow)?.len() == 1
                    && (1..=2).contains(&nt.chars().count())
                    && nt.chars().all(|c| {
                        python_alnum(c) && c != '_' && rapiddoc_native::python_decimal(c).is_none()
                    })
                    && nb.n[2] - nb.n[0] <= 1.5 * h
                    && (0.0..=1.5 * h).contains(&(hb.n[0] - nb.n[2]))
                    && (hb.n[1]..=hb.n[3]).contains(&cy(nb))
                    && optional(host, "_font_signatures", 4096)?.len() >= 2;
                if (!adjacent && fb.n[3].min(sb.n[3]) <= fb.n[1].max(sb.n[1]))
                    || (axis(&fb, &sb, 0) < 0.75 && !prefix)
                    || !eq(&first["_reference_group"], &second["_reference_group"])
                    || (!adjacent
                        && !eq(
                            &first["_hanging_indent_group"],
                            &second["_hanging_indent_group"],
                        ))
                {
                    continue;
                }
                let mut connected = adjacent;
                if !connected {
                    for a in fl {
                        for b in sl {
                            let angle = integer(a, "angle", 0)?;
                            if angle != integer(b, "angle", 0)? {
                                continue;
                            }
                            let ab = ink(a)?;
                            let bb = ink(b)?;
                            let h = reference_height(a, &ab)?.max(reference_height(b, &bb)?);
                            let separated = layouts.get(&angle).ok_or_else(invalid)?.separated(
                                &local(&ab, p, angle),
                                &local(&bb, p, angle),
                                w,
                            )?;
                            if separated && (ab.n[0] - bb.n[2]).max(bb.n[0] - ab.n[2]) > 0.5 * h {
                                continue;
                            }
                            if axis(&ab, &bb, 1) >= 0.25
                                && 0f64.max(ab.n[0] - bb.n[2]).max(bb.n[0] - ab.n[2]) <= 5. * h
                            {
                                connected = true;
                                break;
                            }
                        }
                        if connected {
                            break;
                        }
                    }
                }
                if connected {
                    found = Some((i, j));
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }
        let Some((i, j)) = found else { return Ok(bs) };
        let a = bbox(&bs[i]["bbox"])?;
        let b = bbox(&bs[j]["bbox"])?;
        let widths = [a.n[2] - a.n[0], b.n[2] - b.n[0]];
        let hs = lines(&bs[i])?
            .iter()
            .chain(lines(&bs[j])?)
            .map(height)
            .collect::<Result<Vec<_>, _>>()?;
        let mut merged = c::merge(&bs, &[i, j], false)?;
        if widths[0].min(widths[1]) < 0.4 * widths[0].max(widths[1])
            && hs.iter().copied().fold(f64::INFINITY, f64::min)
                < 0.85 * hs.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        {
            merged["_paragraph_formula_context"] = json!(true)
        }
        merged["content"] = json!(reassemble(lines(&merged)?, text(&merged))?);
        bs[i] = merged;
        bs.remove(j);
    }
}
pub(super) fn truth(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(a) => !a.is_empty(),
    }
}
