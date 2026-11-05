use super::*;
pub(super) fn intersects(a: &Box4, b: &Box4) -> bool {
    a.n[2].min(b.n[2]) > a.n[0].max(b.n[0]) && a.n[3].min(b.n[3]) > a.n[1].max(b.n[1])
}
fn same(a: &Box4, ah: f64, b: &Box4, bh: f64) -> bool {
    same_with_gap(a, ah, b, bh, 3f64.max(0.75 * ah.max(bh)))
}
pub(super) fn same_with_gap(a: &Box4, ah: f64, b: &Box4, bh: f64, gap: f64) -> bool {
    let h = ah.max(bh);
    if ah.min(bh) <= 0. || h / ah.min(bh) > 1.35 {
        return false;
    }
    let overlap = (a.n[3].min(b.n[3]) - a.n[1].max(b.n[1])).max(0.);
    if overlap / (a.n[3] - a.n[1]).min(b.n[3] - b.n[1]).max(0.1) < 0.7
        && (a.n[3] - b.n[3]).abs() > 0.25 * h
    {
        return false;
    }
    let (l, r) = if a.n[0] <= b.n[0] { (a, b) } else { (b, a) };
    (-0.25 * h..=gap).contains(&(r.n[0] - l.n[2]))
}
fn consecutive(a: &Value, b: &Value) -> Result<bool, Error> {
    if a["angle"] != 0
        || b["angle"] != 0
        || a["baseline"].is_null()
        || b["baseline"].is_null()
        || a["preserve_split_boundary"] == true
        || b["preserve_split_boundary"] == true
        || optional(a, "chars", 20000)?.is_empty()
        || optional(b, "chars", 20000)?.is_empty()
        || a["source_bbox"].is_null()
        || b["source_bbox"].is_null()
    {
        return Ok(false);
    }
    let ab = bbox(&a["source_bbox"])?;
    let bb = bbox(&b["source_bbox"])?;
    let ah = ab.n[3] - ab.n[1];
    let bh = bb.n[3] - bb.n[1];
    if (num(&a["baseline"])? - num(&b["baseline"])?).abs() > 0.25 * ah.min(bh)
        || !same(&ab, ah, &bb, bh)
    {
        return Ok(false);
    }
    let (l, r) = if num(&a["bbox"][0])? <= num(&b["bbox"][0])? {
        (a, b)
    } else {
        (b, a)
    };
    let indices = |l: &Value| -> Result<Option<Vec<i64>>, Error> {
        let mut ids = Vec::new();
        for c in optional(l, "chars", 20000)? {
            if c.get("source_indices").is_some() {
                for v in array(&c["source_indices"], 20000)? {
                    if let Some(i) = v.as_i64().or_else(|| v.as_bool().map(i64::from)) {
                        ids.push(i)
                    } else {
                        return Ok(None);
                    }
                }
            } else if let Some(i) = c["char_idx"]
                .as_i64()
                .or_else(|| c["char_idx"].as_bool().map(i64::from))
            {
                ids.push(i)
            } else {
                return Ok(None);
            }
        }
        Ok(Some(ids))
    };
    let (Some(l), Some(r)) = (indices(l)?, indices(r)?) else {
        return Ok(false);
    };
    Ok(!l.is_empty()
        && !r.is_empty()
        && (1..=2).contains(&(r.iter().min().unwrap() - l.iter().max().unwrap())))
}
fn can(a: &Value, ab: &Box4, b: &Value, bb: &Box4, tables: &[Box4]) -> Result<bool, Error> {
    if a["angle"] != b["angle"]
        || a["formula_candidate_only"] != b["formula_candidate_only"]
        || a["semantic_type"] != b["semantic_type"]
    {
        return Ok(false);
    }
    for (seed, other) in [(a, b), (b, a)] {
        let s = bbox(&seed["bbox"])?;
        let o = bbox(&other["bbox"])?;
        let h = height(seed, &s)?;
        if seed["caption_start"] == true && o.n[2] <= s.n[0] + 0.5 * h && o.n[0] < s.n[0] - h {
            return Ok(false);
        }
    }
    if eq(&a["visual_row_id"], &b["visual_row_id"])
        && (a["split_from_row"] == true || b["split_from_row"] == true)
    {
        return Ok(false);
    }
    let ah = height(a, ab)?;
    let bh = height(b, bb)?;
    let compatible = !a["font_signature"].is_null()
        && !b["font_signature"].is_null()
        && num(&a["font_coverage"])? >= 0.75
        && num(&b["font_coverage"])? >= 0.75
        && eq(&a["font_signature"], &b["font_signature"]);
    let h = ah.max(bh);
    let (l, r) = if ab.n[0] <= bb.n[0] {
        (ab, bb)
    } else {
        (bb, ab)
    };
    let touching = (ab.n[3].min(bb.n[3]) - ab.n[1].max(bb.n[1])).max(0.)
        / (ab.n[3] - ab.n[1]).min(bb.n[3] - bb.n[1]).max(0.1)
        >= 0.7
        && ((ab.n[1] + ab.n[3] - bb.n[1] - bb.n[3]) * 0.5).abs() <= 0.5 * h
        && (-0.15 * h..=0.75).contains(&(r.n[0] - l.n[2]));
    if !(consecutive(a, b)? || compatible && same(ab, ah, bb, bh) || touching) {
        return Ok(false);
    }
    let a = bbox(&a["bbox"])?;
    let b = bbox(&b["bbox"])?;
    let (ax, ay) = ((a.n[0] + a.n[2]) * 0.5, (a.n[1] + a.n[3]) * 0.5);
    let (bx, by) = ((b.n[0] + b.n[2]) * 0.5, (b.n[1] + b.n[3]) * 0.5);
    let connector = bbox(&json!([
        ax.min(bx) - 0.1,
        ay.min(by) - 0.1,
        ax.max(bx) + 0.1,
        ay.max(by) + 0.1
    ]))?;
    Ok(!tables.iter().any(|t| intersects(&connector, t)))
}
fn inherit(out: &mut Value, members: &[Value]) -> Result<(), Error> {
    let mut visual = members.iter().collect::<Vec<_>>();
    visual.sort_by(|a, b| {
        a["bbox"][0]
            .as_f64()
            .unwrap()
            .total_cmp(&b["bbox"][0].as_f64().unwrap())
    });
    let label = crate::cached_regex!(r"^(?:[A-Za-z]|[0-9]{1,2})[.)]$");
    let mapped = |s: &str| {
        s.chars()
            .map(|c| python_decimal(c).map_or(c, |d| (b'0' + d) as char))
            .collect::<String>()
    };
    if visual.len() == 2
        && label.is_match(&mapped(text(visual[0])?.trim_matches(python_space)))
        && eq(&visual[0]["font_signature"], &visual[1]["font_signature"])
    {
        out["native_title_label_left"] = visual[1]["native_title_label_left"].clone()
    }
    for k in ["paragraph_group", "title_band_id"] {
        let v = &members[0][k];
        if !v.is_null() && members.iter().all(|l| eq(&l[k], v)) {
            out[k] = v.clone();
            if k == "title_band_id" {
                for f in ["structural_title", "explicit_section_title"] {
                    out[f] = json!(members.iter().all(|l| l[f] == true));
                }
            }
        }
    }
    out["title_suppressed"] = json!(members.iter().all(|l| l["title_suppressed"] == true));
    Ok(())
}
fn materialize(
    indices: &[usize],
    ls: &mut [Value],
    bs: &[Box4],
    p: [f64; 2],
    w: &mut Work,
) -> Result<Value, Error> {
    materialize_mode(indices, ls, bs, p, w, false)
}
pub(super) fn materialize_mode(
    indices: &[usize],
    ls: &mut [Value],
    bs: &[Box4],
    p: [f64; 2],
    w: &mut Work,
    dense: bool,
) -> Result<Value, Error> {
    let mut content = text(&ls[indices[0]])?.trim_matches(python_space).to_owned();
    for pair in indices.windows(2) {
        let widths = if dense {
            vec![]
        } else {
            pair.iter()
                .map(|&i| glyph_width(&mut ls[i], p, w))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
        };
        let width = if widths.is_empty() {
            1.
        } else {
            median(widths)
        };
        if bs[pair[1]].n[0] - bs[pair[0]].n[2] > if dense { 0. } else { 0.5f64.max(0.25 * width) } {
            content.push(' ')
        }
        content.push_str(text(&ls[pair[1]])?.trim_matches(python_space));
    }
    let members = indices.iter().map(|&i| ls[i].clone()).collect::<Vec<_>>();
    let mut out = native_lines::defaults();
    out["text"] = json!(content.trim_matches(python_space));
    out["bbox"] = json!(
        combined(
            members
                .iter()
                .map(|l| bbox(&l["bbox"]))
                .collect::<Result<Vec<_>, _>>()?
                .iter()
        )?
        .raw
    );
    out["source_bbox"] = json!(
        combined(
            members
                .iter()
                .map(|l| bbox(if l["source_bbox"].is_null() {
                    &l["bbox"]
                } else {
                    &l["source_bbox"]
                }))
                .collect::<Result<Vec<_>, _>>()?
                .iter()
        )?
        .raw
    );
    let ink = members
        .iter()
        .filter(|l| !l["ink_bbox"].is_null())
        .map(|l| bbox(&l["ink_bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    if !ink.is_empty() {
        out["ink_bbox"] = json!(combined(ink.iter())?.raw)
    }
    out["source_index"] = json!(members
        .iter()
        .map(|l| integer(l, "source_index", 0))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .min()
        .unwrap());
    out["angle"] = members[0]["angle"].clone();
    out["semantic_type"] = members[0]["semantic_type"].clone();
    out["run_index"] = json!(members
        .iter()
        .map(|l| integer(l, "run_index", 0))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .min()
        .unwrap());
    out["visual_row_id"] = members
        .iter()
        .filter(|l| !l["visual_row_id"].is_null())
        .map(|l| integer(l, "visual_row_id", 0))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .min()
        .map_or(Value::Null, |i| json!(i));
    for k in ["baseline", "dominant_font_weight", "median_glyph_width"] {
        let a = members
            .iter()
            .filter(|l| !l[k].is_null())
            .map(|l| num(&l[k]))
            .collect::<Result<Vec<_>, _>>()?;
        out[k] = if a.is_empty() {
            Value::Null
        } else {
            json!(median(a))
        };
    }
    out["effective_height"] = json!(median(
        members.iter().map(raw_em).collect::<Result<Vec<_>, _>>()?
    ));
    out["em_height"] = json!(median(
        members
            .iter()
            .map(|l| if num(&l["em_height"]).unwrap_or(0.) != 0. {
                num(&l["em_height"])
            } else {
                raw_em(l)
            })
            .collect::<Result<Vec<_>, _>>()?
    ));
    let mut body = 0;
    let first = &members[0];
    if members.len() >= 2
        && matches!(text(first)?.trim_matches(python_space), "•" | "●" | "▪")
        && num(&first["bbox"][2])? - num(&first["bbox"][0])?
            <= height(first, &bbox(&first["bbox"])?)?
    {
        let mut best = 1;
        for i in 2..members.len() {
            if text(&members[i])?
                .trim_matches(python_space)
                .chars()
                .count()
                > text(&members[best])?
                    .trim_matches(python_space)
                    .chars()
                    .count()
            {
                best = i
            }
        }
        let re = crate::cached_regex!(r"[A-Za-z]{2,}|[\u{3400}-\u{9fff}]");
        if re.find_iter(text(&members[best])?).count() >= 2 {
            body = best
        }
    }
    out["font_signature"] = members[body]["font_signature"].clone();
    out["font_coverage"] = if body != 0 {
        members[body]["font_coverage"].clone()
    } else {
        json!(members
            .iter()
            .map(|l| num(&l["font_coverage"]))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .reduce(f64::min)
            .unwrap())
    };
    out["chars"] = json!(members
        .iter()
        .flat_map(|l| l["chars"].as_array().unwrap())
        .collect::<Vec<_>>());
    for k in ["leading_emphasis_width", "leading_typography_width"] {
        out[k] = members[0][k].clone()
    }
    out["paragraph_terminal"] = members.last().unwrap()["paragraph_terminal"].clone();
    for k in [
        "caption_start",
        "paragraph_formula_context",
        "split_from_row",
        "preserve_split_boundary",
        "restored_inline_cluster",
        "compact_formula_cluster",
        "style_scale_repaired",
    ] {
        out[k] = json!(members.iter().any(|l| l[k] == true))
    }
    out["formula_candidate_only"] =
        json!(members.iter().all(|l| l["formula_candidate_only"] == true));
    if dense {
        out["run_index"] = json!(0);
        out["visual_row_id"] = members[0]["visual_row_id"].clone();
        out["split_from_row"] = json!(false);
        let h = median(
            indices
                .iter()
                .map(|&i| height(&ls[i], &bs[i]))
                .collect::<Result<Vec<_>, _>>()?,
        );
        out["effective_height"] = json!(h);
        out["em_height"] = json!(h);
    }
    let words = members
        .iter()
        .flat_map(|l| l["native_math_words"].as_array().unwrap())
        .map(|v| v.as_str().unwrap().to_owned())
        .collect::<BTreeSet<_>>();
    out["native_math_words"] = json!(words);
    out["inline_math_regions"] = json!(members
        .iter()
        .flat_map(|l| l["inline_math_regions"].as_array().unwrap())
        .collect::<Vec<_>>());
    native_lines::markers(&mut out);
    inherit(&mut out, &members)?;
    if !out["chars"].as_array().unwrap().is_empty() {
        let metrics = preparation::page_metrics(&out, p)?;
        out.as_object_mut()
            .unwrap()
            .extend(metrics.as_object().unwrap().clone());
    }
    Ok(out)
}
fn root(parents: &mut [usize], mut i: usize) -> usize {
    while parents[i] != i {
        parents[i] = parents[parents[i]];
        i = parents[i]
    }
    i
}
pub(super) fn merge(
    ls: &mut Vec<Value>,
    aliases: &mut Vec<Option<i64>>,
    p: [f64; 2],
    tables: &[Box4],
    w: &mut Work,
) -> Result<BTreeMap<i64, Value>, Error> {
    if ls.len() < 2 {
        return Ok(BTreeMap::new());
    }
    let bs = ls
        .iter()
        .map(|l| Ok(local(&bbox(&l["bbox"])?, p, integer(l, "angle", 0)?)))
        .collect::<Result<Vec<_>, Error>>()?;
    let mut parents = (0..ls.len()).collect::<Vec<_>>();
    // Original necessary vertical-interval filter, preserving i/j arbitration.
    // Dense buckets use bounded storage and stream the same superset instead.
    let mut intervals = Vec::new();
    let mut safe = true;
    let mut buckets = BTreeMap::<(i64, bool, Option<String>), Vec<usize>>::new();
    for (i, l) in ls.iter().enumerate() {
        let b = &bs[i];
        let h = height(l, b)?;
        let mut low = b.n[1].min(b.n[3] - 0.25 * h);
        let mut high = b.n[3] + 0.25 * h;
        if l["angle"] == 0 && !l["source_bbox"].is_null() {
            if let Ok(s) = bbox(&l["source_bbox"]) {
                let sh = s.n[3] - s.n[1];
                low = low.min(s.n[1]).min(s.n[3] - 0.25 * sh);
                high = high.max(s.n[3] + 0.25 * sh);
            } else {
                safe = false
            }
        }
        intervals.push((low, high));
        buckets
            .entry((
                integer(l, "angle", 0)?,
                l["formula_candidate_only"] == true,
                l["semantic_type"].as_str().map(str::to_owned),
            ))
            .or_default()
            .push(i);
    }
    let mut pairs = Some(vec![Vec::<usize>::new(); ls.len()]);
    let mut count = 0;
    let maximum = 4096usize.max(32 * ls.len());
    if safe {
        'buckets: for ids in buckets.values() {
            let mut ids = ids.clone();
            ids.sort_by(|&a, &b| intervals[a].0.total_cmp(&intervals[b].0).then(a.cmp(&b)));
            let mut active = Vec::<usize>::new();
            for i in ids {
                w.charge(active.len() + 1)?;
                active.retain(|&j| intervals[j].1 >= intervals[i].0);
                count += active.len();
                if count > maximum {
                    pairs = None;
                    break 'buckets;
                }
                for &j in &active {
                    let (a, b) = if i < j { (i, j) } else { (j, i) };
                    pairs.as_mut().unwrap()[a].push(b);
                }
                active.push(i);
            }
        }
    } else {
        pairs = None
    }
    if let Some(pairs) = &mut pairs {
        for p in pairs {
            p.sort_unstable();
        }
    }
    for i in 0..ls.len() {
        let candidates = if let Some(pairs) = &pairs {
            pairs[i].clone()
        } else {
            (i + 1..ls.len())
                .filter(|&j| {
                    ls[i]["angle"] == ls[j]["angle"]
                        && ls[i]["formula_candidate_only"] == ls[j]["formula_candidate_only"]
                        && ls[i]["semantic_type"] == ls[j]["semantic_type"]
                        && (!safe
                            || intervals[i].0 <= intervals[j].1 && intervals[j].0 <= intervals[i].1)
                })
                .collect::<Vec<_>>()
        };
        for j in candidates {
            w.charge(tables.len() + 1)?;
            if can(&ls[i], &bs[i], &ls[j], &bs[j], tables)? {
                let a = root(&mut parents, i);
                let b = root(&mut parents, j);
                if a != b {
                    parents[b] = a;
                }
            }
        }
    }
    let mut groups = Vec::<(usize, Vec<usize>)>::new();
    for i in 0..ls.len() {
        let r = root(&mut parents, i);
        if let Some(g) = groups.iter_mut().find(|g| g.0 == r) {
            g.1.push(i)
        } else {
            groups.push((r, vec![i]))
        }
    }
    let mut out = Vec::new();
    for (_, mut ids) in groups {
        if ids.len() == 1 {
            out.push((ls[ids[0]].clone(), aliases[ids[0]]))
        } else {
            ids.sort_by(|&i, &j| {
                bs[i].n[0]
                    .total_cmp(&bs[j].n[0])
                    .then(bs[i].n[1].total_cmp(&bs[j].n[1]))
                    .then(
                        ls[i]["source_index"]
                            .as_i64()
                            .cmp(&ls[j]["source_index"].as_i64()),
                    )
            });
            out.push((materialize(&ids, ls, &bs, p, w)?, None));
        }
    }
    out.sort_by(|(a, _), (b, _)| {
        let x = local(&bbox(&a["bbox"]).unwrap(), p, a["angle"].as_i64().unwrap());
        let y = local(&bbox(&b["bbox"]).unwrap(), p, b["angle"].as_i64().unwrap());
        a["angle"]
            .as_i64()
            .cmp(&b["angle"].as_i64())
            .then(x.n[1].total_cmp(&y.n[1]))
            .then(x.n[0].total_cmp(&y.n[0]))
            .then(a["source_index"].as_i64().cmp(&b["source_index"].as_i64()))
    });
    let mut updates = BTreeMap::new();
    for (l, id) in ls.iter().zip(aliases.iter()) {
        if let Some(id) = id {
            updates.insert(*id, l["median_glyph_width"].clone());
        }
    }
    let (a, b) = out.into_iter().unzip();
    *ls = a;
    *aliases = b;
    Ok(updates)
}
