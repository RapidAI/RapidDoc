//! Existing Python metric, resolved-title and post-semantic text mergers.
use super::super::{native_layout, native_titles::reference_height as eff};
use super::*;
use unicode_normalization::UnicodeNormalization;
fn axis(a: &Box4, b: &Box4, k: usize) -> f64 {
    let d = (a.n[k + 2] - a.n[k]).min(b.n[k + 2] - b.n[k]);
    if d <= 0. {
        0.
    } else {
        (a.n[k + 2].min(b.n[k + 2]) - a.n[k].max(b.n[k])).max(0.) / d
    }
}
fn cy(b: &Box4) -> f64 {
    (b.n[1] + b.n[3]) * 0.5
}
fn geometries(ls: &[Value], p: [f64; 2]) -> Result<Vec<Box4>, Error> {
    ls.iter()
        .map(|l| Ok(local(&bbox(&l["bbox"])?, p, integer(l, "angle", 0)?)))
        .collect()
}
fn sort(ls: &mut [Value], p: [f64; 2]) -> Result<(), Error> {
    let mut keyed = ls
        .iter()
        .map(|l| {
            Ok((
                integer(l, "angle", 0)?,
                local(&bbox(&l["bbox"])?, p, integer(l, "angle", 0)?),
                integer(l, "source_index", 0)?,
                l.clone(),
            ))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    keyed.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then(a.1.n[1].total_cmp(&b.1.n[1]))
            .then(a.1.n[0].total_cmp(&b.1.n[0]))
            .then(a.2.cmp(&b.2))
    });
    for (out, (_, _, _, l)) in ls.iter_mut().zip(keyed) {
        *out = l
    }
    Ok(())
}
fn order(ids: &mut [usize], ls: &[Value], bs: &[Box4]) {
    ids.sort_by(|&i, &j| {
        bs[i].n[0]
            .total_cmp(&bs[j].n[0])
            .then(bs[i].n[1].total_cmp(&bs[j].n[1]))
            .then(
                integer(&ls[i], "source_index", 0)
                    .unwrap()
                    .cmp(&integer(&ls[j], "source_index", 0).unwrap()),
            )
    })
}
fn dense(
    ids: &[usize],
    ls: &mut [Value],
    bs: &[Box4],
    p: [f64; 2],
    w: &mut Work,
) -> Result<Value, Error> {
    let mut ids = ids.to_vec();
    order(&mut ids, ls, bs);
    baseline::materialize_mode(&ids, ls, bs, p, w, true)
}
fn family(a: &Value, b: &Value) -> Result<bool, Error> {
    if a.is_null() || b.is_null() {
        return Ok(false);
    }
    let get = |f: &Value| -> Result<Option<String>, Error> {
        Ok(preparation::family(&Some((
            f[0].as_str().ok_or_else(invalid)?.to_owned(),
            f[1].clone(),
        ))))
    };
    Ok(get(a)?.zip(get(b)?).is_some_and(|(a, b)| a == b))
}
fn two_run(
    ids: &[usize],
    ls: &[Value],
    bs: &[Box4],
    p: [f64; 2],
    sparse: bool,
) -> Result<bool, Error> {
    if ids.len() != 2
        || ids.iter().any(|&i| {
            ls[i]["split_from_row"] != true
                || ls[i]["preserve_split_boundary"] == true
                || !ls[i]["semantic_type"].is_null()
                || ls[i]["font_signature"].is_null()
        })
    {
        return Ok(false);
    }
    for &i in ids {
        if num(&ls[i]["font_coverage"])? < if sparse { 0.7 } else { 0.75 } {
            return Ok(false);
        }
    }
    let mut ordered = ids.to_vec();
    ordered.sort_by_key(|&i| integer(&ls[i], "run_index", 0).unwrap());
    if integer(&ls[ordered[0]], "run_index", 0)? != 0
        || integer(&ls[ordered[1]], "run_index", 0)? != 1
    {
        return Ok(false);
    }
    if sparse {
        if !family(
            &ls[ordered[0]]["font_signature"],
            &ls[ordered[1]]["font_signature"],
        )? {
            return Ok(false);
        }
    } else if !eq(
        &ls[ordered[0]]["font_signature"],
        &ls[ordered[1]]["font_signature"],
    ) {
        return Ok(false);
    }
    order(&mut ordered, ls, bs);
    let (i, j) = (ordered[0], ordered[1]);
    let ah = eff(&ls[i], &bs[i])?;
    let bh = eff(&ls[j], &bs[j])?;
    let h = ah.max(bh);
    if sparse {
        let width = if ls[i]["angle"] == 90 || ls[i]["angle"] == 270 {
            p[1]
        } else {
            p[0]
        };
        let gap = bs[j].n[0] - bs[i].n[2];
        Ok(bs[i].n[2] - bs[i].n[0] <= 2. * h
            && bs[j].n[2] - bs[j].n[0] >= 0.3 * width
            && 3. * h < gap
            && gap <= 5. * h
            && baseline::same_with_gap(&bs[i], ah, &bs[j], bh, 5. * h))
    } else {
        if !baseline::same_with_gap(&bs[i], ah, &bs[j], bh, 3. * h) {
            return Ok(false);
        }
        let b = combined([&bs[i], &bs[j]])?;
        Ok(
            ((bs[i].n[2] - bs[i].n[0]) + (bs[j].n[2] - bs[j].n[0])) / (b.n[2] - b.n[0]).max(0.1)
                >= 0.85,
        )
    }
}
fn metric(ls: &mut Vec<Value>, p: [f64; 2], w: &mut Work) -> Result<(), Error> {
    let bs = geometries(ls, p)?;
    let raw = ls
        .iter()
        .map(|l| bbox(&l["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let space = r"[\u{9}-\u{d}\u{1c}-\u{20}\u{85}\u{a0}\u{1680}\u{2000}-\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}]";
    let value = constant_regex!(&format!(
        r"\A\d+(?:[.,]\d+)?{space}*(?:[A-Za-z]{{1,4}}|[%×])\z"
    ));
    let arrow = constant_regex!(&format!(r"\A[↑↓↗↘]{space}*\d{{0,3}}\z"));
    let mut consumed = BTreeSet::new();
    let mut merged = Vec::new();
    for i in 0..ls.len() {
        let l = &ls[i];
        let id = integer(l, "source_index", 0)?;
        if consumed.contains(&id)
            || !(l["semantic_type"].is_null() || l["semantic_type"] == "paragraph_title")
            || l["caption_start"] == true
            || l["paragraph_terminal"] == true
            || l["font_signature"].is_null()
            || num(&l["font_coverage"])? < 0.75
            || l["formula_candidate_only"] == true
            || l["compact_formula_cluster"] == true
            || !value.is_match(text(l)?.trim_matches(python_space))
        {
            continue;
        }
        let h = eff(l, &bs[i])?;
        let mut arrows = Vec::new();
        for j in 0..ls.len() {
            w.charge(1)?;
            let l = &ls[j];
            if i == j
                || consumed.contains(&integer(l, "source_index", 0)?)
                || l["angle"] != ls[i]["angle"]
                || l["semantic_type"] != "paragraph_title"
                || !eq(&l["font_signature"], &ls[i]["font_signature"])
                || l["caption_start"] == true
                || l["formula_candidate_only"] == true
                || l["compact_formula_cluster"] == true
                || !arrow.is_match(text(l)?.trim_matches(python_space))
            {
                continue;
            }
            if (0.45 * h..=1.05 * h).contains(&eff(l, &bs[j])?)
                && (0.0..=0.75 * h).contains(&(bs[j].n[0] - bs[i].n[2]))
                && axis(&bs[i], &bs[j], 1) >= 0.7
            {
                arrows.push(j)
            }
        }
        if arrows.len() != 1 {
            continue;
        }
        let j = arrows[0];
        let mut peer = false;
        for k in 0..ls.len() {
            w.charge(1)?;
            let l = &ls[k];
            if i != k
                && j != k
                && l["angle"] == ls[i]["angle"]
                && l["semantic_type"] == "paragraph_title"
                && eq(&l["font_signature"], &ls[i]["font_signature"])
                && l["caption_start"] != true
                && (0.8 * h..=1.25 * h).contains(&eff(l, &raw[k])?)
                && (cy(&raw[k]) - cy(&raw[i])).abs() <= 0.35 * h
                && (raw[k].n[0] - raw[i].n[2])
                    .max(raw[i].n[0] - raw[k].n[2])
                    .max(0.)
                    >= 3. * h
            {
                peer = true;
                break;
            }
        }
        if !peer {
            continue;
        }
        let mut l = dense(&[i, j], ls, &bs, p, w)?;
        l["semantic_type"] = json!("paragraph_title");
        l["structural_title"] = json!(true);
        l["explicit_section_title"] = json!(true);
        l["title_band_id"] = json!(id);
        l["effective_height"] = json!(h);
        l["em_height"] = json!(h);
        merged.push(l);
        consumed.insert(id);
        consumed.insert(integer(&ls[j], "source_index", 0)?);
    }
    if !consumed.is_empty() {
        ls.retain(|l| !consumed.contains(&integer(l, "source_index", 0).unwrap()));
        ls.extend(merged)
    }
    Ok(())
}
fn numbered(ls: &mut Vec<Value>, p: [f64; 2], w: &mut Work) -> Result<(), Error> {
    let bs = geometries(ls, p)?;
    let ids = (0..ls.len())
        .filter(|&i| ls[i]["semantic_type"] == "paragraph_title")
        .collect::<Vec<_>>();
    let marker = crate::cached_regex!(r"\A\d+(?:\s*\.\s*\d+)*\.?\z");
    let mut consumed = BTreeSet::new();
    let mut merged = Vec::new();
    for &i in &ids {
        let id = integer(&ls[i], "source_index", 0)?;
        if consumed.contains(&id) {
            continue;
        }
        let t = text(&ls[i])?
            .nfkc()
            .collect::<String>()
            .split(python_space)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if !marker.is_match(&t) {
            continue;
        }
        let h = eff(&ls[i], &bs[i])?;
        let mut companions = Vec::new();
        for &j in &ids {
            w.charge(1)?;
            if i != j
                && !consumed.contains(&integer(&ls[j], "source_index", 0)?)
                && ls[i]["angle"] == ls[j]["angle"]
                && bs[j].n[0] >= bs[i].n[2]
                && bs[j].n[0] - bs[i].n[2] <= 4. * h
                && axis(&bs[i], &bs[j], 1) >= 0.5
            {
                companions.push(j)
            }
        }
        companions.sort_by(|&a, &b| {
            bs[a].n[0]
                .total_cmp(&bs[b].n[0])
                .then(bs[a].n[1].total_cmp(&bs[b].n[1]))
        });
        if let Some(&j) = companions.first() {
            merged.push(dense(&[i, j], ls, &bs, p, w)?);
            consumed.insert(id);
            consumed.insert(integer(&ls[j], "source_index", 0)?);
        }
    }
    ls.retain(|l| !consumed.contains(&integer(l, "source_index", 0).unwrap()));
    ls.extend(merged);
    sort(ls, p)
}
fn resolved(ls: &mut Vec<Value>, p: [f64; 2], w: &mut Work) -> Result<(), Error> {
    let bs = geometries(ls, p)?;
    let mut groups: Vec<((i64, i64), Vec<usize>)> = Vec::new();
    for (i, l) in ls.iter().enumerate() {
        if l["visual_row_id"].is_null() {
            continue;
        }
        let key = (integer(l, "angle", 0)?, integer(l, "visual_row_id", 0)?);
        if let Some((_, ids)) = groups.iter_mut().find(|(k, _)| *k == key) {
            ids.push(i)
        } else {
            groups.push((key, vec![i]))
        }
    }
    let mut consumed = BTreeSet::new();
    let mut merged = Vec::new();
    for (_, mut ids) in groups {
        if ids.len() < 2
            || ids.iter().any(|&i| {
                ls[i]["split_from_row"] != true || ls[i]["preserve_split_boundary"] == true
            })
        {
            continue;
        }
        let sem = ls[ids[0]]["semantic_type"].clone();
        if ids.iter().any(|&i| !eq(&ls[i]["semantic_type"], &sem)) {
            continue;
        }
        let mut fonts = Vec::new();
        for &i in &ids {
            if !fonts.iter().any(|f| eq(f, &ls[i]["font_signature"])) {
                fonts.push(ls[i]["font_signature"].clone())
            }
        }
        let dense_row = two_run(&ids, ls, &bs, p, false)?;
        let sparse = two_run(&ids, ls, &bs, p, true)?;
        if sem != "paragraph_title" && !(sem.is_null() && (fonts.len() > 1 || dense_row || sparse))
        {
            continue;
        }
        order(&mut ids, ls, &bs);
        let mut valid = true;
        for pair in ids.windows(2) {
            let ah = eff(&ls[pair[0]], &bs[pair[0]])?;
            let bh = eff(&ls[pair[1]], &bs[pair[1]])?;
            if !baseline::same_with_gap(
                &bs[pair[0]],
                ah,
                &bs[pair[1]],
                bh,
                (if sparse { 5. } else { 3. }) * ah.max(bh),
            ) {
                valid = false;
                break;
            }
        }
        if valid {
            merged.push(dense(&ids, ls, &bs, p, w)?);
            for i in ids {
                consumed.insert(integer(&ls[i], "source_index", 0)?);
            }
        }
    }
    ls.retain(|l| !consumed.contains(&integer(l, "source_index", 0).unwrap()));
    ls.extend(merged);
    sort(ls, p)?;
    numbered(ls, p, w)
}
fn post(ls: &mut Vec<Value>, p: [f64; 2], tables: &[Box4], w: &mut Work) -> Result<(), Error> {
    if ls.len() < 2 {
        return Ok(());
    }
    let bs = geometries(ls, p)?;
    let raw = ls
        .iter()
        .map(|l| bbox(&l["bbox"]))
        .collect::<Result<Vec<_>, _>>()?;
    let hs = ls
        .iter()
        .zip(&bs)
        .map(|(l, b)| eff(l, b))
        .collect::<Result<Vec<_>, _>>()?;
    let layout = native_layout::build(ls, p, 0, tables, w)?;
    let mut parents = (0..ls.len()).collect::<Vec<_>>();
    fn root(parents: &mut [usize], mut i: usize) -> usize {
        while parents[i] != i {
            parents[i] = parents[parents[i]];
            i = parents[i]
        }
        i
    }
    for i in 0..ls.len() {
        if !ls[i]["semantic_type"].is_null() {
            continue;
        }
        for j in i + 1..ls.len() {
            w.charge(tables.len() + 1)?;
            if !ls[j]["semantic_type"].is_null()
                || ls[i]["angle"] != ls[j]["angle"]
                || ls[i]["formula_candidate_only"] != ls[j]["formula_candidate_only"]
            {
                continue;
            }
            let mut caption_cross = false;
            for (a, b) in [(i, j), (j, i)] {
                if ls[a]["caption_start"] == true
                    && raw[b].n[2] <= raw[a].n[0] + 0.5 * eff(&ls[a], &raw[a])?
                    && raw[b].n[0] < raw[a].n[0] - eff(&ls[a], &raw[a])?
                {
                    caption_cross = true
                }
            }
            if caption_cross {
                continue;
            }
            let connector = bbox(&json!([
                ((raw[i].n[0] + raw[i].n[2]) * 0.5).min((raw[j].n[0] + raw[j].n[2]) * 0.5) - 0.1,
                cy(&raw[i]).min(cy(&raw[j])) - 0.1,
                ((raw[i].n[0] + raw[i].n[2]) * 0.5).max((raw[j].n[0] + raw[j].n[2]) * 0.5) + 0.1,
                cy(&raw[i]).max(cy(&raw[j])) + 0.1
            ]))?;
            if tables.iter().any(|t| baseline::intersects(&connector, t)) {
                continue;
            }
            let (a, b) = (&bs[i], &bs[j]);
            let h = hs[i].min(hs[j]);
            let (left, right) = if a.n[0] <= b.n[0] { (a, b) } else { (b, a) };
            let physical = !ls[i]["visual_row_id"].is_null()
                && eq(&ls[i]["visual_row_id"], &ls[j]["visual_row_id"]);
            let members = if physical {
                (0..ls.len())
                    .filter(|&k| {
                        ls[k]["angle"] == ls[i]["angle"]
                            && eq(&ls[k]["visual_row_id"], &ls[i]["visual_row_id"])
                    })
                    .collect::<Vec<_>>()
            } else {
                vec![]
            };
            let shared = physical
                && members.iter().all(|&k| {
                    text(&ls[k])
                        .unwrap()
                        .chars()
                        .any(rapiddoc_native::text_boundary::python_alpha)
                });
            let row = if shared {
                combined(members.iter().map(|&i| &bs[i]))?
            } else {
                combined([a, b])?
            };
            let mut justified = eq(&ls[i]["font_signature"], &ls[j]["font_signature"])
                && eq(&ls[i]["paragraph_group"], &ls[j]["paragraph_group"])
                && ls[i]["angle"] == 0
                && (cy(a) - cy(b)).abs() < 0.2 * h
                && (0.0..=(if shared { 3. } else { 2. }) * h).contains(&(right.n[0] - left.n[2]))
                && !layout.separated(a, b, w)?;
            if justified {
                let bottom = a.n[3].max(b.n[3]);
                let mut support = false;
                for k in 0..ls.len() {
                    w.charge(1)?;
                    if k != i
                        && k != j
                        && raw[k].n[1] - bottom > 0.
                        && raw[k].n[1] - bottom <= h
                        && (raw[k].n[0] - row.n[0]).abs() < (if shared { 1.5 } else { 1. }) * h
                        && raw[k].n[2] >= row.n[2] - h
                    {
                        support = true;
                        break;
                    }
                }
                justified = support
            }
            let max = hs[i].max(hs[j]);
            let y = (a.n[3].min(b.n[3]) - a.n[1].max(b.n[1])).max(0.);
            let same = hs[i].min(hs[j]) > 0.
                && max / hs[i].min(hs[j]) <= 1.6
                && y / (a.n[3] - a.n[1]).min(b.n[3] - b.n[1]).max(0.1) >= 0.7
                && (-0.2 * max..=2f64.max(0.35 * max)).contains(&(right.n[0] - left.n[2]));
            if same || justified {
                let a = root(&mut parents, i);
                let b = root(&mut parents, j);
                if a != b {
                    parents[b] = a
                }
            }
        }
    }
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for i in 0..ls.len() {
        let r = root(&mut parents, i);
        if let Some((_, ids)) = groups.iter_mut().find(|(k, _)| *k == r) {
            ids.push(i)
        } else {
            groups.push((r, vec![i]))
        }
    }
    let mut out = Vec::new();
    for (_, mut ids) in groups {
        if ids.len() == 1 {
            out.push(ls[ids[0]].clone())
        } else {
            order(&mut ids, ls, &bs);
            out.push(baseline::materialize_mode(&ids, ls, &bs, p, w, false)?);
        }
    }
    *ls = out;
    sort(ls, p)
}
pub(super) fn run(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let mut ls = array(&v["lines"], 4096)?.to_vec();
    let p = size(&v["page_size"])?;
    let tables = optional(v, "table_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    match v["kind"].as_str().ok_or_else(invalid)? {
        "metric" => metric(&mut ls, p, w)?,
        "resolved" => resolved(&mut ls, p, w)?,
        "post" => post(&mut ls, p, &tables, w)?,
        "sequence" => {
            metric(&mut ls, p, w)?;
            resolved(&mut ls, p, w)?;
            post(&mut ls, p, &tables, w)?
        }
        _ => return Err(invalid()),
    }
    let out = json!({"lines":ls});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        Err(limit())
    } else {
        Ok(out)
    }
}
