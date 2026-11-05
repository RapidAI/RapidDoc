//! Existing grouped-page-footnote splitting, marker matching and bounding boxes.
use super::*;
use rapiddoc_native::text_boundary::python_digit;
use unicode_normalization::UnicodeNormalization;
fn digits(t: &str) -> bool {
    !t.is_empty() && t.chars().all(python_digit)
}
fn compact(s: &State, a: &Row, b: &Row) -> Result<bool, Error> {
    Ok(regex(r"^\s*\d{1,3}[.)]?\s*$").is_match(s.t(a)?)
        && regex(r"[A-Za-z㐀-鿿]").find_iter(s.t(b)?).count() >= 2)
}
fn limit_width(s: &State, r: &Row, h: f64, g: f64) -> Result<f64, Error> {
    let base = (1.5 * g).max(0.75 * h);
    Ok(
        if regex(r"^\d{1,3}$").is_match(s.t(r)?.trim_matches(python_space)) {
            base.max(1.25 * h).max(2.5 * g)
        } else {
            base
        },
    )
}
fn candidate(
    s: &State,
    a: &Row,
    b: &Row,
    h: f64,
    g: f64,
    left: f64,
    geometric: bool,
) -> Result<bool, Error> {
    let gap = b.b.n[0] - a.b.n[2];
    Ok(width(a) <= limit_width(s, a, h, g)?
        && a.b.n[0] - left <= 0.5 * h
        && gap >= 0.
        && gap
            <= (if digits(s.t(a)?.trim_matches(python_space)) {
                2.
            } else {
                1.5
            }) * h
        && b.b.n[0] - left >= (1.5 * width(a)).max((if geometric { 0.4 } else { 0.75 }) * h)
        && (width(b) >= (4. * width(a)).max(4. * g) || compact(s, a, b)?)
        && (cy(&a.b) - cy(&b.b)).abs() <= 0.4 * h
        && (!geometric || fonts(s.l(a), s.l(b))?))
}
fn markers(s: &State, rs: &[Row], h: f64, g: f64, left: f64) -> Result<Vec<(Row, Row)>, Error> {
    let mut groups: Vec<(Value, Vec<Row>)> = vec![];
    for r in rs {
        let l = s.l(r);
        if l["visual_row_id"].is_null() || l["split_from_row"] != true {
            continue;
        }
        if let Some((_, rows)) = groups.iter_mut().find(|(v, _)| eq(v, &l["visual_row_id"])) {
            rows.push(r.clone())
        } else {
            groups.push((l["visual_row_id"].clone(), vec![r.clone()]))
        }
    }
    let mut out = vec![];
    for (_, mut rows) in groups {
        rows.sort_by(|a, b| a.b.n[0].total_cmp(&b.b.n[0]));
        if rows.len() >= 2 && candidate(s, &rows[0], &rows[1], h, g, left, false)? {
            out.push((rows[0].clone(), rows[1].clone()))
        }
    }
    out.sort_by(|(a, b), (c, d)| {
        a.b.n[1]
            .min(b.b.n[1])
            .total_cmp(&c.b.n[1].min(d.b.n[1]))
            .then(a.b.n[0].total_cmp(&c.b.n[0]))
    });
    if !out.is_empty() {
        return Ok(out);
    }
    let mut cs = vec![];
    for a in rs {
        if width(a) > limit_width(s, a, h, g)? || a.b.n[0] - left > 0.5 * h {
            continue;
        }
        for b in rs {
            if a.index != b.index && candidate(s, a, b, h, g, left, true)? {
                cs.push((a.clone(), b.clone()));
            }
        }
    }
    cs.sort_by(|(a, b), (c, d)| {
        a.b.n[1]
            .min(b.b.n[1])
            .total_cmp(&c.b.n[1].min(d.b.n[1]))
            .then((b.b.n[0] - a.b.n[2]).total_cmp(&(d.b.n[0] - c.b.n[2])))
            .then(s.id(a).unwrap().cmp(&s.id(c).unwrap()))
            .then(s.id(b).unwrap().cmp(&s.id(d).unwrap()))
    });
    let mut consumed = BTreeSet::new();
    for (a, b) in cs {
        let (ai, bi) = (s.id(&a)?, s.id(&b)?);
        if consumed.contains(&ai) || consumed.contains(&bi) {
            continue;
        }
        consumed.insert(ai);
        consumed.insert(bi);
        out.push((a, b))
    }
    Ok(out)
}
fn unmarked(
    s: &State,
    rs: &[Row],
    left: f64,
    max_width: f64,
    indent: f64,
) -> Result<Vec<Vec<Row>>, Error> {
    let Some(first) = rs.first() else {
        return Ok(vec![]);
    };
    let indented = first.b.n[0] - left > indent;
    let mut out = vec![vec![first.clone()]];
    for p in rs.windows(2) {
        let (a, b) = (&p[0], &p[1]);
        let compact = !indented
            && width(a) >= 0.7 * max_width
            && (b.b.n[0] - a.b.n[0]).abs() <= 0.5 * indent
            && fonts(s.l(a), s.l(b))?
            && s.gap(a, b)? <= indent;
        let current = b.b.n[0] - left > indent;
        if width(a) >= 0.8 * max_width || compact || if indented { !current } else { current } {
            out.last_mut().unwrap().push(b.clone())
        } else {
            out.push(vec![b.clone()])
        }
    }
    Ok(out)
}
fn marked(
    s: &State,
    rs: &[Row],
    ms: &[(Row, Row)],
    h: f64,
    g: f64,
    left: f64,
    max_width: f64,
    indent: f64,
) -> Result<Vec<Vec<Row>>, Error> {
    let first = ms[0].0.b.n[1].min(ms[0].1.b.n[1]);
    let mut out = unmarked(
        s,
        &rs.iter()
            .filter(|r| r.b.n[1] < first)
            .cloned()
            .collect::<Vec<_>>(),
        left,
        max_width,
        indent,
    )?;
    let mut used = out
        .iter()
        .flatten()
        .map(|r| s.id(r))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let tol = (0.75 * h).max(2. * g);
    for (i, (a, b)) in ms.iter().enumerate() {
        let next = ms
            .get(i + 1)
            .map_or(f64::INFINITY, |(a, b)| a.b.n[1].min(b.b.n[1]));
        let mut entry = vec![a.clone(), b.clone()];
        used.insert(s.id(a)?);
        used.insert(s.id(b)?);
        let mut prev = b;
        for r in rs {
            if used.contains(&s.id(r)?)
                || r.b.n[1] < b.b.n[1]
                || r.b.n[1] >= next
                || (r.b.n[0] - b.b.n[0]).abs() > tol
                || s.gap(prev, r)? > 1.5 * h
            {
                continue;
            }
            entry.push(r.clone());
            used.insert(s.id(r)?);
            prev = r;
        }
        s.ordered(&mut entry, true);
        out.push(entry)
    }
    out.extend(unmarked(
        s,
        &rs.iter()
            .filter(|r| !used.contains(&s.id(r).unwrap()))
            .cloned()
            .collect::<Vec<_>>(),
        left,
        max_width,
        indent,
    )?);
    // Original key searches every geometry row sharing each source, including duplicates.
    let top = |entry: &Vec<Row>| {
        rs.iter()
            .filter(|r| entry.iter().any(|x| s.id(x).unwrap() == s.id(r).unwrap()))
            .map(|r| r.b.n[1])
            .fold(f64::INFINITY, f64::min)
    };
    out.sort_by(|a, b| top(a).total_cmp(&top(b)));
    Ok(out)
}
fn split(s: &State, raw: &[Row], depth: usize) -> Result<Vec<Vec<Row>>, Error> {
    if raw.is_empty() {
        return Ok(vec![]);
    }
    if depth > rapiddoc_core::resources::limit(4096) {
        return Err(limit());
    }
    let mut rs = raw.to_vec();
    s.ordered(&mut rs, true);
    if rs.len() == 1 {
        return Ok(vec![rs]);
    }
    let h = median(rs.iter().map(|r| s.h(r)).collect::<Result<Vec<_>, _>>()?)?;
    let gs = rs
        .iter()
        .filter_map(|r| s.l(r).get("median_glyph_width").filter(|n| !n.is_null()))
        .map(num)
        .collect::<Result<Vec<_>, _>>()?;
    let g = if gs.is_empty() { 0.5 * h } else { median(gs)? };
    let indent = (1.5 * h).max(2. * g);
    let left = rs.iter().map(|r| r.b.n[0]).fold(f64::INFINITY, f64::min);
    let max_width = rs.iter().map(width).fold(f64::NEG_INFINITY, f64::max);
    let role = s
        .t(&rs[0])?
        .nfkc()
        .filter(|&c| !python_space(c) && c != ':' && c != '：')
        .flat_map(char::to_lowercase)
        .collect::<String>();
    let heading = [
        "affiliations",
        "authoraffiliations",
        "作者单位",
        "作者机构",
        "机构信息",
    ]
    .contains(&role.as_str());
    let body = if heading { &rs[1..] } else { &rs[..] };
    let text = body
        .iter()
        .map(|r| s.t(r))
        .collect::<Result<Vec<_>, _>>()?
        .join(" ");
    let separate = body
        .iter()
        .filter(|r| {
            regex(r"^\s*\d+[.)]\s+").is_match(s.t(r).unwrap()) && (r.b.n[0] - left).abs() <= 0.5 * h
        })
        .count();
    if separate >= 2 {
        let mut starts = rs
            .iter()
            .enumerate()
            .filter(|(_, r)| {
                regex(r"^\s*\d+[.)]\s+").is_match(s.t(r).unwrap())
                    && (r.b.n[0] - left).abs() <= 0.5 * h
            })
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        starts[0] = 0;
        starts.push(rs.len());
        return Ok(starts.windows(2).map(|p| rs[p[0]..p[1]].to_vec()).collect());
    }
    let mut continuous = true;
    for p in body.windows(2) {
        continuous &= s.gap(&p[0], &p[1])? <= 1.5 * h;
    }
    if regex(r"(?i)department|university|institute|hospital|centre|center|大学|学院|研究所|医院")
        .is_match(&fold(&text))
        && continuous
        && (heading || regex(r"[;；]\s*\d+\s*\D").is_match(&text))
    {
        return Ok(vec![rs]);
    }
    let mut starts = vec![];
    for (i, r) in rs.iter().enumerate() {
        if s.l(r)["footnote_marker_start"] == true
            || super::super::native_page_preparation::leading_footnote_superscript(s.l(r))?
        {
            starts.push(i)
        }
    }
    if !starts.is_empty() && (starts.len() >= 2 || starts[0] > 0) {
        let mut prefix = split(s, &rs[..starts[0]], depth + 1)?;
        starts.push(rs.len());
        prefix.extend(starts.windows(2).map(|p| rs[p[0]..p[1]].to_vec()));
        return Ok(prefix);
    }
    let ms = markers(s, &rs, h, g, left)?;
    if ms.is_empty() {
        unmarked(s, &rs, left, max_width, indent)
    } else {
        marked(s, &rs, &ms, h, g, left, max_width, indent)
    }
}
fn order(s: &State, raw: &[Row]) -> Result<Vec<Row>, Error> {
    let mut rs = raw.to_vec();
    s.ordered(&mut rs, false);
    let h = median(rs.iter().map(|r| s.h(r)).collect::<Result<Vec<_>, _>>()?)?;
    let mut rows: Vec<Vec<Row>> = vec![];
    for r in rs {
        let same = if let Some(row) = rows.last() {
            (cy(&r.b) - median(row.iter().map(|r| cy(&r.b)).collect())?).abs() <= 0.5 * h
        } else {
            false
        };
        if same {
            rows.last_mut().unwrap().push(r)
        } else {
            rows.push(vec![r])
        }
    }
    for row in &mut rows {
        row.sort_by(|a, b| a.b.n[0].total_cmp(&b.b.n[0]))
    }
    Ok(rows.into_iter().flatten().collect())
}
fn tight_map(s: &State, raw: &[Row], angle: i64) -> Result<BTreeMap<i64, Box4>, Error> {
    let mut rs = raw.to_vec();
    s.ordered(&mut rs, true);
    if rs.is_empty() {
        return Ok(BTreeMap::new());
    }
    let h = median(rs.iter().map(|r| s.h(r)).collect::<Result<Vec<_>, _>>()?)?;
    let ps = rs
        .windows(2)
        .map(|p| p[1].b.n[1] - p[0].b.n[1])
        .filter(|&n| n > 0.25 * h)
        .collect::<Vec<_>>();
    let pitch = if ps.is_empty() {
        None
    } else {
        Some(median(ps)?)
    };
    let mut out = BTreeMap::new();
    for r in rs {
        let mut h = (r.b.n[3] - r.b.n[1]).min(s.h(&r)?);
        if let Some(p) = pitch {
            h = h.min(0.9 * p)
        }
        h = h.max(0.1);
        let c = cy(&r.b);
        let b = bbox(&json!([r.b.n[0], c - 0.5 * h, r.b.n[2], c + 0.5 * h]))?;
        let p = if angle == 90 || angle == 270 {
            [s.page[1], s.page[0]]
        } else {
            s.page
        };
        out.insert(s.id(&r)?, local(&b, p, (360 - angle) % 360));
    }
    Ok(out)
}
pub(super) fn build(s: &State, groups: &[Value]) -> Result<(Vec<Value>, BTreeSet<i64>), Error> {
    let mut by_source = BTreeMap::new();
    for (i, l) in s.lines.iter().enumerate() {
        if l["semantic_type"] == "page_footnote" {
            by_source.insert(integer(l, "source_index", 0)?, i);
        }
    }
    let (mut blocks, mut consumed) = (vec![], BTreeSet::new());
    for group in groups {
        let ids = array(group, 4096)?
            .iter()
            .map(|v| {
                v.as_i64()
                    .or_else(|| v.as_bool().map(i64::from))
                    .ok_or_else(invalid)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if ids
            .iter()
            .any(|v| v.unsigned_abs() > rapiddoc_core::resources::limit(1_000_000_000_000))
        {
            return Err(limit());
        }
        let ids = super::super::table_native_vector::python_set::ordered_signed(&ids);
        let indexes = ids
            .iter()
            .filter_map(|id| by_source.get(id).copied())
            .collect::<Vec<_>>();
        let angles = indexes
            .iter()
            .map(|&i| integer(&s.lines[i], "angle", 0))
            .collect::<Result<BTreeSet<_>, _>>()?;
        for angle in angles {
            let rs = indexes
                .iter()
                .copied()
                .filter(|&i| integer(&s.lines[i], "angle", 0).ok() == Some(angle))
                .map(|index| {
                    Ok(Row {
                        index,
                        b: local(&bbox(&s.lines[index]["bbox"])?, s.page, angle),
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?;
            let tight_boxes = tight_map(s, &rs, angle)?;
            for entry in split(s, &rs, 0)? {
                let rs = order(s, &entry)?;
                let content = join_text(
                    &rs.iter()
                        .map(|r| s.t(r).map(str::to_owned))
                        .collect::<Result<Vec<_>, _>>()?,
                )?;
                if content.is_empty() {
                    continue;
                }
                let visual = block_common::numeric_set(
                    rs.iter()
                        .filter(|r| !s.l(r)["visual_row_id"].is_null())
                        .map(|r| s.l(r)["visual_row_id"].clone()),
                )?;
                let single = if rs.len() == 1 && s.l(&rs[0])["split_from_row"] == true {
                    s.l(&rs[0])["visual_row_id"].clone()
                } else {
                    Value::Null
                };
                let bs = rs
                    .iter()
                    .map(|r| Ok(tight_boxes.get(&s.id(r)?).unwrap().clone()))
                    .collect::<Result<Vec<_>, Error>>()?;
                let inline = rs
                    .iter()
                    .map(|r| optional(s.l(r), "inline_math_regions", 1000))
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>();
                let mut b = json!({"type":"page_footnote","bbox":combined(bs.iter())?.raw,"angle":angle,"content":content,"_visual_row_ids":visual,"_single_run_row_id":single,"_inline_math_regions":inline});
                let mut any = false;
                let mut bs = vec![];
                for r in &rs {
                    let candidate = tight(s.l(r), s.page)?;
                    any |= candidate.is_some();
                    bs.push(candidate.unwrap_or(s.raw(r)?));
                    consumed.insert(s.id(r)?);
                }
                if any {
                    b["_tight_output_bbox"] = json!(combined(bs.iter())?.raw)
                }
                blocks.push(b);
            }
        }
    }
    Ok((blocks, consumed))
}
