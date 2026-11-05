//! 原容器前边缘文字保护、相邻/奇偶页匹配及页眉分隔线确认。
mod separators;
mod text;
use super::{
    page_containers::{height, median, optional, size},
    path_graphics::{array, bbox, integer, num, overlap_first, Box4},
};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::python_space;
use serde_json::{json, Value};
use std::{borrow::Cow, collections::BTreeSet};
struct Page<'a> {
    source: Cow<'a, Value>,
    lines: Cow<'a, [Value]>,
    size: [f64; 2],
}
#[derive(Clone)]
struct Candidate {
    page: usize,
    line: usize,
    b: Box4,
    size: [f64; 2],
    region: &'static str,
}
pub(super) fn local(b: &Box4, p: [f64; 2], a: i64) -> Box4 {
    let n = match a {
        90 => [b.n[1], p[0] - b.n[2], b.n[3], p[0] - b.n[0]],
        180 => [p[0] - b.n[2], p[1] - b.n[3], p[0] - b.n[0], p[1] - b.n[1]],
        270 => [p[1] - b.n[3], b.n[0], p[1] - b.n[1], b.n[2]],
        _ => b.n,
    };
    Box4 {
        n,
        raw: n.map(|n| json!(n)),
    }
}
fn center(b: &Box4, a: usize) -> f64 {
    (b.n[a] + b.n[a + 2]) * 0.5
}
fn axis(a: &Box4, b: &Box4, k: usize) -> f64 {
    let d = (a.n[k + 2] - a.n[k]).min(b.n[k + 2] - b.n[k]);
    if d > 0. {
        (a.n[k + 2].min(b.n[k + 2]) - a.n[k].max(b.n[k])).max(0.) / d
    } else {
        0.
    }
}
fn txt(l: &Value) -> Result<&str, Error> {
    l["text"].as_str().ok_or_else(invalid)
}
fn candidate(page: usize, line: usize, p: &Page, w: &mut Work) -> Result<Option<Candidate>, Error> {
    let l = &p.lines[line];
    let semantic = l["semantic_type"].as_str();
    if !l["semantic_type"].is_null() && semantic != Some("page_footnote")
        || l["caption_start"] == true
        || l["marginal_body_context"] == true
    {
        return Ok(None);
    }
    if text::leader(txt(l)?, w)? {
        return Ok(None);
    }
    let a = integer(l, "angle", 0)?;
    let b = local(&bbox(&l["bbox"])?, p.size, a);
    let size = if [90, 270].contains(&a) {
        [p.size[1], p.size[0]]
    } else {
        p.size
    };
    if size[0] <= 0. || size[1] <= 0. {
        return Ok(None);
    }
    let y = center(&b, 1) / size[1];
    let x = center(&b, 0) / size[0];
    if semantic == Some("page_footnote") && y < 0.94 {
        return Ok(None);
    }
    let region = if y <= 0.15 {
        "header"
    } else if y >= 0.9 {
        "footer"
    } else if y <= 0.18 || y >= 0.82 || (x <= 0.15 || x >= 0.85) && (y <= 0.3 || y >= 0.7) {
        "side"
    } else {
        return Ok(None);
    };
    if height(l, &b)? > 0.06 * size[1] {
        return Ok(None);
    }
    Ok(Some(Candidate {
        page,
        line,
        b,
        size,
        region,
    }))
}
fn geometry(a: &Candidate, b: &Candidate, pages: &[Page]) -> Result<bool, Error> {
    let l = &pages[a.page].lines[a.line];
    let r = &pages[b.page].lines[b.line];
    if a.region != b.region
        || integer(l, "angle", 0)? != integer(r, "angle", 0)?
        || (center(&a.b, 1) / a.size[1] - center(&b.b, 1) / b.size[1]).abs() > 0.025
    {
        return Ok(false);
    }
    let ah = height(l, &a.b)? / a.size[1];
    let bh = height(r, &b.b)? / b.size[1];
    if ah.min(bh) <= 0. || ah.max(bh) / ah.min(bh) > 1.35 {
        return Ok(false);
    }
    if !l["font_signature"].is_null()
        && !r["font_signature"].is_null()
        && l.get("font_coverage").map_or(Ok(0.), num)? >= 0.75
        && r.get("font_coverage").map_or(Ok(0.), num)? >= 0.75
        && !crate::legacy_page::eq(&l["font_signature"], &r["font_signature"])
    {
        return Ok(false);
    }
    let normalize = |c: &Candidate| {
        let n = std::array::from_fn(|i| c.b.n[i] / c.size[i % 2]);
        Box4 {
            n,
            raw: n.map(|x| json!(x)),
        }
    };
    let ab = normalize(a);
    let bb = normalize(b);
    Ok(axis(&ab, &bb, 0) >= 0.4
        || (center(&ab, 0) - center(&bb, 0)).abs() <= 0.08
        || (center(&ab, 0) + center(&bb, 0) - 1.).abs() <= 0.12)
}
fn number_geometry(a: &Candidate, b: &Candidate, p: &[Page]) -> Result<bool, Error> {
    if geometry(a, b, p)? {
        return Ok(true);
    }
    let l = &p[a.page].lines[a.line];
    let r = &p[b.page].lines[b.line];
    if (a.size[0] > a.size[1]) == (b.size[0] > b.size[1])
        || integer(l, "angle", 0)? != integer(r, "angle", 0)?
    {
        return Ok(false);
    }
    let ah = height(l, &a.b)? / a.size[1];
    let bh = height(r, &b.b)? / b.size[1];
    Ok(ah.min(bh) > 0. && ah.max(bh) / ah.min(bh) <= 1.5)
}
fn protect(p: &mut Page, w: &mut Work) -> Result<(), Error> {
    let mut panels = Vec::new();
    for path in optional(&p.source, "path_infos", 5000)? {
        let b = bbox(&path["bbox"])?;
        if path["fill_visible"] != true || path["fill_rgba"].is_null() {
            continue;
        }
        let color = array(&path["fill_rgba"], 4)?;
        if color.is_empty() {
            continue;
        }
        if color.len() != 4 {
            return Err(invalid());
        }
        let c = color.iter().map(num).collect::<Result<Vec<_>, _>>()?;
        if c[3] < 128.
            || c[..3].iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - c[..3].iter().copied().fold(f64::INFINITY, f64::min)
                < 12.
            || b.n[1] < 0.025 * p.size[1]
            || (b.n[2] - b.n[0]) * (b.n[3] - b.n[1]) > 0.5 * p.size[0] * p.size[1]
        {
            continue;
        }
        panels.push(b)
    }
    let angles = p
        .lines
        .iter()
        .map(|l| integer(l, "angle", 0))
        .collect::<Result<BTreeSet<_>, _>>()?;
    for &angle in &angles {
        let h = if [90, 270].contains(&angle) {
            p.size[0]
        } else {
            p.size[1]
        };
        let mut members = Vec::new();
        for (i, l) in p.lines.iter().enumerate() {
            if integer(l, "angle", 0)? == angle {
                members.push((i, local(&bbox(&l["bbox"])?, p.size, angle)))
            }
        }
        let cached_heights = if crate::legacy_page::budget::current().host
            && crate::legacy_page::budget::current().work == usize::MAX
        {
            Some(
                members
                    .iter()
                    .map(|(i, b)| height(&p.lines[*i], b))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            None
        };
        for (member_index, (i, b)) in members.iter().enumerate() {
            let em = match &cached_heights {
                Some(values) => values[member_index],
                None => height(&p.lines[*i], b)?,
            };
            if em <= 0. || b.n[2] - b.n[0] > 8. * em || !(b.n[3] <= 0.18 * h || b.n[1] >= 0.82 * h)
            {
                continue;
            }
            for (peer_index, (j, peer)) in members.iter().enumerate() {
                w.charge(1)?;
                let peer_em = match &cached_heights {
                    Some(values) => values[peer_index],
                    None => height(&p.lines[*j], peer)?,
                };
                if i == j
                    || peer_em < 0.7 * em
                    || peer_em > 1.4 * em
                    || peer.n[1] - b.n[3] < 0.
                    || peer.n[1] - b.n[3] > 1.5 * em
                    || (peer.n[0] - b.n[0]).abs() > 2. * em
                    || peer.n[2] - peer.n[0] < (4. * em).max(1.8 * (b.n[2] - b.n[0]))
                {
                    continue;
                }
                w.charge(members.len() * 2 + panels.len() + 1)?;
                let outer = b.n[3] <= 0.18 * h
                    && members
                        .iter()
                        .any(|(k, a)| k != i && k != j && a.n[3] < b.n[1] - 1.5 * em);
                let continuation = members.iter().any(|(k, a)| {
                    k != i
                        && k != j
                        && a.n[1] - peer.n[3] >= 0.
                        && a.n[1] - peer.n[3] <= 1.5 * em
                        && (a.n[0] - peer.n[0]).abs() <= 2. * em
                        && a.n[2] - a.n[0] >= 4. * em
                });
                let label_box = bbox(&p.lines[*i]["bbox"])?;
                let peer_box = bbox(&p.lines[*j]["bbox"])?;
                let panel = panels.iter().any(|a| {
                    overlap_first(&label_box, a) >= 0.99
                        && overlap_first(&peer_box, a) >= 0.99
                        && a.n[3] - a.n[1] >= 4. * em
                });
                if outer || continuation || panel {
                    p.lines.to_mut()[*i]["marginal_body_context"] = json!(true);
                    break;
                }
            }
        }
    }
    for angle in angles {
        let (width, h) = if [90, 270].contains(&angle) {
            (p.size[1], p.size[0])
        } else {
            (p.size[0], p.size[1])
        };
        if width <= 0. || h <= 0. {
            continue;
        }
        let mut rows = Vec::new();
        for (i, l) in p.lines.iter().enumerate() {
            if integer(l, "angle", 0)? == angle && !txt(l)?.trim_matches(python_space).is_empty() {
                rows.push((i, local(&bbox(&l["bbox"])?, p.size, angle)))
            }
        }
        rows.sort_by(|a, b| {
            a.1.n[1]
                .total_cmp(&b.1.n[1])
                .then(a.1.n[0].total_cmp(&b.1.n[0]))
        });
        if rows.len() < 4 {
            continue;
        }
        let (i, b) = &rows[0];
        let (j, hb) = &rows[1];
        let l = &p.lines[*i];
        let heading = &p.lines[*j];
        let em = height(l, b)?;
        let hem = height(heading, hb)?;
        if !l["semantic_type"].is_null()
            || !heading["semantic_type"].is_null()
            || text::number(txt(l)?, w)?.is_some()
            || l["caption_start"] == true
            || heading["caption_start"] == true
            || em <= 0.
            || b.n[3] > 0.15 * h
            || b.n[2] - b.n[0] < 0.2 * width
            || b.n[2] - b.n[0] > 0.85 * width
            || (center(b, 0) - 0.5 * width).abs() > 0.04 * width
            || (center(hb, 0) - center(b, 0)).abs() > 2. * em
            || hb.n[2] - hb.n[0] > 0.8 * (b.n[2] - b.n[0])
            || hb.n[2] - hb.n[0] < 3. * hem
            || hem < 1.1 * em
            || hb.n[1] - b.n[3] < 0.5 * em
            || hb.n[1] - b.n[3] > 3. * em
        {
            continue;
        }
        let body = rows[2..]
            .iter()
            .filter(|(k, a)| {
                a.n[1] >= hb.n[3]
                    && a.n[1] < 0.85 * h
                    && a.n[2] - a.n[0] >= 0.35 * width
                    && p.lines[*k]["semantic_type"].is_null()
            })
            .collect::<Vec<_>>();
        if body.len() < 2 {
            continue;
        }
        let body_em = median(
            body.iter()
                .map(|(k, a)| height(&p.lines[*k], a))
                .collect::<Result<Vec<_>, _>>()?,
        );
        if body_em <= 0.
            || em < 0.95 * body_em
            || body[0].1.n[1] - hb.n[3] < 0.
            || body[0].1.n[1] - hb.n[3] > 4. * hem
        {
            continue;
        }
        let mut blocked = false;
        for r in optional(&p.source, "drawing_lines", 5000)? {
            w.charge(1)?;
            let rb = local(&bbox(&r["bbox"])?, p.size, angle);
            if rb.n[2] - rb.n[0] >= rb.n[3] - rb.n[1]
                && b.n[3] <= center(&rb, 1)
                && center(&rb, 1) <= hb.n[1]
                && rb.n[2] - rb.n[0] >= 0.6 * width
            {
                blocked = true;
                break;
            }
        }
        if !blocked {
            p.lines.to_mut()[*i]["marginal_body_context"] = json!(true);
            p.lines.to_mut()[*j]["marginal_body_context"] = json!(true)
        }
    }
    Ok(())
}
fn classify(candidates: &[Candidate], pages: &mut [Page], w: &mut Work) -> Result<(), Error> {
    for (i, a) in candidates.iter().enumerate() {
        let Some(n) = text::number(txt(&pages[a.page].lines[a.line])?, w)? else {
            continue;
        };
        for b in &candidates[i + 1..] {
            let delta = b.page as i64 - a.page as i64;
            if delta > 2 {
                break;
            }
            w.charge(1)?;
            if delta > 0 {
                if let Some(m) = text::number(txt(&pages[b.page].lines[b.line])?, w)? {
                    if text::increment(&n, delta as u8) == m && number_geometry(a, b, pages)? {
                        pages[a.page].lines.to_mut()[a.line]["semantic_type"] =
                            json!("page_number");
                        pages[b.page].lines.to_mut()[b.line]["semantic_type"] = json!("page_number")
                    }
                }
            }
        }
    }
    for (i, a) in candidates.iter().enumerate() {
        if pages[a.page].lines[a.line]["semantic_type"] == "page_number"
            || text::number(txt(&pages[a.page].lines[a.line])?, w)?.is_some()
        {
            continue;
        }
        for b in &candidates[i + 1..] {
            let delta = b.page as i64 - a.page as i64;
            if delta > 2 {
                break;
            }
            w.charge(1)?;
            if delta > 0
                && a.region != "side"
                && b.region != "side"
                && pages[b.page].lines[b.line]["semantic_type"] != "page_number"
                && text::number(txt(&pages[b.page].lines[b.line])?, w)?.is_none()
                && geometry(a, b, pages)?
                && text::matches(
                    txt(&pages[a.page].lines[a.line])?,
                    txt(&pages[b.page].lines[b.line])?,
                    w,
                )?
            {
                pages[a.page].lines.to_mut()[a.line]["semantic_type"] = json!(a.region);
                pages[b.page].lines.to_mut()[b.line]["semantic_type"] = json!(b.region)
            }
        }
    }
    Ok(())
}
fn pages(v: &Value) -> Result<Vec<Page<'_>>, Error> {
    pages_from_sources(array(&v["sources"], 1000)?)
}
fn pages_from_sources(sources: &[Value]) -> Result<Vec<Page<'_>>, Error> {
    sources
        .iter()
        .map(|s| {
            let lines = array(&s["lines"], 4096)?;
            for l in lines {
                bbox(&l["bbox"])?;
                integer(l, "angle", 0)?;
                if txt(l)?.chars().count() > rapiddoc_core::resources::limit(20000) {
                    return Err(limit());
                }
            }
            Ok(Page {
                source: Cow::Borrowed(s),
                lines: Cow::Borrowed(lines),
                size: size(&s["page_size"])?,
            })
        })
        .collect()
}
fn sources(pages: &[Page]) -> Vec<Value> {
    pages
        .iter()
        .map(|p| {
            let mut s = (*p.source).clone();
            s["lines"] = json!(p.lines);
            s
        })
        .collect()
}
pub fn prepare(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_with_work(v, &mut Work(0))
}
pub(super) fn prepare_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let protected = protect_sources(v, w)?;
    let separators = detect_separators(&protected, w)?;
    let out = json!({"sources":protected,"header_separator_bboxes":separators});
    if crate::legacy_page::budget::output_exceeds(&out) {
        Err(limit())
    } else {
        Ok(out)
    }
}
pub(super) fn repeat_prepared_in_place(raw: &mut [Value], w: &mut Work) -> Result<(), Error> {
    let source = Value::Null;
    let mut pages = Vec::with_capacity(raw.len());
    for p in raw.iter_mut() {
        let size = size(p.get("page_size").unwrap_or(&p["source"]["page_size"]))?;
        array(&p["remaining_lines"], 4096)?;
        let Value::Array(lines) = p["remaining_lines"].take() else {
            return Err(invalid());
        };
        for l in &lines {
            bbox(&l["bbox"])?;
            integer(l, "angle", 0)?;
            if txt(l)?.chars().count() > rapiddoc_core::resources::limit(20000) {
                return Err(limit());
            }
        }
        pages.push(Page {
            source: Cow::Borrowed(&source),
            lines: Cow::Owned(lines),
            size,
        });
    }
    let mut candidates = Vec::new();
    for (i, p) in pages.iter().enumerate() {
        for j in 0..p.lines.len() {
            if let Some(c) = candidate(i, j, p, w)? {
                candidates.push(c)
            }
        }
    }
    classify(&candidates, &mut pages, w)?;
    for (p, c) in raw.iter_mut().zip(pages) {
        p["remaining_lines"] = Value::Array(c.lines.into_owned())
    }
    Ok(())
}
fn protect_sources(v: &Value, w: &mut Work) -> Result<Vec<Value>, Error> {
    let mut pages = pages(v)?;
    protect_pages(&mut pages, w)?;
    Ok(sources(&pages))
}
fn protect_sources_owned(sources: Vec<Value>, w: &mut Work) -> Result<Vec<Value>, Error> {
    let mut pages = Vec::with_capacity(sources.len());
    for mut source in sources {
        array(&source["lines"], 4096)?;
        let Value::Array(lines) = source["lines"].take() else {
            return Err(invalid());
        };
        for l in &lines {
            bbox(&l["bbox"])?;
            integer(l, "angle", 0)?;
            if txt(l)?.chars().count() > rapiddoc_core::resources::limit(20000) {
                return Err(limit());
            }
        }
        let size = size(&source["page_size"])?;
        pages.push(Page {
            source: Cow::Owned(source),
            lines: Cow::Owned(lines),
            size,
        });
    }
    protect_pages(&mut pages, w)?;
    Ok(pages
        .into_iter()
        .map(|page| {
            let mut source = page.source.into_owned();
            source["lines"] = Value::Array(page.lines.into_owned());
            source
        })
        .collect())
}
fn protect_pages(pages: &mut [Page<'_>], w: &mut Work) -> Result<(), Error> {
    for p in pages.iter_mut() {
        protect(p, w)?
    }
    let mut candidates = Vec::new();
    if pages.len() >= 2 {
        for (i, p) in pages.iter().enumerate() {
            for j in 0..p.lines.len() {
                if let Some(c) = candidate(i, j, p, w)? {
                    let y = center(&c.b, 1) / c.size[1];
                    if y <= 0.08
                        || y >= 0.92
                        || text::number(txt(&p.lines[j])?, w)?.is_some() && (y <= 0.15 || y >= 0.85)
                    {
                        candidates.push(c)
                    }
                }
            }
        }
        classify(&candidates, pages, w)?;
    }
    Ok(())
}
fn detect_separators(sources: &[Value], w: &mut Work) -> Result<Value, Error> {
    let pages = pages_from_sources(sources)?;
    Ok(json!(separators::detect(&pages, w)?))
}
/// Original full document prefix, ending immediately before per-page classifiers.
pub fn prepare_document_sources(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_document_sources_with_work(v, &mut Work(0))
}
pub(super) fn prepare_document_sources_with_work(v: &Value, w: &mut Work) -> Result<Value, Error> {
    prepare_document_sources_owned(v.clone(), w)
}
pub(super) fn prepare_document_sources_owned(v: Value, w: &mut Work) -> Result<Value, Error> {
    let mut profile = crate::processing::NativeStageProfile::new();
    let mut geometry = super::native_geometry::prepare_document_owned(v, w)?;
    profile.mark("source_geometry");
    array(&geometry["pages"], 1000)?;
    let Value::Array(mut pages) = geometry["pages"].take() else {
        return Err(invalid());
    };
    let sources = pages.iter_mut().map(|p| p["source"].take()).collect();
    let protected = protect_sources_owned(sources, w)?;
    profile.mark("source_marginal_protection");
    let normalized = super::page_containers::normalize_form_sources_owned(protected, w)?;
    profile.mark("source_form_normalization");
    let separators = detect_separators(&normalized, w)?;
    profile.mark("source_separators");
    for (p, s) in pages.iter_mut().zip(normalized) {
        p["source"] = s;
    }
    let mut result = json!({"plan":geometry["plan"].take(),"header_separator_bboxes":separators});
    result["pages"] = Value::Array(pages);
    if crate::legacy_page::budget::output_exceeds(&result) {
        Err(limit())
    } else {
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn borrowed_sources_match_installed_prefix_references() {
        let fixture: Value = serde_json::from_reader(flate2::read::GzDecoder::new(
            &include_bytes!(
                "../../../../../../tests/fixtures/upstream_068/native_marginals_owned.json.gz"
            )[..],
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let request = json!({"pages":case["pages"]});
            let actual = prepare_document_sources_owned(request, &mut Work(0)).unwrap();
            assert_eq!(actual, case["expected"], "{}", case["name"]);
        }
    }
    fn diff(a: &Value, b: &Value, path: &str) -> Option<String> {
        if a == b {
            return None;
        }
        match (a, b) {
            (Value::Object(a), Value::Object(b)) => {
                for k in a.keys().chain(b.keys()) {
                    if !a.contains_key(k) || !b.contains_key(k) {
                        return Some(format!("{path}/{k}: missing"));
                    }
                    if let Some(d) = diff(&a[k], &b[k], &format!("{path}/{k}")) {
                        return Some(d);
                    }
                }
            }
            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    return Some(format!("{path}: length {} vs {}", a.len(), b.len()));
                }
                for (i, (a, b)) in a.iter().zip(b).enumerate() {
                    if let Some(d) = diff(a, b, &format!("{path}/{i}")) {
                        return Some(d);
                    }
                }
            }
            _ => return Some(format!("{path}: {a} vs {b}")),
        }
        None
    }
    #[test]
    fn old_document_source_prefix() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-document-source-prefix.json");
        let f: Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let actual = prepare_document_sources(&c["request"])
                .unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = diff(&actual, &c["expected"], "") {
                panic!("{}: {d}", c["label"]);
            }
        }
    }
    #[test]
    fn old_raw_page_marginals() {
        for name in ["native-marginals", "native-marginal-edges"] {
            let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join(format!("../../build/migration/{name}.json"));
            let f: Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
            for c in f["cases"].as_array().unwrap() {
                let out = prepare(&c["request"]).unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
                assert_eq!(out, c["expected"], "{}", c["label"]);
            }
            for c in f["text"].as_array().unwrap() {
                assert_eq!(
                    json!(text::matches(
                        c["first"].as_str().unwrap(),
                        c["second"].as_str().unwrap(),
                        &mut Work(0)
                    )
                    .unwrap()),
                    c["matches"],
                    "{c}"
                );
            }
            for c in f["numbers"].as_array().unwrap() {
                assert_eq!(
                    json!(text::number(c["text"].as_str().unwrap(), &mut Work(0)).unwrap()),
                    c["value"],
                    "{c}"
                );
            }
        }
    }
}
