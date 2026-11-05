//! Original docvortex_adapter pixel-layout, fallback, supplements and visual text direction.
use super::path_graphics::array;
pub fn prepare_table_page(
    library: &std::path::Path,
    bytes: &[u8],
    page_index: usize,
    password: Option<&str>,
) -> Result<Value, Error> {
    let _budget = crate::legacy_page::budget::host_page();
    let limits = rapiddoc_core::Limits::pdf_host();
    let mut g = serde_json::to_value(rapiddoc_pdf::character_primitives::read_visible_geometry(
        library, bytes, page_index, password, true, &limits,
    )?)
    .map_err(|_| invalid())?;
    let vectors = serde_json::to_value(rapiddoc_pdf::legacy_vectors::read(
        library, bytes, page_index, password, &limits,
    )?)
    .map_err(|_| invalid())?;
    let mut rules = vec![];
    for r in array(&vectors["drawing_lines"], 5000)? {
        if !matches!(r["orientation"].as_str(), Some("horizontal" | "vertical")) {
            continue;
        }
        if let Some(b) = super::table_native_text::normalize_bbox(&r["bbox"])? {
            rules.push(json!({"bbox":b,"width":r["width"].as_f64().unwrap_or(0.).max(0.),"orientation":r["orientation"]}));
        }
    }
    let mut rectangles = vec![];
    for r in array(&vectors["path_infos"], 5000)? {
        if let Some(b) = super::table_native_text::normalize_bbox(&r["bbox"])? {
            rectangles.push(json!({"bbox":b,"segment_count":r["segment_count"],"fill_visible":r["fill_visible"],"stroke_visible":r["stroke_visible"],"form_depth":r["form_depth"]}));
        }
    }
    Ok(
        json!({"page_size":g["page_size"].take(),"chars":g["chars"].take(),"tight_bboxes":g["tight_bboxes"].take(),"origins":g["origins"].take(),"drawing_lines":rules,"rectangles":rectangles}),
    )
}
pub fn recover_table_region(v: &Value) -> Result<Value, Error> {
    let _budget = crate::legacy_page::budget::host_page();
    super::table_text_styles::materialization::recover(v)
}
pub(crate) fn recover_table_region_owned(v: Value) -> Result<Value, Error> {
    let _budget = crate::legacy_page::budget::host_page();
    super::table_text_styles::materialization::recover_owned(v)
}
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
fn category(label: &str) -> Option<i64> {
    Some(match label {
        "text" | "code" | "list" | "index" | "ref_text" | "footnote" | "page_footnote"
        | "aside_text" => 1,
        "doc_title" | "paragraph_title" | "title" => 0,
        "image" | "chart" => 3,
        "caption" => 4,
        "table" => 5,
        "table_caption" => 6,
        "equation" => 14,
        "header" | "footer" | "page_number" => 2,
        _ => return None,
    })
}
fn scalar(v: Option<&Value>, empty_false: bool) -> String {
    super::table_native_multiline::spatial::raw_string(v, empty_false)
}
fn number(v: &Value) -> Option<f64> {
    if let Some(b) = v.as_bool() {
        Some(if b { 1. } else { 0. })
    } else {
        v.as_f64().or_else(|| {
            v.as_str()?
                .trim_matches(rapiddoc_native::text_boundary::python_space)
                .parse()
                .ok()
        })
    }
}
fn angle(v: Option<&Value>) -> Result<i64, Error> {
    let Some(v) = v else { return Ok(0) };
    if v.is_null() || v == false || v == "" {
        return Ok(0);
    }
    let n = if let Some(n) = v.as_i64() {
        n
    } else if let Some(b) = v.as_bool() {
        i64::from(b)
    } else if let Some(n) = v.as_f64() {
        if !n.is_finite() || n.abs() > 1e12 {
            return Err(invalid());
        }
        n.trunc() as i64
    } else {
        v.as_str()
            .ok_or_else(invalid)?
            .trim()
            .parse::<i64>()
            .map_err(|_| invalid())?
    };
    Ok(n.rem_euclid(360))
}
fn text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().map(text).collect(),
        Value::Object(o) => o.get("content").map(text).unwrap_or_default(),
        _ => String::new(),
    }
}
fn dimension(v: &Value) -> Result<f64, Error> {
    number(v)
        .filter(|n| n.is_finite() && n.abs() <= 1e9)
        .ok_or_else(invalid)
}
pub fn convert(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let w = dimension(&v["width"])?;
    let h = dimension(&v["height"])?;
    let ignored = v
        .get("ignored_labels")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut out = vec![];
    let table = constant_regex!(r"(?i)^\s*(?:table|tab\.|表\s*\d|表格\s*\d)");
    let figure = constant_regex!(r"(?i)^\s*(?:fig(?:ure)?\.?|图\s*\d)");
    for (order, b) in array(&v["blocks"], 10000)?.iter().enumerate() {
        if !b.is_object() {
            return Err(invalid());
        }
        let label = if b.get("type").is_none() {
            "text".into()
        } else {
            scalar(b.get("type"), false)
        };
        let Some(mut cat) = category(&label) else {
            continue;
        };
        let Some(raw) = b["bbox"].as_array().filter(|a| a.len() == 4) else {
            continue;
        };
        let Some(bb) = raw.iter().map(number).collect::<Option<Vec<_>>>() else {
            continue;
        };
        // Python min(width, value) keeps the original dimension object on a
        // tie or NaN; max(0.0, value) likewise keeps its first argument.
        let clamp = |x: f64, size: f64, original: &Value| {
            let product = x * size;
            if product < size {
                json!(if product > 0. { product } else { 0. })
            } else if size > 0. {
                original.clone()
            } else {
                json!(0.)
            }
        };
        let poly = [
            clamp(bb[0], w, &v["width"]),
            clamp(bb[1], h, &v["height"]),
            clamp(bb[2], w, &v["width"]),
            clamp(bb[3], h, &v["height"]),
        ];
        let bb = poly.each_ref().map(|x| number(x).unwrap());
        if bb[2] <= bb[0] || bb[3] <= bb[1] {
            continue;
        }
        let rotated = angle(b.get("angle"))? != 0;
        if label == "caption" {
            let caption = text(&b["content"])
                .chars()
                .map(|c| {
                    if rapiddoc_native::text_boundary::python_space(c) {
                        ' '
                    } else if matches!(c, 'İ' | 'ı') {
                        'i'
                    } else {
                        c
                    }
                })
                .collect::<String>();
            cat = if rotated {
                1
            } else if table.is_match(&caption) {
                6
            } else if !figure.is_match(&caption) {
                1
            } else {
                cat
            };
        }
        let actual = if ignored.iter().any(|v| v.as_str() == Some(&label)) {
            2
        } else {
            cat
        };
        let mut d = json!({"category_id":actual,"original_label":label,"original_order":order,"poly":[poly[0],poly[1],poly[2],poly[1],poly[2],poly[3],poly[0],poly[3]],"polygon_points":null,"score":1.0});
        if matches!(actual, 0 | 1) {
            d["native_text_region"] = json!(true)
        }
        if (matches!(
            label.as_str(),
            "caption" | "table_caption" | "footnote" | "page_footnote"
        ) || (rotated && matches!(cat, 0 | 1)))
            && actual != 2
        {
            d["native_text"] = json!(text(&b["content"]))
        }
        if label == "table"
            && b["content"]
                .as_str()
                .is_some_and(|s| s.to_lowercase().contains("<table"))
        {
            d["html"] = b["content"].clone();
            d["table_parse_method"] = json!("docvortex_native")
        }
        out.push(d);
    }
    Ok(json!(out))
}
fn bb(v: &Value) -> Result<[f64; 4], Error> {
    let a = array(&v["poly"], 8)?;
    if a.len() != 8 {
        return Err(invalid());
    }
    Ok([
        dimension(&a[0])?,
        dimension(&a[1])?,
        dimension(&a[4])?,
        dimension(&a[5])?,
    ])
}
fn area(v: &Value) -> Result<f64, Error> {
    let b = bb(v)?;
    Ok((b[2] - b[0]).max(0.) * (b[3] - b[1]).max(0.))
}
fn intersection(a: &Value, b: &Value) -> Result<f64, Error> {
    let aa = area(a)?;
    let ba = area(b)?;
    let a = bb(a)?;
    let b = bb(b)?;
    let scale = aa.min(ba);
    Ok(if scale == 0. {
        0.
    } else {
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.)
            / scale
    })
}
fn center(v: &Value) -> Result<[f64; 2], Error> {
    let b = bb(v)?;
    Ok([(b[0] + b[2]) / 2., (b[1] + b[3]) / 2.])
}
fn overlap(a: &Value, b: &Value) -> Result<f64, Error> {
    let a = bb(a)?;
    let b = bb(b)?;
    let h = (a[3] - a[1]).min(b[3] - b[1]);
    Ok(if h > 0. {
        (a[3].min(b[3]) - a[1].max(b[1])).max(0.) / h
    } else {
        0.
    })
}
fn cat(v: &Value) -> i64 {
    v["category_id"].as_i64().unwrap_or(-1)
}
pub fn fallback(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let blocks = array(&v["blocks"], 10000)?;
    let layout = array(&v["layout"], 10000)?;
    if blocks.is_empty() || layout.is_empty() {
        return Ok(json!("empty layout"));
    }
    let pa = dimension(&v["width"])? * dimension(&v["height"])?;
    if pa > 0. {
        for d in layout {
            if cat(d) == 2
                && matches!(
                    d["original_label"].as_str(),
                    Some("header" | "footer" | "aside_text")
                )
            {
                let b = bb(d)?;
                if (b[2] - b[0]) * (b[3] - b[1]) >= 0.4 * pa {
                    return Ok(json!(format!(
                        "oversized discarded {}",
                        d["original_label"].as_str().unwrap()
                    )));
                }
            }
        }
    }
    let retained = layout.iter().filter(|d| cat(d) != 2).collect::<Vec<_>>();
    let discarded: usize = blocks
        .iter()
        .filter(|b| matches!(b["type"].as_str(), Some("header" | "footer" | "aside_text")))
        .map(|b| text(&b["content"]).chars().count())
        .sum();
    let count = v["native_char_count"].as_u64().unwrap_or(0);
    Ok(if retained.is_empty() && (count >= 80 || discarded >= 80) {
        json!("text page has no retained content")
    } else if retained.len() == 1 && cat(retained[0]) == 3 && count >= 80 {
        json!("text page reduced to one image")
    } else {
        Value::Null
    })
}
pub fn direction(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let b = array(&v["table_bbox"], 4)?;
    if b.len() != 4 {
        return Err(invalid());
    }
    let b = b.iter().map(dimension).collect::<Result<Vec<_>, _>>()?;
    let mut counts = Vec::<(i64, usize)>::new();
    for c in array(&v["chars"], 20000)? {
        let Some(cb) = c["bbox"].as_array().filter(|b| b.len() == 4) else {
            continue;
        };
        if scalar(c.get("char"), true)
            .trim_matches(rapiddoc_native::text_boundary::python_space)
            .is_empty()
        {
            continue;
        }
        let cb = cb.iter().map(dimension).collect::<Result<Vec<_>, _>>()?;
        let x = (cb[0] + cb[2]) / 2.;
        let y = (cb[1] + cb[3]) / 2.;
        if !(b[0] <= x && x <= b[2] && b[1] <= y && y <= b[3]) {
            continue;
        }
        let raw = c.get("writing_angle").or_else(|| c.get("rotation"));
        let degrees = if raw.is_none_or(|v| v.is_null() || v == false || v == "") {
            0.
        } else {
            let Some(n) = raw.and_then(number) else {
                continue;
            };
            n * (180. / std::f64::consts::PI)
        };
        if !degrees.is_finite() {
            continue;
        }
        let nearest = (degrees / 90.).round_ties_even() * 90.;
        if (degrees - nearest).abs() <= 5. {
            let a = (nearest as i64).rem_euclid(360);
            if let Some((_, n)) = counts.iter_mut().find(|(k, _)| *k == a) {
                *n += 1
            } else {
                counts.push((a, 1))
            }
        }
    }
    let mut winner = (0, 0);
    for c in counts {
        if c.1 > winner.1 {
            winner = c
        }
    }
    Ok(json!(winner.0))
}
pub fn supplements(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let native = array(&v["native"], 10000)?;
    let pp = array(&v["supplements"], 10000)?;
    let formula = v
        .get("include_formula")
        .map_or(Some(true), Value::as_bool)
        .ok_or_else(invalid)?;
    let seal = v
        .get("include_seal")
        .map_or(Some(true), Value::as_bool)
        .ok_or_else(invalid)?;
    let mut work = Work(0);
    let is_formula = |v: &Value| matches!(cat(v), 13 | 14);
    let score = |v: &Value| {
        v["score"]
            .as_f64()
            .filter(|n| n.is_finite())
            .ok_or_else(invalid)
    };
    let identities = if let Some(a) = v.get("supplement_identities") {
        let a = a
            .as_array()
            .filter(|a| a.len() == pp.len())
            .ok_or_else(invalid)?;
        a.iter()
            .map(|v| v.as_u64().ok_or_else(invalid))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        (0..pp.len() as u64).collect()
    };
    let mut extras = Vec::new();
    for (i, e) in pp.iter().enumerate() {
        if !((formula && matches!(cat(e), 9 | 13 | 14)) || (seal && e["original_label"] == "seal"))
        {
            continue;
        }
        if is_formula(e) && score(e)? < 0.5 {
            continue;
        }
        let mut excluded = false;
        for n in native {
            work.charge(1)?;
            if is_formula(e)
                && matches!(cat(n), 0 | 1 | 4 | 6)
                && intersection(e, n)? >= 0.8
                && area(e)? >= 0.5 * area(n)?
            {
                excluded = true;
                break;
            }
            if e["original_label"] != "seal"
                && n["original_label"] == "table"
                && intersection(e, n)? >= 0.5
            {
                excluded = true;
                break;
            }
        }
        if !excluded {
            bb(e)?;
            extras.push(i)
        }
    }
    if extras.is_empty() {
        return Ok(json!(native));
    }
    let mut used = std::collections::BTreeSet::new();
    let mut replacements = std::collections::BTreeMap::new();
    for (i, n) in native.iter().enumerate() {
        let mut best = None;
        for &j in &extras {
            work.charge(1)?;
            let e = &pp[j];
            if used.contains(&identities[j]) || intersection(n, e)? < 0.5 {
                continue;
            }
            if (n["original_label"] == "equation" && is_formula(e) && area(e)? <= 3. * area(n)?)
                || (matches!(n["original_label"].as_str(), Some("image" | "chart"))
                    && e["original_label"] == "seal"
                    && area(n)? <= 2. * area(e)?)
            {
                let candidate_score = score(e)?;
                let replace = match best {
                    None => true,
                    Some(k) => candidate_score > score(&pp[k])?,
                };
                if replace {
                    best = Some(j)
                }
            }
        }
        if let Some(j) = best {
            used.insert(identities[j]);
            replacements.insert(i, j);
        }
    }
    let mut ordered = native
        .iter()
        .enumerate()
        .map(|(i, n)| {
            (
                i as f64,
                replacements.get(&i).map_or(n, |j| &pp[*j]).clone(),
            )
        })
        .collect::<Vec<_>>();
    let mut anchors = ordered
        .iter()
        .filter(|(_, d)| cat(d) == 14)
        .cloned()
        .collect::<Vec<_>>();
    extras.sort_by(|&a, &b| {
        let a = bb(&pp[a]).unwrap();
        let b = bb(&pp[b]).unwrap();
        a[1].partial_cmp(&b[1])
            .unwrap()
            .then(a[0].partial_cmp(&b[0]).unwrap())
    });
    for j in extras {
        if used.contains(&identities[j]) {
            continue;
        }
        let e = &pp[j];
        let mut duplicate = false;
        for &k in replacements.values() {
            work.charge(1)?;
            if is_formula(e) && is_formula(&pp[k]) && intersection(e, &pp[k])? >= 0.7 {
                duplicate = true;
                break;
            }
        }
        if duplicate {
            continue;
        }
        if cat(e) == 9 {
            let mut best = None;
            let mut distance = f64::INFINITY;
            for (o, a) in &anchors {
                work.charge(1)?;
                if overlap(e, a)? >= 0.5 {
                    let d = (center(e)?[1] - center(a)?[1]).abs();
                    if d < distance {
                        distance = d;
                        best = Some(*o)
                    }
                }
            }
            if let Some(o) = best {
                ordered.push((o + 0.25, e.clone()));
                continue;
            }
        }
        let mut order = 0.;
        if !native.is_empty() {
            let c = center(e)?;
            let mut distance = f64::INFINITY;
            let mut anchor = 0;
            for (i, n) in native.iter().enumerate() {
                work.charge(1)?;
                let nc = center(n)?;
                let d = (c[0] - nc[0]).powi(2) + (c[1] - nc[1]).powi(2);
                if d < distance {
                    distance = d;
                    anchor = i
                }
            }
            order = anchor as f64
                + if c[1] >= center(&native[anchor])?[1] {
                    0.5
                } else {
                    -0.5
                };
        }
        ordered.push((order, e.clone()));
        if cat(e) == 14 {
            anchors.push((order, e.clone()))
        }
    }
    ordered.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let out = ordered
        .into_iter()
        .enumerate()
        .map(|(i, (_, mut d))| {
            d["original_order"] = json!(i);
            d
        })
        .collect::<Vec<_>>();
    if out.len() > rapiddoc_core::resources::limit(10000) {
        return Err(limit());
    }
    Ok(json!(out))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_dense_table_capacity() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
        let bytes = std::fs::read(root.join("demo/pdfs/49－长表格-密集型.pdf")).unwrap();
        let page = prepare_table_page(&library, &bytes, 0, None).unwrap();
        assert_eq!(page["chars"].as_array().unwrap().len(), 20198);
        let mut input = page.clone();
        input["table_bbox"] = json!([48.82, 55.66, 544.57, 786.77]);
        input["angle"] = json!(0);
        let recovered=recover_table_region(&json!({"table_input":input,"tight_bboxes":page["tight_bboxes"],"origins":page["origins"]})).unwrap();
        assert!(recovered.get("recovery_error").is_none(), "{recovered}");
        assert!(
            recovered["content"]
                .as_str()
                .is_some_and(|s| s.len() > rapiddoc_core::resources::limit(10_000)),
            "{recovered}"
        );
    }
    #[test]
    fn original_native_host_tables() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
        let f: Value = serde_json::from_slice(
            &std::fs::read(root.join("build/migration/native-host-tables.json")).unwrap(),
        )
        .unwrap();
        for c in f["pages"].as_array().unwrap() {
            let bytes = std::fs::read(root.join(c["pdf"].as_str().unwrap())).unwrap();
            let got = prepare_table_page(
                &library,
                &bytes,
                c["page_index"].as_u64().unwrap() as usize,
                None,
            )
            .unwrap();
            if let Some(d) = super::super::native_document::tests::diff(&got, &c["expected"], "") {
                panic!("{} page {}: {d}", c["pdf"], c["page_index"])
            }
        }
        for c in f["cases"].as_array().unwrap() {
            let p = &f["pages"]
                .as_array()
                .unwrap()
                .iter()
                .find(|p| p["pdf"] == c["pdf"] && p["page_index"] == c["page_index"])
                .unwrap()["expected"];
            let a = direction(&json!({"chars":p["chars"],"table_bbox":c["bbox"]})).unwrap();
            assert_eq!(a, c["angle"]);
            let got=recover_table_region(&json!({"table_input":{"chars":p["chars"],"page_size":p["page_size"],"drawing_lines":p["drawing_lines"],"rectangles":p["rectangles"],"angle":a,"table_bbox":c["bbox"]},"tight_bboxes":p["tight_bboxes"],"origins":p["origins"]})).unwrap();
            let got = if got.is_null() {
                Value::Null
            } else {
                json!({"html":got["content"],"source":got["result"]["source"],"confidence":got["result"]["confidence"],"diagnostics":got["result"]["diagnostics"]})
            };
            if let Some(d) = super::super::native_document::tests::diff(&got, &c["expected"], "") {
                panic!("{}: {d}", c["label"])
            }
        }
    }
    #[test]
    fn original_native_adapter() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let f: Value = serde_json::from_slice(
            &std::fs::read(root.join("build/migration/native-host-adapter.json")).unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            let result = match c["operation"].as_str().unwrap() {
                "convert" => convert(&c["request"]),
                "fallback" => fallback(&c["request"]),
                "direction" => direction(&c["request"]),
                "supplements" => supplements(&c["request"]),
                _ => unreachable!(),
            };
            if c.get("error").is_some() {
                assert!(result.is_err(), "{}", c["label"]);
                continue;
            }
            let got = result.unwrap_or_else(|e| panic!("{}: {e}", c["label"]));
            if let Some(d) = super::super::native_document::tests::diff(&got, &c["expected"], "") {
                panic!("{}: {d}", c["label"]);
            }
        }
    }
}
