//! Original table/background marginal restoration and document body profiles.
use super::*;
use rapiddoc_native::text_boundary::python_space;
fn stripped(t: &str) -> &str {
    t.trim_matches(python_space)
}
fn compact(t: &str) -> String {
    t.chars().filter(|&c| !python_space(c)).collect()
}
pub(super) fn restore(pages: &mut [Value], w: &mut Work) -> Result<(), Error> {
    for p in pages.iter_mut() {
        let mut lines = ls(p)?.to_vec();
        let drawing = if p.get("drawing_lines").is_some() {
            optional(p, "drawing_lines", 5000)?
        } else {
            optional(&p["source"], "drawing_lines", 5000)?
        };
        for table in optional(p, "table_bboxes", 5000)? {
            let table = bbox(table)?;
            let mut candidates = Vec::new();
            for (i, l) in lines.iter().enumerate() {
                w.charge(1)?;
                if integer(l, "angle", 0)? != 0
                    || !matches!(
                        l["semantic_type"].as_str(),
                        None | Some("header" | "aside_text")
                    )
                    || stripped(text(l)?).chars().count() <= 3
                {
                    continue;
                }
                let b = bbox(&l["bbox"])?;
                let em = l.get("effective_height").map_or(Ok(0.), num)?.max(0.1);
                if table.n[1] - b.n[3] < 0.
                    || table.n[1] - b.n[3] > 4. * em
                    || b.n[0] >= table.n[2]
                    || b.n[2] <= table.n[0]
                {
                    continue;
                }
                let mut barrier = false;
                for r in drawing {
                    w.charge(1)?;
                    if r["orientation"] != "horizontal" {
                        continue;
                    }
                    let rb = bbox(&r["bbox"])?;
                    if b.n[3] < cy(&rb)
                        && cy(&rb) < table.n[1] - 0.25 * em
                        && rb.n[2].min(table.n[2]) - rb.n[0].max(table.n[0])
                            >= 0.6 * (table.n[2] - table.n[0])
                    {
                        barrier = true;
                        break;
                    }
                }
                if !barrier {
                    candidates.push((i, b, em))
                }
            }
            candidates.sort_by(|a, b| b.1.n[3].total_cmp(&a.1.n[3]));
            let mut connected = Vec::new();
            let mut top = table.n[1];
            let mut next = None;
            for item in candidates {
                if top - item.1.n[3] > 1.5 * item.2.max(next.unwrap_or(item.2)) {
                    break;
                }
                top = top.min(item.1.n[1]);
                next = Some(item.2);
                connected.push(item)
            }
            if connected.len() < 2 {
                continue;
            }
            let bounds = combined(connected.iter().map(|(_, b, _)| b))?;
            let em = median(connected.iter().map(|(_, _, h)| *h).collect());
            if bounds.n[3] - bounds.n[1] < 1.5 * em
                || bounds.n[2] - bounds.n[0] < 0.7 * (table.n[2] - table.n[0])
            {
                continue;
            }
            for (i, _, _) in connected {
                lines[i]["semantic_type"] = Value::Null;
                lines[i]["marginal_body_context"] = json!(true)
            }
        }
        p["remaining_lines"] = json!(lines);
    }
    let mut frequencies = BTreeMap::<String, usize>::new();
    for p in pages.iter() {
        let texts = ls(p)?
            .iter()
            .map(|l| Ok(compact(text(l)?)))
            .collect::<Result<BTreeSet<_>, Error>>()?;
        for t in texts {
            *frequencies.entry(t).or_default() += 1
        }
    }
    for p in pages {
        let s = sz(p)?;
        let backgrounds = optional(p, "fixed_blocks", 4000)?
            .iter()
            .filter(|b| b["type"] == "image")
            .map(|b| bbox(&b["bbox"]))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .filter(|b| (b.n[2] - b.n[0]) * (b.n[3] - b.n[1]) >= 0.6 * s[0] * s[1])
            .collect::<Vec<_>>();
        let mut lines = ls(p)?.to_vec();
        for l in &mut lines {
            w.charge(backgrounds.len() + 1)?;
            if !matches!(
                l["semantic_type"].as_str(),
                Some("header" | "footer" | "aside_text")
            ) || stripped(text(l)?).chars().count() <= 3
                || frequencies.get(&compact(text(l)?)) != Some(&1)
            {
                continue;
            }
            let b = bbox(&l["bbox"])?;
            let area = (b.n[2] - b.n[0]) * (b.n[3] - b.n[1]);
            if area > 0.
                && backgrounds.iter().any(|c| {
                    (b.n[2].min(c.n[2]) - b.n[0].max(c.n[0])).max(0.)
                        * (b.n[3].min(c.n[3]) - b.n[1].max(c.n[1])).max(0.)
                        / area
                        >= 0.95
                })
            {
                l["semantic_type"] = Value::Null;
                l["marginal_body_context"] = json!(true)
            }
        }
        p["remaining_lines"] = json!(lines)
    }
    Ok(())
}
struct Sample {
    h: f64,
    page: usize,
    width: f64,
    font: Option<(String, i64)>,
    weight: Option<f64>,
}
fn font(l: &Value) -> Result<Option<(String, i64)>, Error> {
    if l["font_signature"].is_null() || l.get("font_coverage").map_or(Ok(0.), num)? < 0.75 {
        return Ok(None);
    }
    let a = array(&l["font_signature"], 2)?;
    if a.len() != 2 {
        return Err(invalid());
    }
    Ok(Some((
        a[0].as_str().ok_or_else(invalid)?.to_owned(),
        a[1].as_i64()
            .or_else(|| a[1].as_bool().map(i64::from))
            .ok_or_else(invalid)?,
    )))
}
pub(super) fn body(pages: &[Value], canonical_scale: bool, w: &mut Work) -> Result<Value, Error> {
    let sparse = pages
        .iter()
        .map(|p| ls(p).map(|ls| ls.iter().filter(|l| l["semantic_type"].is_null()).count()))
        .collect::<Result<Vec<_>, _>>()?
        .iter()
        .sum::<usize>()
        <= 6;
    let mut samples = Vec::new();
    for (page, p) in pages.iter().enumerate() {
        let s = sz(p)?;
        let mut lines = ls(p)?.iter().collect::<Vec<_>>();
        if sparse {
            lines.extend(optional(p, "table_body_profile_lines", 4096)?)
        }
        for l in lines {
            w.charge(1)?;
            if !l["semantic_type"].is_null() {
                continue;
            }
            let a = integer(l, "angle", 0)?;
            let b = bs(l, s, a, false)?;
            let h = if canonical_scale {
                canonical(l, &b)?
            } else {
                height(l, &b)?
            };
            let width = (b.n[2] - b.n[0]).max(0.) / local_size(s, a)[0].max(0.1);
            if h <= 0. || width <= 0. {
                continue;
            }
            samples.push(Sample {
                h,
                page,
                width,
                font: font(l)?,
                weight: if l["dominant_font_weight"].is_null() {
                    None
                } else {
                    Some(num(&l["dominant_font_weight"])?)
                },
            })
        }
    }
    if samples.is_empty() {
        return Ok(Value::Null);
    }
    // ordered_clusters: stable sort, first median-compatible cluster, inclusive tolerance.
    let mut ordered = (0..samples.len()).collect::<Vec<_>>();
    ordered.sort_by(|&a, &b| samples[a].h.total_cmp(&samples[b].h));
    let mut groups = Vec::<Vec<usize>>::new();
    for i in ordered {
        let mut chosen = None;
        for (j, g) in groups.iter().enumerate() {
            w.charge(g.len() + 1)?;
            let m = median(g.iter().map(|&i| samples[i].h).collect());
            if (samples[i].h - m).abs() <= 0.1 * m {
                chosen = Some(j);
                break;
            }
        }
        if let Some(j) = chosen {
            groups[j].push(i)
        } else {
            groups.push(vec![i])
        }
    }
    let count_pages = |g: &Vec<usize>| {
        g.iter()
            .map(|&i| samples[i].page)
            .collect::<BTreeSet<_>>()
            .len()
    };
    let mut eligible = groups
        .iter()
        .filter(|g| count_pages(g) >= 2)
        .collect::<Vec<_>>();
    if eligible.is_empty() {
        eligible = groups.iter().collect()
    }
    if pages.len() == 1 && samples.len() >= 6 {
        let repeated = eligible
            .iter()
            .copied()
            .filter(|g| g.len() >= 3)
            .collect::<Vec<_>>();
        if !repeated.is_empty() {
            eligible = repeated
        }
    }
    let key = |g: &Vec<usize>| {
        (
            super::super::table_native_candidate::sum(
                &g.iter().map(|&i| samples[i].width).collect::<Vec<_>>(),
            ),
            count_pages(g),
            g.len(),
        )
    };
    let mut chosen = eligible[0];
    for g in eligible.iter().skip(1) {
        let a = key(g);
        let b = key(chosen);
        if a.0 > b.0 || a.0 == b.0 && (a.1, a.2) > (b.1, b.2) {
            chosen = g
        }
    }
    let body_h = median(chosen.iter().map(|&i| samples[i].h).collect());
    let band = |s: &Sample| s.h / body_h >= 0.9 && s.h / body_h <= 1.1;
    let weights = samples
        .iter()
        .filter(|s| band(s))
        .filter_map(|s| s.weight)
        .collect::<Vec<_>>();
    let weight = if weights.is_empty() {
        None
    } else {
        Some(median(weights))
    };
    let mut fonts = BTreeMap::<(String, i64), (BTreeSet<usize>, f64, Vec<f64>)>::new();
    for s in &samples {
        if !band(s) {
            continue;
        }
        if let Some(font) = &s.font {
            let e = fonts.entry(font.clone()).or_default();
            e.0.insert(s.page);
            e.1 += s.width;
            if let Some(weight) = s.weight {
                e.2.push(weight)
            }
        }
    }
    let regular = fonts
        .into_iter()
        .filter(|((_, flags), (pages, width, weights))| {
            pages.len() >= 3
                && *width >= 2.
                && *width >= 0.75 * pages.len() as f64
                && flags & (64 | (1 << 18)) == 0
                && (weights.is_empty()
                    || weight.is_none()
                    || median(weights.clone())
                        < (weight.unwrap() + 100.).max(1.15 * weight.unwrap()))
        })
        .map(|((name, flags), _)| json!([name, flags]))
        .collect::<Vec<_>>();
    let repairs = pages.iter().any(|p| {
        ls(p)
            .unwrap()
            .iter()
            .any(|l| l["style_scale_repaired"] == true)
    });
    Ok(
        json!({"body_height":body_h.max(0.1),"body_weight":weight,"regular_fonts":regular,"has_style_scale_repairs":repairs}),
    )
}
