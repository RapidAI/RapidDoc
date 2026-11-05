//! Original whole-document title-profile probes, seed collection and clustering.
use super::*;
struct Seed {
    family: String,
    flags: i64,
    ratio: f64,
    weight: Option<f64>,
    mode: &'static str,
    anchor: f64,
    page: usize,
}
fn alignment(b: &Box4, left: f64, right: f64, h: f64) -> Option<(&'static str, f64)> {
    let width = (right - left).max(0.1);
    let center = (cx(b) - (left + right) / 2.) / width;
    let offset = (b.n[0] - left) / width;
    if center.abs() <= 0.12 && b.n[2] - b.n[0] <= 0.85 * width {
        Some(("center", center))
    } else if (b.n[0] - left).abs() <= 3. * h {
        Some(("left", offset))
    } else {
        None
    }
}
fn matches(seed: &Seed, cluster: &[Seed]) -> bool {
    let first = &cluster[0];
    let ratio = median(cluster.iter().map(|s| s.ratio).collect());
    let anchor = median(cluster.iter().map(|s| s.anchor).collect());
    let weights = cluster.iter().filter_map(|s| s.weight).collect::<Vec<_>>();
    let compatible = seed.weight.is_none() || weights.is_empty() || {
        let m = median(weights);
        let w = seed.weight.unwrap();
        (w - m).abs() < 100. || w.max(m) < 1.15 * w.min(m)
    };
    seed.family == first.family
        && seed.flags == first.flags
        && seed.mode == first.mode
        && (seed.ratio - ratio).abs() <= 0.1 * ratio
        && (seed.anchor - anchor).abs() <= 0.12
        && compatible
}
pub(super) fn infer(v: &Value, w: &mut Work) -> Result<Value, Error> {
    let body = &v["body_profile"];
    if body.is_null() {
        return Ok(Value::Null);
    }
    let bh = num(&body["body_height"])?;
    if bh == 0. {
        return Err(invalid());
    }
    let mut seeds = Vec::new();
    for (page, p) in array(&v["pages"], 1000)?.iter().enumerate() {
        let size = size(p.get("page_size").unwrap_or(&p["source"]["page_size"]))?;
        let containers = optional(p, "fixed_blocks", 4000)?
            .iter()
            .filter(|b| {
                !(b["_inline_visual_row_id"].is_i64()
                    || b["_inline_visual_row_id"].is_u64()
                    || b["_inline_visual_row_id"].is_boolean())
            })
            .map(|b| bbox(&b["bbox"]))
            .collect::<Result<Vec<_>, _>>()?;
        let out = super::prepare_with_work(
            &json!({"lines":p["remaining_lines"],"page_size":size,"page_index":page,"container_bboxes":containers.iter().map(|b|b.raw.clone()).collect::<Vec<_>>(),"document_body_profile":body,"classify_page":true}),
            w,
        )?;
        let lines = array(&out["lines"], 4096)?.to_vec();
        let angles = lines
            .iter()
            .map(|l| integer(l, "angle", 0))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let mut seen = BTreeSet::new();
        for a in angles {
            let ids = lines
                .iter()
                .enumerate()
                .filter(|(_, l)| integer(l, "angle", 0).ok() == Some(a))
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            if ids.is_empty() {
                continue;
            }
            let boxes = lines
                .iter()
                .map(|l| bbox(&l["bbox"]).map(|b| local(&b, size, a)))
                .collect::<Result<Vec<_>, _>>()?;
            let em = median(
                ids.iter()
                    .map(|&i| height(&lines[i], &boxes[i]))
                    .collect::<Result<Vec<_>, _>>()?,
            );
            let width = if a == 90 || a == 270 {
                size[1]
            } else {
                size[0]
            };
            let lanes = super::super::table_text_styles::lanes::infer_layout(
                &json!({"line_geometry":ids.iter().map(|&i|json!([lane_line(&lines[i]),boxes[i].raw])).collect::<Vec<_>>(),"local_page_width":width,"median_height":em}),
                w,
            )?;
            let local_containers = containers
                .iter()
                .map(|b| local(b, size, a))
                .collect::<Vec<_>>();
            for lane in array(&lanes, 4096)? {
                let left = num(&lane["left"])?;
                let right = num(&lane["right"])?;
                for index in array(&lane["line_indices"], 4096)? {
                    w.charge(local_containers.len() + 1)?;
                    let i = index
                        .as_u64()
                        .and_then(|i| ids.get(i as usize))
                        .copied()
                        .ok_or_else(invalid)?;
                    let l = &lines[i];
                    let b = &boxes[i];
                    let ratio = height(l, b)? / bh;
                    let unresolved = page > 0
                        && l["semantic_type"].is_null()
                        && ratio >= 1.3
                        && b.n[2] - b.n[0] <= 0.8 * width
                        && !inside(b, &local_containers);
                    if l["semantic_type"] != "paragraph_title" && !unresolved
                        || l["font_signature"].is_null()
                        || coverage(l)? < 0.75
                    {
                        continue;
                    }
                    let row = if l["visual_row_id"].is_null() {
                        integer(l, "source_index", 0)?
                    } else {
                        integer(l, "visual_row_id", 0)?
                    };
                    if seen.contains(&(a, row)) {
                        continue;
                    }
                    let Some((mode, anchor)) = alignment(b, left, right, bh) else {
                        continue;
                    };
                    let Some(family) = family(&l["font_signature"])? else {
                        continue;
                    };
                    let flags = l["font_signature"][1]
                        .as_i64()
                        .or_else(|| l["font_signature"][1].as_bool().map(i64::from))
                        .ok_or_else(invalid)?;
                    seeds.push(Seed {
                        family,
                        flags,
                        ratio,
                        weight: weight(l)?,
                        mode,
                        anchor,
                        page,
                    });
                    seen.insert((a, row));
                }
            }
        }
    }
    seeds.sort_by(|a, b| {
        a.family
            .cmp(&b.family)
            .then(a.flags.cmp(&b.flags))
            .then(a.mode.cmp(b.mode))
            .then(a.ratio.total_cmp(&b.ratio))
    });
    let mut clusters = Vec::<Vec<Seed>>::new();
    for seed in seeds {
        let mut target = None;
        for (i, c) in clusters.iter().enumerate() {
            w.charge(c.len() + 1)?;
            if matches(&seed, c) {
                target = Some(i);
                break;
            }
        }
        if let Some(i) = target {
            clusters[i].push(seed)
        } else {
            clusters.push(vec![seed])
        }
    }
    let mut prototypes = Vec::new();
    for c in clusters {
        let pages = c.iter().map(|s| s.page).collect::<BTreeSet<_>>().len();
        if c.len() < 3 && pages < 2 {
            continue;
        }
        let weights = c.iter().filter_map(|s| s.weight).collect::<Vec<_>>();
        prototypes.push(json!({"font_family":c[0].family,"font_flags":c[0].flags,"height_ratio":median(c.iter().map(|s|s.ratio).collect()),"weight":if weights.is_empty(){None}else{Some(median(weights))},"alignment":c[0].mode,"anchor_offset":median(c.iter().map(|s|s.anchor).collect()),"support_count":c.len(),"support_pages":pages}));
    }
    prototypes.sort_by(|a, b| {
        (
            b["support_pages"].as_u64().unwrap(),
            b["support_count"].as_u64().unwrap(),
        )
            .cmp(&(
                a["support_pages"].as_u64().unwrap(),
                a["support_count"].as_u64().unwrap(),
            ))
    });
    Ok(if prototypes.is_empty() {
        Value::Null
    } else {
        json!({"prototypes":prototypes})
    })
}
