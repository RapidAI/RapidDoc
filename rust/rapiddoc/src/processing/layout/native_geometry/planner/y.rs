use super::*;
struct Analysis {
    key: Key,
    line: usize,
    ids: Vec<usize>,
    anchors: Vec<usize>,
    dominant: Vec<usize>,
    baseline: f64,
    tight: [f64; 4],
    source: [f64; 4],
    all: [f64; 4],
    legacy: [f64; 4],
    run: usize,
    coverage: f64,
    share: f64,
    split: bool,
    neighbors: Vec<usize>,
    intrusion: f64,
    candidate: bool,
}
fn clusters(ids: &[usize], s: &[Sample], w: &mut Work) -> Result<Vec<Vec<usize>>, Error> {
    let tol = 0.5f64.max(
        0.25 * quantile(
            ids.iter().map(|&i| s[i].ink[3] - s[i].ink[1]).collect(),
            0.75,
        ),
    );
    w.charge(ids.len().saturating_mul(ids.len() + 1))?;
    Ok(
        ordered_clusters(ids.iter().map(|&i| s[i].base[1]).collect(), tol, 0., false)
            .ok_or_else(invalid)?
            .into_iter()
            .map(|g| g.into_iter().map(|i| ids[i]).collect())
            .collect(),
    )
}
fn mode(ids: &[usize], s: &[Sample]) -> usize {
    let mut counts: Vec<(usize, usize)> = Vec::new();
    for &i in ids {
        if let Some((_, n)) = counts.iter_mut().find(|(r, _)| *r == s[i].run) {
            *n += 1
        } else {
            counts.push((s[i].run, 1))
        }
    }
    let mut best = counts[0];
    for c in counts {
        if c.1 > best.1 {
            best = c
        }
    }
    best.0
}
fn axis(a: [f64; 4], b: [f64; 4]) -> f64 {
    let d = (a[2] - a[0]).min(b[2] - b[0]);
    if d > 0. {
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.) / d
    } else {
        0.
    }
}
fn tiers(mut rows: Vec<[f64; 4]>, em: f64) -> Option<(f64, f64)> {
    if rows.len() < 4 || em <= 0. {
        return None;
    }
    rows.sort_by(|a, b| {
        let key = |a: &[f64; 4]| {
            [
                (a[0] + a[1]) / a[3].max(a[2]).max(0.1),
                a[0],
                a[1],
                a[2],
                a[3],
            ]
        };
        key(a).partial_cmp(&key(b)).unwrap()
    });
    let mut groups: Vec<Vec<[f64; 4]>> = Vec::new();
    for row in &rows {
        let ratio = (row[0] + row[1]) / row[3].max(row[2]).max(0.1);
        if groups.last().is_none_or(|g| {
            ratio
                > 1.35
                    * med(g
                        .iter()
                        .map(|r| (r[0] + r[1]) / r[3].max(r[2]).max(0.1))
                        .collect())
        }) {
            groups.push(vec![*row])
        } else {
            groups.last_mut().unwrap().push(*row)
        }
    }
    if groups.len() < 2 {
        return None;
    }
    let upper = &groups[groups.len() - 1];
    let reference = &groups[groups.len() - 2];
    if upper.len() < 2usize.max((0.15 * rows.len() as f64).ceil() as usize) || reference.len() < 2 {
        return None;
    }
    let loose = med(upper.iter().map(|r| r[0] + r[1]).collect());
    if loose
        <= 1.5
            * med(upper
                .iter()
                .filter_map(|r| (r[3] > 0.).then_some(r[3]))
                .collect())
        || loose <= 1.8 * quantile(upper.iter().map(|r| r[2]).collect(), 0.75).max(0.1)
    {
        return None;
    }
    Some((
        med(reference
            .iter()
            .map(|r| r[0] / r[3].max(r[2]).max(0.1))
            .collect())
            * em,
        med(reference
            .iter()
            .map(|r| r[1] / r[3].max(r[2]).max(0.1))
            .collect())
            * em,
    ))
}
pub(super) fn repair(
    plan: &mut Plan,
    samples: &[Sample],
    by_line: &[(Key, Vec<usize>)],
    keys: &[RunKey],
    pages: &[Page],
    bad: &BTreeSet<usize>,
    w: &mut Work,
) -> Result<(), Error> {
    let mut analyses: Vec<Analysis> = Vec::new();
    for &(k, ref ids) in by_line {
        let s = &samples[ids[0]];
        let Some((line_index, l)) = pages[k.0]
            .lines
            .iter()
            .enumerate()
            .filter(|(_, l)| {
                integer(l, "source_index", 0).ok() == Some(k.1)
                    && integer(l, "angle", 0).ok() == Some(0)
                    && !skip(l)
            })
            .next_back()
        else {
            continue;
        };
        let chars = optional(l, "chars", 20000)?;
        w.charge(chars.len() + ids.len())?;
        let visible = chars
            .iter()
            .filter(|c| {
                let t = super::super::super::table_native_multiline::spatial::raw_string(
                    c.get("char"),
                    true,
                );
                !t.is_empty() && t.chars().all(python_printable) && !t.chars().all(python_space)
            })
            .count();
        let anchors = ids
            .iter()
            .copied()
            .filter(|&i| samples[i].anchor)
            .collect::<Vec<_>>();
        if visible == 0 || anchors.len() < 4 {
            continue;
        }
        let groups = clusters(&anchors, samples, w)?;
        let mut dominant = &groups[0];
        let score = |g: &Vec<usize>| {
            (
                sum(&g
                    .iter()
                    .map(|&i| samples[i].ink[2] - samples[i].ink[0])
                    .collect::<Vec<_>>()),
                g.len(),
            )
        };
        for g in &groups[1..] {
            if score(g) > score(dominant) {
                dominant = g
            }
        }
        let supported = groups
            .iter()
            .filter(|g| g.len() >= 3 && g.len() as f64 / anchors.len() as f64 >= 0.2)
            .count();
        let run = mode(dominant, samples);
        analyses.push(Analysis {
            key: k,
            line: line_index,
            ids: ids.clone(),
            anchors: anchors.clone(),
            dominant: dominant.clone(),
            baseline: med(dominant.iter().map(|&i| samples[i].base[1]).collect()),
            tight: union(dominant.iter().map(|&i| samples[i].ink)),
            source: union(anchors.iter().map(|&i| samples[i].local)),
            all: union(ids.iter().map(|&i| samples[i].local)),
            legacy: g::rotate(
                bbox(&l["bbox"])?.n,
                pages[k.0].size,
                keys[s.run].angle as i32,
            ),
            run,
            coverage: ids.len() as f64 / visible as f64,
            share: dominant.len() as f64 / anchors.len() as f64,
            split: supported >= 2,
            neighbors: Vec::new(),
            intrusion: 0.,
            candidate: false,
        });
    }
    for i in 0..analyses.len() {
        let a = &analyses[i];
        let mut above = None;
        let mut below = None;
        for (j, b) in analyses.iter().enumerate() {
            w.charge(1)?;
            if i == j || a.key.0 != b.key.0 {
                continue;
            }
            let h = (a.tight[3] - a.tight[1]).max(b.tight[3] - b.tight[1]);
            if !(axis(a.source, b.source) >= 0.35
                || (a.source[0] - b.source[0]).abs() <= 2. * h.max(1.))
                || (b.baseline - a.baseline).abs() < 2f64.max(0.6 * (a.tight[3] - a.tight[1]))
            {
                continue;
            }
            if b.baseline < a.baseline
                && above.is_none_or(|k: usize| b.baseline > analyses[k].baseline)
            {
                above = Some(j)
            }
            if b.baseline > a.baseline
                && below.is_none_or(|k: usize| b.baseline < analyses[k].baseline)
            {
                below = Some(j)
            }
        }
        analyses[i].neighbors = above.into_iter().chain(below).collect();
    }
    let mut eligible: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..analyses.len() {
        if analyses[i].anchors.len() < 4 || analyses[i].coverage < 0.8 || analyses[i].share < 0.8 {
            continue;
        }
        let a = &analyses[i];
        let intrusion = a
            .neighbors
            .iter()
            .map(|&j| {
                let b = analyses[j].tight;
                let h = b[3] - b[1];
                if h > 0. {
                    (a.source[3].min(b[3]) - a.source[1].max(b[1])).max(0.) / h
                } else {
                    0.
                }
            })
            .fold(0., f64::max);
        analyses[i].intrusion = intrusion;
        analyses[i].candidate = intrusion >= 0.25;
        eligible.entry(analyses[i].run).or_default().push(i);
    }
    let confirmed = eligible
        .iter()
        .filter_map(|(&r, ids)| {
            let n = ids.iter().filter(|&&i| analyses[i].candidate).count();
            (n >= 3 && n as f64 / ids.len() as f64 >= 0.5).then_some(r)
        })
        .collect::<BTreeSet<_>>();
    let mut healthy = BTreeMap::<usize, (Vec<f64>, Vec<f64>)>::new();
    for a in &analyses {
        if !confirmed.contains(&a.run) || a.candidate {
            continue;
        }
        let v = healthy.entry(a.run).or_default();
        for &i in &a.dominant {
            let s = &samples[i];
            v.0.push((s.base[1] - s.local[1]).max(0.));
            v.1.push((s.local[3] - s.base[1]).max(0.));
        }
    }
    for a in &analyses {
        let l = &pages[a.key.0].lines[a.line];
        let trim = confirmed.contains(&a.run)
            && (a.candidate || bad.contains(&a.run))
            && a.anchors.len() >= 4
            && a.coverage >= 0.8
            && a.share >= 0.8;
        if !plan.lines.contains_key(&a.key) && !a.split && !trim {
            continue;
        }
        let source = bbox(
            l.get("source_bbox")
                .filter(|b| !b.is_null())
                .unwrap_or(&l["bbox"]),
        )?
        .n;
        let ink = union(a.ids.iter().map(|&i| samples[i].tight));
        let mut repair = plan.lines.get(&a.key).cloned().unwrap_or(line_repair(
            source,
            bbox(&l["bbox"])?.n,
            Some(ink),
            Some(a.baseline),
            l.get("effective_height").map_or(Ok(0.), num)?,
        ));
        repair["split_y_candidate"] = json!(a.split);
        repair["baseline"] = json!(a.baseline);
        repair["run_key"] = keys[a.run].value();
        if !trim {
            plan.lines.insert(a.key, repair);
            continue;
        }
        let tight = union(a.ids.iter().map(|&i| samples[i].ink));
        let em = med(a
            .dominant
            .iter()
            .filter_map(|&i| (samples[i].size > 0.).then_some(samples[i].size))
            .collect())
        .max(quantile(
            a.dominant
                .iter()
                .map(|&i| samples[i].ink[3] - samples[i].ink[1])
                .collect(),
            0.75,
        ))
        .max(0.1);
        let (h_asc, h_desc) = healthy.get(&a.run).cloned().unwrap_or_default();
        let height = if !h_asc.is_empty() && !h_desc.is_empty() {
            med(h_asc.clone()) + med(h_desc.clone())
        } else {
            f64::INFINITY
        };
        let tier = tiers(
            a.dominant
                .iter()
                .filter(|&&i| samples[i].run == a.run)
                .map(|&i| {
                    let s = &samples[i];
                    [
                        (s.base[1] - s.local[1]).max(0.),
                        (s.local[3] - s.base[1]).max(0.),
                        s.ink[3] - s.ink[1],
                        s.size,
                    ]
                })
                .collect(),
            em,
        );
        let (asc, desc) = if let Some(t) = tier {
            t
        } else if h_asc.len() >= 20 && height <= 1.5 * em {
            (med(h_asc), med(h_desc))
        } else {
            let pad = 0.5f64.max(0.08 * em);
            (
                quantile(
                    a.dominant
                        .iter()
                        .map(|&i| (samples[i].base[1] - samples[i].ink[1]).max(0.))
                        .collect(),
                    0.9,
                ) + pad,
                quantile(
                    a.dominant
                        .iter()
                        .map(|&i| (samples[i].ink[3] - samples[i].base[1]).max(0.))
                        .collect(),
                    0.9,
                ) + pad,
            )
        };
        let mut top = (a.baseline - asc).min(tight[1]);
        let mut bottom = (a.baseline + desc).max(tight[3]);
        for &j in &a.neighbors {
            let b = &analyses[j];
            let mid = (a.baseline + b.baseline) / 2.;
            if b.baseline < a.baseline {
                top = top.max(mid.min(tight[1]))
            } else {
                bottom = bottom.min(mid.max(tight[3]))
            }
        }
        let tolerance = 1.05f64.max(0.05 * (a.legacy[3] - a.legacy[1]));
        let preserve = (a.all[1] - a.legacy[1]).abs() > tolerance
            || (a.all[3] - a.legacy[3]).abs() > tolerance;
        if preserve {
            top = a.legacy[1];
            bottom = a.legacy[3]
        }
        let p = pages[a.key.0].size;
        let local = g::rotate(bbox(&repair["layout_bbox"])?.n, p, keys[a.run].angle as i32);
        if let Some(layout) = clipped(
            world([local[0], top, local[2], bottom], p, keys[a.run].angle),
            p,
        ) {
            repair["layout_bbox"] = json!(layout);
            repair["ink_bbox"] = json!(ink);
            repair["em_height"] = if preserve {
                l.get("effective_height").cloned().unwrap_or(json!(0.))
            } else {
                json!((bottom - top).max(0.1))
            };
            repair["state"] = json!(if repair["state"] == "repair_x" {
                "repair_xy"
            } else {
                "trim_y"
            });
            repair["confidence"] = json!(a.coverage.min(a.share));
            repair["y_intrusion_ratio"] = json!(a.intrusion)
        }
        plan.lines.insert(a.key, repair);
    }
    Ok(())
}
fn family(s: &str) -> String {
    let mut end = s.len();
    loop {
        let old = end;
        for suffix in [
            "semibold", "demibold", "regular", "regu", "bold", "light", "medium", "italic", "ital",
            "oblique",
        ] {
            if s[..end].ends_with(suffix) {
                end -= suffix.len();
                break;
            }
        }
        if old == end {
            break;
        }
    }
    s[..end].into()
}
pub(super) fn extreme(
    plan: &mut Plan,
    s: &[Sample],
    by_line: &[(Key, Vec<usize>)],
    keys: &[RunKey],
    pages: &[Page],
    w: &mut Work,
) -> Result<(), Error> {
    type Row = (Key, Vec<usize>, f64, f64);
    let mut groups: Vec<((usize, String, u64, i64), Vec<Row>)> = Vec::new();
    for &(key, ref ids) in by_line {
        let anchors = ids
            .iter()
            .copied()
            .filter(|&i| s[i].anchor)
            .collect::<Vec<_>>();
        if anchors.len() < 4 {
            continue;
        }
        let dominant = mode(&anchors, s);
        let k = &keys[dominant];
        let f = family(&k.family);
        let members = anchors
            .iter()
            .copied()
            .filter(|&i| {
                family(&keys[s[i].run].family) == f
                    && keys[s[i].run].size == k.size
                    && keys[s[i].run].angle == k.angle
            })
            .collect::<Vec<_>>();
        w.charge(ids.len() * 2 + 1)?;
        if (members.len() as f64) < 0.8 * anchors.len() as f64 {
            continue;
        }
        let h = quantile(
            members.iter().map(|&i| s[i].ink[3] - s[i].ink[1]).collect(),
            0.75,
        );
        let baseline = med(members.iter().map(|&i| s[i].base[1]).collect());
        if h <= 0.
            || (members
                .iter()
                .filter(|&&i| (s[i].base[1] - baseline).abs() <= 0.2 * h)
                .count() as f64)
                < 0.85 * members.len() as f64
        {
            continue;
        }
        let group_key = (key.0, f, k.size.to_bits(), k.angle);
        if let Some((_, g)) = groups.iter_mut().find(|(key, _)| *key == group_key) {
            g.push((key, members, h, baseline))
        } else {
            groups.push((group_key, vec![(key, members, h, baseline)]));
        }
    }
    for (_, rows) in groups {
        let bad = rows
            .iter()
            .filter(|(_, ids, h, _)| {
                pages[s[ids[0]].page].lines[s[ids[0]].line]["effective_height"]
                    .as_f64()
                    .unwrap_or(0.)
                    >= 4. * h
                    && med(ids.iter().map(|&i| s[i].size).collect()) <= 0.3 * h
            })
            .collect::<Vec<_>>();
        if bad.len() < 6 || (bad.len() as f64) < 0.6 * rows.len() as f64 {
            continue;
        }
        let donors = rows
            .iter()
            .filter(|(_, ids, h, _)| {
                let ratio = pages[s[ids[0]].page].lines[s[ids[0]].line]["effective_height"]
                    .as_f64()
                    .unwrap_or(0.)
                    / h;
                ids.len() >= 12 && (1.2..=2.5).contains(&ratio)
            })
            .collect::<Vec<_>>();
        if donors.is_empty() {
            continue;
        }
        let asc = med(donors
            .iter()
            .map(|(_, ids, h, b)| (b - s[ids[0]].local[1]) / h)
            .collect());
        let desc = med(donors
            .iter()
            .map(|(_, ids, h, b)| (s[ids[0]].local[3] - b) / h)
            .collect());
        if asc <= 0. || desc < 0. || !(1.2..=2.5).contains(&(asc + desc)) {
            continue;
        }
        for &(key, ref members, h, baseline) in bad {
            let ids = &by_line.iter().find(|(k, _)| *k == key).unwrap().1;
            let first = &s[members[0]];
            let line = &pages[key.0].lines[first.line];
            let a = keys[first.run].angle;
            let size = pages[key.0].size;
            let ink = union(ids.iter().map(|&i| s[i].tight));
            let local_ink = g::rotate(ink, size, a as i32);
            let local = g::rotate(bbox(&line["bbox"])?.n, size, a as i32);
            let bounds = [
                local[0],
                local_ink[1].min(baseline - asc * h),
                local[2],
                local_ink[3].max(baseline + desc * h),
            ];
            let Some(layout) = clipped(world(bounds, size, a), size) else {
                continue;
            };
            let scale = h * (asc + desc);
            let mut repair = line_repair(
                bbox(
                    line.get("source_bbox")
                        .filter(|v| !v.is_null())
                        .unwrap_or(&line["bbox"]),
                )?
                .n,
                layout,
                Some(ink),
                Some(baseline),
                scale,
            );
            repair["state"] = json!(if plan
                .lines
                .get(&key)
                .is_some_and(|r| r["state"] == "repair_x" || r["state"] == "repair_xy")
            {
                "repair_xy"
            } else {
                "trim_y"
            });
            repair["run_key"] = keys[first.run].value();
            plan.lines.insert(key, repair);
            plan.scales.insert(key, scale);
            plan.local.insert(key);
            for &i in ids {
                w.charge(1)?;
                let sample = &s[i];
                if sample.local[3] - sample.local[1] < 4. * h {
                    continue;
                }
                let old = plan.chars.get(&(key.0, sample.index));
                let b = if let Some(old) = old {
                    bbox(&old["layout_bbox"])?.n
                } else {
                    sample.source
                };
                let local = g::rotate(b, size, a as i32);
                let delta = sample.base[1] - baseline;
                let Some(layout) = clipped(
                    world(
                        [
                            local[0],
                            (bounds[1] + delta).min(sample.ink[1]),
                            local[2],
                            (bounds[3] + delta).max(sample.ink[3]),
                        ],
                        size,
                        a,
                    ),
                    size,
                ) else {
                    continue;
                };
                let repair = json!({"source_bbox":sample.source,"tight_bbox":sample.tight,"origin":sample.origin,"layout_bbox":layout,"ink_bbox":sample.tight,"baseline":sample.base[1],"advance":old.map(|o|o["advance"].clone()).unwrap_or(Value::Null),"em_height":scale,"x_state":old.map(|o|o["x_state"].clone()).unwrap_or(json!("healthy")),"y_state":"abnormal","confidence":1.});
                plan.chars.insert((key.0, sample.index), repair);
            }
        }
    }
    Ok(())
}
