use super::super::table_native_text::source_index;
use super::*;
use rapiddoc_native::text_boundary::python_space;
fn squeezed(s: &str) -> String {
    s.chars().filter(|c| !python_space(*c)).collect()
}
pub(super) fn build(
    v: &Value,
    text: &Text,
    d: &mut Diagnostic,
    work: &mut Work,
) -> Result<Option<Value>, Error> {
    d.extend(
        json!({"evidence":"grouped_rule_band","first_rejection_gate":"group_bands"})
            .as_object()
            .unwrap()
            .clone(),
    );
    let input = &v["table_input"];
    let (w, h, _) = sparse::dimensions(input)?;
    let rules = sparse::local_rules(input, w, h)?;
    let tol = 0.5f64.max(0.10 * text.median_glyph_height);
    let horizontal: Vec<_> = sparse::clusters(
        rules
            .iter()
            .filter(|r| {
                r.orientation == "horizontal"
                    && r.end - r.start >= 0.90 * w
                    && -3. * tol <= r.coordinate
                    && r.coordinate <= h + 3. * tol
            })
            .map(|r| r.coordinate)
            .collect(),
        tol,
        work,
    )?
    .into_iter()
    .map(|(p, _)| p)
    .collect();
    if horizontal.len() < 4 || horizontal[0] > 3. * tol {
        return Ok(None);
    }
    work.charge(horizontal.len() * (text.rows.len() + rules.len()))?;
    let mut bands = BTreeMap::new();
    for p in horizontal.windows(2) {
        let members: Vec<_> = text
            .rows
            .iter()
            .filter(|r| p[0] < center(r) && center(r) < p[1])
            .collect();
        if members.len() != 1 {
            continue;
        }
        let r = members[0];
        if r.row_index == 0 || r.tokens.len() != 1 {
            continue;
        }
        if r.bbox[1] < p[0] - tol || r.bbox[3] > p[1] + tol {
            return Ok(None);
        }
        if rules.iter().any(|r| {
            r.orientation == "vertical"
                && tol < r.coordinate
                && r.coordinate < w - tol
                && p[1].min(r.end) - p[0].max(r.start) > tol
        }) {
            return Ok(None);
        }
        bands.insert(r.row_index as usize, (p[0], p[1]));
    }
    if bands.len() < 2 || bands.first_key_value().map(|(i, _)| *i) != Some(1) {
        return Ok(None);
    }
    if sparse::overlapping(text, -1.) {
        return Ok(reject(d, "overlapping_formula_rows"));
    }
    let indices: Vec<_> = bands.keys().copied().collect();
    let ends: Vec<_> = indices
        .iter()
        .skip(1)
        .copied()
        .chain([text.rows.len()])
        .collect();
    if indices.iter().zip(ends).any(|(a, b)| b - a < 2) {
        return Ok(None);
    }
    let selected: Vec<_> = text
        .rows
        .iter()
        .filter(|r| r.row_index > 0 && !bands.contains_key(&(r.row_index as usize)))
        .collect();
    let mapping: BTreeMap<_, _> = selected
        .iter()
        .enumerate()
        .map(|(i, r)| (r.row_index, i))
        .collect();
    let data = Text {
        rows: selected
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let mut r = (*r).clone();
                r.row_index = i as i64;
                r
            })
            .collect(),
        glyphs: text
            .glyphs
            .iter()
            .filter_map(|g| {
                mapping.get(&g.visual_row).map(|i| {
                    let mut g = g.clone();
                    g.visual_row = *i as i64;
                    g
                })
            })
            .collect(),
        median_glyph_height: text.median_glyph_height,
        median_glyph_width: text.median_glyph_width,
    };
    let Some(cols) = sparse::target(&data).filter(|c| *c >= 3) else {
        return Ok(reject(d, "column_count"));
    };
    let Some(x) = sparse::text_tracks(&data, w, cols, work)? else {
        return Ok(reject(d, "column_tracks"));
    };
    d.extend(
        json!({"x_tracks":x,"group_rows":indices})
            .as_object()
            .unwrap()
            .clone(),
    );
    let glyphs = text.glyphs.iter().map(|g| (g.glyph_id, g)).collect();
    let full: BTreeSet<_> = (0..cols).collect();
    let suffix: BTreeSet<_> = (1..cols).collect();
    for row in &data.rows {
        let o = sparse::occupancy(row, &glyphs, &x, work)?;
        if (o != full && o != suffix) || row.tokens.len() != o.len() {
            return Ok(reject(d, "data_occupancy"));
        }
    }
    let header = &text.rows[0];
    let o = sparse::occupancy(header, &glyphs, &x, work)?;
    if o != full && o != suffix {
        return Ok(reject(d, "header_occupancy"));
    }
    let margin = 0.15f64.max(0.03 * text.median_glyph_width);
    work.charge(header.tokens.len() * cols)?;
    if header.tokens.iter().any(|t| {
        x[1..cols]
            .iter()
            .any(|b| t.bbox[0] + margin < *b && *b < t.bbox[2] - margin)
    }) {
        return Ok(reject(d, "header_token_split"));
    }
    if !sparse::gutters(&data, &x, &BTreeSet::new(), -1., d, work)? {
        return Ok(reject(d, "gutter_support"));
    }
    let mut y = vec![0.];
    for p in text.rows.windows(2) {
        let current = bands.get(&(p[1].row_index as usize));
        let previous = bands.get(&(p[0].row_index as usize));
        let boundary = if let Some(b) = current {
            b.0
        } else if let Some(b) = previous {
            b.1
        } else {
            table_native_candidate::sum(&[p[0].bbox[1], p[0].bbox[3], p[1].bbox[1], p[1].bbox[3]])
                / 4.
        };
        if (current.is_some() || previous.is_some())
            && !(p[0].bbox[3] - tol <= boundary && boundary <= p[1].bbox[1] + tol)
        {
            return Ok(reject(d, "row_boundary"));
        }
        y.push(boundary);
    }
    y.push(h);
    if text.rows.len() * cols > rapiddoc_core::resources::limit(2000) {
        return Err(limit());
    }
    let mut specs = Vec::new();
    for r in &text.rows {
        let i = r.row_index as usize;
        if bands.contains_key(&i) {
            specs.push(Spec {
                row: i,
                col: 0,
                rowspan: 1,
                colspan: cols,
                bbox: [0., y[i], w, y[i + 1]],
            });
        } else {
            for c in 0..cols {
                specs.push(Spec {
                    row: i,
                    col: c,
                    rowspan: 1,
                    colspan: 1,
                    bbox: [x[c], y[i], x[c + 1], y[i + 1]],
                });
            }
        }
    }
    work.charge(text.glyphs.len() * specs.len() * 3 + text.glyphs.len() * text.glyphs.len() * 2)?;
    let output = table_native_candidate::build(
        &json!({"source":"sparse_multiline","rows":text.rows.len(),"cols":cols,"specs":specs,"text":v["text"],"structure_support":1.,"row_stability":1.,"column_stability":1.,"use_grid_index":true,"issues":["evidence=grouped_rule_band",format!("group_rows={}",indices.iter().map(usize::to_string).collect::<Vec<_>>().join(","))]}),
    )?;
    let mut candidate = output["candidate"].clone();
    let record = &output["diagnostics"];
    d.insert("candidate_diagnostics".into(), record.clone());
    if candidate.is_null()
        || candidate["text_capture"].as_f64().unwrap() < 1.
        || candidate["order_consistency"].as_f64().unwrap() < 1.
        || candidate["score"].as_f64().unwrap() < 0.98
        || record["ambiguous_glyph_ratio"].as_f64().unwrap_or(1.) > 0.
        || record["token_split_count"].as_u64().unwrap_or(1) > 0
    {
        return Ok(reject(d, "verified_integrity"));
    }
    let chars = input.get("chars").map_or(Ok(&[][..]), |r| {
        r.as_array().map(|a| a.as_slice()).ok_or_else(invalid)
    })?;
    if chars.len() > rapiddoc_core::resources::limit(20000) {
        return Err(limit());
    }
    let mut raw: Vec<(i64, &Value)> = Vec::new();
    let mut positions = BTreeMap::new();
    for (i, c) in chars.iter().enumerate() {
        let id = source_index(c, i)?;
        if let Some(p) = positions.get(&id) {
            raw[*p] = (id, c);
        } else {
            positions.insert(id, raw.len());
            raw.push((id, c));
        }
    }
    let mut visible: Vec<_> = raw
        .iter()
        .filter(|(_, c)| {
            !spatial::raw_string(c.get("char"), false)
                .trim_matches(python_space)
                .is_empty()
        })
        .map(|(i, _)| *i)
        .collect();
    visible.sort();
    for cell in candidate["cells"].as_array_mut().unwrap() {
        if cell["colspan"].as_u64() != Some(cols as u64) {
            continue;
        }
        let sources: BTreeSet<_> = cell["source_char_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| n.as_i64().unwrap())
            .collect();
        let Some(first) = sources.first() else {
            return Ok(reject(d, "group_text_integrity"));
        };
        let last = sources.last().unwrap();
        let mut selected = Vec::new();
        work.charge(raw.len())?;
        for (i, c) in &raw {
            if sources.contains(i) {
                selected.push(*c);
            } else if first < i && i < last {
                let s = spatial::raw_string(c.get("char"), false);
                if !s.is_empty() && s.chars().all(python_space) {
                    let p = visible.partition_point(|v| v < i);
                    if p > 0
                        && p < visible.len()
                        && sources.contains(&visible[p - 1])
                        && sources.contains(&visible[p])
                    {
                        selected.push(*c);
                    }
                }
            }
        }
        let content = spatial::project(
            &selected,
            &input["table_bbox"],
            sparse::dimensions(input)?.2,
            work,
        )?;
        let content = content
            .split(python_space)
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if squeezed(&content) != squeezed(cell["content"].as_str().unwrap()) {
            return Ok(reject(d, "group_text_integrity"));
        }
        cell["content"] = json!(content);
    }
    d.extend(json!({"first_rejection_gate":null,"grid":{"rows":candidate["rows"],"cols":candidate["cols"]}}).as_object().unwrap().clone());
    Ok(Some(candidate))
}
