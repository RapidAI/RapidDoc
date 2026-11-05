//! 浅色表头及原生裁剪路径的阶段网格完整旧规则。
use super::*;
fn numeric(text: &str) -> bool {
    !text.is_empty()
        && text
            .chars()
            .all(|c| python_decimal(c).is_some() || python_space(c) || ".,%$€£()−+/-".contains(c))
}
fn small_number(text: &str) -> bool {
    (1..=3).contains(&text.chars().count()) && text.chars().all(|c| python_decimal(c).is_some())
}
fn stage_number(text: &str) -> Option<u32> {
    let chars: Vec<_> = text.trim_start_matches(python_space).chars().collect();
    let mut i = 0;
    let mut n = 0;
    while i < 2 && chars.get(i).is_some_and(|c| python_decimal(*c).is_some()) {
        n = n * 10 + python_decimal(chars[i])? as u32;
        i += 1;
    }
    if i == 0 || !chars.get(i).is_some_and(|c| ".)、".contains(*c)) {
        return None;
    }
    i += 1;
    while chars.get(i).is_some_and(|c| python_space(*c)) {
        i += 1;
    }
    chars
        .get(i)
        .filter(|c| c.is_ascii_alphabetic() || ('\u{3400}'..='\u{9fff}').contains(c))
        .map(|_| n)
}
fn stage_grid(
    v: &Request,
    labels: &[&Value],
    b: &Box4,
    work: &mut Work,
) -> Result<Vec<Value>, Error> {
    if labels.len() < 3 {
        return Ok(vec![]);
    }
    let source = v.source;
    let lines = optional(source, "lines", 4096)?;
    let paths = optional(source, "path_infos", 5000)?;
    let mut heights = Vec::new();
    for l in lines {
        let h = effective(l)?;
        if h > 0. {
            heights.push(h);
        }
    }
    if heights.is_empty() {
        return Err(invalid());
    }
    let em = median(heights);
    let mut columns = Vec::new();
    for label in labels {
        let lb = bbox(&label["bbox"])?;
        let mut boxes = Vec::new();
        for path in paths {
            work.charge(1)?;
            if integer(path, "segment_count", 0)? != 5
                || flag(path, "fill_visible")?
                || flag(path, "stroke_visible")?
            {
                continue;
            }
            let pb = bbox(&path["bbox"])?;
            if (pb.n[0] - lb.n[0]).abs() <= 0.25 * em
                && b.n[1] <= pb.n[1]
                && pb.n[1] < b.n[3]
                && pb.n[2] <= b.n[2]
                && pb.n[2] - pb.n[0] >= 3. * em
            {
                boxes.push(pb);
            }
        }
        if boxes.len() < 6 {
            return Ok(vec![]);
        }
        let width = median(boxes.iter().map(|b| b.n[2] - b.n[0]).collect());
        boxes.retain(|b| ((b.n[2] - b.n[0]) - width).abs() <= 0.3 * em);
        if boxes.len() < 6 {
            return Ok(vec![]);
        }
        columns.push(boxes);
    }
    let label_bottom = labels
        .iter()
        .map(|l| l["bbox"][3].as_f64().unwrap())
        .reduce(f64::max)
        .unwrap();
    let label_left = labels[1]["bbox"][0].as_f64().unwrap();
    let mut function_boxes = Vec::new();
    for pb in &columns[1] {
        work.charge(lines.len() + 1)?;
        if pb.n[1] > label_bottom
            && lines.iter().any(|l| {
                let b = bbox(&l["bbox"]).unwrap();
                (b.n[0] - label_left).abs() <= 0.25 * em
                    && pb.n[1] <= center(&b, 1)
                    && center(&b, 1) <= pb.n[3]
            })
        {
            function_boxes.push(pb);
        }
    }
    function_boxes.sort_by(|a, b| cmp(a.n[1], b.n[1]));
    if function_boxes.len() < 5 || function_boxes.windows(2).any(|p| p[0].n[3] >= p[1].n[1]) {
        return Ok(vec![]);
    }
    let rights: Vec<_> = columns
        .iter()
        .map(|c| c.iter().map(|b| b.n[2]).reduce(f64::max).unwrap())
        .collect();
    let mut grouped = vec![Vec::new(); function_boxes.len()];
    for line in lines {
        work.charge(labels.len() + function_boxes.len() + 1)?;
        let lb = bbox(&line["bbox"])?;
        if labels.contains(&line)
            || line.get("angle").map_or(Ok(0.), num)? != 0.
            || lb.n[1] < function_boxes[0].n[1] - 0.9 * em
        {
            continue;
        }
        let column = labels
            .iter()
            .enumerate()
            .find(|(i, l)| {
                (lb.n[0] - l["bbox"][0].as_f64().unwrap()).abs() <= 0.3 * em
                    && lb.n[2] <= rights[*i] + 0.3 * em
            })
            .map(|(i, _)| i);
        let Some(column) = column else {
            continue;
        };
        let matching = function_boxes
            .iter()
            .rposition(|b| b.n[1] - 0.9 * em <= lb.n[1]);
        if let Some(i) = matching {
            if lb.n[3] <= b.n[3] {
                grouped[i].push((column, line));
            }
        }
    }
    if grouped.iter().any(|row| {
        row.iter()
            .filter(|(i, _)| *i > 0)
            .map(|(i, _)| i)
            .collect::<BTreeSet<_>>()
            .len()
            != labels.len() - 1
    }) {
        return Ok(vec![]);
    }
    let mut stages = Vec::new();
    for (i, row) in grouped.iter().enumerate() {
        let mut first = row
            .iter()
            .filter(|(i, _)| *i == 0)
            .map(|(_, l)| *l)
            .collect::<Vec<_>>();
        first.sort_by(|a, b| {
            cmp(
                a["bbox"][1].as_f64().unwrap(),
                b["bbox"][1].as_f64().unwrap(),
            )
        });
        if let Some(l) = first.first() {
            let Some(n) = stage_number(l["text"].as_str().ok_or_else(invalid)?) else {
                return Ok(vec![]);
            };
            stages.push((i, n));
        }
    }
    if stages.len() < 3
        || stages[0].0 != 0
        || stages.len() == grouped.len()
        || stages.windows(2).any(|p| p[1].1 != p[0].1 + 1)
    {
        return Ok(vec![]);
    }
    let mut xs = vec![b.n[0]];
    for (i, right) in rights[..rights.len() - 1].iter().enumerate() {
        xs.push((right + labels[i + 1]["bbox"][0].as_f64().unwrap()) / 2.);
    }
    xs.push(b.n[2]);
    let mut row_boxes = Vec::new();
    for row in &grouped {
        let mut rb = bbox(&row[0].1["bbox"])?;
        for (_, l) in &row[1..] {
            rb = union(rb, &bbox(&l["bbox"])?);
        }
        row_boxes.push(rb);
    }
    if row_boxes.windows(2).any(|p| p[0].n[3] >= p[1].n[1]) {
        return Ok(vec![]);
    }
    let mut ys = vec![b.n[1], (label_bottom + row_boxes[0].n[1]) / 2.];
    for pair in row_boxes.windows(2) {
        ys.push((pair[0].n[3] + pair[1].n[1]) / 2.);
    }
    ys.push(b.n[3]);
    let mut grid = xs.iter().map(|x| axis(b, *x, false)).collect::<Vec<_>>();
    let starts = stages.iter().map(|(i, _)| i + 1).collect::<BTreeSet<_>>();
    let last = ys.len() - 1;
    for (i, y) in ys.into_iter().enumerate() {
        let left = if i == 0 || i == 1 || i == last || starts.contains(&i) {
            b.raw[0].clone()
        } else {
            json!(xs[1])
        };
        grid.push(
            json!({"bbox":[left,y-0.05,b.raw[2],y+0.05],"width":0.1,"orientation":"horizontal"}),
        );
    }
    Ok(grid)
}
pub(super) fn detect(v: &Request, work: &mut Work) -> Result<Value, Error> {
    let source = v.source;
    let lines = optional(source, "lines", 4096)?;
    let mut eligible = Vec::new();
    let mut heights = Vec::new();
    for l in lines {
        work.charge(1)?;
        if l.get("angle").map_or(Ok(0.), num)? != 0. {
            continue;
        }
        let sem = l.get("semantic_type").filter(|v| !v.is_null());
        if sem.is_some_and(|s| s != "header" && s != "footer") {
            continue;
        }
        let h = effective(l)?;
        if h > 0. {
            heights.push(h);
        }
        eligible.push(l);
    }
    if eligible.is_empty() {
        return Ok(json!({"candidates":[]}));
    }
    if heights.is_empty() {
        return Err(invalid());
    }
    let em = median(heights);
    let p = prepare_source(source, &value_object!({"angle":0,"median_height":em}), work)?;
    let excluded = exclusions(v.options, &p, work)?;
    for l in lines {
        bbox(&l["bbox"])?;
    }
    let images = optional(source, "image_bboxes", 5000)?
        .iter()
        .map(bbox)
        .collect::<Result<Vec<_>, _>>()?;
    let mut cells = Vec::new();
    for path in optional(source, "path_infos", 5000)? {
        work.charge(excluded.len() + 1)?;
        if !flag(path, "fill_visible")? || integer(path, "segment_count", 0)? != 5 {
            continue;
        }
        let rgba = optional(path, "fill_rgba", 4)?;
        if rgba.is_empty() {
            continue;
        }
        if rgba.len() < 3 {
            return Err(invalid());
        }
        let pb = bbox(&path["bbox"])?;
        if rgba[..3]
            .iter()
            .map(num)
            .collect::<Result<Vec<_>, _>>()?
            .iter()
            .all(|x| *x >= 180.)
            && 0.5 * em <= pb.n[3] - pb.n[1]
            && pb.n[3] - pb.n[1] <= 2.5 * em
            && !excluded.iter().any(|e| overlap_smaller(&pb, e) >= 0.5)
        {
            cells.push(pb);
        }
    }
    let mut ordered: Vec<_> = cells.iter().collect();
    ordered.sort_by(|a, b| cmp(a.n[1], b.n[1]).then(cmp(a.n[0], b.n[0])));
    let mut headers: Vec<Box4> = Vec::new();
    for cell in ordered {
        if let Some(last) = headers.last_mut() {
            if (cell.n[1] - last.n[1]).abs() <= 0.2 * em
                && (cell.n[3] - last.n[3]).abs() <= 0.2 * em
                && 0. <= cell.n[0] - last.n[2]
                && cell.n[0] - last.n[2] <= 0.2 * em
            {
                *last = union(last.clone(), cell);
                continue;
            }
        }
        headers.push(cell.clone());
    }
    for image in &images {
        work.charge(eligible.len() + 1)?;
        let inside = eligible
            .iter()
            .filter(|l| overlap_first(&bbox(&l["bbox"]).unwrap(), image) >= 0.9)
            .collect::<Vec<_>>();
        if inside.len() >= 20 {
            let top = inside
                .iter()
                .map(|l| l["bbox"][1].as_f64().unwrap())
                .reduce(f64::min)
                .unwrap();
            headers.push(bbox(&json!([
                image.raw[0],
                top - 0.2 * em,
                image.raw[2],
                top + 1.5 * em
            ]))?);
        }
    }
    let mut ordered: Vec<_> = headers.iter().collect();
    ordered.sort_by(|a, b| cmp(a.n[1], b.n[1]));
    let mut output = Vec::new();
    let mut total_members = 0;
    let mut total_grid = 0;
    for header in ordered {
        work.charge(eligible.len() * 2 + cells.len() + headers.len() + images.len() + 1)?;
        let mut labels = eligible
            .iter()
            .copied()
            .filter(|l| overlap_first(&bbox(&l["bbox"]).unwrap(), header) >= 0.8)
            .collect::<Vec<_>>();
        labels.sort_by(|a, b| {
            cmp(
                a["bbox"][0].as_f64().unwrap(),
                b["bbox"][0].as_f64().unwrap(),
            )
        });
        if !(3..=8).contains(&labels.len())
            || labels
                .windows(2)
                .any(|p| p[0]["bbox"][2].as_f64().unwrap() + em > p[1]["bbox"][0].as_f64().unwrap())
        {
            continue;
        }
        let columns = labels
            .iter()
            .map(|l| l["bbox"][0].as_f64().unwrap())
            .collect::<Vec<_>>();
        let mut header_cells = cells
            .iter()
            .filter(|c| overlap_first(c, header) >= 0.9)
            .collect::<Vec<_>>();
        header_cells.sort_by(|a, b| cmp(a.n[0], b.n[0]));
        let raster = images
            .iter()
            .any(|i| overlap_first(header, i) >= 0.9 && i.n[3] - header.n[3] >= 8. * em);
        let floor = headers
            .iter()
            .filter(|h| {
                h.n[1] > header.n[3] + em
                    && super::super::super::graphic_captions::axis(header, h, 0) >= 0.8
            })
            .map(|h| h.n[1])
            .reduce(f64::min)
            .unwrap_or(num(&p.size[1])?);
        let peers = eligible
            .iter()
            .copied()
            .filter(|l| {
                let b = bbox(&l["bbox"]).unwrap();
                header.n[3] <= center(&b, 1)
                    && center(&b, 1) < floor
                    && header.n[0] <= b.n[0]
                    && b.n[2] <= header.n[2] + 0.25 * em
            })
            .collect::<Vec<_>>();
        let grouped = rows(&peers, em, work)?;
        let mut numeric_rows = 0;
        let mut multi = 0;
        let mut members = labels.clone();
        let mut bottom = header.n[3];
        let mut accepted = vec![labels.clone()];
        for row in grouped {
            work.charge(row.len() * (columns.len() + header_cells.len() + 1))?;
            if row
                .iter()
                .map(|l| l["bbox"][1].as_f64().unwrap())
                .reduce(f64::min)
                .unwrap()
                - bottom
                > 2.5 * em
            {
                break;
            }
            let is_numeric = row.iter().all(|l| numeric(l["text"].as_str().unwrap()));
            let mut bad = false;
            for l in &row {
                let lb = bbox(&l["bbox"])?;
                if columns
                    .iter()
                    .map(|x| (lb.n[0] - x).abs())
                    .reduce(f64::min)
                    .unwrap()
                    > 0.6 * em
                    && !(is_numeric
                        && header_cells
                            .iter()
                            .any(|c| c.n[0] <= center(&lb, 0) && center(&lb, 0) <= c.n[2]))
                    && !(raster
                        && row.len() >= 2
                        && effective(l)? <= em
                        && small_number(l["text"].as_str().unwrap()))
                {
                    bad = true;
                }
            }
            if bad {
                break;
            }
            if is_numeric {
                numeric_rows += 1;
            } else if row.len() >= 2 {
                multi += 1;
            } else if words(row[0]) > 5 && !raster {
                break;
            }
            bottom = bottom.max(
                row.iter()
                    .map(|l| l["bbox"][3].as_f64().unwrap())
                    .reduce(f64::max)
                    .unwrap(),
            );
            members.extend(row.iter().copied());
            accepted.push(row);
        }
        let mut horizontal = 0;
        for r in optional(source, "drawing_lines", 5000)? {
            work.charge(1)?;
            if r["orientation"] == "horizontal" {
                let b = bbox(&r["bbox"])?;
                if header.n[1] <= b.n[1]
                    && b.n[1] <= bottom + em
                    && super::super::super::graphic_captions::axis(header, &b, 0) >= 0.8
                {
                    horizontal += 1;
                }
            }
        }
        let mut min_height = f64::INFINITY;
        for l in &labels {
            min_height = min_height.min(effective(l)?);
        }
        if numeric_rows < 3
            && !(multi >= 4 && horizontal >= 3 || multi >= 8 && raster && min_height >= 1.15 * em)
        {
            continue;
        }
        let b = bbox(&json!([
            header.raw[0],
            header.raw[1],
            header.raw[2],
            bottom + 0.2 * em
        ]))?;
        let mut grid = Vec::new();
        if numeric_rows >= 3
            && header_cells.len() == labels.len()
            && numeric_rows == accepted.len() - 1
        {
            let xs = header_cells
                .iter()
                .map(|c| c.n[0])
                .chain(std::iter::once(header.n[2]))
                .collect::<Vec<_>>();
            grid.extend(xs.into_iter().map(|x| axis(&b, x, false)));
            let mut ys = vec![header.n[1]];
            for pair in accepted.windows(2) {
                let top = pair[0]
                    .iter()
                    .map(|l| l["bbox"][3].as_f64().unwrap())
                    .reduce(f64::max)
                    .unwrap();
                let bottom = pair[1]
                    .iter()
                    .map(|l| l["bbox"][1].as_f64().unwrap())
                    .reduce(f64::min)
                    .unwrap();
                ys.push((top + bottom) / 2.);
            }
            ys.push(b.n[3]);
            grid.extend(ys.into_iter().map(|y| axis(&b, y, true)));
        }
        let stage = if raster && multi >= 8 {
            stage_grid(v, &labels, &b, work)?
        } else {
            Vec::new()
        };
        let authoritative = !stage.is_empty();
        if authoritative {
            grid = stage;
        }
        total_members += members.len();
        total_grid += grid.len();
        work.charge(grid.len() + 1)?;
        if total_members > rapiddoc_core::resources::limit(20000)
            || total_grid > rapiddoc_core::resources::limit(20000)
            || output.len() >= rapiddoc_core::resources::limit(5000)
        {
            return Err(limit());
        }
        let mut c = candidate(b, &members, vec![], grid, 8)?;
        c["preserve_inline_font_styles"] = json!(authoritative);
        c["inferred_grid_authoritative"] = json!(authoritative);
        output.push(c);
    }
    let out = json!({"candidates":output});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(out)
}
