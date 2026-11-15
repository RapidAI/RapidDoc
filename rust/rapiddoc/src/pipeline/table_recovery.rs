use super::table_evidence::{Grid, GridCell, Source, Token};
use super::{error, Error};
use std::collections::BTreeSet;
use rapiddoc_inference::{table_recover::Polygon, wired_match};
use std::collections::HashSet;

pub(super) fn bounds(q: &Polygon) -> [f64; 4] {
    q.iter().fold(
        [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ],
        |[l, t, r, b], &[x, y]| [l.min(x), t.min(y), r.max(x), b.max(y)],
    )
}

pub(super) fn restore_columns(
    before: &[Polygon],
    after: Vec<Polygon>,
    width: usize,
    height: usize,
    clockwise: bool,
) -> Vec<Polygon> {
    let projected: Vec<_> = before
        .iter()
        .map(|q| {
            let q = q.map(|[x, y]| {
                if clockwise {
                    [height as f64 - 1. - y, x]
                } else {
                    [y, width as f64 - 1. - x]
                }
            });
            if clockwise {
                [q[3], q[0], q[1], q[2]]
            } else {
                [q[1], q[2], q[3], q[0]]
            }
        })
        .collect();
    let rects: Vec<_> = projected.iter().map(bounds).collect();
    let center_x = |[l, _, r, _]: [f64; 4]| (l + r) / 2.;
    let size_x = |[l, _, r, _]: [f64; 4]| r - l;
    let mut result = Vec::new();
    for q in after {
        let [left, top, right, bottom] = bounds(&q);
        let (w, h) = (right - left, bottom - top);
        if h <= 0. || w < 4. * h {
            result.push(q);
            continue;
        }
        let mut ids: Vec<_> = rects
            .iter()
            .enumerate()
            .filter_map(|(i, &r)| {
                let overlap = r[3].min(bottom) - r[1].max(top);
                ((left..=right).contains(&center_x(r))
                    && overlap >= 0.65 * h.max(r[3] - r[1])
                    && size_x(r) > 0.
                    && size_x(r) <= w / 2.)
                    .then_some(i)
            })
            .collect();
        ids.sort_by(|&a, &b| center_x(rects[a]).total_cmp(&center_x(rects[b])));
        if ids.len() < 3
            || rects[ids[ids.len() - 1]][2] - rects[ids[0]][0] < 0.8 * w
            || ids.windows(2).any(|p| {
                let (a, b) = (rects[p[0]], rects[p[1]]);
                center_x(b) - center_x(a) < 0.7 * size_x(a).max(size_x(b))
            })
        {
            result.push(q);
            continue;
        }
        let y = (top + bottom) / 2.;
        let aligned = ids
            .iter()
            .filter(|&&i| {
                rects.iter().any(|&peer| {
                    let dy = ((peer[1] + peer[3]) / 2. - y).abs();
                    (h..=4. * h).contains(&dy)
                        && (center_x(rects[i]) - center_x(peer)).abs() <= 0.3 * size_x(rects[i])
                })
            })
            .count();
        if aligned * 4 >= ids.len() * 3 {
            result.extend(ids.into_iter().map(|i| projected[i]));
        } else {
            result.push(q);
        }
    }
    result
}

pub(super) fn compact(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace()).collect()
}

pub(super) fn direction<'a>(
    pairs: impl Iterator<Item = (&'a str, f64, &'a str, f64)>,
    label: &'a str,
) -> &'a str {
    let mut votes = [0; 2];
    let mut numbers = HashSet::new();
    let mut conflict = false;
    for (left, ls, right, rs) in pairs {
        let evidence = |s: &str, score| {
            if s.chars().filter(|c| c.is_alphabetic()).take(2).count() == 2 {
                score
            } else {
                0.
            }
        };
        let (l, r) = (evidence(left, ls), evidence(right, rs));
        votes[0] += usize::from(l >= 0.8 && l - r >= 0.2);
        votes[1] += usize::from(r >= 0.8 && r - l >= 0.2);
        let number = compact(left);
        if number.len() >= 4
            && number.bytes().all(|b| b.is_ascii_digit())
            && number == compact(right)
        {
            let (preferred, opposite) = if label == "90" { (ls, rs) } else { (rs, ls) };
            if preferred >= 0.98 && preferred - opposite >= 0.15 {
                numbers.insert(number);
            }
            conflict |= opposite >= 0.98 && opposite - preferred >= 0.15;
        }
    }
    match votes {
        [left, 0] if left >= 2 => "90",
        [0, right] if right >= 2 => "270",
        [0, 0] if numbers.len() >= 3 && !conflict => label,
        _ => "0",
    }
}

pub(super) fn joined_candidates(
    rects: &[[f64; 4]],
    logic: &[[usize; 4]],
    matched: &wired_match::Matched,
    item: &wired_match::Ocr,
    chars: usize,
) -> Option<Vec<usize>> {
    let [left, top, right, bottom] = bounds(&item.points);
    let h = bottom - top;
    if item.confidence < 0.9 || chars < 2 || h <= 0. || right - left < 2. * h {
        return None;
    }
    let mut ids: Vec<_> = rects
        .iter()
        .enumerate()
        .filter_map(|(i, &[l, t, r, b])| {
            (r.min(right) - l.max(left) > 0.5 * (r - l) && b.min(bottom) - t.max(top) >= 0.75 * h)
                .then_some(i)
        })
        .collect();
    if ids.len() < 2 || ids.len() > chars || ids.iter().any(|&i| !matched.cells[i].is_empty()) {
        return None;
    }
    ids.sort_by_key(|&i| logic[i][2]);
    let row = logic[ids[0]][0];
    if ids.iter().any(|&i| {
        let [r0, r1, c0, c1] = logic[i];
        r0 != row || r1 != row || c0 != c1
    }) || ids.windows(2).any(|p| logic[p[0]][2] + 1 != logic[p[1]][2])
    {
        return None;
    }
    Some(ids)
}

/// 只有模型中的相邻普通格与原图竖线同时支持时，才裁格补识别。
pub(super) fn cross_cell_groups(
    png: &[u8],
    rects: &[[f64; 4]],
    logic: &[[usize; 4]],
    ocr: &[wired_match::Ocr],
    excluded: &[String],
) -> Result<Vec<(usize, Vec<usize>)>, Error> {
    use std::collections::BTreeMap;

    let boxes: Vec<_> = ocr.iter().map(|item| bounds(&item.points)).collect();
    if boxes.is_empty() {
        return Ok(Vec::new());
    }
    let mut heights: Vec<_> = boxes.iter().map(|b| b[3] - b[1]).collect();
    heights.sort_by(f64::total_cmp);
    let mid = heights.len() / 2;
    let em = ((heights[(heights.len() - 1) / 2] + heights[mid]) / 2.).max(2.);
    let mut raster = None;
    let mut groups = BTreeMap::new();
    for (i, &[left, top, right, bottom]) in boxes.iter().enumerate() {
        let h = bottom - top;
        if h <= 0. || ocr[i].text.trim().is_empty() || excluded.contains(&ocr[i].text) {
            continue;
        }
        let mut cells: Vec<_> = rects
            .iter()
            .enumerate()
            .filter_map(|(j, &[l, t, r, b])| {
                (right.min(r) - left.max(l) >= 0.3 * h && bottom.min(b) - top.max(t) >= 0.75 * h)
                    .then_some(j)
            })
            .collect();
        if cells.len() < 2 {
            continue;
        }
        cells.sort_by_key(|&j| logic[j][2]);
        let row = logic[cells[0]][0];
        if cells.iter().any(|&j| {
            let [r0, r1, c0, c1] = logic[j];
            r0 != row || r1 != row || c0 != c1
        }) || cells
            .windows(2)
            .any(|p| logic[p[0]][2] + 1 != logic[p[1]][2])
        {
            continue;
        }
        if raster.is_none() {
            let img = image::load_from_memory(png).map_err(error)?.into_rgb8();
            raster = Some(rapiddoc_inference::table_raster::evidence_rules(
                img.as_raw(),
                img.width() as usize,
                img.height() as usize,
                em,
            ));
        }
        if cells.windows(2).all(|p| {
            let edge = (rects[p[0]][2] + rects[p[1]][0]) / 2.;
            raster.as_ref().unwrap().iter().any(|line| {
                let [x0, y0, x1, y1] = line.bbox;
                let x = (x0 + x1) / 2.;
                line.orientation == "vertical"
                    && (x - edge).abs() <= 0.45 * em
                    && left < x
                    && x < right
                    && y0 <= top + 0.2 * h
                    && y1 >= bottom - 0.2 * h
            })
        }) {
            groups.insert(i, cells);
        }
    }
    if groups.is_empty() {
        return Ok(Vec::new());
    }
    let mut owners = vec![0; rects.len()];
    for (i, &[l, t, r, b]) in boxes.iter().enumerate() {
        if let Some(cells) = groups.get(&i) {
            for &cell in cells {
                owners[cell] += 1;
            }
        } else {
            let (x, y) = ((l + r) / 2., (t + b) / 2.);
            for (cell, &[l, t, r, b]) in rects.iter().enumerate() {
                if l < x && x < r && t < y && y < b {
                    owners[cell] += 1;
                }
            }
        }
    }
    Ok(groups
        .into_iter()
        .filter(|(_, cells)| cells.iter().all(|&j| owners[j] == 1))
        .collect())
}

pub(super) fn restore_trailing_row(
    main: &str,
    candidate: &str,
    main_grid: &Grid,
    grid: &Grid,
    source: &Source,
    lost: &BTreeSet<usize>,
) -> Option<(String, Grid)> {
    if lost.is_empty() {
        return None;
    }
    let last = main_grid.cells.iter().map(|c| c.logic[1]).max()?;
    let mut tail = main_grid.cells.iter().filter(|c| c.logic[1] == last);
    let cell = tail.next()?;
    if tail.next().is_some()
        || cell.logic != [last, last, 0, main_grid.width - 1]
        || cell.tokens.iter().copied().collect::<BTreeSet<_>>() != *lost
        || main_grid
            .cells
            .iter()
            .any(|c| c.logic != cell.logic && c.tokens.iter().any(|id| lost.contains(id)))
    {
        return None;
    }
    let bottom = grid
        .cells
        .iter()
        .map(|c| c.bbox[3])
        .max_by(f64::total_cmp)?;
    let tokens: Vec<_> = source
        .tokens
        .iter()
        .filter(|t| lost.contains(&t.id))
        .collect();
    if tokens.len() != lost.len() || tokens.iter().any(|t| t.bbox[1] < bottom - 0.2 * source.em) {
        return None;
    }
    let row_pattern = crate::cached_regex!(r"(?is)<tr\b[^>]*>.*?</tr\s*>");
    let rows: Vec<_> = row_pattern.find_iter(main).collect();
    let alternative: Vec<_> = row_pattern.find_iter(candidate).collect();
    let next = grid.cells.iter().map(|c| c.logic[1]).max()? + 1;
    if rows.len() != last + 1 || alternative.len() != next {
        return None;
    }
    let cells = crate::cached_regex!(r"(?is)<t([dh])\b[^>]*>(.*?)</t[dh]\s*>");
    let mut tail = cells.captures_iter(rows.last()?.as_str());
    let part = tail.next()?;
    if tail.next().is_some() {
        return None;
    }
    let end = alternative.last()?.end();
    let tag = &part[1];
    let html = format!(
        "{}<tr><t{tag} colspan=\"{}\">{}</t{tag}></tr>{}",
        &candidate[..end],
        grid.width,
        &part[2],
        &candidate[end..]
    );
    super::table_fallback::table_text(&html)?;
    let mut cells: Vec<_> = grid
        .cells
        .iter()
        .map(|c| GridCell {
            bbox: c.bbox,
            logic: c.logic,
            tokens: c.tokens.clone(),
        })
        .collect();
    cells.push(GridCell {
        bbox: cell.bbox,
        logic: [next, next, 0, grid.width - 1],
        tokens: cell.tokens.clone(),
    });
    Some((html, Grid::new(cells)?))
}

pub(super) fn join_aligned_row_bands(
    html: &str,
    grid: &Grid,
    source: &Source,
    texts: &[serde_json::Value],
) -> Option<String> {
    if grid.width < 3 || !grid.cells.iter().any(|c| c.logic[1] > c.logic[0]) {
        return None;
    }
    let x = |t: &Token| (t.bbox[0] + t.bbox[2]) / 2.;
    let y = |t: &Token| (t.bbox[1] + t.bbox[3]) / 2.;
    let mut tokens: Vec<_> = source.tokens.iter().collect();
    tokens.sort_by(|a, b| y(a).total_cmp(&y(b)));
    let mut bands: Vec<Vec<&Token>> = Vec::new();
    for token in tokens {
        if bands
            .last()
            .is_none_or(|band| y(token) - y(band[0]) > 0.4 * source.em)
        {
            bands.push(Vec::new());
        }
        bands.last_mut()?.push(token);
    }
    let assignments = |id| grid.cells.iter().filter(move |c| c.tokens.contains(&id));
    let mut records = Vec::new();
    let mut common = BTreeSet::new();
    let mut columns = Vec::new();
    let mut positions = Vec::<f64>::new();
    let mut header = 0;
    // 从尾部查找相同分布的双行记录，并用已有跨行格确认记录起点。
    for pair in bands.rchunks_exact(2) {
        let mut upper = pair[0].clone();
        let lower = &pair[1];
        if upper.len() != grid.width || lower.len() < 2 || lower.len() >= grid.width {
            break;
        }
        let (y0, y1) = (y(upper[0]), y(lower[0]));
        if !(0.6 * source.em..=1.6 * source.em).contains(&(y1 - y0)) {
            break;
        }
        upper.sort_by(|a, b| x(a).total_cmp(&x(b)));
        let mut matched = std::collections::BTreeMap::new();
        for &t in lower {
            let mut fits = upper
                .iter()
                .enumerate()
                .filter(|(_, u)| (x(t) - x(u)).abs() <= 0.65 * source.em);
            let (col, u) = fits.next()?;
            let (left, right) = (u.bbox[0].min(t.bbox[0]), u.bbox[2].max(t.bbox[2]));
            if fits.next().is_some()
                || matched.contains_key(&col)
                || source.ruling[1].iter().any(|&(y, lo, hi)| {
                    y0 < y && y < y1 && hi.min(right) - lo.max(left) > 0.5 * (right - left)
                })
            {
                break;
            }
            matched.insert(col, t);
        }
        let cols: Vec<_> = matched.keys().copied().collect();
        if matched.len() != lower.len() || (!records.is_empty() && cols != columns) {
            break;
        }
        let mut anchors = BTreeSet::new();
        let mut start = usize::MAX;
        for (col, t) in upper.iter().enumerate() {
            let mut assigned = assignments(t.id);
            let cell = assigned.next()?;
            let [r0, r1, c0, c1] = cell.logic;
            if assigned.next().is_some() || c0 != col || c1 != col {
                return None;
            }
            start = start.min(r0);
            if !matched.contains_key(&col)
                && r1 == r0 + 1
                && cell.bbox[1] <= y0
                && y1 <= cell.bbox[3]
            {
                anchors.insert(col);
            }
        }
        if start == 0 || anchors.is_empty() {
            break;
        }
        if records.is_empty() {
            common = anchors;
            columns = cols;
            positions = upper.iter().map(|t| x(t)).collect();
        } else {
            common.retain(|col| anchors.contains(col));
            if common.is_empty()
                || upper
                    .iter()
                    .zip(&positions)
                    .any(|(t, p)| (x(t) - p).abs() > 0.8 * source.em)
            {
                return None;
            }
        }
        header = start;
        records.push((upper, matched));
    }
    if records.len() < 3 {
        return None;
    }
    let row_pattern = crate::cached_regex!(r"(?is)<tr\b[^>]*>.*?</tr\s*>");
    let rows: Vec<_> = row_pattern.find_iter(html).collect();
    if header >= rows.len()
        || grid
            .cells
            .iter()
            .any(|c| c.logic[0] < header && header <= c.logic[1])
    {
        return None;
    }
    let data: BTreeSet<_> = records
        .iter()
        .flat_map(|(upper, lower)| upper.iter().chain(lower.values()).map(|t| t.id))
        .collect();
    if source.tokens.iter().any(|t| {
        if data.contains(&t.id) {
            return false;
        }
        let cells: Vec<_> = assignments(t.id).collect();
        cells.is_empty() || cells.iter().any(|c| c.logic[1] >= header)
    }) {
        return None;
    }
    let mut output = html[..rows[header].start()].to_owned();
    let text = |t: &Token| -> Option<String> {
        Some(rapiddoc_render::escape_html(
            &crate::processing::inline_output::unescape_html(texts.get(t.id)?.as_str()?),
        ))
    };
    for (upper, lower) in records.into_iter().rev() {
        output.push_str("<tr>");
        for (col, token) in upper.into_iter().enumerate() {
            output.push_str("<td>");
            output.push_str(&text(token)?);
            if let Some(token) = lower.get(&col) {
                output.push_str("<br/>");
                output.push_str(&text(token)?);
            }
            output.push_str("</td>");
        }
        output.push_str("</tr>");
    }
    output.push_str(&html[rows.last()?.end()..]);
    super::table_fallback::table_text(&output)?;
    Some(output)
}

pub(super) fn restore_internal_rows(
    html: &str,
    grid: &Grid,
    source: &Source,
    missing: &BTreeSet<usize>,
    texts: &[serde_json::Value],
) -> Option<(String, Grid)> {
    if missing.len() < 6 || grid.width > grid.cells.len() {
        return None;
    }
    super::table_fallback::table_text(html)?;
    let row_pattern = crate::cached_regex!(r"(?is)<tr\b[^>]*>.*?</tr\s*>");
    let cell_pattern = crate::cached_regex!(r"(?is)<t[dh]\b[^>]*>");
    let span = crate::cached_regex!(r#"(?i)\b(rowspan|colspan)\s*=\s*["']?([0-9]+)["']?"#);
    let rows: Vec<_> = row_pattern.find_iter(html).collect();
    if rows.len() != grid.cells.iter().map(|c| c.logic[1]).max()? + 1 {
        return None;
    }
    let mut ordered: Vec<_> = grid.cells.iter().collect();
    ordered.sort_by_key(|c| (c.logic[0], c.logic[2]));
    let nodes: Vec<_> = cell_pattern.find_iter(html).collect();
    if nodes.len() != ordered.len() {
        return None;
    }
    for (node, cell) in nodes.iter().zip(&ordered) {
        let [r0, r1, c0, c1] = cell.logic;
        let mut sizes = [1, 1];
        for attr in span.captures_iter(node.as_str()) {
            sizes[usize::from(attr[1].eq_ignore_ascii_case("colspan"))] = attr[2].parse().ok()?;
        }
        if sizes != [r1 - r0 + 1, c1 - c0 + 1] {
            return None;
        }
    }
    let median = |mut v: Vec<f64>| {
        v.sort_by(f64::total_cmp);
        (!v.is_empty()).then(|| (v[(v.len() - 1) / 2] + v[v.len() / 2]) / 2.)
    };
    let columns: Vec<_> = (0..grid.width)
        .map(|col| {
            let values = |axis| {
                grid.cells
                    .iter()
                    .filter(|c| c.logic[2..] == [col, col])
                    .map(|c| c.bbox[axis])
                    .collect()
            };
            Some((median(values(0))?, median(values(2))?))
        })
        .collect::<Option<_>>()?;
    let mut tokens: Vec<_> = source
        .tokens
        .iter()
        .filter(|t| missing.contains(&t.id) && t.confidence >= 0.85)
        .collect();
    if tokens.len() != missing.len() {
        return None;
    }
    let y = |t: &Token| (t.bbox[1] + t.bbox[3]) / 2.;
    tokens.sort_by(|a, b| y(a).total_cmp(&y(b)));
    let mut bands: Vec<(f64, std::collections::BTreeMap<usize, &Token>)> = Vec::new();
    let mut occurrences = vec![0; grid.width];
    for token in tokens {
        let [l, _, r, _] = token.bbox;
        let mut fits = columns
            .iter()
            .enumerate()
            .filter(|(_, &(a, b))| b.min(r) - a.max(l) >= 0.75 * (r - l));
        let (col, _) = fits.next()?;
        if fits.next().is_some() {
            return None;
        }
        let pos = y(token);
        if bands
            .last()
            .is_none_or(|(prior, _)| pos - prior > 0.4 * source.em)
        {
            bands.push((pos, Default::default()));
        }
        if bands.last_mut()?.1.insert(col, token).is_some() {
            return None;
        }
        occurrences[col] += 1;
    }
    if bands.len() < 2
        || bands.iter().any(|(_, b)| b.len() < 2)
        || occurrences.iter().filter(|&&n| n >= 2).count() < 2
    {
        return None;
    }
    let top = bands
        .iter()
        .flat_map(|(_, b)| b.values())
        .map(|t| t.bbox[1])
        .min_by(f64::total_cmp)?;
    let bottom = bands
        .iter()
        .flat_map(|(_, b)| b.values())
        .map(|t| t.bbox[3])
        .max_by(f64::total_cmp)?;
    let mut slots = (1..rows.len()).filter(|&row| {
        let edge = |r, axis| {
            median(
                grid.cells
                    .iter()
                    .filter(|c| c.logic[..2] == [r, r])
                    .map(|c| c.bbox[axis])
                    .collect(),
            )
        };
        edge(row - 1, 3)
            .zip(edge(row, 1))
            .is_some_and(|(a, b)| a <= top && bottom <= b && b - a >= 1.5 * source.em)
    });
    let row = slots.next()?;
    if slots.next().is_some() {
        return None;
    }
    let mut occupied = BTreeSet::new();
    let mut changes = Vec::new();
    let n = bands.len();
    let mut cells = Vec::new();
    for (node, cell) in nodes.iter().zip(ordered) {
        let [mut r0, mut r1, c0, c1] = cell.logic;
        if r0 < row && row <= r1 {
            occupied.extend(c0..=c1);
            let mut tag = node.as_str().to_owned();
            let attr = span
                .captures_iter(node.as_str())
                .find(|a| a[1].eq_ignore_ascii_case("rowspan"))?
                .get(0)?;
            tag.replace_range(attr.range(), &format!("rowspan=\"{}\"", r1 - r0 + 1 + n));
            changes.push((node.range(), tag));
        }
        if r0 >= row {
            r0 += n;
        }
        if r1 >= row {
            r1 += n;
        }
        cells.push(GridCell {
            bbox: cell.bbox,
            logic: [r0, r1, c0, c1],
            tokens: cell.tokens.clone(),
        });
    }
    if occupied.iter().any(|&col| occurrences[col] > 0) {
        return None;
    }
    let mut inserted = String::new();
    for (offset, (_, band)) in bands.iter().enumerate() {
        let y0 = band.values().map(|t| t.bbox[1]).min_by(f64::total_cmp)?;
        let y1 = band.values().map(|t| t.bbox[3]).max_by(f64::total_cmp)?;
        inserted.push_str("<tr>");
        for (col, &(l, r)) in columns.iter().enumerate() {
            if occupied.contains(&col) {
                continue;
            }
            inserted.push_str("<td>");
            let mut ids = Vec::new();
            if let Some(t) = band.get(&col) {
                let text =
                    crate::processing::inline_output::unescape_html(texts.get(t.id)?.as_str()?);
                inserted.push_str(&rapiddoc_render::escape_html(&text));
                ids.push(t.id);
            }
            inserted.push_str("</td>");
            cells.push(GridCell {
                bbox: [l, y0, r, y1],
                logic: [row + offset, row + offset, col, col],
                tokens: ids,
            });
        }
        inserted.push_str("</tr>");
    }
    let start = rows[row].start();
    changes.push((start..start, inserted));
    changes.sort_by_key(|(r, _)| r.start);
    let mut result = html.to_owned();
    for (range, replacement) in changes.into_iter().rev() {
        result.replace_range(range, &replacement);
    }
    super::table_fallback::table_text(&result)?;
    Some((result, Grid::new(cells)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(l: f64, t: f64, r: f64, b: f64) -> Polygon {
        [[l, t], [r, t], [r, b], [l, b]]
    }

    #[test]
    fn cross_cell_recognition_needs_a_visible_separator_and_keeps_placeholders() {
        let mut image = image::RgbImage::from_pixel(80, 40, image::Rgb([255; 3]));
        let boxes = [[0., 0., 40., 40.], [40., 0., 80., 40.]];
        let logic = [[0, 0, 0, 0], [0, 0, 1, 1]];
        let token = wired_match::Ocr {
            points: rect(18., 14., 62., 26.),
            text: "甲1".into(),
            confidence: 0.99,
        };
        for ruled in [false, true] {
            if ruled {
                for y in 0..40 {
                    image.put_pixel(39, y, image::Rgb([0; 3]));
                    image.put_pixel(40, y, image::Rgb([0; 3]));
                }
            }
            let mut png = std::io::Cursor::new(Vec::new());
            image.write_to(&mut png, image::ImageFormat::Png).unwrap();
            let found = cross_cell_groups(
                png.get_ref(),
                &boxes,
                &logic,
                std::slice::from_ref(&token),
                &[],
            )
            .unwrap();
            assert_eq!(found, if ruled { vec![(0, vec![0, 1])] } else { vec![] });
            assert!(cross_cell_groups(
                png.get_ref(),
                &boxes,
                &logic,
                std::slice::from_ref(&token),
                &[token.text.clone()]
            )
            .unwrap()
            .is_empty());
        }
    }

    #[test]
    fn rotated_line_splits_only_with_neighboring_column_evidence() {
        let merged = rect(10., 10., 88., 20.);
        for clockwise in [false, true] {
            let before: Vec<_> = [10., 30.]
                .into_iter()
                .flat_map(|y| [10., 40., 70.].map(|x| rect(x, y, x + 18., y + 10.)))
                .map(|q| {
                    q.map(|[x, y]| {
                        if clockwise {
                            [y, 99. - x]
                        } else {
                            [99. - y, x]
                        }
                    })
                })
                .collect();
            let split = restore_columns(&before, vec![merged], 100, 100, clockwise);
            assert_eq!(
                split.iter().map(bounds).collect::<Vec<_>>(),
                [
                    [10., 10., 28., 20.],
                    [40., 10., 58., 20.],
                    [70., 10., 88., 20.]
                ]
            );
            assert_eq!(
                restore_columns(&before[..3], vec![merged], 100, 100, clockwise),
                [merged]
            );
        }
    }

    #[test]
    fn direction_requires_consistent_evidence() {
        let right = [("x", 0.99, "Country", 0.95), ("?", 0.9, "Year", 0.97)];
        assert_eq!(direction(right.into_iter(), "90"), "270");
        assert_eq!(direction(right.into_iter().take(1), "90"), "0");
        let numbers = ["1997", "1998", "1999"].map(|s| (s, 0.8, s, 0.99));
        assert_eq!(direction(numbers.into_iter(), "270"), "270");
        assert_eq!(direction(numbers.into_iter(), "90"), "0");
    }
}
