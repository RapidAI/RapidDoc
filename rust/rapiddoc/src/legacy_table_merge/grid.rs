use super::*;
#[derive(Clone)]
pub(super) struct Metrics {
    pub row: usize,
    pub effective: usize,
    pub actual: usize,
}
#[derive(Default)]
pub(super) struct Scan {
    pub effective: Vec<usize>,
    pub metrics: Vec<Metrics>,
    pub total: usize,
    pub last: Option<Metrics>,
    pub tail: BTreeMap<usize, BTreeSet<usize>>,
}
#[derive(PartialEq)]
struct Signature {
    effective: usize,
    cols: Vec<usize>,
    rows: Vec<usize>,
    texts: Vec<String>,
}
pub(super) struct State {
    pub dom: Dom,
    pub rows: Vec<usize>,
    pub body: Option<usize>,
    pub scan: Scan,
    pub front_metrics: Vec<Metrics>,
    front: Vec<Signature>,
    pub dirty: bool,
}
pub(super) fn span(dom: &Dom, id: usize, key: &str) -> Result<usize, Error> {
    let Some(s) = dom.nodes[id].attrs.get(key) else {
        return Ok(1);
    };
    let s = s.trim_matches(python_space);
    let value = s
        .strip_prefix('+')
        .unwrap_or(s)
        .parse::<usize>()
        .map_err(|_| unsupported("表格跨度必须为正整数"))?;
    if value == 0 || value > rapiddoc_core::resources::limit(1000) {
        return Err(unsupported("表格跨度必须在 1..1000 内"));
    }
    Ok(value)
}
pub(super) fn set_span(dom: &mut Dom, id: usize, key: &str, n: usize) {
    if n > 1 {
        dom.nodes[id].attrs.insert(key.into(), n.to_string());
    } else {
        dom.nodes[id].attrs.remove(key);
    }
}
pub(super) fn scan(
    dom: &Dom,
    rows: &[usize],
    initial: &BTreeMap<usize, BTreeSet<usize>>,
    offset: usize,
    work: &mut Work,
) -> Result<Scan, Error> {
    let mut placements: usize = initial.values().map(BTreeSet::len).sum();
    if placements > rapiddoc_core::resources::limit(200_000) {
        return Err(limit());
    }
    let mut occupied = initial.clone();
    let mut scan = Scan::default();
    for cols in occupied.values() {
        if let Some(&last) = cols.last() {
            scan.total = scan.total.max(last + 1);
        }
    }
    if rows.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    for (r, &row) in rows.iter().enumerate() {
        let cells = dom.find(row, &["td", "th"]);
        let mut col = 0;
        let mut actual = 0;
        for id in &cells {
            while occupied.get(&r).is_some_and(|cols| cols.contains(&col)) {
                col += 1;
                work.charge(1)?;
            }
            let width = span(dom, *id, "colspan")?;
            let height = span(dom, *id, "rowspan")?;
            if col + width > rapiddoc_core::resources::limit(1000) {
                return Err(limit());
            }
            placements += width * height;
            if placements > rapiddoc_core::resources::limit(200_000) {
                return Err(limit());
            }
            work.charge(width * height)?;
            actual += width;
            for target in r..r + height {
                occupied.entry(target).or_default().extend(col..col + width);
            }
            col += width;
            scan.total = scan.total.max(col);
        }
        let effective = occupied
            .get(&r)
            .and_then(|c| c.last().copied())
            .map_or(0, |c| c + 1);
        scan.effective.push(effective);
        scan.total = scan.total.max(effective);
        let metrics = Metrics {
            row: r + offset,
            effective,
            actual,
        };
        scan.metrics.push(metrics.clone());
        if !cells.is_empty() {
            scan.last = Some(metrics);
        }
    }
    scan.tail = occupied
        .into_iter()
        .filter(|(r, c)| *r >= rows.len() && !c.is_empty())
        .map(|(r, c)| (r - rows.len(), c))
        .collect();
    Ok(scan)
}
impl State {
    pub fn from_block(block: &Value, work: &mut Work) -> Result<Option<Self>, Error> {
        let blocks = block["blocks"].as_array().ok_or_else(invalid)?;
        let Some(body) = blocks.iter().find(|b| b["type"] == "table_body") else {
            return Ok(None);
        };
        if body["lines"].as_array().is_some_and(|lines| {
            lines.iter().any(|l| {
                l["spans"]
                    .as_array()
                    .is_some_and(|spans| spans.iter().any(|s| s["text_fallback"] == true))
            })
        }) {
            return Ok(None);
        }
        let Some(html) = body["lines"]
            .get(0)
            .and_then(|l| l["spans"].get(0))
            .and_then(|s| s["html"].as_str())
        else {
            return Ok(None);
        };
        if html.is_empty() {
            return Ok(None);
        }
        let dom = Dom::parse(html, work)?;
        let rows = dom.find(0, &["tr"]);
        // find_all 的嵌套 tr 修复/非结构单元格不在本期严格 HTML 合同内。
        for &row in &rows {
            if !dom.find(row, &["tr"]).is_empty() {
                return Err(unsupported("嵌套行需要 HTML 容错修复"));
            }
        }
        let body = dom
            .find(0, &["tbody"])
            .first()
            .copied()
            .or_else(|| dom.find(0, &["table"]).first().copied());
        let mut state = Self {
            dom,
            rows,
            body,
            scan: Scan::default(),
            front: Vec::new(),
            front_metrics: Vec::new(),
            dirty: false,
        };
        state.refresh(work)?;
        Ok(Some(state))
    }
    pub fn cells(&self, row: usize) -> Vec<usize> {
        self.dom.find(self.rows[row], &["td", "th"])
    }
    pub fn refresh(&mut self, work: &mut Work) -> Result<(), Error> {
        self.scan = scan(&self.dom, &self.rows, &BTreeMap::new(), 0, work)?;
        let front = scan(
            &self.dom,
            &self.rows[..self.rows.len().min(6)],
            &BTreeMap::new(),
            0,
            work,
        )?;
        self.front_metrics = front.metrics;
        self.front.clear();
        for r in 0..self.rows.len().min(5) {
            let cells = self.cells(r);
            let mut cols = Vec::new();
            let mut rows = Vec::new();
            let mut texts = Vec::new();
            for id in cells {
                cols.push(span(&self.dom, id, "colspan")?);
                rows.push(span(&self.dom, id, "rowspan")?);
                texts.push(
                    full_half(&self.dom.text(id))
                        .chars()
                        .filter(|c| !python_space(*c))
                        .collect(),
                );
            }
            self.front.push(Signature {
                effective: front.effective[r],
                cols,
                rows,
                texts,
            });
        }
        Ok(())
    }
}
pub(super) fn headers(a: &State, b: &State) -> (usize, bool) {
    let count = a.front.len().min(b.front.len());
    let mut n = 0;
    for i in 0..count {
        if a.front[i] == b.front[i] {
            n += 1;
        } else {
            break;
        }
    }
    if n == 0 {
        for i in 0..count {
            if a.front[i].texts == b.front[i].texts && a.front[i].effective == b.front[i].effective
            {
                n += 1;
            } else {
                break;
            }
        }
    }
    (n, n > 0 || count == 0)
}
pub(super) fn expanded_header(state: &State, n: usize) -> Result<usize, Error> {
    let mut n = n.min(state.rows.len());
    let mut r = 0;
    while r < n {
        for id in state.cells(r) {
            n = n
                .max(r + span(&state.dom, id, "rowspan")?)
                .min(state.rows.len());
        }
        r += 1;
    }
    Ok(n)
}
type Sources = BTreeMap<usize, (usize, usize)>;
fn sources(state: &State, target: usize, work: &mut Work) -> Result<(Sources, usize), Error> {
    let mut occupied: BTreeMap<usize, Sources> = BTreeMap::new();
    let mut total = 0;
    for r in 0..=target {
        let mut col = 0;
        for (ci, id) in state.cells(r).into_iter().enumerate() {
            while occupied.get(&r).is_some_and(|c| c.contains_key(&col)) {
                col += 1;
                work.charge(1)?;
            }
            let width = span(&state.dom, id, "colspan")?;
            let height = span(&state.dom, id, "rowspan")?;
            if col + width > rapiddoc_core::resources::limit(1000) {
                return Err(limit());
            }
            work.charge(width * height)?;
            for target in r..r + height {
                let row = occupied.entry(target).or_default();
                for c in col..col + width {
                    row.insert(c, (r, ci));
                }
            }
            col += width;
            total = total.max(col);
        }
    }
    Ok((occupied.remove(&target).unwrap_or_default(), total))
}
pub(super) fn mapping(state: &State, row: usize, work: &mut Work) -> Result<Vec<usize>, Error> {
    let (occupied, _) = sources(state, row, work)?;
    let mut col = 0;
    let mut out = Vec::new();
    for id in state.cells(row) {
        while occupied.get(&col).is_some_and(|m| m.0 < row) {
            col += 1;
            work.charge(1)?;
        }
        out.push(col);
        col += span(&state.dom, id, "colspan")?;
    }
    Ok(out)
}
pub(super) fn rendered(state: &State, row: usize, work: &mut Work) -> Result<usize, Error> {
    let (occupied, total) = sources(state, row, work)?;
    let mut previous = None;
    let mut count = 0;
    for i in 0..total {
        let marker = occupied.get(&i).copied();
        if marker.is_some() && marker != previous {
            count += 1;
        }
        previous = marker;
    }
    Ok(count)
}
pub(super) fn visual_cells(
    state: &State,
    row: usize,
    work: &mut Work,
) -> Result<(Vec<usize>, BTreeMap<usize, usize>), Error> {
    let cells = state.cells(row);
    let map = mapping(state, row, work)?;
    let mut out = BTreeMap::new();
    for (i, (&id, &start)) in cells.iter().zip(&map).enumerate() {
        for c in start..start + span(&state.dom, id, "colspan")? {
            out.insert(c, i);
        }
    }
    Ok((cells, out))
}
fn boundaries(state: &State, row: usize, work: &mut Work) -> Result<Option<Vec<usize>>, Error> {
    let (occupied, total) = sources(state, row, work)?;
    if total == 0 || occupied.len() != total {
        return Ok(None);
    }
    let mut out = vec![0];
    for i in 1..total {
        if occupied[&i] != occupied[&(i - 1)] {
            out.push(i);
        }
    }
    out.push(total);
    Ok(Some(out))
}
pub(super) fn align(
    source: &mut State,
    reference: &State,
    sr: usize,
    rr: usize,
    work: &mut Work,
) -> Result<bool, Error> {
    let Some(a) = boundaries(source, sr, work)? else {
        return Ok(false);
    };
    let Some(b) = boundaries(reference, rr, work)? else {
        return Ok(false);
    };
    if a.len() != b.len()
        || a.last() != Some(&source.scan.total)
        || b.last() != Some(&reference.scan.total)
    {
        return Ok(false);
    }
    let mapping: BTreeMap<_, _> = a.into_iter().zip(b).collect();
    let mut occupied: BTreeMap<usize, BTreeSet<usize>> = BTreeMap::new();
    let mut updates = Vec::new();
    for r in 0..source.rows.len() {
        let mut col = 0;
        for id in source.cells(r) {
            while occupied.get(&r).is_some_and(|s| s.contains(&col)) {
                col += 1;
                work.charge(1)?;
            }
            let end = col + span(&source.dom, id, "colspan")?;
            let (Some(&start), Some(&finish)) = (mapping.get(&col), mapping.get(&end)) else {
                return Ok(false);
            };
            updates.push((id, finish - start));
            let height = span(&source.dom, id, "rowspan")?;
            work.charge((end - col) * height)?;
            for target in r..r + height {
                occupied.entry(target).or_default().extend(col..end);
            }
            col = end;
        }
    }
    for (id, width) in updates {
        set_span(&mut source.dom, id, "colspan", width);
    }
    source.refresh(work)?;
    Ok(true)
}
pub(super) fn adjust(
    source: &mut State,
    start: usize,
    reference: &State,
    reference_row: usize,
    total: usize,
    work: &mut Work,
) -> Result<(), Error> {
    let reference = reference
        .cells(reference_row)
        .into_iter()
        .map(|id| span(&reference.dom, id, "colspan"))
        .collect::<Result<Vec<_>, _>>()?;
    for r in start..source.rows.len() {
        let cells = source.cells(r);
        if cells.is_empty() {
            continue;
        }
        let widths = cells
            .iter()
            .map(|&id| span(&source.dom, id, "colspan"))
            .collect::<Result<Vec<_>, _>>()?;
        let effective = source.scan.effective[r];
        if effective >= total || widths.iter().sum::<usize>() >= total {
            continue;
        }
        work.charge(cells.len())?;
        if cells.len() == reference.len() {
            for (&id, &width) in cells.iter().zip(&reference) {
                set_span(&mut source.dom, id, "colspan", width);
            }
        } else {
            let id = *cells.last().unwrap();
            let last = *widths.last().unwrap();
            set_span(&mut source.dom, id, "colspan", last + total - effective);
        }
    }
    Ok(())
}
