//! Original repeated hanging-indent and local-lane grouping, preserving lane identity.
use super::*;
fn body_rows(s: &State, l: &Lane) -> Vec<Row> {
    let mut r = l
        .rows
        .iter()
        .filter(|r| s.l(r)["semantic_type"].is_null())
        .cloned()
        .collect::<Vec<_>>();
    s.ordered(&mut r, true);
    r
}
fn adjacent(s: &State, a: &Row, b: &Row, l: &Lane, h: f64) -> Result<bool, Error> {
    let gap = s.gap(a, b)?;
    let pitch = b.b.n[1] - a.b.n[1];
    Ok(
        ((-0.6 * h..=1.3 * h).contains(&gap) || (0.5 * h..=1.8 * h).contains(&pitch))
            && !s.table_barrier(a, b)?
            && !s.rule_barrier(a, b, l)?,
    )
}
struct Hanging<'a> {
    s: &'a State,
    l: &'a Lane,
    rs: Vec<Row>,
    h: f64,
}
impl Hanging<'_> {
    fn consume(
        &self,
        i: usize,
        left: f64,
        expected: Option<f64>,
        next: bool,
    ) -> Result<Option<(usize, f64)>, Error> {
        let Self { s, l, rs, h } = self;
        let start_tol = 5f64.max(0.65 * h);
        let minimum = 7f64.max(0.8 * h);
        let continuation_tol = 4f64.max(0.55 * h);
        let width = (l.right - l.left).max(0.1);
        let full = rs[i].b.n[2] - rs[i].b.n[0] >= 0.8 * width
            && i + 1 < rs.len()
            && rs[i + 1].b.n[2] - rs[i + 1].b.n[0] <= 0.75 * width;
        if i > 0
            && adjacent(s, &rs[i - 1], &rs[i], l, *h)?
            && (rs[i - 1].b.n[0] - left).abs() <= start_tol
            && !full
        {
            return Ok(None);
        }
        if i > 0 && hyphen(s.t(&rs[i - 1])?) && adjacent(s, &rs[i - 1], &rs[i], l, *h)? {
            return Ok(None);
        }
        let mut j = i + 1;
        if j >= rs.len() || !adjacent(s, &rs[i], &rs[j], l, *h)? {
            return Ok(None);
        }
        let cont = rs[j].b.n[0];
        if cont < left + minimum || expected.is_some_and(|x| (cont - x).abs() > continuation_tol) {
            return Ok(None);
        }
        j += 1;
        while j < rs.len() {
            if !adjacent(s, &rs[j - 1], &rs[j], l, *h)?
                || rs[j].b.n[0] < left + minimum
                || (rs[j].b.n[0] - cont).abs() > continuation_tol
            {
                break;
            }
            j += 1
        }
        if !next {
            return Ok(Some((j, cont)));
        }
        if j >= rs.len()
            || !adjacent(s, &rs[j - 1], &rs[j], l, *h)?
            || (rs[j].b.n[0] - left).abs() > start_tol
        {
            return Ok(None);
        }
        Ok(Some((j, cont)))
    }
}
pub(super) fn hanging(s: &State, l: &Lane) -> Result<BTreeMap<i64, i64>, Error> {
    let mut out = BTreeMap::new();
    if l.rows.len() < 4 {
        return Ok(out);
    }
    let rs = body_rows(s, l);
    if rs.len() < 4 {
        return Ok(out);
    }
    let h = median(rs.iter().map(|r| s.h(r)).collect::<Result<Vec<_>, _>>()?)?;
    let a = Hanging { s, l, rs, h };
    let mut group = 0;
    let mut i = 0;
    while i + 3 < a.rs.len() {
        let left = a.rs[i].b.n[0];
        let Some((_, cont)) = a.consume(i, left, None, true)? else {
            i += 1;
            continue;
        };
        let mut starts = vec![i];
        let mut cur = i;
        let end;
        loop {
            match a.consume(cur, left, Some(cont), true)? {
                None => {
                    end = a.consume(cur, left, Some(cont), false)?.map(|x| x.0);
                    break;
                }
                Some((j, _)) => {
                    if a.consume(j, left, Some(cont), false)?.is_none() {
                        end = Some(j);
                        break;
                    }
                    starts.push(j);
                    cur = j
                }
            }
        }
        if starts.len() < 2 || end.is_none() {
            i += 1;
            continue;
        }
        let end = end.unwrap();
        for (k, &start) in starts.iter().enumerate() {
            let stop = starts.get(k + 1).copied().unwrap_or(end);
            for r in &a.rs[start..stop] {
                out.insert(s.id(r)?, group);
            }
            group += 1
        }
        i = end;
    }
    Ok(out)
}
/// The map stores distinct run indexes, because Python tests lane objects by `is`.
pub(super) fn local_lanes(s: &State, l: &Lane) -> Result<(Vec<Lane>, BTreeMap<i64, usize>), Error> {
    let mut result = Vec::new();
    let mut by_source = BTreeMap::new();
    if l.span || l.rows.len() < 3 {
        return Ok((result, by_source));
    }
    let mut rows = l.rows.clone();
    s.ordered(&mut rows, true);
    let h = median(rows.iter().map(|r| s.h(r)).collect::<Result<Vec<_>, _>>()?)?;
    let tol = 3f64.max(0.75 * h);
    let mut runs: Vec<Vec<Row>> = Vec::new();
    let mut cur: Vec<Row> = Vec::new();
    for r in rows {
        if !s.l(&r)["semantic_type"].is_null() {
            if !cur.is_empty() {
                runs.push(std::mem::take(&mut cur))
            }
            continue;
        }
        if !cur.is_empty() {
            let left = median(cur.iter().map(|r| r.b.n[0]).collect())?;
            let mut hs = cur.iter().map(|r| s.h(r)).collect::<Result<Vec<_>, _>>()?;
            hs.push(s.h(&r)?);
            let min = hs.iter().copied().fold(f64::INFINITY, f64::min).max(0.1);
            let max = hs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if (r.b.n[0] - left).abs() > tol || max / min > 1.25 {
                runs.push(std::mem::take(&mut cur));
            }
        }
        cur.push(r)
    }
    if !cur.is_empty() {
        runs.push(cur)
    }
    let global = (l.right - l.left).max(0.1);
    for run in runs {
        if run.len() < 3 {
            continue;
        }
        let left = median(run.iter().map(|r| r.b.n[0]).collect())?;
        let right = run
            .iter()
            .map(|r| r.b.n[2])
            .fold(f64::NEG_INFINITY, f64::max);
        let width = (right - left).max(0.1);
        if global < 1.4 * width
            || run
                .iter()
                .filter(|r| r.b.n[2] - r.b.n[0] >= 0.7 * width)
                .count()
                < 3
        {
            continue;
        }
        let index = result.len();
        for r in &run {
            by_source.insert(s.id(r)?, index);
        }
        result.push(Lane {
            left,
            right,
            span: false,
            rows: run,
        })
    }
    Ok((result, by_source))
}
pub(super) fn starts_reference(s: &State, a: &Row, b: &Row) -> Result<bool, Error> {
    let ah = s.h(a)?;
    let bh = s.h(b)?;
    let h = ah.max(bh);
    Ok(
        regex(r"^[［\[]\s*\d+\s*[］\]]").is_match(s.t(b)?.trim_matches(python_space))
            && b.b.n[0] <= a.b.n[0] - 5f64.max(0.6 * ah.min(bh))
            && (-0.75 * h..=1.5 * h).contains(&s.gap(a, b)?),
    )
}
pub(super) fn emphasized(s: &State, rs: &[Row]) -> Result<bool, Error> {
    let Some(first) = rs.first() else {
        return Ok(false);
    };
    if !s.l(first)["leading_emphasis_width"].is_null() {
        return Ok(true);
    }
    let weight = &s.l(first)["dominant_font_weight"];
    let following = rs
        .iter()
        .skip(1)
        .filter(|r| !s.l(r)["dominant_font_weight"].is_null())
        .map(|r| num(&s.l(r)["dominant_font_weight"]))
        .collect::<Result<Vec<_>, _>>()?;
    if weight.is_null() || following.is_empty() {
        return Ok(false);
    }
    let body = median(following)?;
    let w = num(weight)?;
    Ok(w - body >= 100. && w >= 1.15 * body.max(1.))
}
