//! 旧外框/最大填充单元格/行带表格候选；不依赖异常CJK字框。
use super::path_graphics::{array, bbox, integer, num, Box4};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};
mod number;
use number::N;
struct Rect {
    b: Box4,
    coords: [N; 4],
    area: N,
}
impl Rect {
    fn new(b: Box4) -> Self {
        let coords = std::array::from_fn(|i| N::value(&b.raw[i]));
        let area = Self::area(coords);
        Self { b, coords, area }
    }
    fn area(c: [N; 4]) -> N {
        N::F(0.)
            .max(c[2].sub(c[0]))
            .mul(N::F(0.).max(c[3].sub(c[1])))
    }
}
fn overlap(a: &Rect, b: &Rect) -> f64 {
    let ac = a.coords;
    let bc = b.coords;
    let area_a = ac[2].sub(ac[0]).mul(ac[3].sub(ac[1]));
    let area_b = bc[2].sub(bc[0]).mul(bc[3].sub(bc[1]));
    let small = area_a.min(area_b);
    if small.float() <= 0. {
        return 0.;
    }
    let intersection = N::F(0.)
        .max(ac[2].min(bc[2]).sub(ac[0].max(bc[0])))
        .mul(N::F(0.).max(ac[3].min(bc[3]).sub(ac[1].max(bc[1]))));
    intersection.div(small)
}
fn same(a: &Rect, b: &Rect) -> bool {
    a.b.n == b.b.n
}
/// Closed-grid Form exclusion needs Python's exact int/float area comparison too.
pub(super) fn closed_grid_form_excludes(region: &Box4, form: &Box4) -> bool {
    let a = Rect::new(region.clone());
    let b = Rect::new(form.clone());
    if a.area.float() <= 0. {
        return false;
    }
    let ac = a.coords;
    let bc = b.coords;
    let intersection = N::F(0.)
        .max(ac[2].min(bc[2]).sub(ac[0].max(bc[0])))
        .mul(N::F(0.).max(ac[3].min(bc[3]).sub(ac[1].max(bc[1]))));
    intersection.div(a.area) >= 0.95
        && N::F(1.5).mul(a.area).cmp(b.area) == std::cmp::Ordering::Less
}
fn contained(a: &Rect, b: &Rect, x: f64, y: f64) -> bool {
    a.b.n[0] >= b.b.n[0] - x
        && a.b.n[1] >= b.b.n[1] - y
        && a.b.n[2] <= b.b.n[2] + x
        && a.b.n[3] <= b.b.n[3] + y
}
fn cells(rects: &[Rect], outer: &Rect, work: &mut Work) -> Result<Vec<usize>, Error> {
    let w = outer.b.n[2] - outer.b.n[0];
    let h = outer.b.n[3] - outer.b.n[1];
    let (x, y) = (0.01 * w, 0.02 * h);
    work.charge(rects.len() + 1)?;
    let nested: Vec<_> = rects
        .iter()
        .enumerate()
        .filter(|(_, r)| {
            !same(r, outer)
                && contained(r, outer, x, y)
                && r.area.le(N::F(0.8).mul(outer.area))
                && r.b.n[2] - r.b.n[0] >= 0.1 * w
                && r.b.n[3] - r.b.n[1] >= 0.06 * h
        })
        .map(|(i, _)| i)
        .collect();
    work.charge(nested.len() * 16 + 1)?;
    let mut sorted = nested.clone();
    sorted.sort_by(|a, b| {
        rects[*b]
            .area
            .cmp(rects[*a].area)
            .then_with(|| rects[*a].b.n[1].partial_cmp(&rects[*b].b.n[1]).unwrap())
            .then_with(|| rects[*a].b.n[0].partial_cmp(&rects[*b].b.n[0]).unwrap())
    });
    let mut output: Vec<usize> = Vec::new();
    for i in sorted {
        let r = &rects[i];
        work.charge(nested.len() + output.len() + 1)?;
        if nested.iter().any(|j| {
            let o = &rects[*j];
            !same(r, o)
                && contained(r, o, x, y)
                && N::F(1.08).mul(r.area).le(o.area)
                && o.area.le(N::F(0.8).mul(outer.area))
                && ((r.b.n[2] - r.b.n[0]) - (o.b.n[2] - o.b.n[0])).abs() <= 0.01 * w
        }) {
            continue;
        }
        if output.iter().any(|j| overlap(r, &rects[*j]) >= 0.95) {
            continue;
        }
        output.push(i);
    }
    work.charge(output.len() * 16 + 1)?;
    output.sort_by(|a, b| {
        let a = &rects[*a].b.n;
        let b = &rects[*b].b.n;
        a[1].partial_cmp(&b[1])
            .unwrap()
            .then(a[0].partial_cmp(&b[0]).unwrap())
            .then(a[3].partial_cmp(&b[3]).unwrap())
            .then(a[2].partial_cmp(&b[2]).unwrap())
    });
    Ok(output)
}
fn bands(
    rects: &[Rect],
    cells: &[usize],
    outer: &Rect,
    work: &mut Work,
) -> Result<Vec<Vec<usize>>, Error> {
    let y = 0.02 * (outer.b.n[3] - outer.b.n[1]);
    let mut sorted = cells.to_vec();
    work.charge(sorted.len() * 16 + 1)?;
    sorted.sort_by(|a, b| {
        let a = &rects[*a].b.n;
        let b = &rects[*b].b.n;
        a[1].partial_cmp(&b[1])
            .unwrap()
            .then(a[0].partial_cmp(&b[0]).unwrap())
            .then(a[3].partial_cmp(&b[3]).unwrap())
    });
    let mut output: Vec<Vec<usize>> = Vec::new();
    for i in sorted {
        work.charge(output.len() + 1)?;
        let r = &rects[i].b.n;
        if let Some(b) = output.iter_mut().find(|b| {
            let p = &rects[b[0]].b.n;
            (r[1] - p[1]).abs() <= y && (r[3] - p[3]).abs() <= y
        }) {
            b.push(i);
        } else {
            output.push(vec![i]);
        }
    }
    Ok(output)
}
fn covers(rects: &[Rect], band: &[usize], outer: &Rect, work: &mut Work) -> Result<bool, Error> {
    if band.len() < 2 {
        return Ok(false);
    }
    let width = outer.b.n[2] - outer.b.n[0];
    let mut segs: Vec<_> = band
        .iter()
        .map(|i| (rects[*i].b.n[0], rects[*i].b.n[2]))
        .collect();
    work.charge(segs.len() * 16 + 1)?;
    segs.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap()
            .then(a.1.partial_cmp(&b.1).unwrap())
    });
    let (mut left, mut right) = segs[0];
    let (mut covered, mut gap) = (0., 0f64);
    for (l, r) in &segs[1..] {
        if *l <= right {
            right = right.max(*r);
        } else {
            covered += right - left;
            gap = gap.max(*l - right);
            left = *l;
            right = *r;
        }
    }
    covered += right - left;
    let min = segs.iter().map(|s| s.0).reduce(f64::min).unwrap();
    let max = segs.iter().map(|s| s.1).reduce(f64::max).unwrap();
    Ok(covered >= 0.9 * width
        && gap <= 0.01 * width
        && (min - outer.b.n[0]).abs() <= 0.02 * width
        && (max - outer.b.n[2]).abs() <= 0.02 * width)
}
struct Evidence {
    outer: usize,
    count: usize,
    coverage: f64,
    score: f64,
}
pub fn detect(v: &Value) -> Result<Value, Error> {
    detect_with_work(v, &mut Work(0))
}
pub(crate) fn detect_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    validate_tree(v)?;
    let size = array(&v["page_size"], 2)?;
    if size.len() != 2 {
        return Err(invalid());
    }
    let (w, h) = (num(&size[0])?, num(&size[1])?);
    let paths = array(&v["path_infos"], 5000)?;
    let exclusions = v
        .get("excluded_bboxes")
        .filter(|v| !v.is_null())
        .map_or(Ok(&[][..]), |v| array(v, 5000))?;
    work.charge(1)?;
    let page_area = N::value(&size[0]).mul(N::value(&size[1]));
    if page_area.float() <= 0. {
        return Ok(json!({"candidates":[]}));
    }
    let excluded = exclusions
        .iter()
        .map(|v| bbox(v).map(Rect::new))
        .collect::<Result<Vec<_>, _>>()?;
    let mut rects = Vec::new();
    for p in paths {
        work.charge(1)?;
        let fill = p
            .get("fill_visible")
            .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))?;
        if integer(p, "form_depth", 0)? == 0 && fill && integer(p, "segment_count", 0)? == 5 {
            let r = Rect::new(bbox(&p["bbox"])?);
            if r.area.float() > 0. {
                rects.push(r);
            }
        }
    }
    let mut evidence = Vec::new();
    for (i, o) in rects.iter().enumerate() {
        work.charge(excluded.len() + 1)?;
        let width = o.b.n[2] - o.b.n[0];
        let height = o.b.n[3] - o.b.n[1];
        let ratio = o.area.div(page_area);
        if !(0.005..=0.25).contains(&ratio)
            || width < 0.12 * w
            || height < 0.03 * h
            || excluded.iter().any(|e| overlap(o, e) >= 0.5)
        {
            continue;
        }
        let selected = cells(&rects, o, work)?;
        let mut accepted = Vec::new();
        for b in bands(&rects, &selected, o, work)? {
            if covers(&rects, &b, o, work)? {
                accepted.push(b);
            }
        }
        if accepted.len() < 4 {
            continue;
        }
        work.charge(selected.len() + accepted.len() * 16 + 1)?;
        let mut keys: Vec<_> = accepted
            .iter()
            .map(|b| {
                let top = b.iter().map(|i| rects[*i].b.n[1]).reduce(f64::min).unwrap();
                let left = b.iter().map(|i| rects[*i].b.n[0]).reduce(f64::min).unwrap();
                let bottom = b.iter().map(|i| rects[*i].b.n[3]).reduce(f64::max).unwrap();
                (top, left, bottom)
            })
            .collect();
        keys.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap()
                .then(a.1.partial_cmp(&b.1).unwrap())
        });
        let covered =
            super::table_native_candidate::sum(&keys.iter().map(|b| b.2 - b.0).collect::<Vec<_>>());
        let coverage = covered / height;
        if coverage < 0.75 || keys.windows(2).any(|p| p[1].0 - p[0].2 > 0.05 * height) {
            continue;
        }
        evidence.push(Evidence {
            outer: i,
            count: accepted.len(),
            coverage,
            score: 100. + accepted.len() as f64 + coverage,
        });
    }
    work.charge(evidence.len() * 16 + 1)?;
    evidence.sort_by(|a, b| {
        b.count
            .cmp(&a.count)
            .then(b.coverage.partial_cmp(&a.coverage).unwrap())
            .then_with(|| rects[b.outer].area.cmp(rects[a.outer].area))
    });
    let mut output: Vec<Evidence> = Vec::new();
    for item in evidence {
        work.charge(output.len() + 1)?;
        if !output
            .iter()
            .any(|a| overlap(&rects[item.outer], &rects[a.outer]) >= 0.9)
        {
            output.push(item);
        }
    }
    work.charge(output.len() * 16 + 1)?;
    output.sort_by(|a, b| {
        let a = &rects[a.outer].b.n;
        let b = &rects[b.outer].b.n;
        a[1].partial_cmp(&b[1])
            .unwrap()
            .then(a[0].partial_cmp(&b[0]).unwrap())
    });
    let candidates:Vec<_>=output.into_iter().map(|e|{let b=&rects[e.outer].b.raw;json!({"bbox":b,"local_bbox":b,"angle":0,"score":e.score,"core_bbox":b,"line_indices":[],"annotations":[],"inferred_grid":[],"preserve_inline_font_styles":false,"inferred_grid_authoritative":false})}).collect();
    let out = json!({"candidates":candidates});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_filled_grid() {
        let f: Value = serde_json::from_slice(
            &std::fs::read(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../build/migration/native-filled-grid.json"),
            )
            .unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                detect(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
        for c in f["numbers"].as_array().unwrap() {
            let a = Rect::new(bbox(&c["a"]).unwrap());
            let b = Rect::new(bbox(&c["b"]).unwrap());
            assert_eq!(
                a.area.cmp(b.area) as i8,
                c["area_compare"].as_i64().unwrap() as i8
            );
            assert_eq!(a.area.div(b.area), c["ratio"].as_f64().unwrap());
            assert_eq!(overlap(&a, &b), c["overlap"].as_f64().unwrap());
        }
    }
    #[test]
    fn shared_budget() {
        let v = json!({"page_size":[400,600],"path_infos":[],"excluded_bboxes":[]});
        assert!(detect_with_work(&v, &mut Work(10_000_000)).is_err());
        assert!(detect(&v).is_ok());
    }
}
