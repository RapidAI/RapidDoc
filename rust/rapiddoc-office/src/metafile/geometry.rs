// Copyright (c) 2026 Xiaomeng Zhao (myhloli). SPDX-License-Identifier: MIT
use super::{malformed, model::*, resource};
use crate::Result;
const K: f64 = 0.5522847498307936;
#[derive(Debug, Default)]
pub struct Builder {
    segments: Vec<Segment>,
    pub current: Option<Point>,
    pub start: Option<Point>,
    closed: bool,
}
impl Builder {
    pub fn move_to(&mut self, p: Point) {
        self.segments.push(Segment {
            verb: Verb::M,
            points: vec![p],
        });
        self.current = Some(p);
        self.start = Some(p);
        self.closed = false;
    }
    pub fn line_to(&mut self, p: Point) {
        if self.current.is_none() {
            self.move_to(p);
            return;
        }
        if self.closed {
            self.move_to(self.current.unwrap());
        }
        self.segments.push(Segment {
            verb: Verb::L,
            points: vec![p],
        });
        self.current = Some(p);
    }
    pub fn cubic_to(&mut self, a: Point, b: Point, end: Point) {
        if self.current.is_none() {
            self.move_to(end);
            return;
        }
        if self.closed {
            self.move_to(self.current.unwrap());
        }
        self.segments.push(Segment {
            verb: Verb::C,
            points: vec![a, b, end],
        });
        self.current = Some(end);
    }
    pub fn close(&mut self) {
        if self.current.is_none() || self.start.is_none() || self.closed {
            return;
        }
        self.segments.push(Segment {
            verb: Verb::Z,
            points: vec![],
        });
        self.current = self.start;
        self.closed = true;
    }
    pub fn figure_open(&self) -> bool {
        self.current.is_some() && self.start.is_some() && !self.closed
    }
    pub fn extend(&mut self, path: &Path) {
        for s in path.segments.iter() {
            match s.verb {
                Verb::M => self.move_to(s.points[0]),
                Verb::L => self.line_to(s.points[0]),
                Verb::C => self.cubic_to(s.points[0], s.points[1], s.points[2]),
                Verb::Z => self.close(),
            }
        }
    }
    pub fn build(&self) -> Path {
        Path {
            segments: self.segments.clone().into(),
        }
    }
    pub fn take(&mut self) -> Path {
        let path = Path {
            segments: std::mem::take(&mut self.segments).into(),
        };
        self.current = None;
        self.start = None;
        self.closed = false;
        path
    }
    pub fn clear(&mut self) {
        self.segments.clear();
        self.current = None;
        self.start = None;
        self.closed = false;
    }
}
pub fn transform(path: &Path, m: Matrix) -> Path {
    Path {
        segments: path
            .segments
            .iter()
            .map(|s| Segment {
                verb: s.verb,
                points: s.points.iter().map(|p| m.point(*p)).collect(),
            })
            .collect::<Vec<_>>()
            .into(),
    }
}
pub fn polygon(points: &[Point], close: bool) -> Path {
    let mut b = Builder::default();
    if let Some(first) = points.first() {
        b.move_to(*first);
        for p in &points[1..] {
            b.line_to(*p);
        }
        if close {
            b.close();
        }
    }
    b.take()
}
pub fn rectangle(rect: Rect) -> Path {
    let r = rect.normalized();
    polygon(
        &[
            [r.left, r.top],
            [r.right, r.top],
            [r.right, r.bottom],
            [r.left, r.bottom],
        ],
        true,
    )
}
pub fn round_rectangle(rect: Rect, rx: f64, ry: f64) -> Path {
    let r = rect.normalized();
    let rx = rx.abs().min(r.width().abs() / 2.);
    let ry = ry.abs().min(r.height().abs() / 2.);
    if rx <= 0. || ry <= 0. {
        return rectangle(r);
    }
    let kx = rx * K;
    let ky = ry * K;
    let mut b = Builder::default();
    let (l, t, r, bt) = (r.left, r.top, r.right, r.bottom);
    b.move_to([l + rx, t]);
    b.line_to([r - rx, t]);
    b.cubic_to([r - rx + kx, t], [r, t + ry - ky], [r, t + ry]);
    b.line_to([r, bt - ry]);
    b.cubic_to([r, bt - ry + ky], [r - rx + kx, bt], [r - rx, bt]);
    b.line_to([l + rx, bt]);
    b.cubic_to([l + rx - kx, bt], [l, bt - ry + ky], [l, bt - ry]);
    b.line_to([l, t + ry]);
    b.cubic_to([l, t + ry - ky], [l + rx - kx, t], [l + rx, t]);
    b.close();
    b.take()
}
pub fn ellipse(rect: Rect) -> Path {
    let r = rect.normalized();
    let cx = (r.left + r.right) / 2.;
    let cy = (r.top + r.bottom) / 2.;
    let rx = r.width() / 2.;
    let ry = r.height() / 2.;
    if rx <= 0. || ry <= 0. {
        return Path::default();
    }
    let kx = rx * K;
    let ky = ry * K;
    let mut b = Builder::default();
    b.move_to([cx + rx, cy]);
    b.cubic_to([cx + rx, cy + ky], [cx + kx, cy + ry], [cx, cy + ry]);
    b.cubic_to([cx - kx, cy + ry], [cx - rx, cy + ky], [cx - rx, cy]);
    b.cubic_to([cx - rx, cy - ky], [cx - kx, cy - ry], [cx, cy - ry]);
    b.cubic_to([cx + kx, cy - ry], [cx + rx, cy - ky], [cx + rx, cy]);
    b.close();
    b.take()
}
pub fn arc(rect: Rect, start: Point, end: Point, direction: u32, close: &str) -> Path {
    let r = rect.normalized();
    let cx = (r.left + r.right) / 2.;
    let cy = (r.top + r.bottom) / 2.;
    let rx = r.width() / 2.;
    let ry = r.height() / 2.;
    if rx <= 0. || ry <= 0. {
        return Path::default();
    }
    let a = ((start[1] - cy) / ry).atan2((start[0] - cx) / rx);
    let mut z = ((end[1] - cy) / ry).atan2((end[0] - cx) / rx);
    let pi = std::f64::consts::PI;
    if direction == 1 {
        while z >= a {
            z -= 2. * pi;
        }
    } else {
        while z <= a {
            z += 2. * pi;
        }
    }
    let sweep = z - a;
    let steps = (sweep.abs() / (pi / 24.)).ceil().clamp(4., 256.) as usize;
    let mut b = Builder::default();
    if close == "pie" {
        b.move_to([cx, cy]);
    }
    for i in 0..=steps {
        let theta = a + sweep * i as f64 / steps as f64;
        let p = [cx + rx * theta.cos(), cy + ry * theta.sin()];
        if i == 0 && close != "pie" {
            b.move_to(p);
        } else {
            b.line_to(p);
        }
    }
    if matches!(close, "pie" | "chord") {
        b.close();
    }
    b.take()
}
pub fn bounds(path: &Path) -> Option<Rect> {
    let mut points = path.segments.iter().flat_map(|s| &s.points);
    let p = points.next()?;
    let mut r = Rect {
        left: p[0],
        top: p[1],
        right: p[0],
        bottom: p[1],
    };
    for p in points {
        r.left = r.left.min(p[0]);
        r.top = r.top.min(p[1]);
        r.right = r.right.max(p[0]);
        r.bottom = r.bottom.max(p[1]);
    }
    Some(r)
}
pub fn close_open(path: &Path) -> Path {
    let mut out = Vec::new();
    let mut open = false;
    for s in path.segments.iter() {
        if s.verb == Verb::M {
            if open {
                out.push(Segment {
                    verb: Verb::Z,
                    points: vec![],
                });
            }
            open = true;
        } else if s.verb == Verb::Z {
            open = false;
        }
        out.push(s.clone());
    }
    if open {
        out.push(Segment {
            verb: Verb::Z,
            points: vec![],
        });
    }
    Path {
        segments: out.into(),
    }
}
pub fn union(rects: impl IntoIterator<Item = Option<Rect>>) -> Option<Rect> {
    let mut r = None;
    for rect in rects.into_iter().flatten().map(Rect::normalized) {
        r = Some(match r {
            None => rect,
            Some(old) => {
                let old: Rect = old;
                Rect {
                    left: old.left.min(rect.left),
                    top: old.top.min(rect.top),
                    right: old.right.max(rect.right),
                    bottom: old.bottom.max(rect.bottom),
                }
            }
        });
    }
    r
}
pub struct FlattenBudget {
    pub limit: usize,
    pub used: usize,
}
impl Default for FlattenBudget {
    fn default() -> Self {
        Self {
            limit: 5_000_000,
            used: 0,
        }
    }
}
impl FlattenBudget {
    fn charge(&mut self) -> Result<()> {
        self.charge_many(1)
    }
    pub(crate) fn charge_many(&mut self, count: usize) -> Result<()> {
        self.used += count;
        if self.used > self.limit {
            Err(resource("metafile exceeds Python max_flattened_points"))
        } else {
            Ok(())
        }
    }
}
fn distance(p: Point, a: Point, b: Point) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let length = dx.hypot(dy);
    if length <= 1e-12 {
        (p[0] - a[0]).hypot(p[1] - a[1])
    } else {
        (dy * p[0] - dx * p[1] + b[0] * a[1] - b[1] * a[0]).abs() / length
    }
}
fn midpoint(a: Point, b: Point) -> Point {
    [(a[0] + b[0]) / 2., (a[1] + b[1]) / 2.]
}
fn cubic(
    a: Point,
    b: Point,
    c: Point,
    d: Point,
    flatness: f64,
    max_depth: u32,
    budget: &mut FlattenBudget,
) -> Result<Vec<Point>> {
    let mut stack = vec![(a, b, c, d, 0)];
    let mut out = Vec::new();
    while let Some((a, b, c, d, depth)) = stack.pop() {
        let error = distance(b, a, d).max(distance(c, a, d));
        if error <= flatness || depth >= max_depth {
            budget.charge()?;
            out.push(d);
            continue;
        }
        let ab = midpoint(a, b);
        let bc = midpoint(b, c);
        let cd = midpoint(c, d);
        let abc = midpoint(ab, bc);
        let bcd = midpoint(bc, cd);
        let center = midpoint(abc, bcd);
        stack.push((center, bcd, cd, d, depth + 1));
        stack.push((a, ab, abc, center, depth + 1));
    }
    Ok(out)
}
pub fn flatten(
    path: &Path,
    flatness: f64,
    max_depth: u32,
    budget: &mut FlattenBudget,
) -> Result<Vec<(Vec<Point>, bool)>> {
    if flatness <= 0. {
        return Err(malformed());
    }
    let mut subpaths = Vec::new();
    let mut points = Vec::new();
    let mut current = None;
    let mut closed = false;
    for s in path.segments.iter() {
        match s.verb {
            Verb::M => {
                if !points.is_empty() {
                    subpaths.push((std::mem::take(&mut points), closed));
                }
                points.push(s.points[0]);
                budget.charge()?;
                current = Some(s.points[0]);
                closed = false;
            }
            Verb::L => {
                if current.is_none() {
                    points.clear();
                }
                points.push(s.points[0]);
                budget.charge()?;
                current = Some(s.points[0]);
            }
            Verb::C => {
                if let Some(a) = current {
                    points.extend(cubic(
                        a,
                        s.points[0],
                        s.points[1],
                        s.points[2],
                        flatness,
                        max_depth,
                        budget,
                    )?);
                    current = Some(s.points[2]);
                }
            }
            Verb::Z => {
                closed = true;
                if !points.is_empty() && points.last() != points.first() {
                    points.push(points[0]);
                    budget.charge()?;
                }
            }
        }
    }
    if !points.is_empty() {
        subpaths.push((points, closed));
    }
    Ok(subpaths)
}
