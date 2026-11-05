// Pillow 12.2.0 Draw.c polygon edge/scanline rules, MIT-CMU.
// Copyright Secret Labs AB, Fredrik Lundh and Pillow contributors.
use super::{clipper::Contour, header, model::Point};
use crate::Result;
pub struct Mask {
    pub(crate) size: [usize; 2],
    pub(crate) pixels: Vec<u8>,
}
#[derive(Clone, Copy)]
struct Edge {
    x0: i32,
    y0: i32,
    xmin: i32,
    xmax: i32,
    ymin: i32,
    ymax: i32,
    dx: f32,
}
fn edge(a: [i32; 2], b: [i32; 2]) -> Edge {
    Edge {
        x0: a[0],
        y0: a[1],
        xmin: a[0].min(b[0]),
        xmax: a[0].max(b[0]),
        ymin: a[1].min(b[1]),
        ymax: a[1].max(b[1]),
        dx: if a[1] == b[1] {
            0.
        } else {
            b[0].wrapping_sub(a[0]) as f32 / b[1].wrapping_sub(a[1]) as f32
        },
    }
}
fn c_int(v: f64) -> i32 {
    if v < i32::MIN as f64 || v >= 2147483648. || !v.is_finite() {
        i32::MIN
    } else {
        v as i32
    }
}
fn up(v: f32) -> i32 {
    c_int(if v >= 0. {
        (v + 0.5).floor() as f64
    } else {
        -((v.abs() + 0.5).floor() as f64)
    })
}
fn down(v: f32) -> i32 {
    c_int(if v >= 0. {
        (v - 0.5).ceil() as f64
    } else {
        -((v.abs() - 0.5).ceil() as f64)
    })
}
impl Mask {
    pub fn new(size: [usize; 2], fill: u8) -> Result<Self> {
        header::check_render_size(size[0] as u64, size[1] as u64, 1)?;
        Ok(Self {
            size,
            pixels: vec![fill; size[0] * size[1]],
        })
    }
    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }
    fn hline(&mut self, x0: i32, y: i32, x1: i32, color: u8) {
        if y < 0 || y >= self.size[1] as i32 {
            return;
        }
        let x0 = x0.max(0);
        let x1 = x1.min(self.size[0] as i32 - 1);
        if x0 <= x1 {
            self.pixels
                [y as usize * self.size[0] + x0 as usize..=y as usize * self.size[0] + x1 as usize]
                .fill(color);
        }
    }
    pub fn polygon(&mut self, points: &[Point], color: u8) {
        if points.is_empty() {
            return;
        }
        let p: Vec<_> = points.iter().map(|p| [c_int(p[0]), c_int(p[1])]).collect();
        let mut edges: Vec<Edge> = Vec::with_capacity(p.len());
        for i in 0..p.len() - 1 {
            let (a, b) = (p[i], p[i + 1]);
            if a[1] == b[1] && i != 0 && a[1] == p[i - 1][1] {
                if b[0] > a[0] && a[0] > p[i - 1][0] {
                    edges.last_mut().unwrap().xmax = b[0];
                    continue;
                }
                if b[0] < a[0] && a[0] < p[i - 1][0] {
                    edges.last_mut().unwrap().xmin = b[0];
                    continue;
                }
            }
            edges.push(edge(a, b));
        }
        if p.last() != p.first() {
            edges.push(edge(*p.last().unwrap(), p[0]));
        }
        let mut ymin = self.size[1] as i32 - 1;
        let mut ymax = 0;
        let mut active = Vec::new();
        for e in &edges {
            ymin = ymin.min(e.ymin);
            ymax = ymax.max(e.ymax);
            if e.ymin == e.ymax {
                self.hline(e.xmin, e.ymin, e.xmax, color);
            } else {
                active.push(e);
            }
        }
        let mut xs: Vec<f32> = Vec::with_capacity(active.len() * 2);
        for y in ymin.max(0)..=ymax.min(self.size[1] as i32) {
            xs.clear();
            for (i, e) in active.iter().enumerate() {
                if y < e.ymin || y > e.ymax {
                    continue;
                }
                let mut x = y.wrapping_sub(e.y0) as f32 * e.dx + e.x0 as f32;
                if y == e.ymax && y < ymax.min(self.size[1] as i32) {
                    xs.extend([x, x]);
                    continue;
                }
                if (y == e.ymin || y == e.ymax) && e.dx != 0. {
                    for other in &active[..i] {
                        if (y != other.ymin && y != other.ymax) || other.dx == 0. {
                            continue;
                        }
                        let ox = y.wrapping_sub(other.y0) as f32 * other.dx + other.x0 as f32;
                        if x.round() == ox.round() {
                            let offset = if y == e.ymax { -1 } else { 1 };
                            let adjacent = y.wrapping_add(offset).wrapping_sub(e.y0) as f32 * e.dx
                                + e.x0 as f32;
                            if y + offset >= other.ymin && y + offset <= other.ymax {
                                let adjacent_other =
                                    y.wrapping_add(offset).wrapping_sub(other.y0) as f32 * other.dx
                                        + other.x0 as f32;
                                if x > adjacent + 1. && x > adjacent_other + 1. {
                                    x = adjacent.max(adjacent_other).round() + 1.;
                                } else if x < adjacent - 1. && x < adjacent_other - 1. {
                                    x = adjacent.min(adjacent_other).round() - 1.;
                                }
                                break;
                            }
                        }
                    }
                }
                xs.push(x);
            }
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            for pair in xs.chunks_exact(2) {
                self.hline(up(pair[0]), y, down(pair[1]), color);
            }
        }
    }
    pub fn contours(&mut self, contours: &[Contour]) {
        for c in contours {
            if c.points.len() >= 3 {
                let points: Vec<_> = c
                    .points
                    .iter()
                    .map(|p| [p[0] as f64 / 256., p[1] as f64 / 256.])
                    .collect();
                self.polygon(&points, if c.hole { 0 } else { 255 });
            }
        }
    }
}
