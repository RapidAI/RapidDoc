//! Original geometric OfficeArt table grid proof, including overlap rejection.
use super::{
    records::{self, Budget},
    shapes::{self, Bbox, Collection, Group},
    styles::MasterLevel,
    text::{Content, Paragraph},
};
use crate::Result;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
#[derive(Debug, Serialize)]
pub struct Cell {
    pub row: usize,
    pub col: usize,
    pub row_span: usize,
    pub col_span: usize,
    pub paragraphs: Vec<Paragraph>,
}
#[derive(Debug, Serialize)]
pub struct Table {
    pub rows: usize,
    pub cols: usize,
    pub cells: Vec<Cell>,
    pub bbox: Bbox,
    pub order: usize,
    pub shape_offsets: BTreeSet<Vec<usize>>,
}
pub fn cluster(mut values: Vec<f64>, tolerance: f64) -> Vec<f64> {
    values.sort_by(f64::total_cmp);
    let mut clusters: Vec<(f64, usize)> = vec![];
    for value in values {
        if let Some((sum, n)) = clusters
            .last_mut()
            .filter(|(sum, n)| (value - *sum / *n as f64).abs() <= tolerance)
        {
            *sum += value;
            *n += 1;
        } else {
            clusters.push((value, 1));
        }
    }
    clusters
        .into_iter()
        .map(|(sum, n)| sum / n as f64)
        .collect()
}
pub fn boundary(values: &[f64], value: f64, tolerance: f64) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .min_by(|a, b| {
            ((*a.1 - value).abs())
                .total_cmp(&(*b.1 - value).abs())
                .then(a.0.cmp(&b.0))
        })
        .filter(|(_, v)| (**v - value).abs() <= tolerance)
        .map(|(i, _)| i)
}
pub fn from_group(
    group: &Group<'_>,
    collection: &Collection<'_>,
    external: &[Content],
    masters: &BTreeMap<u16, Vec<MasterLevel>>,
    links: &BTreeMap<u32, String>,
    budget: &mut Budget,
) -> Result<Option<Table>> {
    let mut cell_shapes = vec![];
    let mut xs = vec![];
    let mut ys = vec![];
    for &index in &group.members {
        let s = &collection.shapes[index];
        let kind = records::children(s.record, budget)?
            .into_iter()
            .find(|r| r.kind == 0xf00a)
            .map(|r| r.instance);
        let [left, top, right, bottom] = s.bbox;
        let w = right - left;
        let h = bottom - top;
        if kind.is_some_and(|k| matches!(k, 20 | 32..=40)) || w <= 2. || h <= 2. {
            if w <= 2. {
                xs.extend([left, right]);
            }
            if h <= 2. {
                ys.extend([top, bottom]);
            }
            continue;
        }
        if kind != Some(1) {
            continue;
        }
        let content = shapes::shape_text(s, external, masters, links, budget)?;
        cell_shapes.push((s, content));
        xs.extend([left, right]);
        ys.extend([top, bottom]);
    }
    if cell_shapes.len() < 4 {
        return Ok(None);
    }
    let xs = cluster(xs, 8.);
    let ys = cluster(ys, 8.);
    let (cols, rows) = (xs.len().saturating_sub(1), ys.len().saturating_sub(1));
    if rows < 2 || cols < 2 || rows.checked_mul(cols).is_none_or(|slots| slots > 4_000_000) {
        return Ok(None);
    }
    let mut covered = vec![false; rows * cols];
    let mut covered_count = 0;
    let mut cells = vec![];
    for (s, content) in cell_shapes {
        let [left, top, right, bottom] = s.bbox;
        let (Some(c0), Some(c1), Some(r0), Some(r1)) = (
            boundary(&xs, left, 16.),
            boundary(&xs, right, 16.),
            boundary(&ys, top, 16.),
            boundary(&ys, bottom, 16.),
        ) else {
            return Ok(None);
        };
        if c1 <= c0 || r1 <= r0 {
            return Ok(None);
        }
        for r in r0..r1 {
            for c in c0..c1 {
                let slot = r * cols + c;
                if covered[slot] {
                    return Ok(None);
                }
                covered[slot] = true;
                covered_count += 1;
            }
        }
        cells.push(Cell {
            row: r0,
            col: c0,
            row_span: r1 - r0,
            col_span: c1 - c0,
            paragraphs: content.map(|c| c.paragraphs).unwrap_or_default(),
        });
    }
    if covered_count != rows * cols {
        return Ok(None);
    }
    cells.sort_by_key(|c| (c.row, c.col));
    Ok(Some(Table {
        rows,
        cols,
        cells,
        bbox: [xs[0], ys[0], *xs.last().unwrap(), *ys.last().unwrap()],
        order: group.order,
        shape_offsets: group
            .members
            .iter()
            .map(|i| collection.shapes[*i].key.clone())
            .collect(),
    }))
}
