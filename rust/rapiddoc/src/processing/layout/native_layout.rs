//! Existing Python layout_evidence: current-stage lanes, corridors and separation.
use super::{
    native_marginals::local,
    native_titles::{reference_height as height, reference_lane_line as line},
    page_containers::median,
    path_graphics::{array, bbox, integer, num, overlap_first, Box4},
};
use crate::legacy_page::Work;
use rapiddoc_core::Error;
use serde_json::{json, Value};
pub(super) struct Layout {
    pub(super) lanes: Vec<Value>,
    pub(super) rows: Vec<(Value, Box4)>,
    width: f64,
    em: f64,
}
fn cx(b: &Box4) -> f64 {
    (b.n[0] + b.n[2]) * 0.5
}
fn cy(b: &Box4) -> f64 {
    (b.n[1] + b.n[3]) * 0.5
}
pub(super) fn infer(
    rows: &[(Value, Box4)],
    width: f64,
    em: f64,
    recalculate: bool,
    w: &mut Work,
) -> Result<Vec<Value>, Error> {
    if rows.is_empty() {
        return Ok(vec![]);
    }
    let v = super::table_text_styles::lanes::infer_formula_layout(
        &json!({"line_geometry":rows.iter().map(|(l,b)|json!([line(l),b.raw])).collect::<Vec<_>>(),"local_page_width":width,"median_height":em,"recalculate_intervals":recalculate}),
        w,
    )?;
    Ok(array(&v, 4096)?.to_vec())
}
pub(super) fn build(
    lines: &[Value],
    size: [f64; 2],
    angle: i64,
    barriers: &[Box4],
    w: &mut Work,
) -> Result<Layout, Error> {
    let width = if angle == 90 || angle == 270 {
        size[1]
    } else {
        size[0]
    };
    let mut rows = Vec::new();
    for l in lines {
        w.charge(barriers.len() + 1)?;
        if integer(l, "angle", 0)? != angle {
            continue;
        }
        let raw = bbox(&l["bbox"])?;
        if barriers.iter().any(|b| overlap_first(&raw, b) >= 0.8) {
            continue;
        }
        let b = local(
            &bbox(if l["ink_bbox"].is_null() {
                &l["bbox"]
            } else {
                &l["ink_bbox"]
            })?,
            size,
            angle,
        );
        rows.push((l.clone(), b))
    }
    let em = if rows.is_empty() {
        10.
    } else {
        median(
            rows.iter()
                .map(|(l, b)| height(l, b))
                .collect::<Result<Vec<_>, _>>()?,
        )
    };
    let mut lanes = infer(&rows, width, em, false, w)?
        .into_iter()
        .filter(|l| l["is_span"] == false && l["line_indices"].as_array().unwrap().len() >= 3)
        .collect::<Vec<_>>();
    lanes.sort_by(|a, b| {
        a["left"]
            .as_f64()
            .unwrap()
            .total_cmp(&b["left"].as_f64().unwrap())
    });
    let mut distinct: Vec<Value> = Vec::new();
    for l in lanes {
        if distinct
            .last()
            .is_some_and(|p| l["left"].as_f64().unwrap() < p["right"].as_f64().unwrap())
        {
            if l["line_indices"].as_array().unwrap().len()
                > distinct.last().unwrap()["line_indices"]
                    .as_array()
                    .unwrap()
                    .len()
            {
                *distinct.last_mut().unwrap() = l
            }
        } else {
            distinct.push(l)
        }
    }
    Ok(Layout {
        lanes: distinct,
        rows,
        width,
        em,
    })
}
impl Layout {
    pub(super) fn corridor(&self, b: &Box4) -> Result<Option<(f64, f64)>, Error> {
        if self.lanes.is_empty() {
            return Ok(None);
        }
        let mut substantial = 0;
        for l in &self.lanes {
            let left = num(&l["left"])?;
            let right = num(&l["right"])?;
            if (b.n[2].min(right) - b.n[0].max(left)).max(0.)
                >= 0.25 * (b.n[2] - b.n[0]).min(right - left)
            {
                substantial += 1
            }
        }
        if substantial > 1 {
            return Ok(Some((0., self.width)));
        }
        let mut left = 0.;
        for k in 0..self.lanes.len() {
            let right = if k + 1 < self.lanes.len() {
                (num(&self.lanes[k]["right"])? + num(&self.lanes[k + 1]["left"])?) * 0.5
            } else {
                self.width
            };
            if left <= cx(b) && cx(b) <= right {
                return Ok(Some((left, right)));
            }
            left = right;
        }
        Ok(None)
    }
    pub(super) fn separated(&self, a: &Box4, b: &Box4, w: &mut Work) -> Result<bool, Error> {
        let ac = self.corridor(a)?;
        let bc = self.corridor(b)?;
        if ac.is_some()
            && bc.is_some()
            && ac != bc
            && ac != Some((0., self.width))
            && bc != Some((0., self.width))
        {
            return Ok(true);
        }
        let (l, r) = if a.n[0] <= b.n[0] { (a, b) } else { (b, a) };
        if r.n[0] - l.n[2] <= 0.5 * self.em || (l.n[2] - l.n[0]).min(r.n[2] - r.n[0]) < 3. * self.em
        {
            return Ok(false);
        }
        let top = a.n[1].min(b.n[1]) - 3. * self.em;
        let bottom = a.n[3].max(b.n[3]) + 3. * self.em;
        let rows = self
            .rows
            .iter()
            .filter(|(_, b)| top <= cy(b) && cy(b) <= bottom)
            .cloned()
            .collect::<Vec<_>>();
        if rows.is_empty() {
            return Ok(false);
        }
        for (seed, b) in &rows {
            if seed["caption_start"] != true {
                continue;
            }
            let followers = rows
                .iter()
                .filter(|(_, c)| {
                    (c.n[0] - b.n[0]).abs() <= 0.75 * self.em
                        && b.n[1] <= c.n[1]
                        && c.n[1] <= b.n[3] + 4. * self.em
                })
                .map(|(_, b)| b)
                .collect::<Vec<_>>();
            w.charge(rows.len() + 1)?;
            if followers.len() >= 2 {
                if l.n[2] <= b.n[0] - 0.5 * self.em && (r.n[0] - b.n[0]).abs() <= self.em {
                    return Ok(true);
                }
                let edge = followers.iter().map(|b| b.n[2]).reduce(f64::max).unwrap();
                if r.n[0] >= edge + 0.5 * self.em && (l.n[0] - b.n[0]).abs() <= self.em {
                    return Ok(true);
                }
            }
        }
        let mut lanes = infer(&rows, self.width, self.em, true, w)?
            .into_iter()
            .filter(|l| l["is_span"] == false && l["line_indices"].as_array().unwrap().len() >= 3)
            .collect::<Vec<_>>();
        lanes.sort_by(|a, b| {
            a["left"]
                .as_f64()
                .unwrap()
                .total_cmp(&b["left"].as_f64().unwrap())
        });
        for pair in lanes.windows(2) {
            if l.n[2] <= num(&pair[0]["right"])? + self.em
                && r.n[0] >= num(&pair[1]["left"])? - self.em
                && num(&pair[0]["right"])? < num(&pair[1]["left"])?
            {
                return Ok(true);
            }
        }
        Ok(false)
    }
}
