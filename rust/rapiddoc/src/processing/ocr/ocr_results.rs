//! 旧sorted_boxes/merge_det_boxes/update_det_boxes/get_ocr_result_list及rec回填。
use crate::legacy_model::native_ocr::{number, useful};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};

// NumPy2标量与Python弱标量运算：min/max保留赢家dtype，算术/比较再按F32提升。
#[derive(Clone, Copy, Debug)]
struct N {
    v: f64,
    f: bool,
}
impl N {
    fn py(v: f64) -> Self {
        Self { v, f: false }
    }
    fn np(v: f64) -> Self {
        Self {
            v: v as f32 as f64,
            f: true,
        }
    }
    fn calc(self, b: Self, op: char) -> Self {
        let f = self.f || b.f;
        let v = if f {
            let a = self.v as f32;
            let b = b.v as f32;
            (match op {
                '-' => a - b,
                '+' => a + b,
                '*' => a * b,
                '/' => a / b,
                _ => unreachable!(),
            }) as f64
        } else {
            match op {
                '-' => self.v - b.v,
                '+' => self.v + b.v,
                '*' => self.v * b.v,
                '/' => self.v / b.v,
                _ => unreachable!(),
            }
        };
        Self { v, f }
    }
    fn lt(self, b: Self) -> bool {
        if self.f || b.f {
            (self.v as f32) < b.v as f32
        } else {
            self.v < b.v
        }
    }
    fn min(self, b: Self) -> Self {
        if b.lt(self) {
            b
        } else {
            self
        }
    }
    fn max(self, b: Self) -> Self {
        if self.lt(b) {
            b
        } else {
            self
        }
    }
    fn sub(self, b: Self) -> Self {
        self.calc(b, '-')
    }
    fn add(self, b: Self) -> Self {
        self.calc(b, '+')
    }
    fn mul(self, b: f64) -> Self {
        self.calc(Self::py(b), '*')
    }
    fn div(self, b: Self) -> Self {
        self.calc(b, '/')
    }
}
type Quad = [[f64; 2]; 4];
type B = [N; 4];
fn angle(q: &Quad) -> bool {
    let n = |i: usize, k: usize| N::np(q[i][k]);
    let h = n(3, 1)
        .sub(n(0, 1))
        .add(n(2, 1).sub(n(1, 1)))
        .div(N::py(2.));
    let d = n(2, 1).sub(n(0, 1));
    d.lt(h.mul(0.8)) || h.mul(1.2).lt(d)
}
fn bbox(q: &Quad) -> B {
    [
        N::np(q[0][0]),
        N::np(q[0][1]),
        N::np(q[1][0]),
        N::np(q[2][1]),
    ]
}
fn points(b: B) -> Quad {
    [
        [b[0].v, b[1].v],
        [b[2].v, b[1].v],
        [b[2].v, b[3].v],
        [b[0].v, b[3].v],
    ]
    .map(|p| p.map(|v| v as f32 as f64))
}
fn y_overlap(a: B, b: B, threshold: f64) -> bool {
    let h = a[3].sub(a[1]).min(b[3].sub(b[1]));
    let overlap = N::py(0.).max(a[3].min(b[3]).sub(a[1].max(b[1])));
    N::py(0.).lt(h) && N::py(threshold).lt(overlap.div(h))
}
fn quads(v: &Value) -> Result<Vec<Quad>, Error> {
    let a = v.as_array().ok_or_else(invalid)?;
    if a.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    a.iter()
        .map(|q| {
            let q = q.as_array().filter(|a| a.len() == 4).ok_or_else(invalid)?;
            let mut out = [[0.; 2]; 4];
            for (i, p) in q.iter().enumerate() {
                let p = p.as_array().filter(|a| a.len() == 2).ok_or_else(invalid)?;
                for k in 0..2 {
                    let n = number(&p[k])?;
                    if n.abs() > rapiddoc_core::resources::limit(1_000_000.) {
                        return Err(invalid());
                    }
                    out[i][k] = n as f32 as f64;
                }
            }
            Ok(out)
        })
        .collect()
}
fn masks(v: &Value) -> Result<Vec<B>, Error> {
    let a = v.as_array().ok_or_else(invalid)?;
    if a.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    a.iter()
        .map(|m| Ok(crate::legacy_model::coordinates(&m["bbox"])?.map(N::py)))
        .collect()
}
fn ordered(mut q: Vec<Quad>, work: &mut Work) -> Result<Vec<Quad>, Error> {
    q.sort_by(|a, b| {
        a[0][1]
            .partial_cmp(&b[0][1])
            .unwrap()
            .then_with(|| a[0][0].partial_cmp(&b[0][0]).unwrap())
    });
    for i in 0..q.len().saturating_sub(1) {
        for j in (0..=i).rev() {
            work.charge(1)?;
            let d = (q[j + 1][0][1] as f32 - q[j][0][1] as f32).abs();
            if d < 10. && q[j + 1][0][0] < q[j][0][0] {
                q.swap(j, j + 1);
            } else {
                break;
            }
        }
    }
    Ok(q)
}
fn merge(q: Vec<Quad>) -> Vec<Quad> {
    let (mut aligned, mut angled) = (Vec::new(), Vec::new());
    for p in q {
        if angle(&p) {
            angled.push(p);
        } else {
            aligned.push(bbox(&p));
        }
    }
    aligned.sort_by(|a, b| a[1].v.partial_cmp(&b[1].v).unwrap());
    let mut lines: Vec<Vec<B>> = Vec::new();
    for b in aligned {
        if let Some(line) = lines
            .last_mut()
            .filter(|l| y_overlap(b, *l.last().unwrap(), 0.6))
        {
            line.push(b);
        } else {
            lines.push(vec![b]);
        }
    }
    let mut out = Vec::new();
    for mut line in lines {
        let mut bounds = line[0];
        for b in &line[1..] {
            for i in 0..4 {
                bounds[i] = if i < 2 {
                    bounds[i].min(b[i])
                } else {
                    bounds[i].max(b[i])
                };
            }
        }
        if bounds[3]
            .sub(bounds[1])
            .mul(4.)
            .lt(bounds[2].sub(bounds[0]))
        {
            line.sort_by(|a, b| a[0].v.partial_cmp(&b[0].v).unwrap());
            let mut merged: Vec<B> = Vec::new();
            for b in line {
                if let Some(last) = merged.last_mut().filter(|l| !l[2].lt(b[0])) {
                    for i in 0..4 {
                        last[i] = if i < 2 {
                            last[i].min(b[i])
                        } else {
                            last[i].max(b[i])
                        };
                    }
                } else {
                    merged.push(b);
                }
            }
            out.extend(merged.into_iter().map(points));
        } else {
            out.extend(line.into_iter().map(points));
        }
    }
    out.append(&mut angled);
    out
}
fn cut(q: Vec<Quad>, masks: &[B], work: &mut Work) -> Result<Vec<Quad>, Error> {
    if masks.is_empty() {
        return Ok(q);
    }
    let (mut out, mut angled) = (Vec::new(), Vec::new());
    for p in q {
        if angle(&p) {
            angled.push(p);
            continue;
        }
        let b = bbox(&p);
        let mut intervals = Vec::new();
        for &m in masks {
            work.charge(1)?;
            if y_overlap(b, m, 0.8) {
                intervals.push([m[0], m[2]]);
            }
        }
        intervals.sort_by(|a, b| a[0].v.partial_cmp(&b[0].v).unwrap());
        let mut merged: Vec<[N; 2]> = Vec::new();
        for m in intervals {
            if let Some(last) = merged.last_mut().filter(|l| !l[1].lt(m[0])) {
                last[1] = last[1].max(m[1]);
            } else {
                merged.push(m);
            }
        }
        let mut start = b[0];
        for m in merged {
            if b[2].lt(m[0]) || m[1].lt(start) {
                continue;
            }
            if start.lt(m[0]) {
                out.push(points([start, b[1], m[0].sub(N::py(1.)), b[3]]));
            }
            start = m[1].add(N::py(1.)).max(start);
        }
        if !b[2].lt(start) {
            out.push(points([start, b[1], b[2], b[3]]));
        }
        if out.len() + angled.len() > rapiddoc_core::resources::limit(20_000) {
            return Err(limit());
        }
    }
    out.append(&mut angled);
    Ok(out)
}
pub fn boxes(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let mut work = Work(0);
    Ok(json!(cut(
        merge(ordered(quads(&request["boxes"])?, &mut work)?),
        &masks(&request["masks"])?,
        &mut work
    )?))
}
/// 表格外层只排序，mfd存在时才update；绝不调用正文merge_det_boxes。
pub fn table_boxes(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let mut work = Work(0);
    let sorted = ordered(quads(&request["boxes"])?, &mut work)?;
    let masks = masks(&request["masks"])?;
    Ok(json!(if masks.is_empty() {
        sorted
    } else {
        cut(sorted, &masks, &mut work)?
    }))
}
fn py_round(v: f64, n: usize) -> f64 {
    format!("{v:.n$}").parse().unwrap()
}
// Python3.14 sum(float)使用补偿和，避免倾斜框中心坐标因简单逐项相加漂移。
fn sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let (mut hi, mut lo) = (0f64, 0f64);
    for x in values {
        let t = hi + x;
        lo += if hi.abs() >= x.abs() {
            (hi - t) + x
        } else {
            (x - t) + hi
        };
        hi = t;
    }
    hi + lo
}
fn mostly_tilted(q: &[Quad]) -> bool {
    if q.is_empty() {
        return false;
    }
    let angles = q.iter().map(|p| {
        py_round(
            (p[1][1] - p[0][1])
                .atan2(p[1][0] - p[0][0])
                .to_degrees()
                .abs()
                % 180.,
            2,
        )
    });
    let avg = py_round(sum(angles) / q.len() as f64, 2);
    avg.abs() > 1. && (avg - 180.).abs() > 1.
}
pub fn plan(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let mut work = Work(0);
    let index = request["page_index"]
        .as_u64()
        .filter(|n| *n <= rapiddoc_core::resources::limit(1_000_000_000))
        .ok_or_else(invalid)?;
    let plan = &request["plan"];
    let jobs = plan["detect"].as_array().ok_or_else(invalid)?;
    let results = request["detections"].as_array().ok_or_else(invalid)?;
    if jobs.len() != results.len() {
        return Err(invalid());
    }
    if jobs.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let lang = request.get("lang").unwrap_or(&Value::Null);
    if !lang.is_null()
        && lang
            .as_str()
            .is_none_or(|s| s.len() > rapiddoc_core::resources::limit(128))
    {
        return Err(invalid());
    }
    let mut model = plan["model"].clone();
    let dets = model["layout_dets"].as_array_mut().ok_or_else(invalid)?;
    let mut crops = Vec::new();
    let mut rec = Vec::new();
    let mut added = Vec::new();
    let mut processed = Vec::new();
    for (job, result) in jobs.iter().zip(results) {
        let q = cut(
            merge(ordered(quads(result)?, &mut work)?),
            &masks(&job["masks"])?,
            &mut work,
        )?;
        let r = dets
            .get(job["layout_index"].as_u64().ok_or_else(invalid)? as usize)
            .ok_or_else(invalid)?;
        let enabled = job["ocr_enable"].as_bool().ok_or_else(invalid)? || mostly_tilted(&q);
        let u = useful(&job["useful_list"])?;
        processed.push(json!({"layout_index":job["layout_index"],"boxes":q,"ocr_enable":enabled}));
        for raw in q {
            let accepted = raw[2][0] - raw[0][0] >= 3.;
            let id = format!("rec-{index}-{}", rec.len());
            if enabled {
                crops.push(json!({"source":job["crop_source"],"points":raw,"output":if accepted{Some(&id)}else{None}}));
            }
            if !accepted {
                continue;
            }
            let mut p = raw;
            // get_ocr_result_list先tolist，以下几何运算是Python FP64而非NumPy F32。
            let h = ((raw[3][1] - raw[0][1]) + (raw[2][1] - raw[1][1])) / 2.;
            let d = raw[2][1] - raw[0][1];
            if !(0.8 * h <= d && d <= 1.2 * h) {
                let x = sum(raw.map(|v| v[0])) / 4.;
                let y = sum(raw.map(|v| v[1])) / 4.;
                let w = raw[2][0] - raw[0][0];
                p = [
                    [x - w / 2., y - h / 2.],
                    [x + w / 2., y - h / 2.],
                    [x + w / 2., y + h / 2.],
                    [x - w / 2., y + h / 2.],
                ];
            }
            let mut poly = Vec::new();
            for p in p {
                for k in 0..2 {
                    poly.push(p[k] - number(&u[k])? + number(&u[2 + k])?);
                }
            }
            let mut span = json!({"category_id":15,"original_label":r["original_label"],"original_order":r["original_order"],"poly":poly,"score":if enabled{json!(1)}else{json!(1.0)},"text":""});
            if enabled {
                span["np_img"] = json!(id);
                span["lang"] = lang.clone();
                rec.push(json!({"source":id,"model_index":dets.len()+added.len()}));
            }
            added.push(span);
            if crops.len() > rapiddoc_core::resources::limit(1000)
                || added.len() > rapiddoc_core::resources::limit(20_000)
            {
                return Err(limit());
            }
        }
    }
    dets.extend(added);
    Ok(json!({"model":model,"processed":processed,"recognition":rec,"crop_attempts":crops}))
}
pub fn apply(request: &Value) -> Result<Value, Error> {
    validate_tree(request)?;
    let mut model = request["model"].clone();
    let results = request["results"].as_array().ok_or_else(invalid)?;
    let dets = model["layout_dets"].as_array_mut().ok_or_else(invalid)?;
    let indices = dets
        .iter()
        .enumerate()
        .filter(|(_, d)| {
            d["category_id"] == 15 && d.get("np_img").is_some() && d.get("lang").is_some()
        })
        .map(|(i, _)| i)
        .collect::<Vec<_>>();
    if indices.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    if indices.len() != results.len() {
        return Err(invalid());
    }
    for (i, result) in indices.into_iter().zip(results) {
        let a = result
            .as_array()
            .filter(|a| a.len() == 2)
            .ok_or_else(invalid)?;
        let text = a[0].as_str().ok_or_else(invalid)?;
        let score = number(&a[1])?;
        if !(0. ..=1.).contains(&score) {
            return Err(invalid());
        }
        let item = &mut dets[i];
        item.as_object_mut().unwrap().remove("np_img");
        item.as_object_mut().unwrap().remove("lang");
        item["text"] = a[0].clone();
        item["score"] = json!(py_round(score, 3));
        let b = &item["poly"];
        if b.as_array().is_none_or(|p| p.len() != 8) {
            return Err(invalid());
        }
        if score < 0.5
            || (["（204号", "（20", "（2", "（2号", "（20号", "号", "（204"].contains(&text)
                && score < 0.8
                && number(&b[4])? - number(&b[0])? < number(&b[5])? - number(&b[1])?)
        {
            item["category_id"] = json!(16);
        }
    }
    Ok(model)
}
