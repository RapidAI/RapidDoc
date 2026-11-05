//! 按旧 pipeline_magic_model.py / magic_model_utils.py 提取页面块与 span。
use super::*;

fn base(v: &Value) -> Value {
    json!({"bbox":v["bbox"],"score":v["score"],"original_label":v["original_label"],"original_order":v["original_order"],"polygon_points":v["polygon_points"]})
}
fn cat(v: &Value) -> i64 {
    v["category_id"].as_i64().unwrap()
}
fn inside(a: &Value, b: &Value) -> bool {
    let a = bb(a);
    let b = bb(b);
    a[0] >= b[0] && a[1] >= b[1] && a[2] <= b[2] && a[3] <= b[3]
}
fn distance(a: &Value, b: &Value) -> f64 {
    let a = bb(a);
    let b = bb(b);
    let left = b[2] < a[0];
    let right = a[2] < b[0];
    let bottom = b[3] < a[1];
    let top = a[3] < b[1];
    let dist = |x: f64, y: f64| (x * x + y * y).sqrt();
    if top && left {
        dist(a[0] - b[2], a[3] - b[1])
    } else if left && bottom {
        dist(a[0] - b[2], a[1] - b[3])
    } else if bottom && right {
        dist(a[2] - b[0], a[1] - b[3])
    } else if right && top {
        dist(a[2] - b[0], a[3] - b[1])
    } else if left {
        a[0] - b[2]
    } else if right {
        b[0] - a[2]
    } else if bottom {
        a[1] - b[3]
    } else if top {
        b[1] - a[3]
    } else {
        0.
    }
}
fn footnote_distance(a: &Value, b: &Value) -> f64 {
    let aa = bb(a);
    let ab = bb(b);
    let horizontal = ab[2] < aa[0] || aa[2] < ab[0];
    let vertical = ab[3] < aa[1] || aa[3] < ab[1];
    if horizontal && vertical {
        return f64::INFINITY;
    }
    let axis = if horizontal { 1 } else { 0 };
    let l1 = aa[axis + 2] - aa[axis];
    let l2 = ab[axis + 2] - ab[axis];
    if l2 > l1 && (l2 - l1) / l1 > 0.3 {
        return f64::INFINITY;
    }
    distance(a, b)
}
fn remove_marked(data: &mut Vec<Value>, marked: &[usize]) {
    let remove: Vec<_> = marked.iter().map(|&i| data[i].clone()).collect();
    remove_values(data, &remove);
}
fn normalize(input: &[Value], scale: f64, work: &mut Work) -> Result<Vec<Value>, Error> {
    let mut data = Vec::new();
    for source in input {
        let mut d = source.clone();
        let p = &d["poly"];
        let bbox = json!([
            ((num(&p[0]) / scale * 100.).floor()) / 100.,
            ((num(&p[1]) / scale * 100.).floor()) / 100.,
            ((num(&p[4]) / scale * 100.).floor()) / 100.,
            ((num(&p[5]) / scale * 100.).floor()) / 100.
        ]);
        coord(&bbox, false)?;
        d["bbox"] = bbox;
        if let Some(points) = d
            .get("polygon_points")
            .and_then(Value::as_array)
            .filter(|p| p.len() >= 3)
        {
            let points: Vec<_> = points
                .iter()
                .map(|p| {
                    json!([
                        format!("{:.2}", num(&p[0]) / scale).parse::<f64>().unwrap(),
                        format!("{:.2}", num(&p[1]) / scale).parse::<f64>().unwrap()
                    ])
                })
                .collect();
            d["polygon_points"] = Value::Array(points);
        }
        let b = bb(&d["bbox"]);
        if b[2] > b[0] && b[3] > b[1] && num(&d["score"]) > 0.05 {
            data.push(d);
        }
    }
    let relevant = |v: &Value| matches!(cat(v), 0 | 1 | 3 | 4 | 5 | 6 | 7 | 8 | 9);
    let mut marked = Vec::new();
    for i in 0..data.len() {
        for j in i + 1..data.len() {
            work.charge(1)?;
            if relevant(&data[i])
                && relevant(&data[j])
                && iou(&data[i]["bbox"], &data[j]["bbox"]) > 0.9
            {
                let k = if num(&data[i]["score"]) < num(&data[j]["score"]) {
                    i
                } else {
                    j
                };
                if !marked.iter().any(|&m| eq(&data[m], &data[k])) {
                    marked.push(k);
                }
            }
        }
    }
    remove_marked(&mut data, &marked);
    let figures: Vec<_> = data
        .iter()
        .filter(|d| cat(d) == 3)
        .map(|d| d["bbox"].clone())
        .collect();
    let tables: Vec<_> = data
        .iter()
        .filter(|d| cat(d) == 5)
        .map(|d| d["bbox"].clone())
        .collect();
    for d in data.iter_mut().filter(|d| cat(d) == 7) {
        if figures.is_empty() {
            break;
        }
        let mut fig = f64::INFINITY;
        let mut table = f64::INFINITY;
        for bbox in &figures {
            work.charge(1)?;
            fig = fig.min(footnote_distance(bbox, &d["bbox"]));
        }
        for bbox in &tables {
            work.charge(1)?;
            table = table.min(footnote_distance(bbox, &d["bbox"]));
        }
        if table > fig {
            d["category_id"] = json!(101);
        }
    }
    marked.clear();
    for kind in [3, 5] {
        let indices: Vec<_> = (0..data.len()).filter(|&i| cat(&data[i]) == kind).collect();
        for (offset, &i) in indices.iter().enumerate() {
            for &j in &indices[offset + 1..] {
                work.charge(1)?;
                if let Some(first) = small(&data[i]["bbox"], &data[j]["bbox"], 0.8) {
                    let (s, l) = if first { (i, j) } else { (j, i) };
                    if !marked.iter().any(|&k| eq(&data[k], &data[s])) {
                        data[l]["bbox"] = union(&data[l]["bbox"], &data[s]["bbox"]);
                        marked.push(s);
                    }
                }
            }
        }
    }
    remove_marked(&mut data, &marked);
    Ok(data)
}
fn reduce(mut values: Vec<Value>, work: &mut Work) -> Result<Vec<Value>, Error> {
    let mut keep = vec![true; values.len()];
    for i in 0..values.len() {
        for j in 0..values.len() {
            work.charge(1)?;
            if i != j && inside(&values[i]["bbox"], &values[j]["bbox"]) {
                keep[i] = false;
            }
        }
    }
    let mut i = 0;
    values.retain(|_| {
        let yes = keep[i];
        i += 1;
        yes
    });
    Ok(values)
}
fn pair(
    mut subjects: Vec<Value>,
    mut objects: Vec<Value>,
    work: &mut Work,
) -> Result<Vec<(usize, Value, Vec<Value>)>, Error> {
    let key = |a: &Value| num(&a["bbox"][0]).powi(2) + num(&a["bbox"][1]).powi(2);
    subjects.sort_by(|a, b| key(a).partial_cmp(&key(b)).unwrap());
    objects.sort_by(|a, b| key(a).partial_cmp(&key(b)).unwrap());
    let n = subjects.len();
    let mut seen = vec![false; n + objects.len()];
    let mut seen_sub = vec![false; n];
    let mut result: Vec<(usize, Value, Vec<Value>)> = Vec::new();
    while seen_sub.iter().any(|b| !*b) {
        let mut candidates: Vec<_> = (0..seen.len()).filter(|&i| !seen[i]).collect();
        if candidates.is_empty() {
            break;
        }
        let object = |i: usize| if i < n { &subjects[i] } else { &objects[i - n] };
        let left = candidates
            .iter()
            .map(|&i| num(&object(i)["bbox"][0]))
            .fold(f64::INFINITY, f64::min);
        let top = candidates
            .iter()
            .map(|&i| num(&object(i)["bbox"][1]))
            .fold(f64::INFINITY, f64::min);
        let origin = |i: usize| {
            (num(&object(i)["bbox"][0]) - left).powi(2) + (num(&object(i)["bbox"][1]) - top).powi(2)
        };
        candidates.sort_by(|&a, &b| origin(a).partial_cmp(&origin(b)).unwrap());
        let first = candidates[0];
        let bbox = &object(first)["bbox"];
        work.charge(candidates.len() * candidates.len().max(1))?;
        candidates.sort_by(|&a, &b| {
            distance(bbox, &object(a)["bbox"])
                .partial_cmp(&distance(bbox, &object(b)["bbox"]))
                .unwrap()
        });
        let Some(next) = candidates
            .iter()
            .skip(1)
            .find(|&&i| (i < n) != (first < n))
            .copied()
        else {
            break;
        };
        let (sub, obj) = if first < n {
            (first, next - n)
        } else {
            (next, first - n)
        };
        work.charge(n + 1)?;
        let pair_distance = distance(&subjects[sub]["bbox"], &objects[obj]["bbox"]);
        let nearest = subjects
            .iter()
            .map(|s| distance(&s["bbox"], &objects[obj]["bbox"]))
            .fold(f64::INFINITY, f64::min);
        if pair_distance >= 3. * nearest {
            seen[sub] = true;
            continue;
        }
        seen[sub] = true;
        seen[obj + n] = true;
        seen_sub[sub] = true;
        result.push((sub, subjects[sub].clone(), vec![objects[obj].clone()]));
    }
    for i in 0..objects.len() {
        if seen[i + n] {
            continue;
        }
        seen[i + n] = true;
        let mut nearest = f64::INFINITY;
        let mut nearest_sub = None;
        for (k, s) in subjects.iter().enumerate() {
            work.charge(1)?;
            let d = distance(&objects[i]["bbox"], &s["bbox"]);
            if d < nearest {
                nearest = d;
                nearest_sub = Some(k);
            }
        }
        if let Some(k) = nearest_sub {
            if seen_sub[k] {
                if let Some(r) = result.iter_mut().find(|r| r.0 == k) {
                    r.2.push(objects[i].clone());
                }
            } else {
                result.push((k, subjects[k].clone(), vec![objects[i].clone()]));
            }
            seen_sub[k] = true;
            seen[k] = true;
        }
    }
    for i in 0..n {
        if !seen_sub[i] {
            result.push((i, subjects[i].clone(), Vec::new()));
        }
    }
    Ok(result)
}
pub(super) fn extract(
    input: &[Value],
    scale: f64,
    formula: bool,
    work: &mut Work,
    writer: &mut AssetWriter<'_>,
) -> Result<(Value, Vec<Value>), Error> {
    let mut data = normalize(input, scale, work)?;
    if let Some(writer) = writer.as_mut() {
        for det in &mut data {
            writer(det, AssetStage::TableFill)?;
        }
    }
    let get = |kind: i64| {
        data.iter()
            .filter(|d| cat(d) == kind)
            .map(base)
            .collect::<Vec<_>>()
    };
    let mut categories = json!({"discarded_blocks":get(2),"text_blocks":get(1),"title_blocks":get(0),"formula_number_blocks":get(9)});
    for (subject, caption, footnote, prefix) in [(3, 4, 101, "img"), (5, 6, 7, "table")] {
        let captions = pair(
            reduce(get(subject), work)?,
            reduce(get(caption), work)?,
            work,
        )?;
        let footnotes = pair(
            reduce(get(subject), work)?,
            reduce(get(footnote), work)?,
            work,
        )?;
        let mut bodies = Vec::new();
        let mut caps = Vec::new();
        let mut feet = Vec::new();
        let mut maybe = Vec::new();
        for (i, (index, mut body, mut child_caps)) in captions.into_iter().enumerate() {
            let mut child_feet = footnotes
                .iter()
                .find(|r| r.0 == index)
                .ok_or_else(invalid)?
                .2
                .clone();
            body["group_id"] = json!(i);
            if subject == 3 && child_caps.is_empty() && child_feet.is_empty() {
                maybe.push(body);
                continue;
            }
            bodies.push(body);
            for c in &mut child_caps {
                c["group_id"] = json!(i);
            }
            for c in &mut child_feet {
                c["group_id"] = json!(i);
            }
            caps.extend(child_caps);
            feet.extend(child_feet);
        }
        bodies.extend(maybe);
        categories[format!("{prefix}_body_blocks")] = json!(bodies);
        categories[format!("{prefix}_caption_blocks")] = json!(caps);
        categories[format!("{prefix}_footnote_blocks")] = json!(feet);
    }
    let mut equations = get(14);
    if !formula && !get(8).is_empty() {
        equations = get(8);
    }
    categories["interline_equations"] = json!(equations);
    let native: Vec<_> = data
        .iter()
        .filter(|d| truth(&d["native_text_region"]) && !d["original_order"].is_null())
        .collect();
    let mut spans = Vec::new();
    for d in &data {
        if !matches!(cat(d), 3 | 5 | 13 | 14 | 15 | 200) {
            continue;
        }
        let mut s = base(d);
        match cat(d) {
            3 => {
                s["type"] = json!("image");
                if d["original_label"] == "seal" {
                    s["content"] = d["text"].clone();
                }
            }
            5 => {
                s["type"] = json!("table");
                if truth(&d["latex"]) {
                    s["latex"] = d["latex"].clone();
                } else if truth(&d["html"]) {
                    s["html"] = d["html"].clone();
                    if d["text_fallback"] == true {
                        s["text_fallback"] = json!(true);
                    }
                    if truth(&d["latex_boxes"]) {
                        s["latex_boxes"] = d["latex_boxes"].clone();
                    } else if truth(&d["img_boxes"]) {
                        s["img_boxes"] = d["img_boxes"].clone();
                    }
                }
            }
            13 | 14 => {
                s["type"] = json!(if cat(d) == 13 {
                    "inline_equation"
                } else {
                    "interline_equation"
                });
                s["content"] = if truth(&d["latex"]) {
                    d["latex"].clone()
                } else {
                    json!("")
                };
            }
            200 => {
                s["type"] = json!("checkbox");
                s["content"] = if truth(&d["checkbox"]) {
                    d["checkbox"].clone()
                } else {
                    json!("")
                };
            }
            15 => {
                s["type"] = json!("text");
                s["content"] = d["text"].clone();
                // 字典赋值后，同 original_order 的最后一个 native owner 获胜。
                if let Some(owner) = native
                    .iter()
                    .rev()
                    .find(|n| eq(&n["original_order"], &d["original_order"]))
                {
                    s["native_text_region"] = json!(true);
                    let bbox = &owner["bbox"];
                    if overlap(&s["bbox"], bbox) <= 0.5 && overlap(bbox, &s["bbox"]) > 0.5 {
                        let a = bb(&s["bbox"]);
                        let b = bb(bbox);
                        s["bbox"] = json!([
                            a[0].max(b[0]),
                            a[1].max(b[1]),
                            a[2].min(b[2]),
                            a[3].min(b[3])
                        ]);
                    }
                }
                if truth(&d["docvortex_native_text"]) {
                    s["docvortex_native_text"] = json!(true);
                }
            }
            _ => unreachable!(),
        }
        if !contains(&spans, &s) {
            spans.push(s);
        }
    }
    if !formula && !get(8).is_empty() {
        for block in get(8) {
            spans.push(json!({"type":"interline_equation","score":block["score"],"bbox":block["bbox"],"content":""}));
        }
    }
    Ok((categories, spans))
}
