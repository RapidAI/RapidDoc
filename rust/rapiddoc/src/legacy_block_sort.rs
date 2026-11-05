//! 旧 block_sort.py 的模型顺序分支，含虚拟行和图表分组；无顺序时明确拒绝。
use rapiddoc_core::{Error, ErrorCode};
use serde_json::{json, Value};

fn error(code: ErrorCode, message: &str) -> Error {
    Error {
        code,
        message: message.into(),
    }
}
fn invalid() -> Error {
    error(ErrorCode::InvalidInput, "旧块排序输入结构或数值无效")
}
fn limit() -> Error {
    error(ErrorCode::ResourceLimit, "旧块排序超过资源预算")
}
fn number(v: &Value) -> Result<f64, Error> {
    v.as_f64()
        .filter(|n| n.is_finite() && n.abs() <= 1_000_000.)
        .ok_or_else(invalid)
}
fn bounds(v: &Value) -> Result<[f64; 4], Error> {
    let a = v.as_array().filter(|a| a.len() == 4).ok_or_else(invalid)?;
    let b = [
        number(&a[0])?,
        number(&a[1])?,
        number(&a[2])?,
        number(&a[3])?,
    ];
    if b[2] <= b[0] || b[3] <= b[1] {
        return Err(invalid());
    }
    Ok(b)
}
fn text(kind: &str) -> bool {
    matches!(
        kind,
        "text" | "title" | "image_caption" | "image_footnote" | "table_caption" | "table_footnote"
    )
}
fn body(kind: &str) -> bool {
    matches!(kind, "image_body" | "table_body" | "interline_equation")
}
fn numeric(v: &Value) -> f64 {
    v.as_bool()
        .map(|b| if b { 1. } else { 0. })
        .or_else(|| v.as_f64())
        .unwrap()
}
fn order(v: Option<&Value>) -> Result<Option<Value>, Error> {
    let Some(v) = v.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    if !v.is_boolean() && !v.is_number() {
        return Err(invalid());
    }
    let n = numeric(v);
    // 保持比较精确；超出可精确表示范围的 Python 整数不属于当前合同。
    if !n.is_finite() || n.abs() > 9_007_199_254_740_991. {
        return Err(invalid());
    }
    Ok((n >= 0.).then(|| v.clone()))
}
fn original(block: &Value) -> Result<Option<Value>, Error> {
    if let Some(o) = order(block.get("original_order"))? {
        return Ok(Some(o));
    }
    let mut best: Option<Value> = None;
    for line in block["lines"].as_array().ok_or_else(invalid)? {
        if let Some(spans) = line.get("spans") {
            for span in spans.as_array().ok_or_else(invalid)? {
                if !span.is_object() {
                    return Err(invalid());
                }
                if let Some(o) = order(span.get("original_order"))? {
                    if best.as_ref().is_none_or(|b| numeric(&o) < numeric(b)) {
                        best = Some(o);
                    }
                }
            }
        }
    }
    Ok(best)
}
fn median(mut values: Vec<Value>) -> Value {
    values.sort_by(|a, b| numeric(a).partial_cmp(&numeric(b)).unwrap());
    let n = values.len();
    if n % 2 == 1 {
        values[n / 2].clone()
    } else {
        json!((numeric(&values[n / 2 - 1]) + numeric(&values[n / 2])) / 2.)
    }
}
fn virtual_lines(
    bbox: &Value,
    height: f64,
    width: f64,
    page_height: f64,
    budget: &mut usize,
) -> Result<Value, Error> {
    let b = bounds(bbox)?;
    let h = b[3] - b[1];
    let w = b[2] - b[0];
    let count = if height * 2. < h {
        if (h > page_height * 0.25 && width * 0.5 > w && w > width * 0.25)
            || (w <= width * 0.4 && w > width * 0.25)
        {
            (h / height) as usize
        } else if w > width * 0.4 {
            3
        } else if h / w > 1.2 {
            1
        } else {
            2
        }
    } else {
        1
    };
    if count == 0 {
        return Err(invalid());
    }
    *budget = budget
        .checked_add(count)
        .filter(|n| *n <= rapiddoc_core::resources::limit(20_000))
        .ok_or_else(limit)?;
    if count == 1 {
        return Ok(json!([{"bbox":bbox,"spans":[]}]));
    }
    let coords = bbox.as_array().unwrap();
    let step = h / count as f64;
    let mut y = coords[1].clone();
    let mut result = Vec::with_capacity(count);
    for _ in 0..count {
        let end = y.as_f64().unwrap() + step;
        result.push(json!({"bbox":[coords[0],y,coords[2],end],"spans":[]}));
        y = json!(end);
    }
    Ok(Value::Array(result))
}
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(_) => numeric(v) != 0.,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}
fn same_key(a: &Value, b: &Value) -> bool {
    if (a.is_number() || a.is_boolean()) && (b.is_number() || b.is_boolean()) {
        numeric(a) == numeric(b)
    } else {
        a == b
    }
}
fn group(blocks: Vec<Value>) -> Result<Vec<Value>, Error> {
    let mut plain = Vec::new();
    let mut images: Vec<(Value, Vec<Value>)> = Vec::new();
    let mut tables: Vec<(Value, Vec<Value>)> = Vec::new();
    for b in blocks {
        let groups = match b["type"].as_str().unwrap() {
            "image_body" | "image_caption" | "image_footnote" => &mut images,
            "table_body" | "table_caption" | "table_footnote" => &mut tables,
            _ => {
                plain.push(b);
                continue;
            }
        };
        let key = b.get("group_id").ok_or_else(invalid)?;
        if key.is_array() || key.is_object() {
            return Err(invalid());
        }
        if key.is_number() {
            order(Some(key))?;
        }
        if let Some((_, list)) = groups.iter_mut().find(|(k, _)| same_key(k, key)) {
            list.push(b);
        } else {
            groups.push((key.clone(), vec![b]));
        }
    }
    for (groups, kind, body_kind) in [
        (images, "image", "image_body"),
        (tables, "table", "table_body"),
    ] {
        for (_, mut children) in groups {
            let index = median(children.iter().map(|b| b["index"].clone()).collect());
            let body = children.iter().find(|b| b["type"] == body_kind);
            let bbox = body.map(|b| b["bbox"].clone()).unwrap_or_else(|| json!([]));
            let polygon = body
                .and_then(|b| b.get("polygon_points"))
                .filter(|p| truthy(p))
                .cloned();
            children.sort_by(|a, b| {
                numeric(&a["index"])
                    .partial_cmp(&numeric(&b["index"]))
                    .unwrap()
            });
            let mut parent = json!({"type":kind,"bbox":bbox,"blocks":children,"index":index});
            if let Some(polygon) = polygon {
                parent["polygon_points"] = polygon;
            }
            plain.push(parent);
        }
    }
    plain.sort_by(|a, b| {
        numeric(&a["index"])
            .partial_cmp(&numeric(&b["index"]))
            .unwrap()
    });
    Ok(plain)
}

/// 输入 {blocks,page_width,page_height,footnote_blocks}；输出旧 sort_blocks_by_bbox 的块数组副本。
/// 仅迁移有 original_order 的分支，不接受无模型顺序的 enhanced/XY-cut 路径。
pub fn sort(request: &Value) -> Result<Value, Error> {
    let mut nodes = 0;
    let mut bytes = 0;
    let mut stack = vec![(request, 0usize)];
    while let Some((v, depth)) = stack.pop() {
        if depth > rapiddoc_core::resources::limit(64) {
            return Err(limit());
        }
        nodes += 1;
        match v {
            Value::Array(a) => stack.extend(a.iter().map(|v| (v, depth + 1))),
            Value::Object(o) => {
                bytes += o.keys().map(String::len).sum::<usize>();
                stack.extend(o.values().map(|v| (v, depth + 1)));
            }
            Value::String(s) => bytes += s.len(),
            _ => {}
        }
        if nodes > rapiddoc_core::resources::limit(200_000)
            || bytes > rapiddoc_core::resources::limit(2_000_000)
        {
            return Err(limit());
        }
    }
    let width = number(&request["page_width"])?;
    let page_height = number(&request["page_height"])?;
    if width <= 0. || page_height <= 0. {
        return Err(invalid());
    }
    let source = request["blocks"].as_array().ok_or_else(invalid)?;
    if source.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let mut heights = Vec::new();
    let mut line_count = 0;
    for b in source {
        let kind = b["type"].as_str().ok_or_else(invalid)?;
        bounds(&b["bbox"])?;
        let lines = b["lines"].as_array().ok_or_else(invalid)?;
        if let Some(real) = b.get("real_lines") {
            let real = real.as_array().ok_or_else(invalid)?;
            line_count += real.len();
            for line in real {
                bounds(&line["bbox"])?;
                if !line.is_object() {
                    return Err(invalid());
                }
            }
        }
        line_count += lines.len();
        if line_count > rapiddoc_core::resources::limit(10_000) {
            return Err(limit());
        }
        for l in lines {
            if !l.is_object() {
                return Err(invalid());
            }
            let bb = bounds(&l["bbox"])?;
            if let Some(spans) = l.get("spans") {
                if !spans.is_array() {
                    return Err(invalid());
                }
            }
            if text(kind) {
                heights.push(json!((bb[3] - bb[1]) as i64));
            }
        }
    }
    let height = if heights.is_empty() {
        10.
    } else {
        numeric(&median(heights))
    };
    if height <= 0. {
        return Err(invalid());
    }
    let mut blocks = source.clone();
    let mut budget = line_count;
    for b in &mut blocks {
        let kind = b["type"].as_str().unwrap();
        let count = b["lines"].as_array().unwrap().len();
        let restore = body(kind)
            || (kind == "title"
                && count == 1
                && bounds(&b["bbox"])?[3] - bounds(&b["bbox"])?[1] > height * 2.);
        let add = restore || (text(kind) && count == 0);
        if restore {
            b["real_lines"] = b["lines"].clone();
        }
        if add {
            b["lines"] = virtual_lines(&b["bbox"], height, width, page_height, &mut budget)?;
        }
    }
    for footnote in request["footnote_blocks"].as_array().ok_or_else(invalid)? {
        let values = footnote
            .as_array()
            .filter(|a| a.len() >= 4)
            .ok_or_else(invalid)?;
        virtual_lines(
            &Value::Array(values[..4].to_vec()),
            height,
            width,
            page_height,
            &mut budget,
        )?;
    }
    let mut orders = Vec::new();
    for b in &mut blocks {
        for x in b["bbox"].as_array_mut().unwrap() {
            if x.as_f64().unwrap() <= 0. {
                *x = json!(0);
            }
        }
        if body(b["type"].as_str().unwrap()) || b["type"] == "title" {
            if let Some(real) = b.as_object_mut().unwrap().remove("real_lines") {
                if !real.is_array() {
                    return Err(invalid());
                }
                b["virtual_lines"] = b["lines"].clone();
                b["lines"] = real;
            }
        }
        orders.push(original(b)?);
    }
    if !orders.iter().any(Option::is_some) {
        return Err(error(
            ErrorCode::Unsupported,
            "缺少 original_order；旧 enhanced/XY-cut 阅读顺序尚未移植",
        ));
    }
    let count = blocks.len();
    for (b, o) in blocks.iter_mut().zip(orders) {
        b["index"] = o.unwrap_or_else(|| json!(count));
    }
    let mut indices: Vec<_> = (0..count).collect();
    indices.sort_by(|a, b| {
        numeric(&blocks[*a]["index"])
            .partial_cmp(&numeric(&blocks[*b]["index"]))
            .unwrap()
    });
    let mut line_index = 1;
    for i in indices {
        for line in blocks[i]["lines"].as_array_mut().ok_or_else(invalid)? {
            if !line.is_object() {
                return Err(invalid());
            }
            line["index"] = json!(line_index);
            line_index += 1;
        }
    }
    Ok(Value::Array(group(blocks)?))
}
