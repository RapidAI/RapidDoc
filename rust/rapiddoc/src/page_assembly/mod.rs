//! 旧 model_json_to_middle_json 的 OCR 已完成、无资产 writer 路径。
//! MagicModel、块预处理、span 过滤/归块、横竖排行和模型排序均沿用旧规则。
use rapiddoc_core::{Error, ErrorCode};
use serde_json::{json, Value};
pub mod assets;
pub(crate) mod budget;
/// Run a single full host page with bounded capacity; restore on error/unwind.
pub fn with_host_page_budget<T>(operation: impl FnOnce() -> T) -> T {
    let _scope = budget::host_page();
    operation()
}
pub use budget::{with_resources as with_host_resources, HostResources};
pub fn pdf_limits() -> rapiddoc_core::Limits {
    if budget::current().host {
        rapiddoc_core::Limits::pdf_host()
    } else {
        rapiddoc_core::Limits::default()
    }
}
pub fn host_json_bytes() -> usize {
    let limits = budget::current();
    if limits.host {
        limits.bytes
    } else {
        32_000_000
    }
}
mod formula;
pub mod inline;
mod magic;
pub(crate) mod native_text;
pub use formula::optimize as optimize_formula_numbers;
mod output;
pub(crate) use output::merge_caption as merge_table_caption;
pub use output::render;
type Prepared = (Vec<Value>, Vec<Value>, Vec<Value>);

pub(crate) fn invalid() -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: "旧页面组装输入结构或数值无效".into(),
    }
}
#[track_caller]
pub(crate) fn limit() -> Error {
    Error {
        code: ErrorCode::ResourceLimit,
        message: format!(
            "旧页面组装超过资源预算（{}:{}）",
            std::panic::Location::caller().file(),
            std::panic::Location::caller().line()
        ),
    }
}
pub(crate) fn unsupported(message: &str) -> Error {
    Error {
        code: ErrorCode::Unsupported,
        message: message.into(),
    }
}
fn num(v: &Value) -> f64 {
    v.as_f64().unwrap()
}
pub(crate) fn bb(v: &Value) -> [f64; 4] {
    [num(&v[0]), num(&v[1]), num(&v[2]), num(&v[3])]
}
fn area(b: [f64; 4]) -> f64 {
    (b[2] - b[0]) * (b[3] - b[1])
}
fn intersection(a: [f64; 4], b: [f64; 4]) -> f64 {
    let x = a[2].min(b[2]) - a[0].max(b[0]);
    let y = a[3].min(b[3]) - a[1].max(b[1]);
    if x < 0. || y < 0. {
        0.
    } else {
        x * y
    }
}
pub(crate) fn overlap(a: &Value, b: &Value) -> f64 {
    let a = bb(a);
    if area(a) == 0. {
        0.
    } else {
        intersection(a, bb(b)) / area(a)
    }
}
fn iou(a: &Value, b: &Value) -> f64 {
    let a = bb(a);
    let b = bb(b);
    let i = intersection(a, b);
    if area(a) == 0. || area(b) == 0. {
        0.
    } else {
        i / (area(a) + area(b) - i)
    }
}
fn small(a: &Value, b: &Value, threshold: f64) -> Option<bool> {
    let aa = area(bb(a));
    let ab = area(bb(b));
    let minimum = aa.min(ab);
    (minimum > 0. && intersection(bb(a), bb(b)) / minimum > threshold).then_some(aa <= ab)
}
pub(crate) fn truth(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(_) => num(v) != 0.,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}
// Python 的列表成员/字典相等比较按数值比较 int/float/bool；不可用 JSON 类型相等替代。
pub(crate) fn eq(a: &Value, b: &Value) -> bool {
    if (a.is_number() || a.is_boolean()) && (b.is_number() || b.is_boolean()) {
        let integer = |v: &Value| {
            v.as_bool()
                .map(i128::from)
                .or_else(|| v.as_i64().map(i128::from))
                .or_else(|| v.as_u64().map(i128::from))
        };
        let float_integer_equal = |i: i128, f: f64| {
            f.is_finite()
                && f.fract() == 0.
                && f >= i64::MIN as f64
                && f < 18_446_744_073_709_551_616.
                && f as i128 == i
        };
        return match (integer(a), integer(b)) {
            (Some(a), Some(b)) => a == b,
            (Some(a), None) => float_integer_equal(a, num(b)),
            (None, Some(b)) => float_integer_equal(b, num(a)),
            (None, None) => num(a) == num(b),
        };
    }
    match (a, b) {
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| eq(x, y))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len() && a.iter().all(|(k, v)| b.get(k).is_some_and(|w| eq(v, w)))
        }
        _ => a == b,
    }
}
fn contains(values: &[Value], v: &Value) -> bool {
    values.iter().any(|x| eq(x, v))
}
fn remove_values(values: &mut Vec<Value>, dropped: &[Value]) {
    for d in dropped {
        if let Some(i) = values.iter().position(|x| eq(x, d)) {
            values.remove(i);
        }
    }
}
fn union(a: &Value, b: &Value) -> Value {
    let mut result = a.as_array().unwrap().clone();
    for i in 0..4 {
        if (i < 2 && num(&b[i]) < num(&a[i])) || (i >= 2 && num(&b[i]) > num(&a[i])) {
            result[i] = b[i].clone();
        }
    }
    Value::Array(result)
}
pub(crate) fn coord(v: &Value, positive: bool) -> Result<(), Error> {
    let a = v.as_array().filter(|a| a.len() == 4).ok_or_else(invalid)?;
    for x in a {
        if !x
            .as_f64()
            .is_some_and(|x| x.is_finite() && x.abs() <= 1_000_000.)
        {
            return Err(invalid());
        }
    }
    if positive && (num(&a[2]) <= num(&a[0]) || num(&a[3]) <= num(&a[1])) {
        return Err(invalid());
    }
    Ok(())
}
pub(crate) struct Work(pub(crate) usize);
enum AssetStage {
    TableFill,
    Crop,
}
type AssetWriter<'a> = Option<&'a mut dyn FnMut(&mut Value, AssetStage) -> Result<(), Error>>;
impl Work {
    #[inline]
    pub(crate) fn charge(&mut self, n: usize) -> Result<(), Error> {
        self.0 = self
            .0
            .checked_add(n)
            .filter(|n| *n <= budget::current().work)
            .ok_or_else(|| Error {
                code: ErrorCode::ResourceLimit,
                message: format!(
                    "旧页面组装工作量超过资源预算：已用 {}，申请 {}，上限 {}",
                    self.0,
                    n,
                    budget::current().work
                ),
            })?;
        Ok(())
    }
}

fn add_boxes(values: &[Value], kind: &str, result: &mut Vec<Value>) {
    for b in values {
        let a = &b["bbox"];
        result.push(json!([
            a[0],
            a[1],
            a[2],
            a[3],
            null,
            null,
            null,
            kind,
            null,
            null,
            b["original_label"],
            b["original_order"],
            b["score"],
            if matches!(
                kind,
                "image_body" | "image_caption" | "table_body" | "table_caption" | "table_footnote"
            ) {
                b["group_id"].clone()
            } else {
                Value::Null
            },
            b["polygon_points"]
        ]));
    }
}
fn box4(b: &Value) -> Value {
    Value::Array(b.as_array().unwrap()[..4].to_vec())
}

fn prepare(
    categories: &Value,
    width: i64,
    height: i64,
    work: &mut Work,
) -> Result<Prepared, Error> {
    let mut blocks = Vec::new();
    for (key, kind) in [
        ("img_body_blocks", "image_body"),
        ("img_caption_blocks", "image_caption"),
        // 旧代码确实把 image_footnote 当作 image_caption；迁移不自行修正。
        ("img_footnote_blocks", "image_caption"),
        ("table_body_blocks", "table_body"),
        ("table_caption_blocks", "table_caption"),
        ("table_footnote_blocks", "table_footnote"),
        ("text_blocks", "text"),
        ("title_blocks", "title"),
        ("interline_equations", "interline_equation"),
        ("formula_number_blocks", "formula_number"),
    ] {
        add_boxes(categories[key].as_array().unwrap(), kind, &mut blocks);
    }
    let discarded = categories["discarded_blocks"].as_array().unwrap();
    let mut drop = Vec::new();
    for text in blocks.iter().filter(|b| b[7] == "text") {
        for title in blocks.iter().filter(|b| b[7] == "title") {
            work.charge(1)?;
            if iou(&box4(text), &box4(title)) > 0.8 && !contains(&drop, title) {
                drop.push(title.clone());
            }
        }
    }
    remove_values(&mut blocks, &drop);
    drop.clear();
    for block in &blocks {
        for d in discarded {
            work.charge(1)?;
            if overlap(&box4(block), &d["bbox"]) > 0.6 {
                if !contains(&drop, block) {
                    drop.push(block.clone());
                }
                break;
            }
        }
    }
    remove_values(&mut blocks, &drop);
    drop.clear();
    for equation in blocks.iter().filter(|b| b[7] == "interline_equation") {
        for text in blocks.iter().filter(|b| b[7] == "text") {
            work.charge(1)?;
            if iou(&box4(equation), &box4(text)) > 0.8 && !contains(&drop, text) {
                drop.push(text.clone());
            }
        }
    }
    remove_values(&mut blocks, &drop);
    let mut dropped = Vec::new();
    add_boxes(discarded, "discarded", &mut dropped);
    let footnotes: Vec<_> = discarded
        .iter()
        .filter(|d| {
            let b = bb(&d["bbox"]);
            b[2] - b[0] > width as f64 / 3. && b[3] - b[1] > 10. && b[1] > height as f64 * 0.7
        })
        .map(|b| b["bbox"].clone())
        .collect();
    // 旧条件只检查粗排序之前的第一个块。
    let model = blocks
        .first()
        .is_some_and(|b| b[11].as_f64().is_some_and(|n| n >= 0.));
    if !model {
        let mut remove = Vec::new();
        for b in &blocks {
            for f in &footnotes {
                work.charge(1)?;
                let a = bb(&box4(b));
                let f = bb(f);
                let width = a[2] - a[0];
                let ratio = if width > 0. {
                    (a[2].min(f[2]) - a[0].max(f[0])).max(0.) / width
                } else {
                    0.
                };
                if a[1] >= f[3] && ratio >= 0.8 {
                    if !contains(&remove, b) {
                        remove.push(b.clone());
                    }
                    break;
                }
            }
        }
        remove_values(&mut blocks, &remove);
        dropped.extend(remove);
        merge_contained(&mut blocks, work)?;
        merge_contained(&mut dropped, work)?;
    }
    blocks.sort_by(|a, b| {
        (num(&a[0]) + num(&a[1]))
            .partial_cmp(&(num(&b[0]) + num(&b[1])))
            .unwrap()
    });
    Ok((blocks, dropped, footnotes))
}
fn merge_contained(blocks: &mut Vec<Value>, work: &mut Work) -> Result<(), Error> {
    let mut indices = Vec::new();
    for i in 0..blocks.len() {
        for j in i + 1..blocks.len() {
            work.charge(1)?;
            if let Some(first) = small(&box4(&blocks[i]), &box4(&blocks[j]), 0.8) {
                let (s, l) = if first { (i, j) } else { (j, i) };
                if !indices.iter().any(|&k| eq(&blocks[k], &blocks[s])) {
                    let b = union(&box4(&blocks[l]), &box4(&blocks[s]));
                    for n in 0..4 {
                        blocks[l][n] = b[n].clone();
                    }
                    indices.push(s);
                }
            }
        }
    }
    let remove: Vec<_> = indices.iter().map(|&i| blocks[i].clone()).collect();
    remove_values(blocks, &remove);
    Ok(())
}
fn filter_spans(
    spans: &mut Vec<Value>,
    blocks: &[Value],
    discarded: &[Value],
    work: &mut Work,
) -> Result<(), Error> {
    let mut kept = Vec::new();
    for span in spans.iter() {
        let mut keep = false;
        for d in discarded {
            work.charge(1)?;
            if d[7] == "discarded" && overlap(&span["bbox"], &box4(d)) > 0.4 {
                keep = true;
                break;
            }
        }
        if !keep {
            for b in blocks {
                work.charge(1)?;
                let compatible = match span["type"].as_str().unwrap() {
                    "image" => b[7] == "image_body",
                    "table" => b[7] == "table_body",
                    _ => b[7] != "image_body" && b[7] != "table_body",
                };
                if compatible && overlap(&span["bbox"], &box4(b)) > 0.5 {
                    keep = true;
                    break;
                }
            }
        }
        if keep {
            kept.push(span.clone());
        }
    }
    *spans = kept;
    for mode in 0..2 {
        let mut dropped = Vec::new();
        for a in spans.iter() {
            for b in spans.iter() {
                work.charge(1)?;
                if eq(a, b) || contains(&dropped, a) || contains(&dropped, b) {
                    continue;
                }
                if mode == 0 {
                    if iou(&a["bbox"], &b["bbox"]) > 0.9 {
                        let d = if num(&a["score"]) < num(&b["score"]) {
                            a
                        } else {
                            b
                        };
                        if !contains(&dropped, d) {
                            dropped.push(d.clone());
                        }
                    }
                } else {
                    if ((a["type"] == "image" && b["type"] == "text")
                        || (a["type"] == "text" && b["type"] == "image"))
                        && ((a["type"] == "text" && truth(&a["native_text_region"]))
                            || (b["type"] == "text" && truth(&b["native_text_region"])))
                    {
                        continue;
                    }
                    if let Some(first) = small(&a["bbox"], &b["bbox"], 0.65) {
                        if a["original_label"] == "seal" || b["original_label"] == "seal" {
                            continue;
                        }
                        let box_to_drop = if first { &a["bbox"] } else { &b["bbox"] };
                        if let Some(d) = spans.iter().find(|s| eq(&s["bbox"], box_to_drop)) {
                            if d["original_label"] != "seal" && !contains(&dropped, d) {
                                dropped.push(d.clone());
                            }
                        }
                    }
                }
            }
        }
        remove_values(spans, &dropped);
    }
    Ok(())
}
fn compatible(span: &str, block: &str) -> bool {
    match span {
        "text" | "inline_equation" | "checkbox" => matches!(
            block,
            "text"
                | "title"
                | "image_caption"
                | "image_footnote"
                | "table_caption"
                | "table_footnote"
                | "formula_number"
                | "discarded"
        ),
        "interline_equation" => matches!(block, "text" | "interline_equation"),
        "image" => block == "image_body",
        "table" => block == "table_body",
        _ => false,
    }
}
fn fill(
    blocks: &[Value],
    spans: &mut Vec<Value>,
    ratio: f64,
    work: &mut Work,
) -> Result<Vec<Value>, Error> {
    let mut result = Vec::new();
    for b in blocks {
        let kind = b[7].as_str().unwrap();
        let bbox = box4(b);
        let mut chosen = Vec::new();
        for s in spans.iter() {
            work.charge(1)?;
            let span_kind = s["type"].as_str().unwrap();
            let threshold = if matches!(span_kind, "image" | "table") {
                0.9
            } else {
                ratio
            };
            if overlap(&s["bbox"], &bbox) > threshold && compatible(span_kind, kind) {
                chosen.push(s.clone());
            }
        }
        remove_values(spans, &chosen);
        let mut block = json!({"type":kind,"bbox":bbox,"spans":chosen,"original_label":b[10],"original_order":b[11],"polygon_points":b[14]});
        if matches!(
            kind,
            "image_body"
                | "image_caption"
                | "image_footnote"
                | "table_body"
                | "table_caption"
                | "table_footnote"
        ) {
            block["group_id"] = b[13].clone();
        }
        result.push(block);
    }
    Ok(result)
}
fn fix(mut block: Value, text: bool) -> Value {
    let mut spans = block
        .as_object_mut()
        .unwrap()
        .remove("spans")
        .unwrap()
        .as_array()
        .unwrap()
        .clone();
    if text {
        for s in &mut spans {
            if s["type"] == "interline_equation" {
                s["type"] = json!("inline_equation");
            }
        }
    }
    let vertical = text
        && !spans.is_empty()
        && (spans
            .iter()
            .filter(|s| {
                let b = bb(&s["bbox"]);
                (b[3] - b[1]) / (b[2] - b[0]) > 2.
            })
            .count() as f64
            / spans.len() as f64)
            > 0.8;
    spans.sort_by(|a, b| {
        if vertical {
            num(&b["bbox"][2]).partial_cmp(&num(&a["bbox"][2])).unwrap()
        } else {
            num(&a["bbox"][1]).partial_cmp(&num(&b["bbox"][1])).unwrap()
        }
    });
    let mut lines: Vec<Vec<Value>> = Vec::new();
    for span in spans {
        let special = |s: &Value| {
            matches!(
                s["type"].as_str().unwrap(),
                "interline_equation" | "image" | "table"
            )
        };
        let append = lines.last().is_some_and(|line| {
            if special(&span) || line.iter().any(special) {
                return false;
            }
            let a = bb(&span["bbox"]);
            let b = bb(&line.last().unwrap()["bbox"]);
            let axis = if vertical { 0 } else { 1 };
            let minimum = (a[axis + 2] - a[axis]).min(b[axis + 2] - b[axis]);
            minimum > 0.
                && (a[axis + 2].min(b[axis + 2]) - a[axis].max(b[axis])).max(0.) / minimum > 0.6
        });
        if append {
            lines.last_mut().unwrap().push(span);
        } else {
            lines.push(vec![span]);
        }
    }
    let lines: Vec<_> = lines
        .into_iter()
        .map(|mut line| {
            let axis = if vertical { 1 } else { 0 };
            line.sort_by(|a, b| {
                num(&a["bbox"][axis])
                    .partial_cmp(&num(&b["bbox"][axis]))
                    .unwrap()
            });
            let mut bbox = line[0]["bbox"].clone();
            for span in &line[1..] {
                bbox = union(&bbox, &span["bbox"]);
            }
            json!({"bbox":bbox,"spans":line})
        })
        .collect();
    block["lines"] = Value::Array(lines);
    block
}

/// 原始 layout_dets 与 scale 输入；OCR 文字必须已识别，资产 writer 为 None 的旧分支。
/// 不触发 OCR、原生字符填充、图片写出、公式编号后处理或跨页表格合并。
pub fn assemble(request: &Value) -> Result<Value, Error> {
    assemble_with_work(request, &mut Work(0), None)
}

fn assemble_with_work(
    request: &Value,
    work: &mut Work,
    writer: AssetWriter<'_>,
) -> Result<Value, Error> {
    assemble_with_native(request, work, writer, None)
}

type NativeFiller<'a> =
    Option<&'a mut dyn FnMut(&mut Vec<Value>, &[Value], &[Value], &mut Work) -> Result<(), Error>>;
fn assemble_with_native(
    request: &Value,
    work: &mut Work,
    mut writer: AssetWriter<'_>,
    mut native: NativeFiller<'_>,
) -> Result<Value, Error> {
    validate_tree(request)?;
    assemble_validated(request, work, &mut writer, &mut native)
}

pub(crate) fn validate_tree(request: &Value) -> Result<(), Error> {
    if unlimited_tree_budget() {
        return Ok(());
    }
    validate_tree_stack(vec![(request, 0usize)], 0, 0)
}
/// Validate the same virtual object as {"source": source, ..options}, borrowing
/// its values instead of serializing/cloning the complete source into a request.
pub(crate) fn validate_source_request(source: &Value, options: &Value) -> Result<(), Error> {
    if unlimited_tree_budget() {
        return Ok(());
    }
    let mut stack = vec![(source, 1usize)];
    let mut bytes = "source".len();
    for (key, value) in options.as_object().ok_or_else(invalid)? {
        if key != "source" {
            bytes += key.len();
            stack.push((value, 1));
        }
    }
    validate_tree_stack(stack, 1, bytes)
}
fn unlimited_tree_budget() -> bool {
    let limits = budget::current();
    limits.nodes == usize::MAX
        && limits.bytes == usize::MAX
        && rapiddoc_core::resources::limit(64usize) == usize::MAX
}
fn validate_tree_stack(
    mut stack: Vec<(&Value, usize)>,
    mut nodes: usize,
    mut bytes: usize,
) -> Result<(), Error> {
    let limits = budget::current();
    if limits.nodes == usize::MAX
        && limits.bytes == usize::MAX
        && rapiddoc_core::resources::limit(64usize) == usize::MAX
    {
        // This pass only counts budgeted depth/nodes/bytes; shape and numeric
        // validation belongs to the individual operations below. With no tree
        // budget, walking every character again cannot reject any input.
        return Ok(());
    }
    while let Some((v, depth)) = stack.pop() {
        nodes += 1;
        if depth > rapiddoc_core::resources::limit(64) || nodes > budget::current().nodes {
            return Err(limit());
        }
        match v {
            Value::Array(a) => {
                if nodes + stack.len() + a.len() > budget::current().nodes {
                    return Err(limit());
                }
                stack.extend(a.iter().map(|v| (v, depth + 1)));
            }
            Value::Object(o) => {
                bytes += o.keys().map(String::len).sum::<usize>();
                stack.extend(o.values().map(|v| (v, depth + 1)));
            }
            Value::String(s) => bytes += s.len(),
            _ => {}
        }
        if bytes > budget::current().bytes {
            return Err(limit());
        }
    }
    Ok(())
}

fn assemble_validated(
    request: &Value,
    work: &mut Work,
    writer: &mut AssetWriter<'_>,
    native: &mut NativeFiller<'_>,
) -> Result<Value, Error> {
    let width = request["page_width"]
        .as_f64()
        .filter(|n| n.is_finite() && *n >= 1. && *n <= 1_000_000.)
        .ok_or_else(invalid)? as i64;
    let height = request["page_height"]
        .as_f64()
        .filter(|n| n.is_finite() && *n >= 1. && *n <= 1_000_000.)
        .ok_or_else(invalid)? as i64;
    let page_idx = request["page_index"].as_u64().ok_or_else(invalid)?;
    let scale = request["scale"]
        .as_f64()
        .filter(|n| n.is_finite() && *n > 0. && *n <= 1000.)
        .ok_or_else(invalid)?;
    let formula = request["formula_enabled"].as_bool().ok_or_else(invalid)?;
    let dets = request["model"]["layout_dets"]
        .as_array()
        .ok_or_else(invalid)?;
    if dets.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    for d in dets {
        if !d.is_object()
            || d["category_id"].as_i64().is_none()
            || !d["score"].as_f64().is_some_and(|n| n.is_finite())
        {
            return Err(invalid());
        }
        let poly = d["poly"]
            .as_array()
            .filter(|p| p.len() == 8)
            .ok_or_else(invalid)?;
        for v in poly {
            if !v
                .as_f64()
                .is_some_and(|n| n.is_finite() && n.abs() <= 1_000_000.)
            {
                return Err(invalid());
            }
        }
        if truth(&d["vl_ocr"]) {
            return Err(unsupported("VL OCR 不在 Rust 旧页面组装范围内"));
        }
        if let Some(order) = d.get("original_order").filter(|v| !v.is_null()) {
            if !order
                .as_f64()
                .is_some_and(|n| n.is_finite() && n.abs() <= 9_007_199_254_740_991.)
            {
                return Err(invalid());
            }
        }
        if d["category_id"] == 15 && !d["text"].is_string() {
            return Err(invalid());
        }
        if let Some(polygon) = d.get("polygon_points").filter(|p| !p.is_null()) {
            for point in polygon.as_array().ok_or_else(invalid)? {
                let p = point
                    .as_array()
                    .filter(|p| p.len() == 2)
                    .ok_or_else(invalid)?;
                for c in p {
                    if !c
                        .as_f64()
                        .is_some_and(|n| n.is_finite() && n.abs() <= 1_000_000.)
                    {
                        return Err(invalid());
                    }
                }
            }
        }
        for key in ["html", "latex", "checkbox"] {
            if let Some(v) = d.get(key).filter(|v| !v.is_null()) {
                if !v.is_string() {
                    return Err(invalid());
                }
            }
        }
        if writer.is_none()
            && d["html"]
                .as_str()
                .is_some_and(|s| s.contains("data:image/"))
        {
            return Err(unsupported("内联图片资产须先完成 writer 处理"));
        }
    }
    let (categories, mut spans) = magic::extract(dets, scale, formula, work, writer)?;
    let (blocks, discarded, footnotes) = prepare(&categories, width, height, work)?;
    filter_spans(&mut spans, &blocks, &discarded, work)?;
    if let Some(fill) = native.as_mut() {
        fill(&mut spans, &blocks, &discarded, work)?;
    }
    let discarded = fill(&discarded, &mut spans, 0.4, work)?
        .into_iter()
        .map(|b| fix(b, true))
        .collect::<Vec<_>>();
    if blocks.is_empty() && discarded.is_empty() {
        return Ok(Value::Null);
    }
    // 与 image_writer=None 的旧 cut_image_and_table 一致；不伪造图片已写出的路径。
    for span in &mut spans {
        if matches!(
            span["type"].as_str().unwrap(),
            "image" | "table" | "interline_equation"
        ) {
            if let Some(writer) = writer.as_mut() {
                writer(span, AssetStage::Crop)?;
            } else {
                span["image_path"] = json!("");
            }
        }
    }
    let fixed = fill(&blocks, &mut spans, 0.5, work)?
        .into_iter()
        .filter_map(|b| match b["type"].as_str().unwrap() {
            "text" | "title" | "image_caption" | "table_caption" | "table_footnote"
            | "formula_number" => Some(fix(b, true)),
            "image_body" | "table_body" | "interline_equation" => Some(fix(b, false)),
            _ => None,
        })
        .collect::<Vec<_>>();
    // 纯丢弃页不需要 enhanced：旧版空主块列表排序返回空数组。
    let sorted = if fixed.is_empty() {
        json!([])
    } else {
        crate::legacy_block_sort::sort(
            &json!({"blocks":fixed,"page_width":width,"page_height":height,"footnote_blocks":footnotes}),
        )?
    };
    Ok(
        json!({"preproc_blocks":sorted,"page_idx":page_idx,"page_size":[width,height],"discarded_blocks":discarded}),
    )
}

/// 多页仅组合已验证的单页组装和 para_split；不冒充 result_to_middle_json 的全部后处理。
pub fn assemble_many(requests: &Value) -> Result<Value, Error> {
    assemble_many_inner(requests, false)
}

pub fn assemble_many_with_formulas(requests: &Value) -> Result<Value, Error> {
    assemble_many_inner(requests, true)
}

fn assemble_many_inner(requests: &Value, optimize_formulas: bool) -> Result<Value, Error> {
    let requests = requests.as_array().ok_or_else(invalid)?;
    if requests.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let mut pages = Vec::with_capacity(requests.len());
    let mut work = Work(0);
    for request in requests {
        let mut page = assemble_with_work(request, &mut work, None)?;
        if page.is_null() {
            page = json!({"preproc_blocks":[],"page_idx":request["page_index"],
                "page_size":[num(&request["page_width"]) as i64,num(&request["page_height"]) as i64],"discarded_blocks":[]});
        }
        pages.push(page);
    }
    let pages = Value::Array(pages);
    let pages = if optimize_formulas {
        optimize_formula_numbers(&pages)?
    } else {
        pages
    };
    crate::legacy_paragraphs::split(&pages)
}
