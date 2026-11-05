//! 逐规则移植 backend/pipeline/para_split.py；保留未知 JSON 字段，不使用实验几何段落器。
use rapiddoc_core::{Error, ErrorCode};
use rapiddoc_native::text_boundary::{python_digit, python_space, python_upper, uses_unspaced_script};
use serde_json::{Map, Value};
use std::collections::BTreeSet;

fn invalid(message: &str) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: message.into(),
    }
}
fn limit(message: &str) -> Error {
    Error {
        code: ErrorCode::ResourceLimit,
        message: message.into(),
    }
}
fn object(value: Value) -> Result<Map<String, Value>, Error> {
    if let Value::Object(map) = value {
        Ok(map)
    } else {
        Err(invalid("段落页、块、行和 span 必须是对象"))
    }
}
fn array(value: Value) -> Result<Vec<Value>, Error> {
    if let Value::Array(values) = value {
        Ok(values)
    } else {
        Err(invalid("段落 pages/blocks/lines/spans 必须是数组"))
    }
}

/// 校验几何但保留原数值的 JSON 类型；重算 bbox_fs 时只复制获胜坐标。
fn bbox(value: &Value) -> Result<([f64; 4], [Value; 4]), Error> {
    let values = value
        .as_array()
        .filter(|a| a.len() == 4)
        .ok_or_else(|| invalid("段落 bbox 必须有四个坐标"))?;
    let mut b = [0.; 4];
    for i in 0..4 {
        b[i] = values[i]
            .as_f64()
            .filter(|n| n.is_finite() && n.abs() <= 1_000_000.)
            .ok_or_else(|| invalid("段落坐标必须有限且在范围内"))?;
    }
    if b[2] <= b[0] || b[3] <= b[1] {
        return Err(invalid("段落 bbox 必须有正宽高"));
    }
    Ok((
        b,
        [
            values[0].clone(),
            values[1].clone(),
            values[2].clone(),
            values[3].clone(),
        ],
    ))
}

struct Line {
    data: Map<String, Value>,
    bounds: [f64; 4],
    coordinates: [Value; 4],
    spans: Vec<Value>,
}
impl Line {
    fn parse(value: Value) -> Result<Self, Error> {
        let mut data = object(value)?;
        let (bounds, coordinates) =
            bbox(data.get("bbox").ok_or_else(|| invalid("段落行缺少 bbox"))?)?;
        let spans = array(
            data.remove("spans")
                .ok_or_else(|| invalid("段落行缺少 spans"))?,
        )?;
        for s in &spans {
            if !s.is_object() || s.get("type").and_then(Value::as_str).is_none() {
                return Err(invalid("span 缺少 type"));
            }
            if s["type"] == "text" && !s.get("content").is_some_and(Value::is_string) {
                return Err(invalid("text span 必须包含字符串 content"));
            }
        }
        Ok(Self {
            data,
            bounds,
            coordinates,
            spans,
        })
    }
    fn start(&mut self) {
        self.data
            .insert("is_list_start_line".into(), Value::Bool(true));
    }
    fn end(&mut self) {
        self.data
            .insert("is_list_end_line".into(), Value::Bool(true));
    }
    fn into_json(mut self) -> Value {
        self.data.insert("spans".into(), Value::Array(self.spans));
        Value::Object(self.data)
    }
}
struct Block {
    data: Map<String, Value>,
    kind: String,
    lines: Option<Vec<Line>>,
    bounds: [f64; 4],
    frame: [f64; 4],
    page_idx: i64,
    page_size: [f64; 2],
    owner: usize,
    allow_uppercase: bool,
}
impl Block {
    fn parse(
        value: Value,
        page_idx: i64,
        page_size: [f64; 2],
        owner: usize,
    ) -> Result<Self, Error> {
        let mut data = object(value)?;
        let kind = data
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("段落块缺少 type"))?
            .to_string();
        let (bounds, lines) = if kind == "text" {
            let (b, _) = bbox(
                data.get("bbox")
                    .ok_or_else(|| invalid("text 块缺少 bbox"))?,
            )?;
            let rows = array(
                data.remove("lines")
                    .ok_or_else(|| invalid("text 块缺少 lines"))?,
            )?;
            (
                b,
                Some(
                    rows.into_iter()
                        .map(Line::parse)
                        .collect::<Result<Vec<_>, _>>()?,
                ),
            )
        } else {
            ([0.; 4], None)
        };
        Ok(Self {
            data,
            kind,
            lines,
            bounds,
            frame: bounds,
            page_idx,
            page_size,
            owner,
            allow_uppercase: false,
        })
    }
    /// 仅处理原始 text 块；Python min/max 平局保留第一个坐标的类型和值。
    fn reframe(&mut self) {
        let Some(lines) = self.lines.as_ref() else {
            return;
        };
        let Some(first) = lines.first() else {
            if let Some(v) = self.data.get("bbox").cloned() {
                self.data.insert("bbox_fs".into(), v);
            }
            return;
        };
        let mut frame = first.bounds;
        let mut coordinates = first.coordinates.clone();
        for l in &lines[1..] {
            for i in 0..4 {
                if (i < 2 && l.bounds[i] < frame[i]) || (i >= 2 && l.bounds[i] > frame[i]) {
                    frame[i] = l.bounds[i];
                    coordinates[i] = l.coordinates[i].clone();
                }
            }
        }
        self.frame = frame;
        self.data
            .insert("bbox_fs".into(), Value::Array(coordinates.into()));
    }
    fn into_json(mut self) -> Value {
        self.data.insert("type".into(), Value::String(self.kind));
        self.data.remove("page_num");
        self.data.remove("page_size");
        if let Some(lines) = self.lines {
            self.data.insert(
                "lines".into(),
                Value::Array(lines.into_iter().map(Line::into_json).collect()),
            );
        }
        Value::Object(self.data)
    }
}

/// 原函数的列表/目录分类和行标记；严格保留分支优先级、浮点比例和不清除旧标记的行为。
fn classify(block: &mut Block) {
    let Some(lines) = block.lines.as_mut() else {
        return;
    };
    if lines.len() < 2 {
        return;
    }
    let n = lines.len() as f64;
    let first = lines[0].bounds;
    let last = lines[lines.len() - 1].bounds;
    let h = first[3] - first[1];
    let width = block.frame[2] - block.frame[0];
    let height = block.frame[3] - block.frame[1];
    let width_ratio = if block.page_size[0] == 0. {
        0.
    } else {
        width / block.page_size[0]
    };
    let multiple = first[0] - block.frame[0] > h / 2.
        && (last[0] - block.frame[0]).abs() < h / 2.
        && block.frame[2] - last[2] > h;
    let texts: Vec<String> = lines
        .iter()
        .map(|l| {
            l.spans
                .iter()
                .filter(|s| s["type"] == "text")
                .filter_map(|s| s["content"].as_str())
                .map(|s| s.trim_matches(python_space))
                .collect::<String>()
        })
        .collect();
    let cjk = uses_unspaced_script(&texts.concat());
    let (
        mut left_close,
        mut left_not_close,
        mut right_close,
        mut right_not_close,
        mut centered,
        mut inside,
    ) = (0, 0, 0, 0, 0, 0);
    for l in lines.iter() {
        let b = l.bounds;
        if b[0] - block.frame[0] > 0.7 * h && block.frame[2] - b[2] > 0.7 * h {
            inside += 1;
        }
        if ((b[0] + b[2]) / 2. - (block.frame[0] + block.frame[2]) / 2.).abs() < h / 2. {
            centered += 1;
        }
        if (block.frame[0] - b[0]).abs() < h / 2. {
            left_close += 1;
        } else if b[0] - block.frame[0] > h {
            left_not_close += 1;
        }
        if (block.frame[2] - b[2]).abs() < h {
            right_close += 1;
        } else {
            let closed = if cjk || width_ratio >= 0.5 {
                0.26 * width
            } else {
                0.36 * width
            };
            if block.frame[2] - b[2] > closed {
                right_not_close += 1;
            }
        }
    }
    let starts = texts
        .iter()
        .filter(|s| s.chars().next().is_some_and(python_digit))
        .count();
    let ends = texts
        .iter()
        .filter(|s| s.chars().last().is_some_and(python_digit))
        .count();
    let stops = texts
        .iter()
        .filter(|s| s.chars().last().is_some_and(|c| ".。;；".contains(c)))
        .count();
    let end_flag = stops as f64 / n >= 0.8;
    let num_flag = starts as f64 / n >= 0.8 || ends as f64 / n >= 0.8;
    if (left_close as f64 / n >= 0.8 || right_close as f64 / n >= 0.8) && num_flag {
        for l in lines {
            l.start();
        }
        block.kind = "index".into();
    } else if inside >= 2
        && centered == lines.len()
        && inside as f64 / n >= 0.5
        && height / width > 0.4
    {
        for l in lines {
            l.start();
        }
        block.kind = "list".into();
    } else if left_close >= 2
        && (right_not_close >= 2 || end_flag || left_not_close >= 2)
        && !multiple
    {
        if left_close as f64 / n > 0.8 {
            if stops == 0 && (right_close as f64 / n) < 0.5 {
                for l in lines.iter_mut() {
                    if (block.frame[0] - l.bounds[0]).abs() < h / 2. {
                        l.start();
                    }
                }
            } else if end_flag {
                for i in 0..lines.len() {
                    if texts[i]
                        .chars()
                        .last()
                        .is_some_and(|c| ".。;；".contains(c))
                    {
                        lines[i].end();
                        if i + 1 < lines.len() {
                            lines[i + 1].start();
                        }
                    }
                }
            } else {
                let mut start = false;
                for l in lines.iter_mut() {
                    if start {
                        l.start();
                        start = false;
                    }
                    if (block.frame[2] - l.bounds[2]).abs() > 0.1 * width {
                        l.end();
                        start = true;
                    }
                }
            }
        } else if starts >= 2 && starts == stops {
            for (l, s) in lines.iter_mut().zip(&texts) {
                if s.chars().next().is_some_and(python_digit) {
                    l.start();
                }
                if s.chars().last().is_some_and(|c| ".。;；".contains(c)) {
                    l.end();
                }
            }
        } else {
            for l in lines.iter_mut() {
                if (block.frame[0] - l.bounds[0]).abs() < h / 2. {
                    l.start();
                }
                if (block.frame[2] - l.bounds[2]).abs() > h {
                    l.end();
                }
            }
        }
        block.kind = "list".into();
    }
}

// DocVortex 0.6.12 indentation evidence, frozen before classification and reverse merging.
fn body_content(block: &Block) -> String {
    block
        .lines
        .as_ref()
        .into_iter()
        .flatten()
        .flat_map(|l| &l.spans)
        .filter_map(|s| s["content"].as_str())
        .collect::<String>()
        .trim_matches(python_space)
        .into()
}
fn list_start(block: &Block) -> bool {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    RE.get_or_init(||regex::Regex::new(r"^(?:[-*+•●○▪▫◦‣⁃–—](?:\s|$)|[（(]?(?:\d+|[A-Za-z]|[IVXLCDMivxlcdm]+)[.．、)）]|[\[［]\d+[\]］]|[一二三四五六七八九十百]+[、.．)）]|[①-⑳])").unwrap()).is_match(&body_content(block))
}
fn vertical(lines: &[Line]) -> bool {
    !lines.is_empty()
        && lines
            .iter()
            .filter(|l| (l.bounds[3] - l.bounds[1]) / (l.bounds[2] - l.bounds[0]).max(1e-12) > 2.)
            .count() as f64
            / lines.len() as f64
            > 0.8
}
fn reference(block: &Block) -> bool {
    block
        .data
        .get("_reference_start")
        .is_some_and(|v| !v.is_null())
}
fn indented(block: &Block) -> bool {
    let Some(lines) = &block.lines else {
        return false;
    };
    if block.kind != "text"
        || reference(block)
        || list_start(block)
        || body_content(block).is_empty()
        || lines.len() < 2
        || vertical(lines)
    {
        return false;
    }
    let first = lines[0].bounds;
    let h = first[3] - first[1];
    let left = lines[1..]
        .iter()
        .map(|l| l.bounds[0])
        .fold(f64::INFINITY, f64::min);
    lines
        .iter()
        .map(|l| l.bounds[2])
        .fold(f64::NEG_INFINITY, f64::max)
        - first[2]
        < h
        && first[0] - left >= h / 2.
        && lines[1..]
            .iter()
            .all(|l| (l.bounds[0] - left).abs() < (l.bounds[3] - l.bounds[1]) / 2.)
}
fn uppercase_evidence(blocks: &[Block]) -> Vec<bool> {
    let pages: BTreeSet<_> = blocks
        .iter()
        .filter(|b| indented(b))
        .map(|b| b.page_idx)
        .collect();
    let mut flags = vec![false; blocks.len()];
    for i in 1..blocks.len() {
        let current = &blocks[i];
        let previous = &blocks[i - 1];
        if !pages.contains(&current.page_idx)
            || [current, previous]
                .iter()
                .any(|b| b.kind != "text" || reference(b) || list_start(b))
        {
            continue;
        }
        let (Some(lines), Some(prior)) = (&current.lines, &previous.lines) else {
            continue;
        };
        if lines.is_empty() || prior.is_empty() || vertical(lines) || vertical(prior) {
            continue;
        }
        if current.page_idx == previous.page_idx
            && !(lines
                .iter()
                .map(|l| l.bounds[0])
                .fold(f64::INFINITY, f64::min)
                >= prior
                    .iter()
                    .map(|l| l.bounds[2])
                    .fold(f64::NEG_INFINITY, f64::max)
                && lines[0].bounds[3] < prior.last().unwrap().bounds[1])
        {
            continue;
        }
        if lines.len() == 1 {
            let mut following = vec![];
            for b in &blocks[i + 1..] {
                let Some(rows) = &b.lines else { break };
                if b.page_idx != current.page_idx || b.kind != "text" || vertical(rows) {
                    break;
                }
                following.extend(rows);
                if following.len() >= 5 {
                    break;
                }
            }
            let h = lines[0].bounds[3] - lines[0].bounds[1];
            let mut aligned: Vec<f64> = following
                .into_iter()
                .take(5)
                .filter(|l| {
                    let other = l.bounds[3] - l.bounds[1];
                    h.min(other) > 0.
                        && h.max(other) / h.min(other) <= 1.5
                        && (l.bounds[0] - lines[0].bounds[0]).abs() <= h.max(other)
                })
                .map(|l| l.bounds[0])
                .collect();
            if aligned.len() < 3 {
                continue;
            }
            aligned.sort_by(f64::total_cmp);
            let n = aligned.len() / 2;
            let mid = if aligned.len() % 2 == 1 {
                aligned[n]
            } else {
                (aligned[n - 1] + aligned[n]) / 2.
            };
            if (lines[0].bounds[0] - mid).abs() >= h / 2. {
                continue;
            }
        }
        flags[i] = true;
    }
    flags
}

/// 原始 span 内容参与判断，不 trim、不重排，也不假设 span 一定是 text。
fn merge_text_allowed(current: &Block, previous: &Block) -> Result<bool, Error> {
    let (Some(a), Some(b)) = (&current.lines, &previous.lines) else {
        return Ok(false);
    };
    let (Some(first), Some(last)) = (a.first(), b.last()) else {
        return Ok(false);
    };
    let h = first.bounds[3] - first.bounds[1];
    if (current.frame[0] - first.bounds[0]).abs() >= h / 2. {
        return Ok(false);
    }
    let (Some(first_span), Some(last_span)) = (first.spans.first(), last.spans.last()) else {
        return Ok(false);
    };
    let content = first_span
        .get("content")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("正文合并的首 span content 必须是字符串"))?;
    let Some(c) = content.chars().next() else {
        return Ok(false);
    };
    let last_content = last_span
        .get("content")
        .and_then(Value::as_str)
        .ok_or_else(|| invalid("正文合并的末 span content 必须是字符串"))?;
    let last_h = last.bounds[3] - last.bounds[1];
    let w1 = current.bounds[2] - current.bounds[0];
    let w2 = previous.bounds[2] - previous.bounds[0];
    Ok((previous.frame[2] - last.bounds[2]).abs() < last_h
        && !last_content
            .chars()
            .last()
            .is_some_and(|c| ".!?。！？)）\"”:：;；".contains(c))
        && (w1 - w2).abs() < w1.min(w2)
        && !python_digit(c)
        && (!python_upper(c) || current.allow_uppercase)
        && current.bounds[1] < previous.bounds[3]
        && (a.len() > 1 || b.len() > 1))
}

/// 倒序合并时移动行所有权；仅移动的跨页 spans 写 cross_page，删除标记留在原块。
fn merge_into_previous(current: &mut Block, previous: &mut Block) {
    let (Some(lines), Some(destination)) = (&mut current.lines, &mut previous.lines) else {
        return;
    };
    if current.page_idx != previous.page_idx {
        for l in lines.iter_mut() {
            for s in &mut l.spans {
                if let Value::Object(map) = s {
                    map.insert("cross_page".into(), Value::Bool(true));
                }
            }
        }
    }
    destination.append(lines);
    current
        .data
        .insert("lines_deleted".into(), Value::Bool(true));
}

/// 对旧 page_info_list 运行 para_split 的复制语义：输出完整页对象，preproc_blocks 不变。
/// 正几何、有效 Unicode 和唯一整数页号属于当前合同；无效输入/资源超限明确报错。
/// 最多 1000 页/块、10000 行、200000 JSON 节点、2MB 字符串；不调用模型或实验段落器。
pub fn split(pages: &Value) -> Result<Value, Error> {
    let mut stack = vec![pages];
    let (mut nodes, mut bytes) = (0usize, 0usize);
    while let Some(v) = stack.pop() {
        nodes += 1;
        match v {
            Value::String(s) => bytes += s.len(),
            Value::Array(a) => stack.extend(a),
            Value::Object(o) => {
                bytes += o.keys().map(String::len).sum::<usize>();
                stack.extend(o.values());
            }
            _ => {}
        }
        if nodes > rapiddoc_core::resources::limit(200_000)
            || bytes > rapiddoc_core::resources::limit(2_000_000)
        {
            return Err(limit("旧段落 JSON 节点或字符串超过预算"));
        }
    }
    let source = pages
        .as_array()
        .ok_or_else(|| invalid("旧段落输入必须是 page_info_list"))?;
    if source.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit("旧段落页数超过预算"));
    }
    let mut output = Vec::with_capacity(source.len());
    let mut blocks = Vec::new();
    let mut seen = BTreeSet::new();
    let mut line_count = 0;
    for (owner, p) in source.iter().enumerate() {
        let page = p.as_object().ok_or_else(|| invalid("旧段落页必须是对象"))?;
        let idx = page
            .get("page_idx")
            .and_then(Value::as_i64)
            .ok_or_else(|| invalid("page_idx 必须是整数"))?;
        if !seen.insert(idx) {
            return Err(invalid("page_idx 必须唯一"));
        }
        let size = page
            .get("page_size")
            .and_then(Value::as_array)
            .filter(|a| a.len() == 2)
            .ok_or_else(|| invalid("page_size 必须有两个数值"))?;
        let mut page_size = [0.; 2];
        for i in 0..2 {
            page_size[i] = size[i]
                .as_f64()
                .filter(|v| v.is_finite() && (0. ..=1_000_000.).contains(v))
                .ok_or_else(|| invalid("page_size 无效"))?;
        }
        let raw = page
            .get("preproc_blocks")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("页缺少 preproc_blocks"))?;
        for b in raw {
            let block = Block::parse(b.clone(), idx, page_size, owner)?;
            line_count += block.lines.as_ref().map_or(0, Vec::len);
            blocks.push(block);
            if blocks.len() > rapiddoc_core::resources::limit(1000)
                || line_count > rapiddoc_core::resources::limit(10_000)
            {
                return Err(limit("旧段落块或行数超过预算"));
            }
        }
        let mut copy = page.clone();
        copy.insert("para_blocks".into(), Value::Array(Vec::new()));
        output.push(copy);
    }
    let flags = uppercase_evidence(&blocks);
    for (block, allow) in blocks.iter_mut().zip(flags) {
        block.allow_uppercase = allow;
    }
    let mut start = 0;
    while start < blocks.len() {
        if blocks[start].kind != "text" {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < blocks.len() && blocks[end].kind == "text" {
            end += 1;
        }
        for b in &mut blocks[start..end] {
            b.reframe();
            classify(b);
        }
        let short_group = blocks[start..end]
            .iter()
            .all(|b| b.lines.as_ref().is_some_and(|l| l.len() <= 3));
        for i in (start + 1..end).rev() {
            let (before, after) = blocks.split_at_mut(i);
            let previous = &mut before[i - 1];
            let current = &mut after[0];
            let list_pair = (current.kind == "list" && previous.kind == "list")
                || (current.kind == "index" && previous.kind == "index");
            if list_pair
                || (current.kind == "text"
                    && previous.kind == "text"
                    && !short_group
                    && merge_text_allowed(current, previous)?)
            {
                merge_into_previous(current, previous);
            }
        }
        start = end;
    }
    let mut buckets: Vec<Vec<Value>> = (0..output.len()).map(|_| Vec::new()).collect();
    for b in blocks {
        let owner = b.owner;
        buckets[owner].push(b.into_json());
    }
    for (page, values) in output.iter_mut().zip(buckets) {
        page.insert("para_blocks".into(), Value::Array(values));
    }
    Ok(Value::Array(
        output.into_iter().map(Value::Object).collect(),
    ))
}
