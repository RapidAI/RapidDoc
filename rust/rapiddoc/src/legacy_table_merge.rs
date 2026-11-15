//! 旧 table_merge.py：跨页候选、表头、行列跨度与续行语义；不启用 enhanced。
use crate::legacy_page::{invalid, limit, unsupported, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::python_space;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
mod grid;
pub(crate) mod html;
use grid::*;
use html::Dom;

fn full_half(s: &str) -> String {
    s.chars()
        .map(|c| {
            if (0xff01..=0xff5e).contains(&(c as u32)) {
                char::from_u32(c as u32 - 0xfee0).unwrap()
            } else {
                c
            }
        })
        .collect()
}

pub fn merge(pages: &Value) -> Result<Value, Error> {
    validate_tree(pages)?;
    let source = pages.as_array().ok_or_else(invalid)?;
    if source.len() > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let mut pages = source.clone();
    let mut work = Work(0);
    let mut states: BTreeMap<usize, State> = BTreeMap::new();
    let mut removed = BTreeSet::new();
    for p in &pages {
        let blocks = p["para_blocks"].as_array().ok_or_else(invalid)?;
        if blocks.len() > rapiddoc_core::resources::limit(1000) {
            return Err(limit());
        }
        for block in blocks {
            if !block.is_object() || !block["type"].is_string() {
                return Err(invalid());
            }
            if let Some(flags) = block.get("cell_merge").filter(|v| !v.is_null()) {
                let flags = flags.as_array().ok_or_else(invalid)?;
                if flags.len() > rapiddoc_core::resources::limit(1000) {
                    return Err(limit());
                }
            }
        }
    }
    let mut ids: Vec<Vec<usize>> = pages
        .iter()
        .enumerate()
        .map(|(i, p)| {
            (0..p["para_blocks"].as_array().unwrap().len())
                .map(|j| i * 1001 + j)
                .collect()
        })
        .collect();
    for i in (1..pages.len()).rev() {
        let previous = pages[i - 1]["para_blocks"].as_array().unwrap();
        let current = pages[i]["para_blocks"].as_array().unwrap();
        if previous.last().is_none_or(|b| b["type"] != "table")
            || current.first().is_none_or(|b| b["type"] != "table")
        {
            continue;
        }
        let previous_idx = previous.len() - 1;
        // 每页末表可在随后一轮成为首页表；键包括页与块，不以内容代替对象身份。
        let current_key = ids[i][0];
        let previous_key = ids[i - 1][previous_idx];
        if let std::collections::btree_map::Entry::Vacant(entry) = states.entry(current_key) {
            if let Some(state) = State::from_block(&current[0], &mut work)? {
                entry.insert(state);
            } else {
                continue;
            }
        }
        if let std::collections::btree_map::Entry::Vacant(entry) = states.entry(previous_key) {
            if let Some(state) = State::from_block(&previous[previous_idx], &mut work)? {
                entry.insert(state);
            } else {
                continue;
            }
        }
        let mut current_state = states.remove(&current_key).unwrap();
        let previous_state = states.get_mut(&previous_key).unwrap();
        if !can_merge(
            previous_state,
            &current_state,
            &previous[previous_idx],
            &current[0],
            &mut work,
        )? {
            states.insert(current_key, current_state);
            continue;
        }
        let mut current_block = current[0].clone();
        let mut previous_block = previous[previous_idx].clone();
        perform(
            previous_state,
            &mut current_state,
            &mut previous_block,
            &mut current_block,
            &mut work,
        )?;
        let post = post_captions(&current_block)?;
        if !post.is_empty() {
            let mut restored = Vec::new();
            for &j in &post {
                let mut block = current_block["blocks"][j].clone();
                block["type"] = json!("text");
                restored.push(block);
            }
            let blocks = current_block["blocks"].as_array_mut().unwrap();
            let mut pos = 0;
            blocks.retain(|_| {
                let keep = !post.contains(&pos);
                pos += 1;
                keep
            });
            ids[i].splice(1..1, std::iter::repeat_n(usize::MAX, restored.len()));
            pages[i]["para_blocks"]
                .as_array_mut()
                .unwrap()
                .splice(1..1, restored);
        }
        for block in current_block["blocks"].as_array_mut().unwrap() {
            block["lines"] = json!([]);
            block["lines_deleted"] = json!(true);
        }
        pages[i - 1]["para_blocks"][previous_idx] = previous_block;
        pages[i]["para_blocks"][0] = current_block;
        removed.insert(current_key);
    }
    for (key, state) in states {
        if state.dirty && !removed.contains(&key) {
            let page = key / 1001;
            work.charge(ids[page].len())?;
            let index = ids[page]
                .iter()
                .position(|&id| id == key)
                .ok_or_else(invalid)?;
            let blocks = pages[page]["para_blocks"][index]["blocks"]
                .as_array_mut()
                .unwrap();
            let body = blocks
                .iter_mut()
                .find(|b| b["type"] == "table_body")
                .ok_or_else(invalid)?;
            body["lines"][0]["spans"][0]["html"] = json!(state.dom.all_html());
        }
    }
    Ok(Value::Array(pages))
}

fn caption(block: &Value) -> Result<String, Error> {
    crate::legacy_page::merge_table_caption(block)
}
fn continuation(block: &Value) -> Result<bool, Error> {
    let text = full_half(caption(block)?.trim_matches(python_space)).to_lowercase();
    Ok([
        "(续)",
        "(续表)",
        "(续上表)",
        "(continued)",
        "(cont.)",
        "(cont’d)",
        "(…continued)",
        "续表",
    ]
    .iter()
    .any(|s| text.ends_with(s))
        || text.contains("(continued)"))
}
fn post_captions(block: &Value) -> Result<Vec<usize>, Error> {
    let blocks = block["blocks"].as_array().ok_or_else(invalid)?;
    let body = blocks.iter().find(|b| b["type"] == "table_body");
    let mut out = Vec::new();
    for (i, b) in blocks
        .iter()
        .enumerate()
        .filter(|(_, b)| b["type"] == "table_caption")
    {
        if continuation(b)? {
            continue;
        }
        if let Some(body) = body {
            if let (Some(a), Some(bb)) = (b["bbox"].as_array(), body["bbox"].as_array()) {
                if a.len() != 4 || bb.len() != 4 {
                    return Err(invalid());
                }
                if a[1].as_f64().ok_or_else(invalid)? >= bb[3].as_f64().ok_or_else(invalid)? {
                    out.push(i);
                }
            }
        }
    }
    Ok(out)
}
fn can_merge(
    previous: &State,
    current: &State,
    pb: &Value,
    cb: &Value,
    work: &mut Work,
) -> Result<bool, Error> {
    let previous_blocks = pb["blocks"].as_array().ok_or_else(invalid)?;
    let current_blocks = cb["blocks"].as_array().ok_or_else(invalid)?;
    let footnotes = previous_blocks
        .iter()
        .filter(|b| b["type"] == "table_footnote")
        .count();
    let post = post_captions(cb)?;
    let captions: Vec<_> = current_blocks
        .iter()
        .enumerate()
        .filter(|(i, b)| b["type"] == "table_caption" && !post.contains(i))
        .map(|(_, b)| b)
        .collect();
    if !captions.is_empty() {
        let mut has = false;
        for b in captions {
            has |= continuation(b)?;
        }
        if !has || footnotes > 1 {
            return Ok(false);
        }
    } else if footnotes > 0 {
        return Ok(false);
    }
    let width = |b: &Value| -> Result<f64, Error> {
        let a = b["bbox"]
            .as_array()
            .filter(|a| a.len() == 4)
            .ok_or_else(invalid)?;
        let x = a[0].as_f64().ok_or_else(invalid)?;
        let end = a[2].as_f64().ok_or_else(invalid)?;
        let width = end - x;
        if !width.is_finite() || width <= 0. {
            return Err(invalid());
        }
        Ok(width)
    };
    let a = width(pb)?;
    let b = width(cb)?;
    if (a - b).abs() / a.min(b) >= 0.1 {
        return Ok(false);
    }
    if previous.scan.total == current.scan.total {
        return Ok(true);
    }
    let Some(last) = previous.scan.last.as_ref() else {
        return Ok(false);
    };
    let header = expanded_header(current, headers(previous, current).0)?;
    // 旧 front cache 最多保留 6 行，不在缓存中的首数据行不能回退到全表扫描。
    let Some(first) = current.front_metrics.get(header) else {
        return Ok(false);
    };
    Ok(last.effective == first.effective
        || last.actual == first.actual
        || rendered(previous, last.row, work)? == rendered(current, first.row, work)?)
}

fn perform(
    previous: &mut State,
    current: &mut State,
    pb: &mut Value,
    cb: &mut Value,
    work: &mut Work,
) -> Result<(), Error> {
    let header = expanded_header(current, headers(previous, current).0)?;
    let a = previous.scan.total;
    let b = current.scan.total;
    if !previous.rows.is_empty() && header < current.rows.len() {
        if a > b {
            let last = previous.rows.len() - 1;
            if !align(current, previous, header, last, work)? {
                adjust(current, header, previous, last, a, work)?;
            }
        } else if b > a {
            let last = previous.rows.len() - 1;
            if !align(previous, current, last, header, work)? {
                adjust(previous, 0, current, header, b, work)?;
                previous.refresh(work)?;
            }
        }
    }
    if let Some(flags) = infer(previous, current, header, cb, work)? {
        cb["cell_merge"] = json!(flags);
    }
    apply_cells(previous, current, header, cb, work)?;
    let offset = previous.rows.len();
    if offset + current.rows.len().saturating_sub(header) > rapiddoc_core::resources::limit(1000) {
        return Err(limit());
    }
    let mut copied = Vec::new();
    if let (Some(parent), Some(_)) = (previous.body, current.body) {
        for &row in current.rows.iter().skip(header) {
            work.charge(current.dom.serialize(row).len())?;
            let id = previous.dom.copy_from(&current.dom, row, parent);
            copied.push(id);
        }
    }
    previous.rows.extend(copied.iter());
    if !copied.is_empty() {
        let scan = scan(&previous.dom, &copied, &previous.scan.tail, offset, work)?;
        previous.scan.effective.extend(scan.effective);
        previous.scan.metrics.extend(scan.metrics);
        previous.scan.total = previous.scan.total.max(scan.total);
        if scan.last.is_some() {
            previous.scan.last = scan.last;
        }
        previous.scan.tail = scan.tail;
    }
    let footnotes: Vec<_> = cb["blocks"]
        .as_array()
        .ok_or_else(invalid)?
        .iter()
        .filter(|b| b["type"] == "table_footnote")
        .cloned()
        .collect();
    let blocks = pb["blocks"].as_array_mut().ok_or_else(invalid)?;
    blocks.retain(|b| b["type"] != "table_footnote");
    for mut note in footnotes {
        note["cross_page"] = json!(true);
        blocks.push(note);
    }
    previous.dirty = true;
    Ok(())
}

fn apply_cells(
    previous: &mut State,
    current: &mut State,
    header: usize,
    block: &Value,
    work: &mut Work,
) -> Result<(), Error> {
    let Some(flags) = block
        .get("cell_merge")
        .and_then(Value::as_array)
        .filter(|f| !f.is_empty())
    else {
        return Ok(());
    };
    if header >= current.rows.len() || previous.rows.is_empty() {
        return Ok(());
    }
    let last = previous.rows.len() - 1;
    let (previous_columns, _) = sources(previous, last, work)?;
    let (next_columns, _) = scan_sources(current, header, header, &previous.scan.tail, work)?;
    let source_cells = current.cells(header);
    let mut transferred = false;
    for (&start, &marker) in &next_columns {
        let before = start.checked_sub(1);
        if marker.0 != header || before.is_some_and(|col| next_columns.get(&col) == Some(&marker)) {
            continue;
        }
        let source = source_cells[marker.1];
        let end = start + span(&current.dom, source, "colspan")?;
        let Some(target) = previous_columns.get(&start) else {
            continue;
        };
        if end > flags.len()
            || !flags[start..end]
                .iter()
                .all(|flag| flag.as_f64() == Some(1.) || flag.as_bool() == Some(true))
            || (start..end).any(|col| next_columns.get(&col) != Some(&marker))
            || before.is_some_and(|col| previous_columns.get(&col) == Some(target))
            || previous_columns.get(&end) == Some(target)
            || (start..end).any(|col| previous_columns.get(&col) != Some(target))
        {
            continue;
        }
        let target_id = previous.cells(target.0)[target.1];
        if current.dom.semantic(source) {
            previous
                .dom
                .append_merged_cell(target_id, &current.dom, source, work)?;
        }
        // 目标可以起始于上一页更早的一行，使用全表行号计算跨度。
        let height = previous.rows.len() + span(&current.dom, source, "rowspan")? - target.0;
        set_span(&mut previous.dom, target_id, "rowspan", height);
        current.dom.detach(source);
        transferred = true;
    }
    if transferred {
        if current.cells(header).is_empty() {
            for (row, index) in previous_columns.values().copied().collect::<BTreeSet<_>>() {
                let id = previous.cells(row)[index];
                let height = span(&previous.dom, id, "rowspan")?;
                if row + height > previous.rows.len() {
                    set_span(&mut previous.dom, id, "rowspan", height - 1);
                }
            }
            let id = current.rows.remove(header);
            current.dom.detach(id);
        }
        previous.refresh(work)?;
    }
    Ok(())
}
fn infer(
    previous: &State,
    current: &State,
    header: usize,
    block: &Value,
    work: &mut Work,
) -> Result<Option<Vec<usize>>, Error> {
    if block
        .get("cell_merge")
        .is_some_and(|f| f.as_array().is_some_and(|a| !a.is_empty()))
        || previous.rows.is_empty()
        || header + 1 >= current.rows.len()
        || previous.scan.total != current.scan.total
    {
        return Ok(None);
    }
    let (pc, pm) = visual_cells(previous, previous.rows.len() - 1, work)?;
    let (cc, cm) = visual_cells(current, header, work)?;
    let (fc, fm) = visual_cells(current, header + 1, work)?;
    let pc: Vec<_> = pc.iter().map(|&id| previous.dom.semantic(id)).collect();
    let cc: Vec<_> = cc.iter().map(|&id| current.dom.semantic(id)).collect();
    let fc: Vec<_> = fc.iter().map(|&id| current.dom.semantic(id)).collect();
    let populated =
        |c: &[bool], m: &BTreeMap<usize, usize>, i: usize| m.get(&i).is_some_and(|&j| c[j]);
    if !populated(&pc, &pm, 0) || populated(&cc, &cm, 0) || !populated(&fc, &fm, 0) {
        return Ok(None);
    }
    let nonempty: Vec<_> = cm
        .keys()
        .copied()
        .filter(|&i| populated(&cc, &cm, i))
        .collect();
    if nonempty.is_empty()
        || nonempty.len() > 2.max(previous.scan.total / 3)
        || nonempty.iter().any(|&i| !populated(&pc, &pm, i))
    {
        return Ok(None);
    }
    let mut flags = vec![0; previous.scan.total];
    for &col in cm.keys() {
        if col < flags.len()
            && (nonempty.contains(&col) || (pm.contains_key(&col) && !populated(&cc, &cm, col)))
        {
            flags[col] = 1;
        }
    }
    Ok(Some(flags))
}
