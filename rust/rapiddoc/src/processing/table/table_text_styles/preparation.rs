//! 旧表格脚本的字符索引、视觉行、排版度量及非网格分数线准备。
use super::{numeric, spatial};
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{python_decimal, python_printable, text_boundary::python_space};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, HashMap},
    sync::OnceLock,
};
type Box4 = [f64; 4];

// Preserve the exact serde_json UTF-8/escaping/number byte budget without
// allocating a temporary JSON String for every selected glyph and character.
fn json_bytes(v: &Value) -> usize {
    // This value only enforces max clone bytes. Production has no such cap;
    // avoid serializing every selected glyph for an unused counter.
    if rapiddoc_core::resources::limit(8_000_000usize) == usize::MAX {
        return 0;
    }
    struct Counter(usize);
    impl std::io::Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0 = self.0.saturating_add(bytes.len());
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut count = Counter(0);
    serde_json::to_writer(&mut count, v).expect("JSON Value serialization cannot fail");
    count.0
}
fn median(mut a: Vec<f64>) -> f64 {
    a.sort_by(f64::total_cmp);
    if a.len() % 2 == 1 {
        a[a.len() / 2]
    } else {
        (a[a.len() / 2 - 1] + a[a.len() / 2]) / 2.
    }
}
fn bbox(v: &Value) -> Result<Option<Box4>, Error> {
    let Some(a) = v.as_array().filter(|a| a.len() == 4) else {
        return Ok(None);
    };
    let mut b = [0.; 4];
    for (i, n) in a.iter().enumerate() {
        let Some(x) = numeric(Some(n)) else {
            return Ok(None);
        };
        if x.abs() > 1e9 {
            return Err(limit());
        }
        b[i] = x;
    }
    let out = [
        b[0].min(b[2]),
        b[1].min(b[3]),
        b[0].max(b[2]),
        b[1].max(b[3]),
    ];
    Ok((out[2] > out[0] && out[3] > out[1]).then_some(out))
}
fn coordinates(v: &Value) -> Result<Box4, Error> {
    let a = v.as_array().filter(|a| a.len() == 4).ok_or_else(invalid)?;
    let mut out = [0.; 4];
    for (i, n) in a.iter().enumerate() {
        out[i] = n.as_f64().filter(|n| n.is_finite()).ok_or_else(invalid)?;
        if out[i].abs() > 1e9 {
            return Err(limit());
        }
    }
    Ok(out)
}
fn page(v: &Value) -> Result<[f64; 2], Error> {
    let a = v.as_array().filter(|a| a.len() == 2).ok_or_else(invalid)?;
    let p = [
        numeric(Some(&a[0])).ok_or_else(invalid)?,
        numeric(Some(&a[1])).ok_or_else(invalid)?,
    ];
    if p.iter().any(|&n| n <= 0. || n > 1e9) {
        return Err(limit());
    }
    Ok(p)
}
fn angle(v: &Value) -> Result<i64, Error> {
    let n = v.as_i64().unwrap_or(0);
    if !v.is_null() && !v.is_i64() {
        return Err(invalid());
    }
    Ok(n)
}
fn rotate(b: Box4, p: [f64; 2], a: i64) -> Box4 {
    match a {
        90 => [b[1], p[0] - b[2], b[3], p[0] - b[0]],
        180 => [p[0] - b[2], p[1] - b[3], p[0] - b[0], p[1] - b[1]],
        270 => [p[1] - b[3], b[0], p[1] - b[1], b[2]],
        _ => b,
    }
}
fn union(a: Box4, b: Box4) -> Box4 {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}
fn flags(v: &Value) -> Result<Value, Error> {
    if v.is_null() {
        return Ok(json!(0));
    }
    if let Some(b) = v.as_bool() {
        return Ok(json!(i64::from(b)));
    }
    if v.is_i64() || v.is_u64() {
        return Ok(v.clone());
    }
    let s = if let Some(s) = v.as_str() {
        let Some(s) = super::table_native_text::numeric_string(s)? else {
            return Ok(json!(0));
        };
        let digits = s.trim_start_matches(['+', '-']);
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return Ok(json!(0));
        }
        s
    } else if let Some(n) = v.as_f64() {
        format!("{:.0}", n.trunc())
    } else {
        return Ok(json!(0));
    };
    if let Ok(n) = s.parse::<i64>() {
        Ok(json!(n))
    } else if let Ok(n) = s.strip_prefix('+').unwrap_or(&s).parse::<u64>() {
        Ok(json!(n))
    } else {
        Err(limit())
    }
}
pub(in crate::processing) fn fold(s: &str) -> String {
    static FOLD: OnceLock<HashMap<char, String>> = OnceLock::new();
    let map =
        FOLD.get_or_init(|| serde_json::from_str(include_str!("font_casefold.json")).unwrap());
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if let Some(value) = map.get(&c) {
            out.push_str(value)
        } else {
            out.push(c)
        }
    }
    out
}
pub(in crate::processing) fn family(signature: &Option<(String, Value)>) -> Option<String> {
    let s = &signature.as_ref()?.0;
    let b = s.as_bytes();
    let s = if b.len() >= 7 && b[..6].iter().all(u8::is_ascii_uppercase) && b[6] == b'+' {
        &s[7..]
    } else {
        s
    };
    let filtered: String = s
        .chars()
        .filter(|&c| !python_space(c) && c != '_' && c != '-')
        .collect();
    let out = fold(&filtered);
    (!out.is_empty()).then_some(out)
}
fn terminal(line: &Value, chars: &[Value], a: i64) -> Result<bool, Error> {
    let raw = spatial::raw_string(line.get("text"), true);
    let text = raw.trim_end_matches(python_space);
    let punct = |c: char| ".!?。！？".contains(c);
    let trimmed = text.trim_end_matches(|c| "])}）】》”’'\"".contains(c));
    if trimmed.chars().last().is_some_and(punct) {
        return Ok(true);
    }
    let suffix = text;
    let split = suffix.rfind(|c: char| python_decimal(c).is_none() && !",–—-".contains(c));
    let Some(pos) = split else { return Ok(false) };
    let c = suffix[pos..].chars().next().unwrap();
    if !punct(c) {
        return Ok(false);
    }
    let refs = &suffix[pos + c.len_utf8()..];
    if !refs
        .chars()
        .next()
        .is_some_and(|c| python_decimal(c).is_some())
    {
        return Ok(false);
    }
    let count = refs.chars().count();
    let selected: Vec<_> = chars
        .iter()
        .filter(|c| {
            !spatial::raw_string(c.get("char"), false)
                .trim_matches(python_space)
                .is_empty()
        })
        .collect();
    if selected.len() <= count {
        return Ok(false);
    }
    let axis = usize::from(a != 90 && a != 270);
    let mut sizes = Vec::new();
    for c in &selected {
        let b = c["bbox"]
            .as_array()
            .filter(|b| b.len() == 4)
            .ok_or_else(invalid)?;
        let lo = b[axis].as_f64().ok_or_else(invalid)?;
        let hi = b[axis + 2].as_f64().ok_or_else(invalid)?;
        sizes.push(hi - lo);
    }
    let body: Vec<_> = sizes[..sizes.len() - count]
        .iter()
        .copied()
        .filter(|&n| n > 0.)
        .collect();
    if body.is_empty() {
        return Ok(false);
    }
    let scale = median(body);
    Ok(sizes[sizes.len() - count..]
        .iter()
        .all(|&n| n > 0. && n <= 0.8 * scale))
}
pub(in crate::processing) fn metrics(line: &Value, p: [f64; 2]) -> Result<Value, Error> {
    metrics_bounded(line, p, 1500)
}
pub(in crate::processing) fn page_metrics(line: &Value, p: [f64; 2]) -> Result<Value, Error> {
    metrics_bounded(line, p, 20000)
}
fn metrics_bounded(line: &Value, p: [f64; 2], maximum: usize) -> Result<Value, Error> {
    let maximum = crate::legacy_page::budget::records(maximum);
    let a = angle(&line["angle"])?;
    let chars = line["chars"]
        .as_array()
        .filter(|a| a.len() <= maximum)
        .ok_or_else(limit)?;
    let mut heights = Vec::new();
    let mut widths = Vec::new();
    let mut signatures: Vec<(String, Value, usize, Vec<f64>)> = Vec::new();
    let mut signature_ids = HashMap::new();
    let mut glyphs: Vec<(Box4, Option<usize>, Option<f64>)> = Vec::new();
    let mut valid = 0;
    for c in chars {
        if !c.is_object() {
            return Err(invalid());
        }
        let text = spatial::raw_string(c.get("char"), true);
        if !text.chars().all(python_printable)
            || (!text.is_empty() && text.chars().all(python_space))
        {
            continue;
        }
        let Some(b) = bbox(&c["bbox"])? else { continue };
        let b = [
            b[0].clamp(0., p[0]),
            b[1].clamp(0., p[1]),
            b[2].clamp(0., p[0]),
            b[3].clamp(0., p[1]),
        ];
        if b[2] <= b[0] || b[3] <= b[1] {
            continue;
        }
        let b = rotate(b, p, a);
        heights.push((b[3] - b[1]).max(0.1));
        widths.push((b[2] - b[0]).max(0.1));
        let name = spatial::raw_string(c["font"].get("name"), true);
        if name.is_empty() {
            glyphs.push((b, None, None));
            continue;
        }
        let flag = flags(&c["font"]["flags"])?;
        let key = (name.clone(), flag.to_string());
        let id = *signature_ids.entry(key).or_insert_with(|| {
            signatures.push((name, flag, 0, Vec::new()));
            signatures.len() - 1
        });
        signatures[id].2 += 1;
        valid += 1;
        let weight = numeric(c["font"].get("weight")).filter(|&w| w > 0.);
        if let Some(w) = weight {
            signatures[id].3.push(w)
        }
        glyphs.push((b, Some(id), weight));
    }
    let line_box = rotate(bbox(&line["bbox"])?.ok_or_else(invalid)?, p, a);
    let height = if heights.is_empty() {
        (line_box[3] - line_box[1]).max(0.1)
    } else {
        median(heights)
    };
    let em = numeric(line.get("em_height")).unwrap_or(0.);
    let winner = signatures
        .iter()
        .enumerate()
        .max_by(|(i, a), (j, b)| a.2.cmp(&b.2).then(j.cmp(i)))
        .map(|(i, _)| i);
    let (sig, coverage, weight) = if let Some(i) = winner {
        let s = &signatures[i];
        (
            json!([s.0, s.1]),
            s.2 as f64 / valid as f64,
            if s.3.is_empty() {
                Value::Null
            } else {
                json!(median(s.3.clone()))
            },
        )
    } else {
        (Value::Null, 0., Value::Null)
    };
    let mut emphasis = Value::Null;
    let mut typography = Value::Null;
    if glyphs.len() >= 4 {
        if let Some(first) = glyphs[0].1 {
            let n = glyphs.iter().take_while(|g| g.1 == Some(first)).count();
            if n >= 2 && glyphs.len() - n >= 2 {
                let boxes = glyphs[..n].iter().map(|g| g.0).reduce(union).unwrap();
                let width = (boxes[2] - boxes[0]).max(0.1);
                let prefix: Vec<_> = glyphs[..n].iter().filter_map(|g| g.2).collect();
                let body: Vec<_> = glyphs[n..].iter().filter_map(|g| g.2).collect();
                if !prefix.is_empty() && !body.is_empty() {
                    let pw = median(prefix);
                    let bw = median(body);
                    if pw - bw >= 100. && pw >= 1.15 * bw.max(1.) {
                        emphasis = json!(width)
                    }
                }
                let mut body_counts = Vec::<(usize, usize)>::new();
                for g in &glyphs[n..] {
                    if let Some(i) = g.1 {
                        if let Some(v) = body_counts.iter_mut().find(|v| v.0 == i) {
                            v.1 += 1
                        } else {
                            body_counts.push((i, 1))
                        }
                    }
                }
                let best = body_counts
                    .iter()
                    .enumerate()
                    .max_by(|(i, a), (j, b)| a.1.cmp(&b.1).then(j.cmp(i)))
                    .map(|(_, x)| x);
                let fs = Some((signatures[first].0.clone(), signatures[first].1.clone()));
                if let Some(&(id, count)) = best {
                    let bs = Some((signatures[id].0.clone(), signatures[id].1.clone()));
                    if count >= 2 && family(&fs).is_some() && family(&fs) != family(&bs) {
                        typography = json!(width)
                    }
                }
            }
        }
    }
    Ok(
        json!({"effective_height":height,"em_height":if em>0.{line["em_height"].clone()}else{json!(height)},"median_glyph_width":if widths.is_empty(){Value::Null}else{json!(median(widths))},"font_signature":sig,"font_coverage":coverage,"dominant_font_weight":weight,"leading_emphasis_width":emphasis,"leading_typography_width":typography,"paragraph_terminal":terminal(line,chars,a)?}),
    )
}
pub fn measure(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    metrics(&v["line"], page(&v["page_size"])?)
}
fn array(v: &Value, n: usize) -> Result<&Vec<Value>, Error> {
    let n = if n > 8 {
        crate::legacy_page::budget::records(n)
    } else {
        n
    };
    v.as_array().filter(|a| a.len() <= n).ok_or_else(limit)
}
pub fn prepare(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    prepare_with_work(v, &mut Work(0))
}
pub(super) fn prepare_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    let input = &v["table_input"];
    let result = &v["result"];
    let p = page(&input["page_size"])?;
    let a = angle(&input["angle"])?;
    let table = coordinates(&input["table_bbox"])?;
    if table[2] <= table[0] || table[3] <= table[1] {
        return Err(invalid());
    }
    let cells = array(&result["cells"], 2000)?;
    let rules = array(&input["drawing_lines"], 5000)?;
    let chars = array(&input["chars"], 20000)?;
    let glyphs = array(&result["text"]["glyphs"], 3000)?;
    let height = numeric(result["text"].get("median_glyph_height"))
        .filter(|&n| n > 0. && n <= 1e9)
        .ok_or_else(invalid)?;
    let mut by_source = BTreeMap::new();
    for (i, c) in chars.iter().enumerate() {
        if !c.is_object() {
            return Err(invalid());
        }
        let id = c.get("char_idx").map_or(Some(i as i64), Value::as_i64);
        if let Some(id) = id {
            by_source.entry(id).or_insert(c);
        }
    }
    let mut glyph_by_source = BTreeMap::new();
    for g in glyphs {
        let id = g["source_index"].as_i64().ok_or_else(invalid)?;
        coordinates(&g["bbox"])?;
        g["visual_row"].as_i64().ok_or_else(invalid)?;
        g["glyph_id"].as_i64().ok_or_else(invalid)?;
        g["text"].as_str().ok_or_else(invalid)?;
        if g["bbox"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n.as_f64().is_none())
        {
            return Err(invalid());
        }
        glyph_by_source.insert(id, g);
    }
    let boundaries: Vec<_> = cells
        .iter()
        .map(|c| coordinates(&c["bbox"]))
        .collect::<Result<_, _>>()?;
    let mut kept = Vec::new();
    let tolerance = 0.75f64.max(0.12 * height);
    for r in rules {
        if r["orientation"] != "horizontal" {
            continue;
        }
        let raw = coordinates(&r["bbox"])?;
        let Some(b) = super::super::table_native_vector::local(raw, table, a, 0.) else {
            continue;
        };
        let width = numeric(r.get("width"))
            .filter(|&n| (0. ..=1e9).contains(&n))
            .ok_or_else(invalid)?;
        let mut grid = false;
        for c in &boundaries {
            work.charge(2)?;
            for y in [c[1], c[3]] {
                let overlap = b[2].min(c[2]) - b[0].max(c[0]);
                if ((b[1] + b[3]) / 2. - y).abs() <= tolerance.max(width)
                    && overlap > 0.
                    && overlap / (b[2] - b[0]).min(c[2] - c[0]).max(1e-6) >= 0.5
                {
                    grid = true;
                    break;
                }
            }
            if grid {
                break;
            }
        }
        if !grid {
            kept.push(r.clone())
        }
    }
    let mut output = Vec::new();
    let mut clone_bytes = 0usize;
    let mut selected_count = 0usize;
    let mut keys = std::collections::BTreeSet::new();
    for c in cells {
        let row = c["row"].as_i64().ok_or_else(invalid)?;
        let col = c["col"].as_i64().ok_or_else(invalid)?;
        if !keys.insert((row, col)) {
            return Err(invalid());
        }
        let mut selected = Vec::new();
        for id in array(&c["source_char_indices"], 20000)? {
            if let Some(g) = glyph_by_source.get(&id.as_i64().ok_or_else(invalid)?) {
                selected.push(*g)
            }
        }
        selected.sort_by(|x, y| {
            x["visual_row"]
                .as_i64()
                .cmp(&y["visual_row"].as_i64())
                .then_with(|| {
                    x["bbox"][0]
                        .as_f64()
                        .unwrap()
                        .partial_cmp(&y["bbox"][0].as_f64().unwrap())
                        .unwrap()
                })
                .then_with(|| {
                    x["bbox"][1]
                        .as_f64()
                        .unwrap()
                        .partial_cmp(&y["bbox"][1].as_f64().unwrap())
                        .unwrap()
                })
                .then_with(|| x["glyph_id"].as_i64().cmp(&y["glyph_id"].as_i64()))
        });
        if selected.len() > rapiddoc_core::resources::limit(1500) {
            return Err(limit());
        }
        selected_count += selected.len();
        if selected_count > rapiddoc_core::resources::limit(10000) {
            return Err(limit());
        }
        let requires_direction = selected.iter().any(|g| {
            g.get("writing_angle")
                .is_some_and(|v| v.as_i64() != Some(0))
        });
        let (direction, reading) = if requires_direction {
            let typed: Vec<super::super::table_native_text::Glyph> = selected
                .iter()
                .map(|g| serde_json::from_value((*g).clone()).map_err(|_| invalid()))
                .collect::<Result<_, _>>()?;
            let direction = super::super::table_native_text::cell_writing_angle(&typed);
            let mut reading = Vec::new();
            if direction.is_some_and(|d| d != 0) {
                let (glyphs, rows, _) = super::super::table_native_text::cell_text_rows(typed, 1.)?;
                for row in rows {
                    for i in row {
                        reading.push(json!(glyphs[i]));
                    }
                }
            }
            (direction, reading)
        } else {
            (Some(0), Vec::<Value>::new())
        };
        let line_angle = (a + direction.unwrap_or(0) as i64).rem_euclid(360);
        let mut cell_chars = Vec::new();
        let mut groups = BTreeMap::<i64, Vec<&Value>>::new();
        for g in &selected {
            clone_bytes = clone_bytes.saturating_add(json_bytes(g));
            if clone_bytes > rapiddoc_core::resources::limit(8_000_000) {
                return Err(limit());
            }
            if direction == Some(0) {
                groups
                    .entry(g["visual_row"].as_i64().unwrap())
                    .or_default()
                    .push(g);
            }
            if let Some(c) = by_source.get(&g["source_index"].as_i64().unwrap()) {
                clone_bytes = clone_bytes.saturating_add(json_bytes(c));
                if clone_bytes > rapiddoc_core::resources::limit(8_000_000) {
                    return Err(limit());
                }
                cell_chars.push((*c).clone());
            }
        }
        for g in &reading {
            groups
                .entry(g["visual_row"].as_i64().unwrap())
                .or_default()
                .push(g);
        }
        let mut lines = Vec::new();
        for (row, gs) in groups {
            let cs: Vec<_> = gs
                .iter()
                .filter_map(|g| {
                    by_source
                        .get(&g["source_index"].as_i64().unwrap())
                        .map(|c| (*c).clone())
                })
                .collect();
            let bs: Vec<_> = cs
                .iter()
                .map(|c| bbox(&c["bbox"]))
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten()
                .collect();
            if cs.is_empty() || bs.is_empty() {
                continue;
            }
            let text: String = gs.iter().map(|g| g["text"].as_str().unwrap()).collect();
            lines.push(json!({"text":text,"bbox":bs.into_iter().reduce(union).unwrap(),"angle":line_angle,"source_index":row,"visual_row_id":row,"chars":cs,"effective_height":0.,"em_height":0.,"median_glyph_width":null,"font_signature":null,"font_coverage":0.,"dominant_font_weight":null,"leading_emphasis_width":null,"leading_typography_width":null,"paragraph_terminal":false}));
        }
        if lines.len() > 1 {
            for line in &mut lines {
                let m = metrics(line, p)?;
                line.as_object_mut()
                    .unwrap()
                    .extend(m.as_object().unwrap().clone());
            }
        }
        work.charge(cell_chars.len().saturating_mul(kept.len()))?;
        let members = rapiddoc_native::script_geometry::refine::fraction_members(
            &json!({"chars":cell_chars,"tight_bboxes":super::cell_side_map(v.get("tight_bboxes").unwrap_or(&json!({})),&cell_chars),"drawing_lines":kept,"page_size":p,"angle":a}),
        )?;
        output.push(json!({"row":row,"col":col,"glyphs":selected,"chars":cell_chars,"visual_lines":lines,"fraction_members":members}));
    }
    Ok(json!({"fraction_rules":kept,"cells":output}))
}
/// 旧表格专用单行角色：完整脚本分类后抑制偏移词首并聚合同源冲突。
pub(super) fn line_roles(v: &Value) -> Result<(Vec<Value>, Vec<String>), Error> {
    // 旧行级入口先解包并旋转侧映射，不能把解包失败当成“缺少几何”。
    // 低层build_script_features仍保留其自身的容错，不更改基础分类器。
    let angle = v["line"]["angle"].as_i64().unwrap_or(0).rem_euclid(360);
    if let Some(chars) = v["line"]["chars"].as_array() {
        for c in chars {
            let id = c["char_idx"]
                .as_i64()
                .or_else(|| c["char_idx"].as_bool().map(i64::from));
            for (key, size) in [("tight_bboxes", 4), ("origins", 2)] {
                let Some(value) = super::source_value(&v[key], id).filter(|v| !v.is_null()) else {
                    continue;
                };
                let values = if let Some(a) = value.as_array() {
                    if a.len() != size {
                        return Err(invalid());
                    }
                    a.clone()
                } else if let Some(s) = value.as_str() {
                    if s.chars().take(size + 1).count() != size {
                        return Err(invalid());
                    }
                    s.chars().map(|c| json!(c.to_string())).collect()
                } else {
                    return Err(invalid());
                };
                if values.len() != size {
                    return Err(invalid());
                }
                let components: &[usize] = match (size, angle) {
                    (2, 90) => &[0],
                    (2, 270) => &[1],
                    (2, 180) => &[0, 1],
                    (4, 90) => &[0, 2],
                    (4, 270) => &[1, 3],
                    (4, 180) => &[0, 1, 2, 3],
                    _ => &[],
                };
                if components
                    .iter()
                    .any(|&i| !values[i].is_number() && !values[i].is_boolean())
                {
                    return Err(invalid());
                }
            }
        }
    }
    let mut result = rapiddoc_native::script_geometry::refine::classify_line(v)?;
    let Value::Array(chars) = result["chars"].take() else {
        unreachable!()
    };
    let mut roles: Vec<String> =
        serde_json::from_value(result["roles"].clone()).map_err(|_| invalid())?;
    let alpha = |i: usize| {
        let s = spatial::raw_string(chars[i].get("char"), false);
        !s.is_empty() && s.chars().all(rapiddoc_native::text_boundary::python_alpha)
    };
    let mut start = 0;
    while start < chars.len() {
        if !alpha(start) {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < chars.len() && alpha(end) {
            end += 1;
        }
        if end - start >= 2
            && roles[start] != "body"
            && roles[start + 1..end].iter().all(|r| r == "body")
        {
            roles[start] = "body".into();
        }
        start = end;
    }
    Ok((chars, roles))
}
pub(super) fn accumulate_roles(
    output: &mut BTreeMap<i64, String>,
    chars: &[Value],
    roles: Vec<String>,
) {
    for (c, r) in chars.iter().zip(roles) {
        if r == "body" {
            continue;
        }
        if let Some(id) = c["char_idx"]
            .as_i64()
            .or_else(|| c["char_idx"].as_bool().map(i64::from))
        {
            let role = if output.get(&id).is_none_or(|p| p == &r) {
                r
            } else {
                "body".into()
            };
            output.insert(id, role);
        }
    }
}
pub(super) fn roles_value(output: BTreeMap<i64, String>) -> Value {
    json!(output
        .into_iter()
        .filter(|(_, r)| r != "body")
        .map(|(i, r)| (i.to_string(), r))
        .collect::<BTreeMap<_, _>>())
}
pub fn classify_cell_line(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let (chars, roles) = line_roles(v)?;
    let mut output = BTreeMap::new();
    accumulate_roles(&mut output, &chars, roles);
    Ok(roles_value(output))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn clone_byte_counter_keeps_finite_limits_and_skips_unlimited_scope() {
        let value = json!({"text":"中文\n\"\\","number":1.25,"chars":[1,2,3]});
        let expected = serde_json::to_vec(&value).unwrap().len();
        assert_eq!(json_bytes(&value), expected);
        {
            let _scope = rapiddoc_core::resources::python_compatible();
            assert_eq!(json_bytes(&value), 0);
        }
        assert_eq!(json_bytes(&value), expected);
    }
    #[test]
    fn old_single_line_cell_html_chains() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../build/migration");
        let f: Value = serde_json::from_slice(
            &std::fs::read(root.join("table-script-preparation.json")).unwrap(),
        )
        .unwrap();
        let extra: Value = serde_json::from_slice(
            &std::fs::read(root.join("table-script-preparation-extra.json")).unwrap(),
        )
        .unwrap();
        let prepared: Vec<_> = f["preparation"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| prepare(&c["request"]).unwrap())
            .collect();
        for c in extra["chains"].as_array().unwrap() {
            let index = c["preparation_index"].as_u64().unwrap() as usize;
            let ci = c["cell_index"].as_u64().unwrap() as usize;
            let v = &f["preparation"][index]["request"];
            let cell = &prepared[index]["cells"][ci];
            let roles=classify_cell_line(&json!({"line":cell["visual_lines"][0],"page_size":v["table_input"]["page_size"],"tight_bboxes":v["tight_bboxes"],"origins":c["origins"],"fraction_members":cell["fraction_members"]})).unwrap();
            assert_eq!(roles, c["expected_roles"], "{}", c["label"]);
            let mut chars = serde_json::Map::new();
            for (i, q) in v["table_input"]["chars"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
            {
                if let Some(id) = q.get("char_idx").map_or(Some(i as i64), Value::as_i64) {
                    chars.entry(id.to_string()).or_insert(q.clone());
                }
            }
            let source = v["result"]["cells"]
                .as_array()
                .unwrap()
                .iter()
                .find(|s| s["row"] == cell["row"] && s["col"] == cell["col"])
                .unwrap();
            let html=super::super::render_cell(&json!({"cell":source,"glyphs":cell["glyphs"],"median_height":v["result"]["text"]["median_glyph_height"],"roles":roles,"chars_by_source":chars,"tight_bboxes":v["tight_bboxes"],"origins":c["origins"],"preserve_font_styles":c["preserve_font_styles"]})).unwrap();
            assert_eq!(html, c["expected_html"].as_str().unwrap(), "{}", c["label"]);
        }
    }
    #[test]
    fn old_table_script_preparation() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/table-script-preparation.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/table-script-preparation-extra.json");
        let extra: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        assert_eq!(
            f["casefold"],
            serde_json::from_str::<Value>(include_str!("font_casefold.json")).unwrap()
        );
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/table-typography-canonical-em.json");
        let canonical: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in canonical["cases"].as_array().unwrap() {
            assert_eq!(
                measure(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
        for c in extra["metrics"].as_array().unwrap() {
            assert_eq!(
                measure(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
        for (key, op) in [
            ("metrics", measure as fn(&Value) -> Result<Value, Error>),
            ("preparation", prepare),
        ] {
            for c in f[key].as_array().unwrap() {
                assert_eq!(op(&c["request"]).unwrap(), c["expected"], "{}", c["label"]);
            }
        }
    }
}
