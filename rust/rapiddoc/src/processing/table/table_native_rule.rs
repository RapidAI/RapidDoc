//! 旧DocVortex表格捷径的守卫/接纳，以及已有原生候选的阈值/拓扑仲裁。
//! 不生成原生候选，不调用实验网格恢复器替代旧恢复算法。
use super::coordinates;
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use serde_json::{json, Value};

fn flag(v: &Value, key: &str, default: bool) -> Result<bool, Error> {
    match v.get(key) {
        None => Ok(default),
        Some(x) => x.as_bool().ok_or_else(invalid),
    }
}
fn rect(v: &Value) -> Result<[f64; 4], Error> {
    let b = coordinates(v)?;
    if b.iter().any(|x| x.abs() > 1e9) {
        return Err(invalid());
    }
    Ok(b)
}
pub fn plan(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let config = &v["config"];
    let scale = v["scale"]
        .as_f64()
        .filter(|x| x.is_finite() && *x > 0. && *x <= 1000.)
        .ok_or_else(invalid)?;
    let poly = v["table"]["poly"]
        .as_array()
        .filter(|a| a.len() == 8)
        .ok_or_else(invalid)?;
    let b = rect(&json!([poly[0], poly[1], poly[4], poly[5]]))?;
    let point = b.map(|x| x / scale);
    let reason = if !flag(config, "use_rule_table", true)? {
        Some("disabled")
    } else if config
        .get("rule_table_engine")
        .is_some_and(|x| x != "docvortex")
    {
        Some("other_engine")
    } else if flag(config, "force_ocr", false)? {
        Some("force_ocr")
    } else if flag(v, "ocr_enable", false)? {
        Some("ocr_enabled")
    } else if matches!(v["rotate_label"].as_str(), Some("90" | "180" | "270")) {
        Some("page_rotation")
    } else if !flag(v, "has_source_pdf", false)? {
        Some("missing_pdf")
    } else {
        None
    };
    if let Some(reason) = reason {
        return Ok(json!({"attempt":false,"reason":reason,"table_bbox":point}));
    }
    let list = |key: &str| -> Result<&Vec<Value>, Error> {
        v[key]
            .as_array()
            .filter(|a| a.len() <= rapiddoc_core::resources::limit(1000))
            .ok_or_else(invalid)
    };
    for formula in list("formulas")?.iter().chain(list("checkboxes")?) {
        if let Some(box_) = formula
            .get("bbox")
            .filter(|x| !x.is_null() && x.as_array().is_none_or(|a| !a.is_empty()))
        {
            let f = rect(box_)?;
            let (cx, cy) = ((f[0] + f[2]) / 2., (f[1] + f[3]) / 2.);
            if b[0] <= cx && cx <= b[2] && b[1] <= cy && cy <= b[3] {
                return Ok(
                    json!({"attempt":false,"reason":"formula_or_checkbox","table_bbox":point}),
                );
            }
        }
    }
    for image in list("images")? {
        if let Some(box_) = image
            .get("bbox")
            .filter(|x| !x.is_null() && x.as_array().is_none_or(|a| !a.is_empty()))
        {
            let i = rect(box_)?;
            if point[2].min(i[2]) > point[0].max(i[0]) && point[3].min(i[3]) > point[1].max(i[1]) {
                return Ok(json!({"attempt":false,"reason":"native_image","table_bbox":point}));
            }
        }
    }
    Ok(json!({"attempt":true,"reason":"eligible","table_bbox":point}))
}
pub fn accept(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let threshold = v["threshold"]
        .as_f64()
        .filter(|x| x.is_finite() && (0. ..=1.).contains(x))
        .ok_or_else(invalid)?;
    let mut table = v["table"].clone();
    let obj = table.as_object_mut().ok_or_else(invalid)?;
    let result = &v["result"];
    if result.is_null() {
        return Ok(json!({"accepted":false,"table":table}));
    }
    let html = result["html"].as_str().ok_or_else(invalid)?;
    let score = result["confidence"]
        .as_f64()
        .filter(|x| x.is_finite())
        .ok_or_else(invalid)?;
    let accepted = !html
        .trim_matches(rapiddoc_native::text_boundary::python_space)
        .is_empty()
        && score >= threshold;
    if accepted {
        obj.remove("layout_image_list");
        obj.insert("html".into(), json!(html));
        obj.insert("rule_table_score".into(), result["confidence"].clone());
        obj.insert("table_parse_method".into(), json!("docvortex_native"));
    }
    Ok(json!({"accepted":accepted,"table":table}))
}

struct Candidate<'a> {
    index: usize,
    value: &'a Value,
    source: &'a str,
    score: f64,
    rows: usize,
    cols: usize,
    topology: Vec<[u64; 4]>,
}
impl Candidate<'_> {
    fn verified(&self, evidence: &str) -> bool {
        self.source == "vector_grid"
            && self.score >= 0.95
            && self.value["issues"].as_array().is_some_and(|a| {
                a.iter().any(|x| x == evidence)
                    && a.iter().any(|x| x == "ambiguous_separator_ratio=0.0000")
            })
    }
    fn same_topology(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.topology == other.topology
    }
    fn priority(&self) -> usize {
        match self.source {
            "vector_grid" => 6,
            "sparse_hybrid" => 5,
            "sparse_grid" => 4,
            "key_value" => 3,
            "text_grid" => 2,
            _ => 1,
        }
    }
    fn threshold(&self) -> f64 {
        if matches!(self.source, "sparse_hybrid" | "sparse_multiline") {
            0.98
        } else {
            0.95
        }
    }
}
pub fn select(v: &Value) -> Result<Value, Error> {
    validate_tree(v)?;
    let array = v["candidates"]
        .as_array()
        .filter(|a| a.len() <= rapiddoc_core::resources::limit(1000))
        .ok_or_else(invalid)?;
    let mut candidates = Vec::new();
    let mut count = 0;
    let mut work = Work(0);
    let number = |v: &Value| v.as_f64().filter(|x| x.is_finite()).ok_or_else(invalid);
    for (index, c) in array.iter().enumerate() {
        let source = c["source"]
            .as_str()
            .filter(|s| {
                matches!(
                    *s,
                    "vector_grid"
                        | "sparse_hybrid"
                        | "sparse_grid"
                        | "key_value"
                        | "text_grid"
                        | "sparse_multiline"
                )
            })
            .ok_or_else(invalid)?;
        let dim = |key: &str| {
            c[key]
                .as_u64()
                .filter(|x| *x <= rapiddoc_core::resources::limit(1000))
                .map(|x| x as usize)
                .ok_or_else(invalid)
        };
        let mut topology = Vec::new();
        for cell in c["cells"].as_array().ok_or_else(invalid)? {
            count += 1;
            if count > rapiddoc_core::resources::limit(20000) {
                return Err(limit());
            }
            let mut span = [0; 4];
            for (i, key) in ["row", "col", "rowspan", "colspan"].iter().enumerate() {
                span[i] = cell[*key]
                    .as_u64()
                    .filter(|x| *x <= rapiddoc_core::resources::limit(1000))
                    .ok_or_else(invalid)?;
            }
            topology.push(span);
        }
        topology.sort();
        number(&c["row_stability"])?;
        number(&c["column_stability"])?;
        candidates.push(Candidate {
            index,
            value: c,
            source,
            score: number(&c["score"])?,
            rows: dim("rows")?,
            cols: dim("cols")?,
            topology,
        });
    }
    let remove = flag(v, "remove_undercounted", true)?;
    let kept: Vec<_> = candidates
        .iter()
        .filter(|c| {
            !remove
                || c.source != "vector_grid"
                || c.verified("evidence=line_grid")
                || !candidates.iter().any(|text| {
                    text.source != "vector_grid"
                        && text.value["row_stability"].as_f64().unwrap() >= 0.8
                        && text.value["column_stability"].as_f64().unwrap() >= 0.8
                        && (text.rows as f64 >= (1.5 * c.rows as f64).ceil()
                            || text.cols as f64 >= (1.3 * c.cols as f64).ceil())
                })
        })
        .collect();
    work.charge(candidates.len() * candidates.len())?;
    let mut accepted: Vec<_> = kept
        .iter()
        .copied()
        .filter(|c| c.score >= c.threshold())
        .collect();
    let mut conflict = false;
    for line in &accepted {
        if line.verified("evidence=line_grid") {
            for rect in &accepted {
                work.charge(line.topology.len() + rect.topology.len() + 1)?;
                if rect.verified("evidence=rect_grid") && !line.same_topology(rect) {
                    conflict = true;
                }
            }
        }
    }
    let lines: Vec<_> = accepted
        .iter()
        .filter(|c| c.verified("evidence=line_grid"))
        .collect();
    let selected = if accepted.is_empty() || conflict {
        None
    } else if lines.len() == 1 {
        Some(lines[0].index)
    } else {
        accepted.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then(b.priority().cmp(&a.priority()))
        });
        let best = accepted[0];
        let mut blocked = false;
        for other in &accepted[1..] {
            work.charge(best.topology.len() + other.topology.len() + 1)?;
            if !best.same_topology(other) && best.score - other.score < 0.05 {
                blocked = true;
                break;
            }
        }
        if blocked {
            None
        } else {
            Some(best.index)
        }
    };
    Ok(
        json!({"kept":kept.iter().map(|c|c.index).collect::<Vec<_>>(),"selected":selected,"physical_topology_conflict":conflict}),
    )
}
