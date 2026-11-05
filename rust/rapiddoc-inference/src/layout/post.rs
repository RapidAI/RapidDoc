//! PP-DocLayoutV3 的 rect 模式后处理；不将掩码多边形伪装为矩形等价。
use rapiddoc_core::{Error, ErrorCode};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Region {
    pub cls_id: usize,
    pub label: String,
    pub score: f64,
    pub coordinate: [f64; 4],
    pub order: usize,
}
fn error(message: &str) -> Error {
    Error {
        code: ErrorCode::InvalidInput,
        message: message.into(),
    }
}

/// 单页有序七列框：class/score/xmin/ymin/xmax/ymax/order。最多 1000 个候选。
pub fn rectangles(
    rows: &[[f32; 7]],
    labels: &[String],
    width: usize,
    height: usize,
    threshold: f32,
) -> Result<Vec<Region>, Error> {
    if rows.len() > rapiddoc_core::resources::limit(1000) {
        return Err(Error {
            code: ErrorCode::ResourceLimit,
            message: "版面候选框超过预算".into(),
        });
    }
    if width == 0
        || height == 0
        || width > rapiddoc_core::resources::limit(1_000_000)
        || height > rapiddoc_core::resources::limit(1_000_000)
        || labels.is_empty()
        || labels.len() > rapiddoc_core::resources::limit(1000)
        || !threshold.is_finite()
        || !(0. ..=1.).contains(&threshold)
        || rows
            .iter()
            .flatten()
            .any(|v| !v.is_finite() || v.abs() > 1e9)
    {
        return Err(error("版面后处理配置或数值无效"));
    }
    let mut rows: Vec<_> = rows
        .iter()
        .copied()
        .filter(|r| r[1] > threshold && r[0] > -1.)
        .collect();
    if rows
        .iter()
        .any(|r| r[0].fract() != 0. || r[0] as usize >= labels.len() || !(0. ..=1.).contains(&r[1]))
    {
        return Err(error("版面类别或评分无效"));
    }
    // 相同评分明确选择后出现的框；避免依赖平台排序实现的偶然顺序。
    let mut indices: Vec<_> = (0..rows.len()).collect();
    indices.sort_by(|&a, &b| rows[b][1].total_cmp(&rows[a][1]).then(b.cmp(&a)));
    let mut kept: Vec<usize> = Vec::new();
    for i in indices {
        if kept
            .iter()
            .all(|&j| iou(&rows[i], &rows[j]) < if rows[i][0] == rows[j][0] { 0.6 } else { 0.98 })
        {
            kept.push(i);
        }
    }
    rows = kept.into_iter().map(|i| rows[i]).collect();
    if rows.len() > 1 {
        let image = labels.iter().position(|s| s == "image");
        let area_threshold =
            (width as f64 * height as f64) as f32 * if width > height { 0.82 } else { 0.93 };
        let filtered: Vec<_> = rows
            .iter()
            .copied()
            .filter(|r| {
                Some(r[0] as usize) != image
                    || (r[4].min(width as f32) - r[2].max(0.))
                        * (r[5].min(height as f32) - r[3].max(0.))
                        <= area_threshold
            })
            .collect();
        if !filtered.is_empty() {
            rows = filtered;
        }
    }
    // 默认 large 类别中的外框优先；匹配源实现跨类别的包含淘汰。
    let large = [
        "chart",
        "display_formula",
        "doc_title",
        "inline_formula",
        "paragraph_title",
    ];
    let mut dropped = vec![false; rows.len()];
    for (i, a) in rows.iter().enumerate() {
        for (j, b) in rows.iter().enumerate() {
            if i != j && large.contains(&labels[b[0] as usize].as_str()) && contained(a, b) {
                dropped[i] = true;
            }
        }
    }
    rows = rows
        .into_iter()
        .enumerate()
        .filter_map(|(i, r)| (!dropped[i]).then_some(r))
        .collect();
    rows.sort_by(|a, b| a[6].total_cmp(&b[6]));
    let mut result = Vec::new();
    for (i, r) in rows.into_iter().enumerate() {
        // unclip_ratio=[1,1] 仍保留源 FP32 的中心/半宽运算顺序。
        let w = r[4] - r[2];
        let h = r[5] - r[3];
        let cx = r[2] + w / 2.;
        let cy = r[3] + h / 2.;
        let bounds = [
            (cx - w / 2.).max(0.),
            (cy - h / 2.).max(0.),
            (cx + w / 2.).min(width as f32),
            (cy + h / 2.).min(height as f32),
        ];
        if bounds[2] <= bounds[0] || bounds[3] <= bounds[1] {
            continue;
        }
        result.push(Region {
            cls_id: r[0] as usize,
            label: labels[r[0] as usize].clone(),
            score: r[1] as f64,
            coordinate: bounds.map(f64::from),
            order: i + 1,
        });
    }
    Ok(result)
}

/// 源 NMS 使用包含端点的 +1 像素面积。
fn iou(a: &[f32; 7], b: &[f32; 7]) -> f32 {
    let intersection = (a[4].min(b[4]) - a[2].max(b[2]) + 1.).max(0.)
        * (a[5].min(b[5]) - a[3].max(b[3]) + 1.).max(0.);
    let aa = (a[4] - a[2] + 1.) * (a[5] - a[3] + 1.);
    let ba = (b[4] - b[2] + 1.) * (b[5] - b[3] + 1.);
    let union = aa + ba - intersection;
    if union > 0. {
        intersection / union
    } else {
        0.
    }
}

/// 包含关系使用连续面积且阈值为 90%，与 NMS 的面积约定分离。
fn contained(a: &[f32; 7], b: &[f32; 7]) -> bool {
    let area = (a[4] - a[2]) * (a[5] - a[3]);
    let intersection =
        (a[4].min(b[4]) - a[2].max(b[2])).max(0.) * (a[5].min(b[5]) - a[3].max(b[3])).max(0.);
    area > 0. && intersection / area >= 0.9
}
