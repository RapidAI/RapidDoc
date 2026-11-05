//! 旧页源图片区域裁决：先全文水印统计，再逐页背景/标题白边过滤。
//! 输入是已提取的图片证据；本模块不解码图片、不重新识别图中文字。
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::text_boundary::python_space;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

type Box4 = [f64; 4];
fn number(v: &Value) -> Result<f64, Error> {
    let n = v.as_f64().filter(|n| n.is_finite()).ok_or_else(invalid)?;
    if n.abs() > 1e9 {
        return Err(limit());
    }
    Ok(n)
}
fn bbox(v: &Value) -> Result<Box4, Error> {
    let a = v.as_array().filter(|a| a.len() == 4).ok_or_else(invalid)?;
    Ok([
        number(&a[0])?,
        number(&a[1])?,
        number(&a[2])?,
        number(&a[3])?,
    ])
}
fn array(v: &Value, maximum: usize) -> Result<&[Value], Error> {
    let a = v.as_array().ok_or_else(invalid)?;
    let maximum = if maximum > 8 {
        crate::legacy_page::budget::records(maximum)
    } else {
        maximum
    };
    if a.len() > maximum {
        return Err(limit());
    }
    Ok(a)
}
struct Image<'a> {
    raw: &'a Value,
    b: Box4,
    fingerprint: Option<&'a str>,
    smooth: bool,
    blank: Option<Box4>,
}
impl Image<'_> {
    fn large(&self, page: [f64; 2]) -> bool {
        // _coerce_bbox规范化只用于面积判定，最终输出仍是原始bbox。
        let area = (self.b[2] - self.b[0]).abs() * (self.b[3] - self.b[1]).abs();
        let page_area = page[0].max(0.) * page[1].max(0.);
        area > 0. && page_area > 0. && area / page_area >= 0.08
    }
}
struct Line {
    b: Box4,
    height: f64,
    weight: f64,
    upright: bool,
}
struct Page<'a> {
    size: [f64; 2],
    images: Vec<Image<'a>>,
    lines: Vec<Line>,
    text_count: usize,
}
fn page<'a>(v: &'a Value, work: &mut Work) -> Result<Page<'a>, Error> {
    v.as_object().ok_or_else(invalid)?;
    let size = array(&v["page_size"], 2)?;
    if size.len() != 2 {
        return Err(invalid());
    }
    let size = [number(&size[0])?, number(&size[1])?];
    let mut images = Vec::new();
    for raw in array(&v["image_infos"], 5000)? {
        raw.as_object().ok_or_else(invalid)?;
        let fingerprint = match raw.get("fingerprint") {
            None | Some(Value::Null) => None,
            Some(Value::String(s)) => Some(s.as_str()),
            _ => return Err(invalid()),
        };
        let smooth = raw
            .get("smooth_background")
            .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))?;
        let blank = match raw.get("blank_top_bbox") {
            None | Some(Value::Null) => None,
            Some(v) => Some(bbox(v)?),
        };
        images.push(Image {
            raw,
            b: bbox(&raw["bbox"])?,
            fingerprint,
            smooth,
            blank,
        });
    }
    let mut lines = Vec::new();
    let mut text_count = 0;
    for raw in array(&v["lines"], 4096)? {
        raw.as_object().ok_or_else(invalid)?;
        let text = raw["text"].as_str().ok_or_else(invalid)?;
        work.charge(text.len())?;
        if text.chars().count() > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        text_count += text.trim_matches(python_space).chars().count();
        let angle = raw.get("angle").map_or(Ok(0.), number)?;
        let height = raw.get("effective_height").map_or(Ok(0.), number)?;
        let weight = match raw.get("dominant_font_weight") {
            None | Some(Value::Null) => 400.,
            Some(v) => {
                let n = number(v)?;
                if n == 0. {
                    400.
                } else {
                    n
                }
            }
        };
        lines.push(Line {
            b: bbox(&raw["bbox"])?,
            height,
            weight,
            upright: angle == 0.,
        });
    }
    work.charge(images.len() + lines.len())?;
    Ok(Page {
        size,
        images,
        lines,
        text_count,
    })
}
fn key(b: Box4) -> [u64; 4] {
    // Python tuple数值相等，包括-0.0和0.0。
    b.map(|n| if n == 0. { 0 } else { n.to_bits() })
}
fn title(line: &Line, box_: Box4, trimmed: Box4, body: f64) -> bool {
    if !line.upright || line.height < 1.4 * body || line.weight < 600. {
        return false;
    }
    let overlap = (line.b[2].min(box_[2]) - line.b[0].max(box_[0])).max(0.);
    let shorter = (line.b[2] - line.b[0]).min(box_[2] - box_[0]);
    shorter > 0.
        && overlap / shorter >= 0.5
        && line.b[1] < trimmed[1]
        && box_[1] + 0.1 * body < line.b[3]
        && line.b[3] <= trimmed[1]
}
pub fn prepare(v: &Value) -> Result<Value, Error> {
    prepare_with_work(v, &mut Work(0))
}
pub(crate) fn prepare_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    validate_tree(v)?;
    let mut pages = Vec::new();
    for p in array(&v["pages"], 10000)? {
        pages.push(page(p, work)?);
    }
    let mut by_fingerprint: BTreeMap<&str, BTreeSet<usize>> = BTreeMap::new();
    // 所有原始图片都参与全文统计；不得先删掉某页的smooth背景。
    for (i, page) in pages.iter().enumerate() {
        for image in &page.images {
            if let Some(fp) = image.fingerprint.filter(|_| image.large(page.size)) {
                work.charge(fp.len() + 1)?;
                by_fingerprint.entry(fp).or_default().insert(i);
            }
        }
    }
    let watermarks: BTreeSet<&str> = by_fingerprint
        .into_iter()
        .filter_map(|(fp, pages)| (pages.len() >= 3).then_some(fp))
        .collect();
    let mut result = Vec::new();
    for page in pages {
        let mut heights: Vec<_> = page
            .lines
            .iter()
            .filter(|l| l.upright && l.height > 0.)
            .map(|l| l.height)
            .collect();
        heights.sort_by(f64::total_cmp);
        let body = if heights.is_empty() {
            None
        } else {
            let n = heights.len();
            Some(if n % 2 == 1 {
                heights[n / 2]
            } else {
                (heights[n / 2 - 1] + heights[n / 2]) / 2.
            })
        };
        let filter_smooth = page.lines.len() >= 3 && page.text_count >= 60;
        let candidates: Vec<_> = page
            .images
            .iter()
            .filter(|i| !(filter_smooth && i.smooth))
            .collect();
        // 此映射包含未过水印过滤的候选，重复bbox最后一项覆盖前一项（含None）。
        let evidence: BTreeMap<_, _> = candidates.iter().map(|i| (key(i.b), *i)).collect();
        let mut boxes = Vec::new();
        for image in candidates {
            if image.fingerprint.is_some_and(|fp| watermarks.contains(fp)) && image.large(page.size)
            {
                continue;
            }
            let proof = evidence[&key(image.b)];
            let trimmed = if let (Some(body), Some(b)) = (body, proof.blank) {
                work.charge(page.lines.len())?;
                page.lines
                    .iter()
                    .any(|l| title(l, image.b, b, body))
                    .then_some(&proof.raw["blank_top_bbox"])
            } else {
                None
            };
            boxes.push(trimmed.unwrap_or(&image.raw["bbox"]).clone());
        }
        result.push(boxes);
    }
    let result = json!({"watermark_fingerprints":watermarks,"image_bboxes":result});
    if crate::legacy_page::budget::output_exceeds_limit(
        &result,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(result)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_image_region_decisions() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-image-regions.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            let got = prepare(&c["request"]).unwrap_or_else(|e| panic!("{}: {e:?}", c["label"]));
            assert_eq!(got, c["expected"], "{}", c["label"]);
        }
    }
    #[test]
    fn preserves_callers_budget() {
        let q = json!({"pages":[{"page_size":[400,600],"lines":[],"image_infos":[{"bbox":[0,0,400,600],"fingerprint":"a"}]}]});
        assert_eq!(
            prepare_with_work(&q, &mut Work(9_999_999))
                .unwrap_err()
                .code,
            rapiddoc_core::ErrorCode::ResourceLimit
        );
        assert!(prepare(&q).is_ok());
    }
}
