//! 旧异常CJK图形尺度校准和堆叠柱系列核心；不替代完整强图形仲裁。
use crate::legacy_page::{invalid, limit, validate_tree, Work};
use rapiddoc_core::Error;
use rapiddoc_native::{python_decimal, text_boundary::python_space};
use serde_json::{json, Value};
type B = [f64; 4];
fn num(v: &Value) -> Result<f64, Error> {
    let n = v.as_f64().filter(|n| n.is_finite()).ok_or_else(invalid)?;
    if n.abs() > 1e9 {
        return Err(limit());
    }
    Ok(n)
}
fn array(v: &Value, max: usize) -> Result<&[Value], Error> {
    let a = v.as_array().ok_or_else(invalid)?;
    let max = if max > 8 {
        crate::legacy_page::budget::records(max)
    } else {
        max
    };
    if a.len() > max {
        return Err(limit());
    }
    Ok(a)
}
fn optional<'a>(v: &'a Value, k: &str, max: usize) -> Result<&'a [Value], Error> {
    v.get(k).map_or(Ok(&[]), |v| array(v, max))
}
fn bbox(v: &Value) -> Result<B, Error> {
    let a = array(v, 4)?;
    if a.len() != 4 {
        return Err(invalid());
    }
    Ok([num(&a[0])?, num(&a[1])?, num(&a[2])?, num(&a[3])?])
}
fn boolean(v: &Value, k: &str) -> Result<bool, Error> {
    v.get(k)
        .map_or(Ok(false), |v| v.as_bool().ok_or_else(invalid))
}
fn median(mut a: Vec<f64>) -> f64 {
    a.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = a.len();
    if n % 2 == 1 {
        a[n / 2]
    } else {
        (a[n / 2 - 1] + a[n / 2]) / 2.
    }
}
fn union(a: B, b: B) -> B {
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ]
}
fn overlap(a: B, b: B) -> f64 {
    let short = (a[2] - a[0]).min(b[2] - b[0]);
    if short > 0. {
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.) / short
    } else {
        0.
    }
}
fn numeric(s: &str) -> bool {
    let cs: Vec<_> = s.trim_matches(python_space).chars().collect();
    let mut i = 0;
    while i < cs.len() && python_decimal(cs[i]).is_some() {
        i += 1
    }
    if i == 0 {
        return false;
    }
    if i < cs.len() && matches!(cs[i], '.' | ',') {
        i += 1;
        let start = i;
        while i < cs.len() && python_decimal(cs[i]).is_some() {
            i += 1
        }
        if i == start {
            return false;
        }
    }
    i == cs.len()
}
struct Path {
    b: B,
    segments: i64,
    fill: bool,
    bars: Vec<B>,
}
pub(super) fn measure_with_work(
    source: &Value,
    fallback: f64,
    work: &mut Work,
) -> Result<(f64, bool), Error> {
    // 两个旧判定的锚点条件不同：graphic_em只要求tight框，inflated还要求loose框。
    let (mut widths, mut inks, mut strict_widths, mut loose_heights, mut strict_inks) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for c in optional(source, "chars", 20000)? {
        c.as_object().ok_or_else(invalid)?;
        let s = c
            .get("char")
            .map_or(Ok(""), |v| v.as_str().ok_or_else(invalid))?;
        work.charge(s.len() + 1)?;
        let mut chars = s.chars();
        let cjk = chars
            .next()
            .is_some_and(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
            && chars.next().is_none();
        if !cjk {
            continue;
        }
        let Some(tight) = c
            .get("tight_bbox")
            .filter(|v| !v.is_null() && v.as_array().is_none_or(|v| !v.is_empty()))
        else {
            continue;
        };
        let tight = bbox(tight)?;
        let loose = match c.get("bbox") {
            None | Some(Value::Null) => None,
            Some(Value::Array(a)) if a.is_empty() => None,
            Some(v) => Some(bbox(v)?),
        };
        let Some(b) = loose else {
            return Err(invalid());
        };
        widths.push(b[2] - b[0]);
        inks.push(tight[3] - tight[1]);
        strict_widths.push(b[2] - b[0]);
        loose_heights.push(b[3] - b[1]);
        strict_inks.push(tight[3] - tight[1]);
    }
    let count = widths.len();
    work.charge(count * log(count) * 5)?;
    let graphic_em = if count >= 20 {
        let w = median(widths);
        let ink = median(inks);
        if w > 0. && fallback >= 2.2 * w && 0.5 * w <= ink && ink <= 1.4 * w {
            w
        } else {
            fallback
        }
    } else {
        fallback
    };
    let inflated = if strict_widths.len() >= 20 {
        let w = median(strict_widths);
        let loose = median(loose_heights);
        let ink = median(strict_inks);
        w > 0. && loose >= 2.2 * w && 0.5 * w <= ink && ink <= 1.4 * w
    } else {
        false
    };
    Ok((graphic_em, inflated))
}
pub fn detect(v: &Value) -> Result<Value, Error> {
    detect_with_work(v, &mut Work(0))
}
pub(crate) fn detect_with_work(v: &Value, work: &mut Work) -> Result<Value, Error> {
    validate_tree(v)?;
    detect_source(&v["source"], v, work)
}
pub(super) fn detect_source(source: &Value, v: &Value, work: &mut Work) -> Result<Value, Error> {
    crate::legacy_page::validate_source_request(source, v)?;
    source.as_object().ok_or_else(invalid)?;
    let fallback = num(&v["fallback_em"])?;
    let em = num(&v["em"])?;
    let (graphic_em, inflated) = measure_with_work(source, fallback, work)?;
    let mut paths = Vec::new();
    let mut total_bars = 0;
    for p in optional(source, "path_infos", 5000)? {
        let bars = optional(p, "rectangle_bboxes", 20000)?
            .iter()
            .map(bbox)
            .collect::<Result<Vec<_>, _>>()?;
        total_bars += bars.len();
        if total_bars > rapiddoc_core::resources::limit(20000) {
            return Err(limit());
        }
        paths.push(Path {
            b: bbox(&p["bbox"])?,
            segments: p["segment_count"].as_i64().ok_or_else(invalid)?,
            fill: boolean(p, "fill_visible")?,
            bars,
        });
    }
    let lines = optional(source, "lines", 4096)?
        .iter()
        .map(|l| {
            let text = l["text"].as_str().ok_or_else(invalid)?;
            work.charge(text.len() + 1)?;
            if text.chars().count() > rapiddoc_core::resources::limit(20000) {
                return Err(limit());
            }
            Ok((bbox(&l["bbox"])?, numeric(text)))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let rules = optional(source, "drawing_lines", 5000)?
        .iter()
        .map(|l| {
            Ok((
                bbox(&l["bbox"])?,
                l["orientation"].as_str().ok_or_else(invalid)? == "horizontal",
            ))
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let mut output = Vec::new();
    if inflated {
        for path in &paths {
            let bars = &path.bars;
            if !path.fill || bars.len() < 4 {
                continue;
            }
            work.charge(bars.len() * (log(bars.len()) + 4))?;
            if !bars
                .iter()
                .all(|b| 0.25 * em <= b[2] - b[0] && b[2] - b[0] <= 2. * em)
            {
                continue;
            }
            let baseline = median(bars.iter().map(|b| b[3]).collect());
            let mut order: Vec<_> = bars.iter().collect();
            order.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
            if (bars
                .iter()
                .filter(|b| (b[3] - baseline).abs() <= 0.2 * em)
                .count() as f64)
                < 0.75 * bars.len() as f64
                || !order.windows(2).all(|p| p[1][0] - p[0][2] >= 0.2 * em)
                || bars.iter().map(|b| b[3] - b[1]).reduce(f64::max).unwrap() < 3. * em
            {
                continue;
            }
            let mut bounds = bars.iter().copied().reduce(union).unwrap();
            work.charge(
                bars.len()
                    .checked_mul(total_bars)
                    .and_then(|v| v.checked_add(paths.len() + lines.len() + rules.len()))
                    .ok_or_else(limit)?,
            )?;
            for other in &paths {
                if other.fill {
                    for &b in &other.bars {
                        if bars.iter().any(|bar| {
                            (b[0] - bar[0]).abs() <= 0.1 * em
                                && (b[2] - bar[2]).abs() <= 0.1 * em
                                && b[3] <= baseline + 0.2 * em
                        }) {
                            bounds = union(bounds, b)
                        }
                    }
                }
            }
            let ticks: Vec<_> = lines
                .iter()
                .filter(|(b, n)| {
                    *n && 0. <= bounds[0] - b[2]
                        && bounds[0] - b[2] <= 4. * em
                        && bounds[1] - 2. * em <= (b[1] + b[3]) / 2.
                        && (b[1] + b[3]) / 2. <= bounds[3] + em
                })
                .map(|(b, _)| *b)
                .collect();
            if ticks.len() < 3 {
                continue;
            }
            let axes: Vec<_> = rules
                .iter()
                .filter(|(b, h)| {
                    *h && overlap(*b, bounds) >= 0.9
                        && bounds[1] - 2. * em <= b[1]
                        && b[1] <= bounds[3] + 0.2 * em
                })
                .map(|(b, _)| *b)
                .collect();
            let labels: Vec<_> = paths
                .iter()
                .filter(|p| {
                    p.fill
                        && p.segments >= 8
                        && bounds[0] - 2. * em <= p.b[0]
                        && p.b[0] <= bounds[2] + em
                        && baseline <= p.b[1]
                        && p.b[1] <= baseline + 4. * em
                        && p.b[2] - p.b[0] <= 3. * em
                        && p.b[3] - p.b[1] <= 4. * em
                })
                .map(|p| p.b)
                .collect();
            output.push(
                ticks
                    .into_iter()
                    .chain(axes)
                    .chain(labels)
                    .fold(bounds, union),
            );
        }
    }
    let out = json!({"graphic_em":graphic_em,"inflated_cjk_metrics":inflated,"stacked_column_bboxes":output});
    if crate::legacy_page::budget::output_exceeds_limit(
        &out,
        rapiddoc_core::resources::limit(8_000_000),
    ) {
        return Err(limit());
    }
    Ok(out)
}
fn log(n: usize) -> usize {
    if n < 2 {
        1
    } else {
        usize::BITS as usize - (n - 1).leading_zeros() as usize + 1
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_stacked_graphics() {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../build/migration/native-stacked-graphics.json");
        let f: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
        for c in f["cases"].as_array().unwrap() {
            assert_eq!(
                detect(&c["request"]).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
    }
    #[test]
    fn shared_budget() {
        let q = json!({"source":{"chars":[{"char":"中","bbox":[0,0,10,30],"tight_bbox":[0,0,10,10]}]},"em":10,"fallback_em":30});
        assert!(detect_with_work(&q, &mut Work(9_999_999)).is_err());
        assert!(detect(&q).is_ok());
    }
}
