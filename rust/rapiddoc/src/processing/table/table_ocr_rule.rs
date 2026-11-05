//! Python docvortex_ocr policy with native raster preparation/recovery.
use super::{inline_output, table_native_engine as engine, table_native_text as text};
use crate::legacy_page::invalid;
use rapiddoc_core::Error;
use rapiddoc_inference::table_raster::{self, Rule, Word};
use serde_json::{json, Value};
fn rect(v: &Value) -> Result<[f64; 4], Error> {
    serde_json::from_value(v.clone()).map_err(|_| invalid())
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.
    } else {
        v[n / 2]
    }
}
fn frame(cells: &[Value], rules: &[Rule]) -> Result<bool, Error> {
    if cells.is_empty() {
        return Ok(false);
    }
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for c in cells {
        let a = rect(&c["bbox"])?;
        b = [
            b[0].min(a[0]),
            b[1].min(a[1]),
            b[2].max(a[2]),
            b[3].max(a[3]),
        ];
    }
    for (horizontal, coord, start, end) in [
        (true, b[1], b[0], b[2]),
        (true, b[3], b[0], b[2]),
        (false, b[0], b[1], b[3]),
        (false, b[2], b[1], b[3]),
    ] {
        let mut intervals = vec![];
        let mut allowance = 3f64;
        for r in rules {
            if (r.orientation == "horizontal") != horizontal {
                continue;
            }
            let box_ = r.bbox;
            let center = if horizontal {
                (box_[1] + box_[3]) / 2.
            } else {
                (box_[0] + box_[2]) / 2.
            };
            if (center - coord).abs() > 2f64.max(r.width * 1.5) {
                continue;
            }
            let (a, z) = if horizontal {
                (box_[0].max(start), box_[2].min(end))
            } else {
                (box_[1].max(start), box_[3].min(end))
            };
            if z > a {
                intervals.push((a, z));
                allowance = allowance.max(r.width * 2.);
            }
        }
        if intervals.is_empty() {
            return Ok(false);
        }
        intervals.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let mut cursor = start;
        for (a, z) in intervals {
            if a - cursor > allowance {
                return Ok(false);
            }
            cursor = cursor.max(z);
        }
        if end - cursor > allowance {
            return Ok(false);
        }
    }
    Ok(true)
}
pub fn recover(png: &[u8], ocr: &Value) -> Result<Value, Error> {
    let (w, h) = super::table_direction::dimensions(png)?;
    let lists = ocr
        .as_array()
        .filter(|a| a.len() == 3)
        .ok_or_else(invalid)?;
    let boxes = lists[0].as_array().ok_or_else(invalid)?;
    let texts = lists[1].as_array().ok_or_else(invalid)?;
    let scores = lists[2].as_array().ok_or_else(invalid)?;
    if boxes.len() != texts.len() || boxes.len() != scores.len() {
        return Err(invalid());
    }
    let mut words = vec![];
    let mut widths = vec![];
    for (i, box_) in boxes.iter().enumerate() {
        let score = scores[i]
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(invalid)?;
        let points: Vec<[f64; 2]> = serde_json::from_value(box_.clone()).map_err(|_| invalid())?;
        if points.is_empty() || points.iter().flatten().any(|v| !v.is_finite()) {
            return Err(invalid());
        }
        let value = inline_output::unescape_html(texts[i].as_str().ok_or_else(invalid)?);
        let value = value.trim_matches(rapiddoc_native::text_boundary::python_space);
        if value.is_empty() {
            continue;
        }
        if score < 0.5 {
            return Ok(Value::Null);
        }
        let mut b = [
            points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min),
            points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min),
            points
                .iter()
                .map(|p| p[0])
                .fold(f64::NEG_INFINITY, f64::max),
            points
                .iter()
                .map(|p| p[1])
                .fold(f64::NEG_INFINITY, f64::max),
        ];
        if b[0] < -1.
            || b[1] < -1.
            || b[2] > w as f64 + 1.
            || b[3] > h as f64 + 1.
            || b[2] <= b[0]
            || b[3] <= b[1]
        {
            return Err(invalid());
        }
        b = [
            b[0].max(0.),
            b[1].max(0.),
            b[2].min(w as f64),
            b[3].min(h as f64),
        ];
        widths.push((b[2] - b[0]) / value.chars().count().max(1) as f64);
        words.push(Word {
            bbox: b,
            text: value.into(),
            source_index: words.len(),
        });
    }
    if words.is_empty() {
        return Ok(Value::Null);
    }
    let prepared = table_raster::prepare(png, words)?;
    let input = json!({"page_size":[prepared.width,prepared.height],"table_bbox":[0.,0.,prepared.width as f64,prepared.height as f64],"angle":0,
        "chars":prepared.words.iter().map(|w|json!({"bbox":w.bbox,"char":w.text,"char_idx":w.source_index})).collect::<Vec<_>>(),"drawing_lines":prepared.rules,"rectangles":[]});
    let Some(mut txt) = text::prepare(&input)? else {
        return Ok(Value::Null);
    };
    if txt.glyphs.len() != prepared.words.len()
        || txt.glyphs.iter().any(|g| {
            !prepared
                .words
                .iter()
                .any(|w| w.source_index as i64 == g.source_index)
        })
    {
        return Ok(Value::Null);
    }
    txt.median_glyph_width = median(widths).max(0.1);
    for row in &mut txt.rows {
        let glyphs = row
            .glyph_ids
            .iter()
            .map(|i| {
                txt.glyphs
                    .iter()
                    .find(|g| g.glyph_id == *i)
                    .unwrap()
                    .clone()
            })
            .collect::<Vec<_>>();
        row.tokens = text::tokens(&glyphs, txt.median_glyph_width, txt.median_glyph_height);
    }
    let result = engine::recover_prepared(&input, txt)?;
    if result.is_null() || result["source"] != "vector_grid" {
        return Ok(Value::Null);
    }
    let cells = result["cells"].as_array().ok_or_else(invalid)?;
    if !frame(cells, &prepared.rules)? {
        return Ok(Value::Null);
    }
    for w in &prepared.words {
        let b = w.bbox;
        let area = (b[2] - b[0]) * (b[3] - b[1]);
        let mut owners = 0;
        for c in cells {
            let a = rect(&c["bbox"])?;
            if (b[2].min(a[2]) - b[0].max(a[0])).max(0.) * (b[3].min(a[3]) - b[1].max(a[1])).max(0.)
                / area
                >= 0.8
            {
                owners += 1;
            }
        }
        if owners != 1 {
            return Ok(Value::Null);
        }
    }
    Ok(
        json!({"html":result["html"],"confidence":result["confidence"],"source":result["source"],"result":result,"deskew_angle":prepared.angle}),
    )
}
