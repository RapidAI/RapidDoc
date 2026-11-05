//! 旧DocVortex完整字符去重适配；固定vendor源码不修改。
use super::{error, Character};
use docvortex_core::text_snapshot::{self as kernel, Angle, Properties};
use rapiddoc_core::{Error, ErrorCode, Limits};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::Path,
    sync::LazyLock,
};
use unicode_normalization::UnicodeNormalization;

#[derive(Deserialize)]
struct Unicode {
    version: String,
    spaces: Vec<u32>,
    han: Vec<[u32; 2]>,
    radicals: HashMap<u32, String>,
}
static UNICODE: LazyLock<Unicode> = LazyLock::new(|| {
    let data: Unicode =
        serde_json::from_str(include_str!("unicode.json")).expect("frozen Unicode policy");
    assert_eq!(data.version, "16.0.0");
    data
});
fn space(c: char) -> bool {
    UNICODE.spaces.binary_search(&(c as u32)).is_ok()
}
fn han(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(c) = chars.next() else {
        return false;
    };
    if chars.next().is_some() {
        return false;
    }
    let code = c as u32;
    UNICODE.han.iter().any(|r| r[0] <= code && code <= r[1])
}
fn canonical(s: &str) -> String {
    if s.chars().count() != 1 {
        return s.into();
    }
    let code = s.chars().next().unwrap() as u32;
    if (0x2e80..=0x2fdf).contains(&code) {
        UNICODE
            .radicals
            .get(&code)
            .cloned()
            .unwrap_or_else(|| s.into())
    } else {
        s.nfkc().collect()
    }
}
fn property(s: &str) -> Properties {
    Properties {
        space: !s.is_empty() && s.chars().all(space),
        canonical: canonical(s),
        han: han(s),
    }
}
fn bits(n: f64) -> u64 {
    if n == 0. {
        0
    } else {
        n.to_bits()
    }
}
fn angle(n: f64) -> Angle {
    Angle {
        cos: n.cos(),
        sin: n.sin(),
        rounded: bits(format!("{n:.3}").parse().expect("finite angle")),
        bucket: (n * 1000.).round_ties_even() as i64,
    }
}
fn invalid(message: &str) -> Error {
    error(ErrorCode::InvalidInput, message)
}
fn limit(message: &str) -> Error {
    error(ErrorCode::ResourceLimit, message)
}
fn kernel_error(e: kernel::SnapshotError) -> Error {
    invalid(&format!("字符去重内核拒绝输入：{e:?}"))
}

/// 预检真正会访问的九邻桶，保留旧64候选桶饱和规则。保护复用内核的临时候选分配。
fn budget(
    glyphs: &[kernel::Glyph],
    angles: &HashMap<u64, Angle>,
    properties: &HashMap<String, Properties>,
) -> Result<(), Error> {
    let mut signatures = HashMap::new();
    let mut buckets: HashMap<(usize, i64, i64), Vec<usize>> = HashMap::new();
    let (mut work, mut hidden, mut visible) = (0usize, 0usize, 0usize);
    for g in glyphs {
        let c = &g.chars[0];
        if g.bbox.is_none() || g.text.is_empty() || kernel::is_space(&g.text, properties) {
            continue;
        }
        if c.mode == Some(3) {
            hidden += 1;
        }
        if c.mode.is_some_and(|v| [0, 1, 2, 4, 5, 6].contains(&v)) && c.visible.unwrap_or(true) {
            visible += 1;
        }
        if c.object.is_none() || c.origin.is_none() || !c.mode.is_some_and(|v| (0..=6).contains(&v))
        {
            continue;
        }
        let key = (
            g.text.as_str(),
            c.font,
            angles[&g.angle.to_bits()].rounded,
            c.mode,
            c.visible,
        );
        let next = signatures.len();
        let sig = *signatures.entry(key).or_insert(next);
        let b = g.bbox.unwrap();
        let (x, y) = ((b[0] / 2.5).floor() as i64, (b[1] / 2.5).floor() as i64);
        for xx in x - 1..=x + 1 {
            for yy in y - 1..=y + 1 {
                if let Some(items) = buckets.get(&(sig, xx, yy)) {
                    if items.len() < 64 {
                        work += items.len();
                    }
                }
            }
        }
        if work > rapiddoc_core::resources::limit(2_000_000) {
            return Err(limit("字符去重绘制候选工作量超限"));
        }
        let items = buckets.entry((sig, x, y)).or_default();
        if items.len() < 64 {
            items.push(0);
        }
    }
    if hidden
        .checked_mul(visible)
        .is_none_or(|n| n > rapiddoc_core::resources::limit(20_000_000))
    {
        return Err(limit("隐藏字符配对工作量超限"));
    }
    Ok(())
}

/// 输入为已完成代理对恢复的普通字符记录；不重新读取PDF，不过滤页面可见区域。
pub fn deduplicate(chars: Vec<Character>) -> Result<Vec<Character>, Error> {
    deduplicate_bounded(chars, 5000, 20_000)
}
fn deduplicate_bounded(
    chars: Vec<Character>,
    records: usize,
    sources: usize,
) -> Result<Vec<Character>, Error> {
    if chars.len() > records {
        return Err(limit("完整字符去重上限5000条"));
    }
    let mut seen = HashSet::new();
    let (mut units, mut source_count) = (0usize, 0usize);
    for c in &chars {
        if c.char.len() > rapiddoc_core::resources::limit(256) {
            return Err(limit("单条字符映射超过256字节"));
        }
        units = units
            .checked_add(c.char.len() + c.font.name.len())
            .ok_or_else(|| limit("字符内容超限"))?;
        source_count = source_count
            .checked_add(c.source_indices.len())
            .ok_or_else(|| limit("字符来源超限"))?;
        if units > rapiddoc_core::resources::limit(8_000_000) || source_count > sources {
            return Err(limit("字符内容或来源超限"));
        }
        if !seen.insert(c.char_idx) || c.source_indices.is_empty() {
            return Err(invalid("字符索引必须唯一且来源不能为空"));
        }
        if c.bbox
            .iter()
            .chain(c.origin.iter().flatten())
            .chain(c.loose_bbox.iter().flatten().flatten())
            .chain(c.tight_bbox.iter().flatten().flatten())
            .any(|v| !v.is_finite() || v.abs() > 1e15)
            || !c.rotation.is_finite()
            || !c.font.size.is_finite()
            || !c.writing_angle.is_finite()
            || c.writing_angle.abs() > 1e9
        {
            return Err(invalid("字符几何或字体数值无效"));
        }
    }
    let mut properties = HashMap::new();
    let mut fonts = HashMap::new();
    let mut records = Vec::with_capacity(chars.len());
    for c in &chars {
        properties
            .entry(c.char.clone())
            .or_insert_with(|| property(&c.char));
        for ch in c.char.chars() {
            let s = ch.to_string();
            properties.entry(s.clone()).or_insert_with(|| property(&s));
        }
        let next = fonts.len();
        let font = *fonts
            .entry((&c.font.name, c.font.flags, bits(c.font.size), c.font.weight))
            .or_insert(next);
        records.push(kernel::Character {
            text: c.char.clone(),
            bbox: c.bbox,
            rotation: c.rotation,
            font,
            index: c.char_idx,
            sources: c.source_indices.clone(),
            code: c.raw_code,
            generated: c.is_generated,
            object: c.text_object_id,
            mode: c.text_render_mode,
            writing_angle: c.writing_angle,
            origin: c.origin,
            loose: c.loose_bbox.flatten(),
            tight: c.tight_bbox.flatten(),
            visible: c.text_is_visible,
        });
    }
    let canonical: Vec<_> = properties.values().map(|p| p.canonical.clone()).collect();
    for s in canonical {
        properties.entry(s.clone()).or_insert_with(|| property(&s));
    }
    let glyphs = kernel::mapping_groups(records, &properties);
    let angles: HashMap<_, _> = glyphs
        .iter()
        .map(|g| (g.angle.to_bits(), angle(g.angle)))
        .collect();
    budget(&glyphs, &angles, &properties)?;
    let glyphs = kernel::collapse_paints(glyphs, &angles, &properties).map_err(kernel_error)?;
    let pairs = kernel::hidden_pairs(&glyphs, &angles, &properties).map_err(kernel_error)?;
    let mut normalized = HashMap::new();
    for (a, b) in &pairs {
        for indices in [a, b] {
            let text: String = indices.iter().map(|&i| glyphs[i].text.as_str()).collect();
            normalized
                .entry(text.clone())
                .or_insert_with(|| text.nfkc().filter(|c| !space(*c)).collect::<String>());
        }
    }
    let normalized_chars: Vec<_> = normalized.values().flat_map(|s| s.chars()).collect();
    for c in normalized_chars {
        let s = c.to_string();
        properties.entry(s.clone()).or_insert_with(|| property(&s));
    }
    let result = kernel::suppress_hidden(glyphs, pairs, &normalized, &properties);
    let originals: HashMap<_, _> = chars.into_iter().map(|c| (c.char_idx, c)).collect();
    result
        .into_iter()
        .map(|c| {
            let mut original = originals
                .get(&c.index)
                .ok_or_else(|| invalid("去重来源索引无效"))?
                .clone();
            original.char = c.text;
            original.source_indices = c.sources;
            Ok(original)
        })
        .collect()
}

#[derive(Debug, Serialize)]
pub struct Geometry {
    pub page_size: [f64; 2],
    pub page_bbox: [f64; 4],
    pub rotation: i32,
    pub raw_char_count: usize,
    pub chars: Vec<Character>,
    pub tight_bboxes: BTreeMap<usize, [f64; 4]>,
    pub loose_bboxes: BTreeMap<usize, [f64; 4]>,
    pub origins: BTreeMap<usize, [f64; 2]>,
}
pub fn read_geometry(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    password: Option<&str>,
    extended: bool,
    limits: &Limits,
) -> Result<Geometry, Error> {
    geometry_mode(
        library, bytes, page_index, password, extended, false, limits,
    )
}
pub fn read_visible_geometry(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    password: Option<&str>,
    extended: bool,
    limits: &Limits,
) -> Result<Geometry, Error> {
    geometry_mode(library, bytes, page_index, password, extended, true, limits)
}
fn geometry_mode(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    password: Option<&str>,
    extended: bool,
    visible: bool,
    limits: &Limits,
) -> Result<Geometry, Error> {
    let raw = if visible {
        super::read_visible(library, bytes, page_index, password, extended, limits)?
    } else {
        super::read(library, bytes, page_index, password, extended, limits)?
    };
    let chars = if limits.max_native_chars_per_page > 20_000 {
        deduplicate_bounded(
            raw.restored_chars,
            limits.max_native_chars_per_page,
            limits.max_native_chars_per_page.saturating_mul(4),
        )?
    } else {
        deduplicate(raw.restored_chars)?
    };
    let (mut tight_bboxes, mut loose_bboxes, mut origins) =
        (BTreeMap::new(), BTreeMap::new(), BTreeMap::new());
    if extended {
        for c in &chars {
            if let Some(b) = c.tight_bbox.flatten() {
                tight_bboxes.insert(c.char_idx, b);
            }
            if c.rotation.abs() > 1e-9 {
                if let Some(b) = c.loose_bbox.flatten() {
                    loose_bboxes.insert(c.char_idx, b);
                }
            }
            if let Some(p) = c.origin {
                origins.insert(c.char_idx, p);
            }
        }
    }
    Ok(Geometry {
        page_size: raw.page_size,
        page_bbox: raw.page_bbox,
        rotation: raw.rotation,
        raw_char_count: raw.raw_char_count,
        chars,
        tight_bboxes,
        loose_bboxes,
        origins,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn original_visible_pdf_geometry() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let f: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("build/migration/native-visible-text.json")).unwrap(),
        )
        .unwrap();
        let bytes = std::fs::read(root.join(f["pdf"].as_str().unwrap())).unwrap();
        let library = root.join(".venv/Lib/site-packages/pypdfium2_raw/pdfium.dll");
        for c in f["cases"].as_array().unwrap() {
            let got = read_visible_geometry(
                &library,
                &bytes,
                c["page_index"].as_u64().unwrap() as usize,
                None,
                true,
                &Limits::default(),
            )
            .unwrap();
            let got = serde_json::to_value(got).unwrap();
            for key in [
                "page_size",
                "chars",
                "tight_bboxes",
                "loose_bboxes",
                "origins",
            ] {
                assert_eq!(got[key], c["expected"][key], "{} / {key}", c["label"]);
            }
        }
    }
    #[test]
    fn frozen_complete_dedup_unicode_and_angles() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let f: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("build/migration/character-dedup.json")).unwrap(),
        )
        .unwrap();
        for c in f["cases"].as_array().unwrap() {
            let chars: Vec<Character> = serde_json::from_value(c["chars"].clone()).unwrap();
            assert_eq!(
                serde_json::to_value(deduplicate(chars).unwrap()).unwrap(),
                c["expected"],
                "{}",
                c["label"]
            );
        }
        let extra: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("build/migration/pdf-native-dedup-extra.json")).unwrap(),
        )
        .unwrap();
        for c in extra["raw_cases"].as_array().unwrap() {
            let chars: Vec<Character> = serde_json::from_value(c["chars"].clone()).unwrap();
            assert_eq!(
                serde_json::to_value(deduplicate(chars).unwrap()).unwrap(),
                c["expected"]
            );
        }
        let ties: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("build/migration/character-dedup-ties.json")).unwrap(),
        )
        .unwrap();
        for c in ties["cases"].as_array().unwrap() {
            let chars: Vec<Character> = serde_json::from_value(c["chars"].clone()).unwrap();
            assert_eq!(
                serde_json::to_value(deduplicate(chars).unwrap()).unwrap(),
                c["expected"]
            );
        }
        for c in f["nfkc"].as_array().unwrap() {
            let s = char::from_u32(c[0].as_u64().unwrap() as u32)
                .unwrap()
                .to_string();
            assert_eq!(s.nfkc().collect::<String>(), c[1].as_str().unwrap());
        }
        for c in f["numerics"].as_array().unwrap() {
            let a = angle(c["angle"].as_f64().unwrap());
            assert_eq!(a.cos.to_bits(), c["cos"].as_f64().unwrap().to_bits());
            assert_eq!(a.sin.to_bits(), c["sin"].as_f64().unwrap().to_bits());
            assert_eq!(a.rounded, bits(c["rounded"].as_f64().unwrap()));
            assert_eq!(a.bucket, c["bucket"].as_i64().unwrap());
        }
    }
}
