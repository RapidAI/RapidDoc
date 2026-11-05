//! Original _svg_raster.py sizing and resvg_py 0.5.0 rendering parameters.
//! Same published resvg 0.48.0/usvg 0.48.1 engine, no Python adapter at runtime.
use crate::{xml, Result};
use std::sync::OnceLock;
fn length(value: Option<&str>) -> Option<f64> {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        regex::Regex::new(r"(?i)^([0-9]*\.?[0-9]+)\s*(px|pt|pc|mm|cm|in|q|%)?$").unwrap()
    });
    let m = re.captures(value?.trim())?;
    let v = m[1].parse::<f64>().ok()?;
    Some(
        v * match m
            .get(2)
            .map(|m| m.as_str().to_ascii_lowercase())
            .as_deref()
            .unwrap_or("")
        {
            "" | "px" => 1.,
            "pt" => 96. / 72.,
            "pc" => 16.,
            "mm" => 96. / 25.4,
            "cm" => 96. / 2.54,
            "in" => 96.,
            "q" => 96. / 101.6,
            _ => return None,
        },
    )
}
fn target(root: roxmltree::Node<'_, '_>) -> Option<[u32; 2]> {
    let size = length(root.attribute("width"))
        .zip(length(root.attribute("height")))
        .filter(|(w, h)| *w > 0. && *h > 0.)
        .or_else(|| {
            let value = root.attribute("viewBox")?.replace(',', " ");
            let parts = value.split_whitespace().collect::<Vec<_>>();
            if parts.len() != 4 {
                return None;
            }
            let w = parts[2].parse::<f64>().ok()?;
            let h = parts[3].parse::<f64>().ok()?;
            if w > 0. && h > 0. {
                Some((w, h))
            } else {
                None
            }
        });
    let (w, h) = size?;
    if !w.is_finite() || !h.is_finite() {
        return None;
    }
    let (mut w, mut h) = (w.round_ties_even().max(1.), h.round_ties_even().max(1.));
    let long = w.max(h);
    if long < 512. {
        w = (w * 512. / long).round_ties_even().max(1.);
        h = (h * 512. / long).round_ties_even().max(1.);
    }
    let scale = 1f64
        .min(8192. / w)
        .min(8192. / h)
        .min((16_000_000. / (w * h)).sqrt());
    Some([
        (w * scale).round_ties_even().max(1.) as u32,
        (h * scale).round_ties_even().max(1.) as u32,
    ])
}
pub fn serialize(data: &[u8]) -> Result<Option<String>> {
    if data.len() > 64 * 1024 * 1024 {
        return Ok(None);
    }
    let text = match xml::decode(data) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let doc = match xml::parse(&text) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    let root = doc.root_element();
    if !root.tag_name().name().eq_ignore_ascii_case("svg") {
        return Ok(None);
    }
    let target = target(root);
    let mut options = resvg::usvg::Options::default();
    options.dpi = 0.;
    options.font_size = 16.;
    options.font_family = if cfg!(target_os = "linux") {
        "Liberation Serif"
    } else if cfg!(any(windows, target_os = "macos")) {
        "Times New Roman"
    } else {
        "serif"
    }
    .into();
    options.languages = vec![];
    options.style_sheet = Some(String::new());
    options.default_size = resvg::usvg::Size::from_wh(
        target.map_or(100., |s| s[0] as f32),
        target.map_or(100., |s| s[1] as f32),
    )
    .unwrap();
    // Empty fontdb is the original skip_system_fonts=True contract.
    let tree = match resvg::usvg::Tree::from_str(&text, &options) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    let original = tree.size().to_int_size();
    let final_size = target
        .and_then(|s| resvg::tiny_skia::IntSize::from_wh(s[0], s[1]))
        .map_or(original, |s| original.scale_to(s));
    let Some(mut pixmap) = resvg::tiny_skia::Pixmap::new(final_size.width(), final_size.height())
    else {
        return Ok(None);
    };
    let transform = resvg::tiny_skia::Transform::from_scale(
        final_size.width() as f32 / original.width() as f32,
        final_size.height() as f32 / original.height() as f32,
    );
    resvg::render(&tree, transform, &mut pixmap.as_mut());
    if !pixmap.data().chunks_exact(4).any(|p| p[3] != 0) {
        return Ok(None);
    }
    let bytes = match pixmap.encode_png() {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    Ok(Some(format!(
        "data:image/png;base64,{}",
        base64::encode(bytes)
    )))
}
