//! Native PPTX context. Geometry is kept in EMU, as in the Python converter;
//! neither rotations nor flips change its axis-aligned reading-order boxes.
pub mod body;
pub mod context;
pub mod geometry;
pub mod math_fallback;
pub mod normalize;
pub mod reading_order;
pub mod resources;
pub mod styles;
pub mod text;
pub mod titles;

/// Normal inputs use the already opened package. Retry only after failure,
/// using ZIP overlays; no full archive rewrite or Python callback.
pub fn convert_package(mut package: crate::package::Package) -> crate::Result<body::Body> {
    // python-pptx opens every ZIP member, including unreferenced entries.
    // Stream CRC validation without retaining an extra media/package copy.
    let initial = (|| {
        for name in package.names().to_vec() {
            package.validate_member(&name)?;
        }
        body::convert(&mut package)
    })();
    match initial {
        Ok(body) => Ok(body),
        Err(original) if original.kind == crate::ErrorKind::ResourceLimit => Err(original),
        Err(original) => {
            let changed = normalize::apply(&mut package)?;
            if changed.is_empty() {
                return Err(original);
            }
            let mut body = body::convert(&mut package)?;
            body.diagnostics.extend(
                changed
                    .into_iter()
                    .map(|name| format!("PPTX package normalized: {name}")),
            );
            Ok(body)
        }
    }
}
