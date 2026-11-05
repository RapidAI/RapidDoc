//! Actual Clipper 6 core, preserving Pyclipper output point order.
use imageproc::point::Point;
use rapiddoc_core::Error;

pub(crate) fn expand(rect: &[[f64; 2]; 4], delta: f64) -> Result<Vec<Point<i32>>, Error> {
    Ok(crate::seal::offset_polygon(rect, delta)?
        .unwrap_or_default()
        .into_iter()
        .map(|p| Point::new(p[0] as i32, p[1] as i32))
        .collect())
}
