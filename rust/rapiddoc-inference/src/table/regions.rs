//! UNET 抗锯齿线图→角度估计→旋转/逆变换→闭合区域→排序及逻辑格。
use crate::{
    error, preprocess::check_size, table_components, table_draw, table_lines, table_recover,
    unet_lines, Model,
};
use image::{GrayImage, Luma};
use imageproc::{
    contours::{find_contours, BorderType},
    geometry::convex_hull,
    point::Point,
};
use rapiddoc_core::{Error, ErrorCode};
use serde::Serialize;

#[derive(Serialize)]
pub struct Cells {
    pub rotation_degrees: f32,
    /// 原图像素；逆时针 [左上、左下、右下、右上]，与 topology 来源索引相同。
    pub polygons: Vec<[[f32; 2]; 4]>,
    pub rotated_polygons: Vec<[[f32; 2]; 4]>,
    pub topology: Option<table_recover::Topology>,
}

fn matrix(width: usize, height: usize, degrees: f64) -> [f64; 6] {
    let angle = degrees * std::f64::consts::PI / 180.;
    let c = angle.cos();
    let s = angle.sin();
    let x = (width / 2) as f64;
    let y = (height / 2) as f64;
    [c, s, (1. - c) * x - s * y, -s, c, s * x + (1. - c) * y]
}

pub fn rotate(bytes: &[u8], width: usize, height: usize, degrees: f64) -> Result<Vec<u8>, Error> {
    let count = check_size(width, height, 4_000_000)?;
    if width > rapiddoc_core::resources::limit(4096)
        || height > rapiddoc_core::resources::limit(4096)
    {
        return Err(error(ErrorCode::ResourceLimit, "UNET 旋转图边长超过预算"));
    }
    if bytes.len() != count || !degrees.is_finite() || degrees.abs() > 180. {
        return Err(error(ErrorCode::InvalidInput, "UNET 旋转参数无效"));
    }
    let mut m = matrix(width, height, degrees);
    let det = 1. / (m[0] * m[4] - m[1] * m[3]);
    let a = m[4] * det;
    let d = m[0] * det;
    m[0] = a;
    m[1] *= -det;
    m[3] *= -det;
    m[4] = d;
    let bx = -m[0] * m[2] - m[1] * m[5];
    let by = -m[3] * m[2] - m[4] * m[5];
    m[2] = bx;
    m[5] = by;
    let m = m.map(|v| v as f32);
    let mut output = vec![0; count];
    for y in 0..height {
        let x_offset = y as f32 * m[1] + m[2];
        let y_offset = y as f32 * m[4] + m[5];
        for x in 0..width {
            // OpenCV 5 / AVX2 的 16 像素 SIMD 块使用 FMA，剩余标量为两次加法。
            let (sx, sy) = if x < width / 16 * 16 {
                (
                    m[0].mul_add(x as f32, x_offset),
                    m[3].mul_add(x as f32, y_offset),
                )
            } else {
                (
                    (x as f32 * m[0] + y as f32 * m[1]) + m[2],
                    (x as f32 * m[3] + y as f32 * m[4]) + m[5],
                )
            };
            let sx = sx.round_ties_even().clamp(0., (width - 1) as f32) as usize;
            let sy = sy.round_ties_even().clamp(0., (height - 1) as f32) as usize;
            output[y * width + x] = bytes[sy * width + sx];
        }
    }
    Ok(output)
}

pub fn angle(bytes: &[u8], width: usize, height: usize) -> Result<f32, Error> {
    let count = check_size(width, height, 4_000_000)?;
    if bytes.len() != count
        || width > rapiddoc_core::resources::limit(4096)
        || height > rapiddoc_core::resources::limit(4096)
    {
        return Err(error(ErrorCode::InvalidInput, "UNET 角度图像无效"));
    }
    let mut image = GrayImage::new(width as u32 + 2, height as u32 + 2);
    for y in 0..height {
        for x in 0..width {
            if bytes[y * width + x] > 0 {
                image.put_pixel(x as u32 + 1, y as u32 + 1, Luma([1]));
            }
        }
    }
    let contours = find_contours::<i32>(&image);
    let mut best_area = -1f64;
    let mut best = None;
    for contour in contours
        .iter()
        .rev()
        .filter(|c| c.border_type == BorderType::Outer && c.parent.is_none())
    {
        let p = &contour.points;
        let area = (0..p.len())
            .map(|i| {
                let a = p[i];
                let b = p[(i + 1) % p.len()];
                a.x as f64 * b.y as f64 - b.x as f64 * a.y as f64
            })
            .sum::<f64>()
            .abs();
        if area > best_area {
            best_area = area;
            best = Some(p);
        }
    }
    let Some(points) = best else {
        return Ok(0.);
    };
    // CHAIN_APPROX_SIMPLE 保留转向点，避免直边所有像素进入凸包原索引查找。
    let mut simple = Vec::new();
    for i in 0..points.len() {
        let a = points[(i + points.len() - 1) % points.len()];
        let b = points[i];
        let c = points[(i + 1) % points.len()];
        if ((b.x - a.x).signum(), (b.y - a.y).signum())
            != ((c.x - b.x).signum(), (c.y - b.y).signum())
        {
            simple.push(Point::new(b.x - 1, b.y - 1));
        }
    }
    if simple.len() > rapiddoc_core::resources::limit(200_000)
        || simple
            .len()
            .saturating_mul(convex_hull(simple.as_slice()).len() + 16)
            > rapiddoc_core::resources::limit(50_000_000)
    {
        return Err(error(
            ErrorCode::ResourceLimit,
            "UNET 角度轮廓超过工作量预算",
        ));
    }
    let Some((_, _, mut angle)) = crate::detection_geometry::rectangle_angle(&simple) else {
        return Ok(0.);
    };
    if angle < -45. {
        angle += 90.;
    } else if angle > 45. {
        angle -= 90.;
    }
    Ok(angle)
}

pub fn regions_from_lines(
    lines: &[table_lines::Line],
    width: usize,
    height: usize,
    rotated_fix: bool,
) -> Result<Cells, Error> {
    let image = table_draw::draw(lines, width, height)?;
    let degrees = angle(&image, width, height)?;
    let apply = rotated_fix && degrees.abs() > 0.3;
    let rotated = if apply {
        rotate(&image, width, height, degrees as f64)?
    } else {
        image
    };
    let binary: Vec<_> = rotated.iter().map(|v| u8::from(*v < 255)).collect();
    let mut rotated_polygons = table_components::analyze(&binary, width, height, None, 0)?.polygons;
    let transform = matrix(width, height, -(degrees as f64)).map(|v| v as f32);
    let mut polygons = rotated_polygons.clone();
    if apply {
        for polygon in &mut polygons {
            for point in polygon {
                let [x, y] = *point;
                *point = [
                    (x * transform[0] + y * transform[1]) + transform[2],
                    (x * transform[3] + y * transform[4]) + transform[5],
                ];
            }
        }
    }
    for p in polygons.iter_mut().chain(&mut rotated_polygons) {
        p.swap(1, 3);
    }
    let mut indices: Vec<_> = (0..polygons.len()).collect();
    indices.sort_by(|&a, &b| {
        rotated_polygons[a][0][1]
            .total_cmp(&rotated_polygons[b][0][1])
            .then_with(|| rotated_polygons[a][0][0].total_cmp(&rotated_polygons[b][0][0]))
    });
    for i in 0..indices.len().saturating_sub(1) {
        for j in (0..=i).rev() {
            let a = rotated_polygons[indices[j]];
            let b = rotated_polygons[indices[j + 1]];
            let ha = a[2][1] - a[0][1];
            let hb = b[2][1] - b[0][1];
            let overlap = a[2][1].min(b[2][1]) - a[0][1].max(b[0][1]);
            let ra = if ha > 0. { (ha - overlap) / ha } else { 0. };
            let rb = if hb > 0. { (hb - overlap) / hb } else { 0. };
            if (ra < 0.4 || rb < 0.4) && b[0][0] < a[0][0] && (a[0][1] - b[0][1]).abs() < 20. {
                indices.swap(j, j + 1);
            } else {
                break;
            }
        }
    }
    let polygons = indices.iter().map(|&i| polygons[i]).collect();
    let rotated_polygons: Vec<_> = indices.iter().map(|&i| rotated_polygons[i]).collect();
    Ok(Cells {
        rotation_degrees: degrees,
        polygons,
        rotated_polygons,
        topology: None,
    })
}

pub fn from_lines(
    lines: &[table_lines::Line],
    width: usize,
    height: usize,
    rotated_fix: bool,
) -> Result<Cells, Error> {
    let mut cells = regions_from_lines(lines, width, height, rotated_fix)?;
    if !cells.rotated_polygons.is_empty() {
        cells.topology = Some(table_recover::recover(
            &cells
                .rotated_polygons
                .iter()
                .map(|p| p.map(|xy| xy.map(f64::from)))
                .collect::<Vec<_>>(),
            10.,
            15.,
        )?);
    }
    Ok(cells)
}

impl Model {
    pub fn table_cells_png(&mut self, bytes: &[u8], rotated_fix: bool) -> Result<Cells, Error> {
        let (bgr, width, height) = crate::preprocess::decode_png(bytes, 4_000_000)?;
        let mask = self.table_line_mask_bgr(&bgr, width, height)?;
        let lines =
            unet_lines::geometry(&mask, width, height, true, &table_lines::Config::default())?;
        let lines: Vec<_> = lines.rows.into_iter().chain(lines.columns).collect();
        from_lines(&lines, width, height, rotated_fix)
    }
}
