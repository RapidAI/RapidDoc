//! 页面表格裁图规划、像素所有权及裁图坐标到页面坐标的映射。
use crate::{error, preprocess, table_recover::Polygon, wired_match::Fragment};
use image::{codecs::png::PngEncoder, ExtendedColorType, ImageEncoder};
use rapiddoc_core::{Error, ErrorCode};
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayoutRegion {
    pub cls_id: usize,
    pub label: String,
    pub score: f64,
    pub coordinate: [f64; 4],
    pub order: usize,
}
#[derive(Serialize)]
pub struct CropPlan {
    pub layout_index: usize,
    pub bounds: [usize; 4],
    pub width: usize,
    pub height: usize,
}
#[derive(Serialize)]
pub struct Plan {
    pub width: usize,
    pub height: usize,
    pub crop_pixels: usize,
    pub tables: Vec<CropPlan>,
}
#[derive(Serialize)]
pub struct BlockPlan {
    pub width: usize,
    pub height: usize,
    pub crop_pixels: usize,
    pub regions: Vec<CropPlan>,
}
/// 当前预览按旧布局类别映射选择文字，默认忽略页眉、页脚等边缘内容。
pub fn is_text_label(label: &str) -> bool {
    matches!(
        label,
        "abstract"
            | "algorithm"
            | "content"
            | "doc_title"
            | "figure_title"
            | "paragraph_title"
            | "reference"
            | "reference_content"
            | "text"
            | "vertical_text"
            | "vision_footnote"
    )
}
pub struct PageImage {
    bgr: Vec<u8>,
    pub width: usize,
    pub height: usize,
}
impl PageImage {
    pub fn decode(data: &[u8], max_pixels: usize) -> Result<Self, Error> {
        let max_pixels = rapiddoc_core::resources::limit(max_pixels);
        if max_pixels > rapiddoc_core::resources::limit(16_000_000) {
            return Err(error(
                ErrorCode::InvalidInput,
                "页面像素预算不得超过 1600 万",
            ));
        }
        let (bgr, width, height) = preprocess::decode_png(data, max_pixels)?;
        Ok(Self { bgr, width, height })
    }
    pub fn plan(
        &self,
        regions: &[LayoutRegion],
        grid: usize,
        max_tables: usize,
        max_total_pixels: usize,
    ) -> Result<Plan, Error> {
        let max_tables = rapiddoc_core::resources::limit(max_tables);
        let max_total_pixels = rapiddoc_core::resources::limit(max_total_pixels);
        self.plan_selected(regions, grid, max_tables, max_total_pixels, false)
    }
    pub fn plan_blocks(
        &self,
        regions: &[LayoutRegion],
        grid: usize,
        max_regions: usize,
        max_total_pixels: usize,
    ) -> Result<BlockPlan, Error> {
        let max_regions = rapiddoc_core::resources::limit(max_regions);
        let max_total_pixels = rapiddoc_core::resources::limit(max_total_pixels);
        let plan = self.plan_selected(regions, grid, max_regions, max_total_pixels, true)?;
        Ok(BlockPlan {
            width: plan.width,
            height: plan.height,
            crop_pixels: plan.crop_pixels,
            regions: plan.tables,
        })
    }
    fn plan_selected(
        &self,
        regions: &[LayoutRegion],
        grid: usize,
        max_tables: usize,
        max_total_pixels: usize,
        include_text: bool,
    ) -> Result<Plan, Error> {
        let max_tables = rapiddoc_core::resources::limit(max_tables);
        let max_total_pixels = rapiddoc_core::resources::limit(max_total_pixels);
        if regions.len() > rapiddoc_core::resources::limit(1000) {
            return Err(error(ErrorCode::ResourceLimit, "页面布局候选超过 1000"));
        }
        if !(1..=64).contains(&grid)
            || max_tables > rapiddoc_core::resources::limit(1000)
            || max_total_pixels > rapiddoc_core::resources::limit(128_000_000)
        {
            return Err(error(ErrorCode::InvalidInput, "页面表格裁图配置无效"));
        }
        let mut tables = Vec::new();
        let mut crop_pixels = 0usize;
        for (layout_index, region) in regions.iter().enumerate() {
            if !region.score.is_finite()
                || !(0. ..=1.).contains(&region.score)
                || region.coordinate.iter().any(|v| {
                    !v.is_finite() || v.abs() > rapiddoc_core::resources::limit(1_000_000.)
                })
            {
                return Err(error(ErrorCode::InvalidInput, "布局候选评分/坐标无效"));
            }
            if region.label != "table" && !(include_text && is_text_label(&region.label)) {
                continue;
            }
            let [x0, y0, x1, y1] = region.coordinate;
            if x1 <= x0 || y1 <= y0 {
                return Err(error(ErrorCode::InvalidInput, "待识别区域必须有正面积"));
            }
            // 与 batch_analyze 的 normalize_to_int_bbox(bbox/5) 再乘 5 一致；先扩网格再裁页面。
            let g = grid as f64;
            let bounds = [
                ((x0 / g).floor() * g).clamp(0., self.width as f64) as usize,
                ((y0 / g).floor() * g).clamp(0., self.height as f64) as usize,
                ((x1 / g).ceil() * g).clamp(0., self.width as f64) as usize,
                ((y1 / g).ceil() * g).clamp(0., self.height as f64) as usize,
            ];
            let (width, height) = self.dimensions(bounds)?;
            crop_pixels = crop_pixels
                .checked_add(width * height)
                .ok_or_else(|| error(ErrorCode::ResourceLimit, "裁图累计像素溢出"))?;
            if tables.len() >= max_tables || crop_pixels > max_total_pixels {
                return Err(error(
                    ErrorCode::ResourceLimit,
                    "页面识别区域数量/累计像素超过预算",
                ));
            }
            tables.push(CropPlan {
                layout_index,
                bounds,
                width,
                height,
            });
        }
        Ok(Plan {
            width: self.width,
            height: self.height,
            crop_pixels,
            tables,
        })
    }
    fn dimensions(&self, b: [usize; 4]) -> Result<(usize, usize), Error> {
        if b[0] >= b[2] || b[1] >= b[3] || b[2] > self.width || b[3] > self.height {
            return Err(error(ErrorCode::InvalidInput, "裁图范围为空或越界"));
        }
        let (w, h) = (b[2] - b[0], b[3] - b[1]);
        preprocess::check_size(w, h, 4_000_000)?;
        Ok((w, h))
    }
    pub fn crop_rgb(&self, b: [usize; 4]) -> Result<Vec<u8>, Error> {
        let (w, h) = self.dimensions(b)?;
        let mut rgb = Vec::with_capacity(w * h * 3);
        for y in b[1]..b[3] {
            for x in b[0]..b[2] {
                let i = (y * self.width + x) * 3;
                rgb.extend_from_slice(&[self.bgr[i + 2], self.bgr[i + 1], self.bgr[i]]);
            }
        }
        Ok(rgb)
    }
    pub fn crop_png(&self, b: [usize; 4]) -> Result<Vec<u8>, Error> {
        let (w, h) = self.dimensions(b)?;
        let rgb = self.crop_rgb(b)?;
        let mut png = Vec::new();
        PngEncoder::new(&mut png)
            .write_image(&rgb, w as u32, h as u32, ExtendedColorType::Rgb8)
            .map_err(|e| error(ErrorCode::InvalidInput, e))?;
        Ok(png)
    }
    pub fn project_ocr(&self, lines: &str, b: [usize; 4]) -> Result<Vec<OcrLine>, Error> {
        self.dimensions(b)?;
        if lines.len() > rapiddoc_core::resources::limit(16_000_000) {
            return Err(error(ErrorCode::ResourceLimit, "OCR JSON 超过预算"));
        }
        let mut lines: Vec<OcrLine> =
            serde_json::from_str(lines).map_err(|e| error(ErrorCode::InvalidInput, e))?;
        if lines.len() > rapiddoc_core::resources::limit(1000) {
            return Err(error(ErrorCode::ResourceLimit, "区域 OCR 行数超过预算"));
        }
        for line in &mut lines {
            if !line.confidence.is_finite() || !(0. ..=1.).contains(&line.confidence) {
                return Err(error(ErrorCode::InvalidInput, "OCR 评分无效"));
            }
            translate(&mut line.points, b)?;
        }
        Ok(lines)
    }
    pub fn project(&self, recognition: &str, b: [usize; 4]) -> Result<PageGeometry, Error> {
        self.dimensions(b)?;
        if recognition.len() > rapiddoc_core::resources::limit(16_000_000) {
            return Err(error(ErrorCode::ResourceLimit, "表格几何 JSON 超过预算"));
        }
        let mut r: Recognition =
            serde_json::from_str(recognition).map_err(|e| error(ErrorCode::InvalidInput, e))?;
        if r.ocr_lines.len() > rapiddoc_core::resources::limit(1000) {
            return Err(error(ErrorCode::ResourceLimit, "表格 OCR 数量超过预算"));
        }
        for line in &mut r.ocr_lines {
            if !line.confidence.is_finite() || !(0. ..=1.).contains(&line.confidence) {
                return Err(error(ErrorCode::InvalidInput, "表格 OCR 评分无效"));
            }
            translate(&mut line.points, b)?;
        }
        let (mut polygons, mut matched_cells) = match r.route.as_str() {
            "wired" => (
                r.cells
                    .ok_or_else(|| error(ErrorCode::InvalidInput, "有线结果缺少格框"))?
                    .polygons,
                Some(
                    serde_json::from_value::<Vec<Vec<Fragment>>>(r.matches.cells)
                        .map_err(|e| error(ErrorCode::InvalidInput, e))?,
                ),
            ),
            "wireless" => {
                let boxes = r
                    .structure
                    .ok_or_else(|| error(ErrorCode::InvalidInput, "无线结果缺少格框"))?
                    .cell_bboxes;
                if boxes.len() > rapiddoc_core::resources::limit(2000) {
                    return Err(error(ErrorCode::ResourceLimit, "表格格框数量超过预算"));
                }
                let mut polygons = Vec::with_capacity(boxes.len());
                for bbox in boxes {
                    polygons.push(match bbox.as_slice() {
                        [x0, y0, x1, y1] => [[*x0, *y0], [*x1, *y0], [*x1, *y1], [*x0, *y1]],
                        [a, b, c, d, e, f, g, h] => [[*a, *b], [*c, *d], [*e, *f], [*g, *h]],
                        _ => return Err(error(ErrorCode::InvalidInput, "无线格框须为四/八值")),
                    });
                }
                let indices: Vec<Vec<usize>> = serde_json::from_value(r.matches.cells)
                    .map_err(|e| error(ErrorCode::InvalidInput, e))?;
                if indices.len() != polygons.len()
                    || indices.iter().map(Vec::len).sum::<usize>()
                        > rapiddoc_core::resources::limit(10000)
                    || indices.iter().flatten().any(|&i| i >= r.ocr_lines.len())
                {
                    return Err(error(ErrorCode::InvalidInput, "无线匹配索引无效"));
                }
                (polygons, None)
            }
            _ => return Err(error(ErrorCode::InvalidInput, "未知表格分支")),
        };
        if polygons.len() > rapiddoc_core::resources::limit(2000) {
            return Err(error(ErrorCode::ResourceLimit, "表格格框数量超过预算"));
        }
        for p in &mut polygons {
            translate(p, b)?;
        }
        if let Some(cells) = &mut matched_cells {
            if cells.len() != polygons.len()
                || cells.iter().map(Vec::len).sum::<usize>()
                    > rapiddoc_core::resources::limit(10000)
            {
                return Err(error(ErrorCode::InvalidInput, "表格匹配格数/片段数无效"));
            }
            for f in cells.iter_mut().flatten() {
                if f.ocr_index.is_some_and(|i| i >= r.ocr_lines.len())
                    || !f.confidence.is_finite()
                    || !(0. ..=1.).contains(&f.confidence)
                {
                    return Err(error(ErrorCode::InvalidInput, "表格匹配来源/评分无效"));
                }
                translate(&mut f.points, b)?;
            }
        }
        Ok(PageGeometry {
            coordinate_space: "page_pixels",
            crop_to_page: [1., 0., 0., 1., b[0] as f64, b[1] as f64],
            cell_polygons: polygons,
            ocr_lines: r.ocr_lines,
            matched_cells,
        })
    }
}
fn translate(p: &mut Polygon, b: [usize; 4]) -> Result<(), Error> {
    for xy in p {
        if xy
            .iter()
            .any(|v| !v.is_finite() || v.abs() > rapiddoc_core::resources::limit(1_000_000.))
        {
            return Err(error(ErrorCode::InvalidInput, "表格局部坐标无效"));
        }
        xy[0] += b[0] as f64;
        xy[1] += b[1] as f64;
    }
    Ok(())
}
#[derive(Serialize, Deserialize)]
pub struct OcrLine {
    pub points: Polygon,
    pub text: String,
    pub confidence: f64,
}
#[derive(Deserialize)]
struct Cells {
    polygons: Vec<Polygon>,
}
#[derive(Deserialize)]
struct Structure {
    cell_bboxes: Vec<Vec<f64>>,
}
#[derive(Deserialize)]
struct Matches {
    cells: serde_json::Value,
}
#[derive(Deserialize)]
struct Recognition {
    route: String,
    ocr_lines: Vec<OcrLine>,
    cells: Option<Cells>,
    structure: Option<Structure>,
    matches: Matches,
}
#[derive(Serialize)]
pub struct PageGeometry {
    pub coordinate_space: &'static str,
    pub crop_to_page: [f64; 6],
    pub cell_polygons: Vec<Polygon>,
    pub ocr_lines: Vec<OcrLine>,
    pub matched_cells: Option<Vec<Vec<Fragment>>>,
}
