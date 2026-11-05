//! 已有布局候选的原生网格优先选择；结构不可靠时保留原因并交给模型。
use rapiddoc_core::{document::Page, geometry::Affine, Error, ErrorCode, Rect};
use rapiddoc_native::grid::{self, Glyph, GridConfig, Separator, Table};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Deserialize)]
pub struct Proposal {
    pub layout_index: usize,
    pub bounds: Rect,
}
#[derive(Serialize)]
pub struct Selection {
    pub layout_index: usize,
    pub method: &'static str,
    pub reason: &'static str,
    pub native: Option<NativeOutput>,
}
impl Selection {
    fn model(index: usize, reason: &'static str) -> Self {
        Self {
            layout_index: index,
            method: "model",
            reason,
            native: None,
        }
    }
}
#[derive(Serialize)]
pub struct NativeOutput {
    pub route: &'static str,
    pub coordinate_space: &'static str,
    pub table: Table,
    pub x_tracks: Vec<f64>,
    pub y_tracks: Vec<f64>,
    pub html: String,
    pub text: String,
    pub page_geometry: Geometry,
}
#[derive(Serialize)]
pub struct Geometry {
    pub coordinate_space: &'static str,
    pub point_to_page: [f64; 6],
    pub cell_polygons: Vec<[[f64; 2]; 4]>,
}
#[derive(Serialize)]
pub struct Plan {
    pub selections: Vec<Selection>,
    pub work: usize,
}
pub struct PageTables {
    page: Page,
    to_points: Affine,
    to_pixels: Affine,
}
fn error(code: ErrorCode, message: &str) -> Error {
    Error {
        code,
        message: message.into(),
    }
}
impl PageTables {
    pub fn new(mut page: Page, width: usize, height: usize) -> Result<Self, Error> {
        if width == 0
            || height == 0
            || width
                .checked_mul(height)
                .is_none_or(|n| n > rapiddoc_core::resources::limit(16_000_000))
            || !page.visual_width_pt.is_finite()
            || page.visual_width_pt <= 0.
            || !page.visual_height_pt.is_finite()
            || page.visual_height_pt <= 0.
            || ![0, 90, 180, 270].contains(&page.rotation)
        {
            return Err(error(ErrorCode::InvalidInput, "原生表格页面尺寸或方向无效"));
        }
        if page.characters.len() > rapiddoc_core::resources::limit(100_000)
            || page.drawing_lines.len() > rapiddoc_core::resources::limit(100_000)
            || page.images.len() > rapiddoc_core::resources::limit(10_000)
        {
            return Err(error(ErrorCode::ResourceLimit, "原生表格页面事实超过预算"));
        }
        for (index, glyph) in page.characters.iter().enumerate() {
            if glyph.index != index
                || !glyph.angle_degrees.is_finite()
                || glyph
                    .text
                    .as_ref()
                    .is_some_and(|t| t.len() > rapiddoc_core::resources::limit(64))
            {
                return Err(error(ErrorCode::InvalidInput, "原生表格字符索引或方向无效"));
            }
        }
        for line in &page.drawing_lines {
            if !["vertical", "horizontal"].contains(&line.orientation.as_str())
                || line.start.iter().chain(&line.end).any(|v| !v.is_finite())
                || (line.orientation == "vertical" && (line.start[0] - line.end[0]).abs() > 1e-6)
                || (line.orientation == "horizontal" && (line.start[1] - line.end[1]).abs() > 1e-6)
            {
                return Err(error(ErrorCode::InvalidInput, "原生表格轴线无效"));
            }
        }
        let to_pixels = Affine::new([
            width as f64 / page.visual_width_pt,
            0.,
            0.,
            height as f64 / page.visual_height_pt,
            0.,
            0.,
        ])?;
        // 不持有无关的路径/链接对象；页面和字符均为已释放 PDFium 句柄后的自有事实。
        page.paths = Vec::new();
        page.links = Vec::new();
        page.text = String::new();
        Ok(Self {
            page,
            to_points: to_pixels.inverse()?,
            to_pixels,
        })
    }
    pub fn plan(
        &self,
        proposals: &[Proposal],
        margin_pt: f64,
        max_work: usize,
    ) -> Result<Plan, Error> {
        let max_work = rapiddoc_core::resources::limit(max_work);
        if !margin_pt.is_finite()
            || !(0. ..=32.).contains(&margin_pt)
            || max_work > rapiddoc_core::resources::limit(10_000_000)
        {
            return Err(error(ErrorCode::InvalidInput, "原生表格余量或工作预算无效"));
        }
        if proposals.len() > rapiddoc_core::resources::limit(1000) {
            return Err(error(ErrorCode::ResourceLimit, "原生表格候选超过预算"));
        }
        let mut seen = BTreeSet::new();
        let mut selections = Vec::with_capacity(proposals.len());
        let mut work = 0;
        for proposal in proposals {
            if !seen.insert(proposal.layout_index) {
                return Err(error(ErrorCode::InvalidInput, "原生表格候选索引重复"));
            }
            let cost =
                self.page.characters.len() + self.page.drawing_lines.len() + self.page.images.len();
            work += cost;
            if work > max_work {
                return Err(error(
                    ErrorCode::ResourceLimit,
                    "原生表格累计工作量超过预算",
                ));
            }
            let model: [f64; 4] = self.to_points.rect(proposal.bounds)?.into();
            let model = [
                model[0].clamp(0., self.page.visual_width_pt),
                model[1].clamp(0., self.page.visual_height_pt),
                model[2].clamp(0., self.page.visual_width_pt),
                model[3].clamp(0., self.page.visual_height_pt),
            ];
            if area(model) <= 0. {
                return Err(error(ErrorCode::InvalidInput, "原生表格候选必须有正面积"));
            }
            let region = [
                (model[0] - margin_pt).max(0.),
                (model[1] - margin_pt).max(0.),
                (model[2] + margin_pt).min(self.page.visual_width_pt),
                (model[3] + margin_pt).min(self.page.visual_height_pt),
            ];
            let (selected, used) =
                self.select(proposal.layout_index, model, region, max_work - work)?;
            work += used;
            selections.push(selected);
        }
        Ok(Plan { selections, work })
    }
    fn select(
        &self,
        index: usize,
        model: [f64; 4],
        region: [f64; 4],
        max_work: usize,
    ) -> Result<(Selection, usize), Error> {
        let max_work = rapiddoc_core::resources::limit(max_work);
        let reject = |reason| Ok((Selection::model(index, reason), 0));
        if self.page.rotation != 0 {
            return reject("rotated_page");
        }
        if self
            .page
            .images
            .iter()
            .any(|image| overlap(image.visual_bounds.into(), region) > 0.)
        {
            return reject("raster_overlap");
        }
        let mut separators = Vec::new();
        for line in &self.page.drawing_lines {
            let vertical = line.orientation == "vertical";
            let coordinate = if vertical {
                line.start[0]
            } else {
                line.start[1]
            };
            let low = if vertical { region[0] } else { region[1] };
            let high = if vertical { region[2] } else { region[3] };
            let (start, end) = if vertical {
                (line.start[1], line.end[1])
            } else {
                (line.start[0], line.end[0])
            };
            let first = if vertical { region[1] } else { region[0] };
            let last = if vertical { region[3] } else { region[2] };
            if coordinate >= low - 1.
                && coordinate <= high + 1.
                && start.min(end) < last
                && start.max(end) > first
            {
                separators.push(Separator {
                    vertical,
                    coordinate,
                    start,
                    end,
                });
            }
        }
        let mut glyphs = Vec::new();
        let mut required = BTreeSet::new();
        for glyph in &self.page.characters {
            let tight: [f64; 4] = glyph.visual_tight_bounds.into();
            let center = [(tight[0] + tight[2]) / 2., (tight[1] + tight[3]) / 2.];
            let whitespace = glyph
                .text
                .as_ref()
                .is_some_and(|t| t.chars().all(char::is_whitespace));
            if !whitespace
                && (center[0] < region[0]
                    || center[0] > region[2]
                    || center[1] < region[1]
                    || center[1] > region[3])
            {
                continue;
            }
            let Some(text) = &glyph.text else {
                return reject("unreliable_native_text");
            };
            if text.chars().all(char::is_control) {
                continue;
            }
            if text.contains('\u{fffd}')
                || (!text.chars().all(char::is_whitespace)
                    && (glyph.is_generated || glyph.angle_degrees.abs() > 5. || area(tight) <= 0.))
            {
                return reject("unreliable_native_text");
            }
            if !text.chars().all(char::is_whitespace) {
                required.insert(glyph.index);
            }
            glyphs.push(Glyph {
                index: glyph.index,
                bounds: glyph.visual_loose_bounds,
                text: text.clone(),
                generated: glyph.is_generated,
            });
        }
        if required.is_empty() {
            return reject("no_native_text");
        }
        let config = GridConfig {
            max_cells: 2000,
            max_separators: 2000,
            max_coverage_work: max_work,
            ..Default::default()
        };
        let tracks = grid::plan_tracks(Rect::try_from(region)?, &separators, &config)?;
        let used = tracks.coverage_work;
        let table = match grid::assemble(
            &tracks.x_tracks,
            &tracks.y_tracks,
            &tracks.lines,
            glyphs,
            &config,
        ) {
            Ok(table) => table,
            Err(e) if e.code == ErrorCode::InvalidInput => {
                return Ok((Selection::model(index, "incomplete_grid"), used))
            }
            Err(e) => return Err(e),
        };
        let grid = grid::RecoveredGrid {
            table,
            x_tracks: tracks.x_tracks,
            y_tracks: tracks.y_tracks,
            coverage_work: used,
        };
        let actual: [f64; 4] = grid.table.bounds.into();
        let intersect = overlap(model, actual);
        if intersect / (area(model) + area(actual) - intersect) < 0.8 {
            return Ok((Selection::model(index, "layout_grid_mismatch"), used));
        }
        let assigned: BTreeSet<_> = grid
            .table
            .cells
            .iter()
            .flat_map(|cell| cell.character_indices.iter().copied())
            .collect();
        if !required.is_subset(&assigned) {
            return Ok((Selection::model(index, "unassigned_native_text"), used));
        }
        let html = rapiddoc_render::table_html(&grid.table, config.max_cells)?;
        let mut polygons = Vec::with_capacity(grid.table.cells.len());
        for cell in &grid.table.cells {
            let x0 = grid.x_tracks[cell.column];
            let x1 = grid.x_tracks[cell.column + cell.column_span];
            let y0 = grid.y_tracks[cell.row];
            let y1 = grid.y_tracks[cell.row + cell.row_span];
            polygons.push([
                self.to_pixels.point([x0, y0])?,
                self.to_pixels.point([x1, y0])?,
                self.to_pixels.point([x1, y1])?,
                self.to_pixels.point([x0, y1])?,
            ]);
        }
        let text = grid
            .table
            .cells
            .iter()
            .map(|cell| cell.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let native = NativeOutput {
            route: "native",
            coordinate_space: "visual_points",
            table: grid.table,
            x_tracks: grid.x_tracks,
            y_tracks: grid.y_tracks,
            html,
            text,
            page_geometry: Geometry {
                coordinate_space: "page_pixels",
                point_to_page: self.to_pixels.values(),
                cell_polygons: polygons,
            },
        };
        Ok((
            Selection {
                layout_index: index,
                method: "native",
                reason: "closed_native_grid",
                native: Some(native),
            },
            used,
        ))
    }
}
fn area(b: [f64; 4]) -> f64 {
    (b[2] - b[0]) * (b[3] - b[1])
}
fn overlap(a: [f64; 4], b: [f64; 4]) -> f64 {
    (a[2].min(b[2]) - a[0].max(b[0])).max(0.) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.)
}
