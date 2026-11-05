//! 旧实际 Form 调用树与声明核验；结果只持有稳定编号，不泄漏 PDFium 地址。
mod declarations;
use crate::with_raw;
use pdfium_render::prelude::*;
use rapiddoc_core::{Error, ErrorCode, Limits};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    path::Path,
};
type B = [f64; 4];
type M = [f64; 6];
const ID: M = [1., 0., 0., 1., 0., 0.];
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FormInfo {
    pub instance_id: usize,
    pub parent_id: Option<usize>,
    pub occurrence: Vec<usize>,
    pub bbox: B,
    pub declared_bbox: Option<B>,
    pub media_bbox: B,
    pub matrix: M,
    pub clip: Option<B>,
    pub paint_order: usize,
    pub text_indices: BTreeSet<usize>,
    pub path_indices: BTreeSet<usize>,
    pub image_bboxes: Vec<B>,
    pub structure_valid: bool,
}
#[derive(Debug, Serialize)]
pub struct Geometry {
    pub page_size: [f64; 2],
    pub page_bbox: B,
    pub rotation: i32,
    pub form_infos: Vec<FormInfo>,
    pub form_bboxes: Vec<B>,
}
fn error(code: ErrorCode, message: &str) -> Error {
    Error {
        code,
        message: message.into(),
    }
}
fn limited() -> Error {
    error(ErrorCode::ResourceLimit, "Form结构超过资源预算")
}
fn multiply(a: M, b: M) -> M {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[4] * b[0] + a[5] * b[2] + b[4],
        a[4] * b[1] + a[5] * b[3] + b[5],
    ]
}
fn point(p: [f64; 2], m: M) -> [f64; 2] {
    [
        m[0] * p[0] + m[2] * p[1] + m[4],
        m[1] * p[0] + m[3] * p[1] + m[5],
    ]
}
fn transform(b: B, f: impl Fn([f64; 2]) -> [f64; 2]) -> B {
    let mut it = [(b[0], b[1]), (b[0], b[3]), (b[2], b[1]), (b[2], b[3])]
        .into_iter()
        .map(|(x, y)| f([x, y]));
    let p = it.next().unwrap();
    let mut b = [p[0], p[1], p[0], p[1]];
    for p in it {
        if p[0] < b[0] {
            b[0] = p[0]
        }
        if p[1] < b[1] {
            b[1] = p[1]
        }
        if p[0] > b[2] {
            b[2] = p[0]
        }
        if p[1] > b[3] {
            b[3] = p[1]
        }
    }
    b
}
fn visual(b: B, m: M, frame: B, r: i32) -> B {
    transform(b, |p| {
        let [x, y] = point(p, m);
        match r {
            90 => [y - frame[1], x - frame[0]],
            180 => [frame[2] - x, y - frame[1]],
            270 => [frame[3] - y, frame[2] - x],
            _ => [x - frame[0], frame[3] - y],
        }
    })
}
fn intersection(a: B, b: B) -> B {
    [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
}
fn visible(b: B) -> bool {
    b[2] > b[0] && b[3] > b[1]
}
fn clipped(b: B, clip: Option<B>, frame: B, r: i32) -> Option<B> {
    if clip.is_some_and(|b| !visible(b)) {
        return None;
    }
    let b = clip.map_or(b, |c| intersection(b, visual(c, ID, frame, r)));
    visible(b).then_some(b)
}
struct Handles<'a> {
    api: &'a dyn PdfiumLibraryBindings,
    doc: FPDF_DOCUMENT,
    page: FPDF_PAGE,
    text: FPDF_TEXTPAGE,
}
impl Drop for Handles<'_> {
    fn drop(&mut self) {
        unsafe {
            if !self.text.is_null() {
                self.api.FPDFText_ClosePage(self.text)
            }
            if !self.page.is_null() {
                self.api.FPDF_ClosePage(self.page)
            }
            if !self.doc.is_null() {
                self.api.FPDF_CloseDocument(self.doc)
            }
        }
    }
}
struct Reader<'a> {
    api: &'a dyn PdfiumLibraryBindings,
    limits: &'a Limits,
    objects: usize,
    segments: usize,
    paths: usize,
    paint: usize,
    forms: Vec<FormInfo>,
    owners: HashMap<usize, usize>,
    frame: B,
    rotation: i32,
    size: [f64; 2],
}
impl Reader<'_> {
    fn charge(&mut self, objects: usize, segments: usize) -> Result<(), Error> {
        self.objects = self.objects.checked_add(objects).ok_or_else(limited)?;
        self.segments = self.segments.checked_add(segments).ok_or_else(limited)?;
        if self.objects > self.limits.max_objects_per_page
            || self.segments > self.limits.max_path_segments_per_page
        {
            Err(limited())
        } else {
            Ok(())
        }
    }
    unsafe fn matrix(&self, obj: FPDF_PAGEOBJECT) -> Option<M> {
        let mut m = FS_MATRIX {
            a: 0.,
            b: 0.,
            c: 0.,
            d: 0.,
            e: 0.,
            f: 0.,
        };
        (self.api.FPDFPageObj_GetMatrix(obj, &mut m) != 0)
            .then_some([
                m.a as f64, m.b as f64, m.c as f64, m.d as f64, m.e as f64, m.f as f64,
            ])
            .filter(|m| m.iter().all(|v| v.is_finite()))
    }
    unsafe fn bounds(&self, obj: FPDF_PAGEOBJECT) -> B {
        let (mut l, mut b, mut r, mut t) = (0., 0., 0., 0.);
        if self
            .api
            .FPDFPageObj_GetBounds(obj, &mut l, &mut b, &mut r, &mut t)
            == 0
        {
            [0.; 4]
        } else {
            [l as f64, b as f64, r as f64, t as f64]
        }
    }
    unsafe fn clip(
        &mut self,
        obj: FPDF_PAGEOBJECT,
        parent: M,
        inherited: Option<B>,
    ) -> Result<Option<B>, Error> {
        let mut out = inherited;
        let c = self.api.FPDFPageObj_GetClipPath(obj);
        if c.is_null() {
            return Ok(out);
        }
        let n = self.api.FPDFClipPath_CountPaths(c).max(0);
        self.charge(0, n as usize)?;
        for i in 0..n {
            let count = self.api.FPDFClipPath_CountPathSegments(c, i).max(0);
            self.charge(0, count as usize)?;
            let mut points = Vec::new();
            for j in 0..count {
                let seg = self.api.FPDFClipPath_GetPathSegment(c, i, j);
                let (mut x, mut y) = (0., 0.);
                if self.api.FPDFPathSegment_GetPoint(seg, &mut x, &mut y) != 0 {
                    points.push(point([x as f64, y as f64], parent));
                }
            }
            if !points.is_empty() && points.iter().flatten().all(|v| v.is_finite()) {
                let p = points[0];
                let mut b = [p[0], p[1], p[0], p[1]];
                for p in points.into_iter().skip(1) {
                    b = [
                        b[0].min(p[0]),
                        b[1].min(p[1]),
                        b[2].max(p[0]),
                        b[3].max(p[1]),
                    ];
                }
                out = Some(out.map_or(b, |a| intersection(b, a)));
            }
        }
        Ok(out)
    }
    unsafe fn extent(
        &mut self,
        container: FPDF_PAGEOBJECT,
        parent: M,
        inherited: Option<B>,
        depth: usize,
        boxes: &mut Vec<B>,
        had_clip: &mut bool,
    ) -> Result<(), Error> {
        if depth >= 15 {
            return Ok(());
        }
        let count = self.api.FPDFFormObj_CountObjects(container).max(0);
        self.charge(count as usize, 0)?;
        for i in 0..count {
            let obj = self.api.FPDFFormObj_GetObject(container, i as _);
            if obj.is_null() {
                continue;
            }
            let Some(m) = self.matrix(obj) else { continue };
            let clip = self.clip(obj, parent, inherited)?;
            let combined = multiply(m, parent);
            if self.api.FPDFPageObj_GetType(obj) == 5 {
                self.extent(obj, combined, clip, depth + 1, boxes, had_clip)?;
            } else {
                *had_clip |= clip.is_some();
                let b = transform(self.bounds(obj), |p| point(p, parent));
                let b = clip.map_or(b, |c| intersection(b, c));
                if visible(b) {
                    boxes.push(visual(b, ID, self.frame, self.rotation));
                }
            }
        }
        Ok(())
    }
    unsafe fn walk(
        &mut self,
        container: usize,
        parent: M,
        ancestors: &[usize],
        occurrence: &[usize],
        inherited: Option<B>,
        depth: usize,
    ) -> Result<(), Error> {
        if depth >= 15 {
            return Ok(());
        }
        let is_form = !ancestors.is_empty();
        let count = if is_form {
            self.api.FPDFFormObj_CountObjects(container as _)
        } else {
            self.api.FPDFPage_CountObjects(container as _)
        }
        .max(0);
        self.charge(count as usize, 0)?;
        let mut form_index = 0;
        for i in 0..count {
            let obj = if is_form {
                self.api.FPDFFormObj_GetObject(container as _, i as _)
            } else {
                self.api.FPDFPage_GetObject(container as _, i)
            };
            if obj.is_null() {
                continue;
            }
            let Some(m) = self.matrix(obj) else { continue };
            let combined = multiply(m, parent);
            let clip = self.clip(obj, parent, inherited)?;
            let kind = self.api.FPDFPageObj_GetType(obj);
            let order = self.paint;
            self.paint += 1;
            match kind {
                5 => {
                    let mut path = occurrence.to_vec();
                    path.push(form_index);
                    form_index += 1;
                    let mut b = clipped(
                        visual(self.bounds(obj), parent, self.frame, self.rotation),
                        clip,
                        self.frame,
                        self.rotation,
                    );
                    if ancestors.is_empty() {
                        let mut bs = Vec::new();
                        let mut had = false;
                        let top_clip = self.clip(obj, ID, None)?;
                        self.extent(obj, m, top_clip, 1, &mut bs, &mut had)?;
                        if had {
                            let extent = if bs.is_empty() {
                                [0.; 4]
                            } else {
                                let mut r = bs[0];
                                for b in &bs[1..] {
                                    r = [
                                        r[0].min(b[0]),
                                        r[1].min(b[1]),
                                        r[2].max(b[2]),
                                        r[3].max(b[3]),
                                    ];
                                }
                                r
                            };
                            b = b.map(|b| intersection(b, extent));
                        }
                    }
                    b = b
                        .map(|b| intersection(b, [0., 0., self.size[0], self.size[1]]))
                        .filter(|b| visible(*b));
                    let id = self.forms.len();
                    if id >= rapiddoc_core::resources::limit(5000) {
                        return Err(limited());
                    }
                    self.forms.push(FormInfo {
                        instance_id: id,
                        parent_id: ancestors.last().copied(),
                        occurrence: path.clone(),
                        bbox: b.unwrap_or([0.; 4]),
                        declared_bbox: None,
                        media_bbox: [0.; 4],
                        matrix: combined,
                        clip,
                        paint_order: order,
                        text_indices: BTreeSet::new(),
                        path_indices: BTreeSet::new(),
                        image_bboxes: Vec::new(),
                        structure_valid: false,
                    });
                    let mut next = ancestors.to_vec();
                    next.push(id);
                    self.walk(obj as usize, combined, &next, &path, clip, depth + 1)?;
                }
                1 => {
                    if let Some(&id) = ancestors.last() {
                        self.owners.insert(obj as usize, id);
                    }
                }
                2 => {
                    for &id in ancestors {
                        self.forms[id].path_indices.insert(self.paths);
                    }
                    self.paths += 1;
                }
                3 => {
                    let b = intersection(
                        visual([0., 0., 1., 1.], combined, self.frame, self.rotation),
                        [0., 0., self.size[0], self.size[1]],
                    );
                    if let Some(b) = clipped(b, clip, self.frame, self.rotation) {
                        for &id in ancestors {
                            self.forms[id].image_bboxes.push(b);
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }
}
pub fn read(
    library: &Path,
    bytes: &[u8],
    page_index: usize,
    password: Option<&str>,
    limits: &Limits,
) -> Result<Geometry, Error> {
    limits.check_input(bytes)?;
    if password.is_some_and(|p| p.contains('\0')) {
        return Err(error(ErrorCode::InvalidInput, "密码包含NUL"));
    }
    // The complete PDF parser supplies declarations; malformed resources only disable structural proof.
    let declarations = declarations::read(bytes, page_index, password, limits);
    with_raw(library, |api| unsafe {
        let mut h = Handles {
            api,
            doc: api.FPDF_LoadMemDocument64(bytes, password),
            page: std::ptr::null_mut(),
            text: std::ptr::null_mut(),
        };
        if h.doc.is_null() {
            return Err(error(
                if matches!(api.FPDF_GetLastError(), 4 | 5) {
                    ErrorCode::Encrypted
                } else {
                    ErrorCode::CorruptDocument
                },
                "无法打开PDF",
            ));
        }
        let count = api.FPDF_GetPageCount(h.doc);
        if count < 0 {
            return Err(error(ErrorCode::CorruptDocument, "无效页数"));
        }
        if count as usize > limits.max_pages {
            return Err(limited());
        }
        if page_index >= count as usize {
            return Err(error(ErrorCode::InvalidInput, "页号越界"));
        }
        h.page = api.FPDF_LoadPage(h.doc, page_index as _);
        if h.page.is_null() {
            return Err(error(ErrorCode::CorruptDocument, "无法打开PDF页"));
        }
        let mut rect = FS_RECTF {
            left: 0.,
            bottom: 0.,
            right: 0.,
            top: 0.,
        };
        if api.FPDF_GetPageBoundingBox(h.page, &mut rect) == 0 {
            return Err(error(ErrorCode::CorruptDocument, "无法读取页框"));
        }
        let frame = [
            rect.left as f64,
            rect.bottom as f64,
            rect.right as f64,
            rect.top as f64,
        ];
        let rotation = api.FPDFPage_GetRotation(h.page) * 90;
        let size = [
            api.FPDF_GetPageWidthF(h.page) as f64,
            api.FPDF_GetPageHeightF(h.page) as f64,
        ];
        if frame
            .iter()
            .chain(&size)
            .any(|n| !n.is_finite() || n.abs() > 1e9)
            || !visible(frame)
            || !matches!(rotation, 0 | 90 | 180 | 270)
        {
            return Err(error(ErrorCode::CorruptDocument, "PDF页面几何无效"));
        }
        let mut reader = Reader {
            api,
            limits,
            objects: 0,
            segments: 0,
            paths: 0,
            paint: 0,
            forms: Vec::new(),
            owners: HashMap::new(),
            frame,
            rotation,
            size,
        };
        reader.walk(h.page as usize, ID, &[], &[], None, 0)?;
        let boxes = reader
            .forms
            .iter()
            .filter(|f| f.parent_id.is_none() && visible(f.bbox))
            .map(|f| f.bbox)
            .collect();
        if !reader.forms.is_empty() {
            h.text = api.FPDFText_LoadPage(h.page);
            if h.text.is_null() {
                return Err(error(ErrorCode::CorruptDocument, "无法打开文字页"));
            }
            let n = api.FPDFText_CountChars(h.text);
            if n < 0 || n as usize > limits.max_chars_per_page.min(limits.max_total_chars) {
                return Err(limited());
            }
            for i in 0..n {
                let obj = api.FPDFText_GetTextObject(h.text, i);
                let mut owner = reader.owners.get(&(obj as usize)).copied();
                while let Some(id) = owner {
                    reader.forms[id].text_indices.insert(i as usize);
                    owner = reader.forms[id].parent_id;
                }
            }
        }
        let (decls, media) = match declarations {
            Ok((d, m)) => (d, visual(m, ID, frame, rotation)),
            Err(e) if e.code == rapiddoc_core::ErrorCode::ResourceLimit => return Err(e),
            Err(_) => (BTreeMap::new(), [0., 0., size[0], size[1]]),
        };
        let matched = decls.keys().cloned().collect::<BTreeSet<_>>()
            == reader.forms.iter().map(|f| f.occurrence.clone()).collect();
        for i in 0..reader.forms.len() {
            let d = decls.get(&reader.forms[i].occurrence);
            let valid =
                matched
                    && d.is_some_and(|(_, m)| {
                        reader.forms[i].matrix.iter().zip(m).all(|(a, b)| {
                            (a - b).abs() <= 0.001f64.max(1e-5 * a.abs().max(b.abs()))
                        })
                    });
            if valid {
                let (bounds, m) = d.unwrap();
                let declared = visual(*bounds, *m, frame, rotation);
                let mut clip = declared;
                if let Some(c) = reader.forms[i].clip {
                    clip = intersection(clip, visual(c, ID, frame, rotation));
                }
                if let Some(parent) = reader.forms[i].parent_id {
                    if let Some(c) = reader.forms[parent].clip {
                        clip = intersection(clip, c);
                    }
                }
                reader.forms[i].declared_bbox = Some(declared);
                reader.forms[i].clip = Some(clip);
                reader.forms[i].bbox = intersection(reader.forms[i].bbox, clip);
                reader.forms[i].image_bboxes = reader.forms[i]
                    .image_bboxes
                    .iter()
                    .map(|b| intersection(*b, clip))
                    .filter(|b| visible(*b))
                    .collect();
            }
            reader.forms[i].structure_valid = valid;
            reader.forms[i].media_bbox = media;
        }
        Ok(Geometry {
            page_size: size,
            page_bbox: frame,
            rotation,
            form_infos: reader.forms,
            form_bboxes: boxes,
        })
    })
}
