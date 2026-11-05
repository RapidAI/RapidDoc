// metafile-render 0.3.0 fonts/index.py, resolve.py (MIT, Xiaomeng Zhao) and
// Pillow 12.2.0 _imagingft.c BASIC layout (MIT-CMU; licenses/PILLOW.txt).
use super::{malformed, model::Font, resource};
use crate::Result;
use freetype_sys as ft;
use std::{
    cell::RefCell,
    collections::{BTreeMap, HashMap},
    ffi::CStr,
    path::PathBuf,
    rc::Rc,
    sync::{Arc, OnceLock, Weak},
};
fn key(s: &str) -> String {
    crate::ole::casefold(s).replace(' ', "")
}
fn roots() -> Vec<PathBuf> {
    let home = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" })
        .map(PathBuf::from)
        .unwrap_or_default();
    let roots = if cfg!(windows) {
        vec![
            PathBuf::from(std::env::var_os("WINDIR").unwrap_or_else(|| "C:/Windows".into()))
                .join("Fonts"),
        ]
    } else if cfg!(target_os = "macos") {
        vec![
            "/System/Library/Fonts".into(),
            "/Library/Fonts".into(),
            home.join("Library/Fonts"),
        ]
    } else {
        vec![
            "/usr/share/fonts".into(),
            "/usr/local/share/fonts".into(),
            home.join(".fonts"),
            home.join(".local/share/fonts"),
        ]
    };
    roots.into_iter().filter(|p| p.is_dir()).collect()
}
fn index() -> &'static BTreeMap<String, PathBuf> {
    static INDEX: OnceLock<BTreeMap<String, PathBuf>> = OnceLock::new();
    INDEX.get_or_init(|| {
        let mut result = BTreeMap::new();
        for root in roots() {
            let mut stack = vec![root];
            let mut files = vec![];
            while let Some(dir) = stack.pop() {
                if let Ok(entries) = std::fs::read_dir(dir) {
                    for e in entries.flatten() {
                        let p = e.path();
                        if p.is_dir() {
                            if !e.file_type().is_ok_and(|f| f.is_symlink()) {
                                stack.push(p);
                            }
                        } else {
                            files.push(p);
                        }
                    }
                }
            }
            files.sort_by_key(|p| {
                if cfg!(windows) {
                    crate::ole::casefold(&p.to_string_lossy())
                } else {
                    p.to_string_lossy().into_owned()
                }
            });
            for p in files {
                if p.extension().is_some_and(|e| {
                    matches!(
                        e.to_string_lossy().to_ascii_lowercase().as_str(),
                        "ttf" | "ttc" | "otf"
                    )
                }) {
                    if let Some(stem) = p.file_stem() {
                        result.entry(key(&stem.to_string_lossy())).or_insert(p);
                    }
                }
            }
        }
        result
    })
}
pub fn aliases(name: &str, charset: u8) -> Vec<String> {
    let names: &[&str] = match key(name).as_str() {
        "arial" => &["Arial", "LiberationSans-Regular", "DejaVuSans"],
        "calibri" => &["Calibri", "Carlito-Regular", "Arial", "DejaVuSans"],
        "cambria" => &["Cambria", "Caladea-Regular", "DejaVuSerif"],
        "timesnewroman" => &["Times New Roman", "LiberationSerif-Regular", "DejaVuSerif"],
        "couriernew" => &["Courier New", "LiberationMono-Regular", "DejaVuSansMono"],
        "simsun" => &[
            "SimSun",
            "Songti SC",
            "NotoSerifCJKsc-Regular",
            "DejaVuSans",
        ],
        "microsoftyahei" => &[
            "Microsoft YaHei",
            "PingFang SC",
            "NotoSansCJKsc-Regular",
            "DejaVuSans",
        ],
        _ => &[name],
    };
    let mut out: Vec<String> = names.iter().map(|s| s.to_string()).collect();
    if matches!(charset, 128 | 129 | 134 | 136) {
        out.extend(
            [
                "PingFang SC",
                "NotoSansCJKsc-Regular",
                "NotoSansCJK-Regular",
                "DejaVuSans",
            ]
            .map(String::from),
        );
    }
    out.push("DejaVuSans".into());
    dedup(out)
}
fn dedup(values: Vec<String>) -> Vec<String> {
    let mut out = vec![];
    for v in values {
        if !out.contains(&v) {
            out.push(v);
        }
    }
    out
}
pub fn style_candidates(alias: &str, weight: i32, italic: bool) -> Vec<String> {
    let bold = weight >= 700;
    if !bold && !italic {
        return vec![alias.into()];
    }
    let suffixes: &[&str] = if bold && italic {
        &["BoldItalic", "BoldOblique"]
    } else if bold {
        &["Bold"]
    } else {
        &["Italic", "Oblique"]
    };
    let base = alias.strip_suffix("-Regular").unwrap_or(alias);
    let base = base.strip_suffix("Regular").unwrap_or(base);
    let mut out = vec![];
    for suffix in suffixes {
        out.push(format!("{base}{suffix}"));
        out.push(format!("{base}-{suffix}"));
    }
    let short = match key(alias).as_str() {
        "arial" => Some(["arialbd", "ariali", "arialbi"]),
        "timesnewroman" => Some(["timesbd", "timesi", "timesbi"]),
        "couriernew" => Some(["courbd", "couri", "courbi"]),
        "calibri" => Some(["calibrib", "calibrii", "calibriz"]),
        "cambria" => Some(["cambriab", "cambriai", "cambriaz"]),
        _ => None,
    };
    if let Some(short) = short {
        out.insert(
            0,
            short[if bold && italic {
                2
            } else if bold {
                0
            } else {
                1
            }]
            .into(),
        );
    }
    out.push(alias.into());
    dedup(out)
}
struct Library(ft::FT_Library);
impl Drop for Library {
    fn drop(&mut self) {
        unsafe {
            ft::FT_Done_FreeType(self.0);
        }
    }
}
pub struct Face {
    raw: ft::FT_Face,
    _data: Arc<[u8]>,
    _library: Rc<Library>,
    pub family: String,
    pub style: String,
    pub reason: Option<String>,
}
impl Drop for Face {
    fn drop(&mut self) {
        unsafe {
            ft::FT_Done_Face(self.raw);
        }
    }
}
struct Cache {
    library: Rc<Library>,
    faces: HashMap<(String, u32, i32, bool, u8), (Rc<Face>, u64)>,
    files: HashMap<PathBuf, Weak<[u8]>>,
    clock: u64,
}
thread_local! {static CACHE:RefCell<Option<Cache>>=const{RefCell::new(None)};}
fn ok(code: ft::FT_Error) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(crate::Error::malformed(format!("FreeType error {code}")))
    }
}
impl Face {
    pub fn metrics(&self) -> (i32, i32) {
        unsafe {
            let metrics = (*(*self.raw).size).metrics;
            (
                pixel(metrics.ascender as i64) as i32,
                -pixel(metrics.descender as i64) as i32,
            )
        }
    }
    fn new(library: Rc<Library>, data: Arc<[u8]>, size: u32) -> Result<Self> {
        unsafe {
            let mut raw = std::ptr::null_mut();
            ok(ft::FT_New_Memory_Face(
                library.0,
                data.as_ptr(),
                data.len()
                    .try_into()
                    .map_err(|_| resource("font bytes exceed platform address range"))?,
                0,
                &mut raw,
            ))?;
            let mut face = Self {
                raw,
                _data: data,
                _library: library,
                family: String::new(),
                style: String::new(),
                reason: None,
            };
            let mut request = ft::FT_Size_RequestRec {
                size_request_type: ft::FT_SIZE_REQUEST_TYPE_NOMINAL,
                width: (size * 64) as _,
                height: (size * 64) as _,
                horiResolution: 0,
                vertResolution: 0,
            };
            ok(ft::FT_Request_Size(raw, &mut request))?;
            face.family = if (*raw).family_name.is_null() {
                String::new()
            } else {
                CStr::from_ptr((*raw).family_name)
                    .to_string_lossy()
                    .into_owned()
            };
            face.style = if (*raw).style_name.is_null() {
                "Regular".into()
            } else {
                CStr::from_ptr((*raw).style_name)
                    .to_string_lossy()
                    .into_owned()
            };
            Ok(face)
        }
    }
}
pub fn resolve(font: &Font, size: u32) -> Result<Rc<Face>> {
    CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.is_none() {
            let mut library = std::ptr::null_mut();
            unsafe {
                ok(ft::FT_Init_FreeType(&mut library))?;
            }
            *cache = Some(Cache {
                library: Rc::new(Library(library)),
                faces: HashMap::new(),
                files: HashMap::new(),
                clock: 0,
            });
        }
        let cache = cache.as_mut().unwrap();
        cache.clock += 1;
        let size = size.clamp(1, 4096);
        let k = (
            font.face_name.clone(),
            size,
            font.weight,
            font.italic,
            font.charset,
        );
        if let Some((face, stamp)) = cache.faces.get_mut(&k) {
            *stamp = cache.clock;
            return Ok(Rc::clone(face));
        }
        let mut found = None;
        'search: for alias in aliases(&font.face_name, font.charset) {
            for variant in style_candidates(&alias, font.weight, font.italic) {
                let mut candidates = vec![];
                if let Some(path) = index().get(&key(&variant)) {
                    candidates.push(path.clone());
                }
                candidates.push(PathBuf::from(&variant));
                candidates.push(PathBuf::from(format!("{variant}.ttf")));
                // Pillow also searches platform font directories for a bare filename.
                for root in roots() {
                    let path = root.join(format!("{variant}.ttf"));
                    if path.is_file() {
                        candidates.push(path);
                    }
                }
                for path in candidates {
                    let data = if let Some(data) = cache.files.get(&path).and_then(Weak::upgrade) {
                        data
                    } else {
                        let Ok(data) = std::fs::read(&path) else {
                            continue;
                        };
                        let data: Arc<[u8]> = data.into();
                        cache.files.insert(path, Arc::downgrade(&data));
                        data
                    };
                    if let Ok(mut face) = Face::new(Rc::clone(&cache.library), data, size) {
                        if face.family.is_empty() {
                            face.family = alias.clone();
                        }
                        let style = crate::ole::casefold(&face.style);
                        face.reason = if key(&face.family) != key(&font.face_name) {
                            Some("font family substituted".into())
                        } else if (font.weight >= 700 && !style.contains("bold"))
                            || (font.italic
                                && !style.contains("italic")
                                && !style.contains("oblique"))
                        {
                            Some("font style substituted".into())
                        } else {
                            None
                        };
                        found = Some(face);
                        break 'search;
                    }
                }
            }
        }
        let face = if let Some(face) = found {
            face
        } else {
            static DEFAULT: OnceLock<Arc<[u8]>> = OnceLock::new();
            let mut face = Face::new(
                Rc::clone(&cache.library),
                Arc::clone(
                    DEFAULT.get_or_init(|| Arc::from(&include_bytes!("aileron-default.ttf")[..])),
                ),
                size,
            )?;
            face.family = "Pillow default".into();
            face.style = "Regular".into();
            face.reason = Some("requested font unavailable".into());
            face
        };
        let face = Rc::new(face);
        if cache.faces.len() >= 256 {
            let oldest = cache
                .faces
                .iter()
                .min_by_key(|(_, (_, s))| *s)
                .map(|(k, _)| k.clone())
                .unwrap();
            cache.faces.remove(&oldest);
        }
        cache.faces.insert(k, (Rc::clone(&face), cache.clock));
        Ok(face)
    })
}
/// Pillow ImageFont.truetype's filename route, without metafile family aliases.
/// Used by the original Office placeholder font fallback sequence.
pub(crate) fn office_placeholder(size: u32) -> Result<Rc<Face>> {
    let mut library = std::ptr::null_mut();
    unsafe {
        ok(ft::FT_Init_FreeType(&mut library))?;
    }
    let library = Rc::new(Library(library));
    for name in ["DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"] {
        let mut paths = vec![PathBuf::from(name)];
        for root in roots() {
            paths.push(root.join(name));
        }
        // ImageFont also searches subdirectories on Unix platforms.
        if !cfg!(windows) {
            if let Some(path) = index().get(&key(name.trim_end_matches(".ttf"))) {
                paths.push(path.clone());
            }
        }
        for path in paths {
            if let Ok(bytes) = std::fs::read(path) {
                if let Ok(face) = Face::new(Rc::clone(&library), bytes.into(), size) {
                    return Ok(Rc::new(face));
                }
            }
        }
    }
    // ImageFont.load_default() has a fixed 10px size when size is omitted.
    Face::new(
        library,
        Arc::from(&include_bytes!("aileron-default.ttf")[..]),
        10,
    )
    .map(Rc::new)
}
#[derive(Clone, Copy)]
struct Glyph {
    index: u32,
    advance: i64,
}
fn pixel(v: i64) -> i64 {
    (v + 32) >> 6
}
impl Face {
    fn layout(&self, text: &str) -> Result<Vec<Glyph>> {
        unsafe {
            let mut out: Vec<Glyph> = vec![];
            let mut previous = 0;
            for ch in text.chars() {
                let index = ft::FT_Get_Char_Index(self.raw, ch as _);
                ok(ft::FT_Load_Glyph(self.raw, index, ft::FT_LOAD_DEFAULT))?;
                if (*self.raw).face_flags & ft::FT_FACE_FLAG_KERNING != 0
                    && previous != 0
                    && index != 0
                {
                    let mut delta = ft::FT_Vector { x: 0, y: 0 };
                    if ft::FT_Get_Kerning(
                        self.raw,
                        previous,
                        index,
                        ft::FT_KERNING_DEFAULT,
                        &mut delta,
                    ) == 0
                    {
                        out.last_mut().unwrap().advance += pixel(delta.x as i64);
                    }
                }
                out.push(Glyph {
                    index,
                    advance: (*(*self.raw).glyph).metrics.horiAdvance as i64,
                });
                previous = index;
            }
            Ok(out)
        }
    }
    pub fn length(&self, text: &str) -> Result<f64> {
        Ok(self.layout(text)?.iter().map(|g| g.advance).sum::<i64>() as f64 / 64.)
    }
    pub fn bounds(&self, text: &str, anchor: &str) -> Result<[i32; 4]> {
        let glyphs = self.layout(text)?;
        self.bounds_glyphs(&glyphs, anchor)
    }
    fn bounds_glyphs(&self, glyphs: &[Glyph], anchor: &str) -> Result<[i32; 4]> {
        unsafe {
            let (mut position, mut xmin, mut ymin, mut xmax, mut ymax) =
                (0i64, 0i64, 0i64, 0i64, 0i64);
            for g in glyphs {
                let px = pixel(position);
                position += g.advance;
                xmax = xmax.max(pixel(position));
                ok(ft::FT_Load_Glyph(self.raw, g.index, ft::FT_LOAD_DEFAULT))?;
                let mut glyph = std::ptr::null_mut();
                ok(ft::FT_Get_Glyph((*self.raw).glyph, &mut glyph))?;
                let mut bbox = std::mem::zeroed();
                ft::FT_Glyph_Get_CBox(glyph, ft::FT_GLYPH_BBOX_PIXELS, &mut bbox);
                ft::FT_Done_Glyph(glyph);
                xmin = xmin.min(bbox.xMin as i64 + px);
                xmax = xmax.max(bbox.xMax as i64 + px);
                ymin = ymin.min(bbox.yMin as i64);
                ymax = ymax.max(bbox.yMax as i64);
            }
            let a = anchor.as_bytes();
            if a.len() != 2 {
                return Err(malformed());
            }
            let xa = if glyphs.is_empty() {
                0
            } else {
                match a[0] {
                    b'l' => 0,
                    b'm' => pixel(position / 2),
                    b'r' => pixel(position),
                    _ => return Err(malformed()),
                }
            };
            let metrics = (*(*self.raw).size).metrics;
            let ya = if glyphs.is_empty() {
                0
            } else {
                match a[1] {
                    b'a' => pixel(metrics.ascender as i64),
                    b't' => ymax,
                    b'm' => pixel((metrics.ascender as i64 + metrics.descender as i64) / 2),
                    b's' => 0,
                    b'b' => ymin,
                    b'd' => pixel(metrics.descender as i64),
                    _ => return Err(malformed()),
                }
            };
            Ok([
                (xmin - xa) as i32,
                (ya - ymax) as i32,
                (xmax - xa) as i32,
                (ya - ymin) as i32,
            ])
        }
    }
    pub fn mask(
        &self,
        text: &str,
        anchor: &str,
        start: [f64; 2],
    ) -> Result<([i32; 2], [usize; 2], Vec<u8>)> {
        unsafe {
            let glyphs = self.layout(text)?;
            let bbox = self.bounds_glyphs(&glyphs, anchor)?;
            let width = (bbox[2] - bbox[0] + start[0].ceil() as i32).max(0) as usize;
            let height = (bbox[3] - bbox[1] + start[1].ceil() as i32).max(0) as usize;
            if width == 0 || height == 0 {
                return Ok(([bbox[0], bbox[1]], [width, height], vec![]));
            }
            let pixels = width
                .checked_mul(height)
                .ok_or_else(|| resource("font mask exceeds platform address range"))?;
            let (mut x, mut xmin, mut ymax) = (0i64, 0i64, 0i64);
            for g in &glyphs {
                let px = pixel(x);
                ok(ft::FT_Load_Glyph(
                    self.raw,
                    g.index,
                    ft::FT_LOAD_DEFAULT | ft::FT_LOAD_RENDER,
                ))?;
                let slot = &*(*self.raw).glyph;
                xmin = xmin.min(slot.bitmap_left as i64 + px);
                ymax = ymax.max(slot.bitmap_top as i64);
                x += g.advance;
            }
            let mut x = ((-xmin as f64 + start[0]) * 64.).round() as i64;
            let y = ((-ymax as f64 - start[1]) * 64.).round() as i64;
            let mut mask = vec![0u8; pixels];
            for g in glyphs {
                let px = pixel(x);
                let py = pixel(y);
                ok(ft::FT_Load_Glyph(
                    self.raw,
                    g.index,
                    ft::FT_LOAD_DEFAULT | ft::FT_LOAD_RENDER,
                ))?;
                let slot = &*(*self.raw).glyph;
                let bitmap = &slot.bitmap;
                let xx = px + slot.bitmap_left as i64;
                let yy = -(py + slot.bitmap_top as i64);
                if !bitmap.buffer.is_null() {
                    let mode = bitmap.pixel_mode as u32;
                    if !matches!(
                        mode,
                        ft::FT_PIXEL_MODE_GRAY
                            | ft::FT_PIXEL_MODE_MONO
                            | ft::FT_PIXEL_MODE_GRAY2
                            | ft::FT_PIXEL_MODE_GRAY4
                    ) {
                        return Err(crate::Error::new(
                            crate::ErrorKind::Unsupported,
                            "FreeType bitmap pixel mode not connected",
                        ));
                    }
                    let pitch = bitmap.pitch.unsigned_abs() as usize;
                    for row in 0..bitmap.rows as usize {
                        let dy = yy + row as i64;
                        if dy < 0 || dy >= height as i64 {
                            continue;
                        }
                        let sy = if bitmap.pitch < 0 {
                            bitmap.rows as usize - 1 - row
                        } else {
                            row
                        };
                        let bytes =
                            std::slice::from_raw_parts(bitmap.buffer.add(sy * pitch), pitch);
                        for col in 0..bitmap.width as usize {
                            let dx = xx + col as i64;
                            if dx < 0 || dx >= width as i64 {
                                continue;
                            }
                            let alpha = match mode {
                                ft::FT_PIXEL_MODE_MONO => {
                                    ((bytes[col / 8] >> (7 - col % 8)) & 1) * 255
                                }
                                ft::FT_PIXEL_MODE_GRAY2 => {
                                    ((bytes[col / 4] >> (6 - (col % 4) * 2)) & 3) * 85
                                }
                                ft::FT_PIXEL_MODE_GRAY4 => {
                                    ((bytes[col / 2] >> (4 - (col % 2) * 4)) & 15) * 17
                                }
                                _ => bytes[col],
                            };
                            let dst = &mut mask[dy as usize * width + dx as usize];
                            let v = u32::from(*dst) * (255 - u32::from(alpha)) + 128;
                            *dst = (u32::from(alpha) + (((v >> 8) + v) >> 8)).min(255) as u8;
                        }
                    }
                }
                x += g.advance;
            }
            Ok(([bbox[0], bbox[1]], [width, height], mask))
        }
    }
}
pub fn advance(font: &Font, text: &str) -> Result<f64> {
    if text.is_empty() {
        return Ok(0.);
    }
    let size = if font.height == 0. {
        12.
    } else {
        font.height.abs()
    }
    .round_ties_even()
    .max(1.) as u32;
    let face = resolve(font, size)?;
    let mut advance = face.length(text)?;
    if font.width != 0. {
        advance *= font.width.abs() / face.length("0")?.max(1e-9);
    }
    Ok(advance.max(0.))
}
