//! Same GEOS operations used by the original Shapely overlap decision.
//! Dynamic linking allows replacement of the LGPL runtime by downstream users.
use libloading::Library;
use rapiddoc_core::{Error, ErrorCode};
use std::{
    ffi::{c_char, c_int, c_uint, c_void},
    path::Path,
};
type Ptr = *mut c_void;
type Init = unsafe extern "C" fn() -> Ptr;
type Finish = unsafe extern "C" fn(Ptr);
type Create = unsafe extern "C" fn(Ptr, c_uint, c_uint) -> Ptr;
type Set = unsafe extern "C" fn(Ptr, Ptr, c_uint, f64) -> c_int;
type Ring = unsafe extern "C" fn(Ptr, Ptr) -> Ptr;
type Polygon = unsafe extern "C" fn(Ptr, Ptr, *mut Ptr, c_uint) -> Ptr;
type Destroy = unsafe extern "C" fn(Ptr, Ptr);
type Valid = unsafe extern "C" fn(Ptr, Ptr) -> c_char;
type Buffer = unsafe extern "C" fn(Ptr, Ptr, f64, c_int) -> Ptr;
type Binary = unsafe extern "C" fn(Ptr, Ptr, Ptr) -> Ptr;
type Area = unsafe extern "C" fn(Ptr, Ptr, *mut f64) -> c_int;
fn error(message: impl ToString) -> Error {
    Error {
        code: ErrorCode::RuntimeUnavailable,
        message: format!("GEOS: {}", message.to_string()),
    }
}
unsafe fn load_library(path: &Path) -> Result<Library, Error> {
    #[cfg(target_os = "windows")]
    {
        libloading::os::windows::Library::load_with_flags(path, 0x00000100 | 0x00001000)
            .map(Into::into)
            .map_err(|e| error(format!("{}: {e}", path.display())))
    }
    #[cfg(not(target_os = "windows"))]
    {
        Library::new(path).map_err(error)
    }
}
pub struct Context {
    context: Ptr,
    finish: Finish,
    create: Create,
    set_x: Set,
    set_y: Set,
    ring: Ring,
    polygon: Polygon,
    destroy: Destroy,
    destroy_seq: Destroy,
    valid: Valid,
    buffer: Buffer,
    intersection: Binary,
    union: Binary,
    area: Area,
    _libraries: Vec<Library>,
}
impl Context {
    pub fn load(path: &Path) -> Result<Self, Error> {
        unsafe {
            let mut libraries = vec![];
            #[cfg(target_os = "windows")]
            {
                let parent = path
                    .parent()
                    .ok_or_else(|| error("Missing runtime directory"))?;
                let mut deps = std::fs::read_dir(parent)
                    .map_err(error)?
                    .filter_map(Result::ok)
                    .map(|e| e.path())
                    .filter(|p| {
                        let n = p.file_name().unwrap_or_default().to_string_lossy();
                        p.extension().is_some_and(|e| e == "dll")
                            && (n.starts_with("msvcp") || n.starts_with("geos-"))
                    })
                    .collect::<Vec<_>>();
                deps.sort_by_key(|p| {
                    (
                        !p.file_name()
                            .unwrap()
                            .to_string_lossy()
                            .starts_with("msvcp"),
                        p.clone(),
                    )
                });
                for dependency in deps {
                    libraries.push(load_library(&dependency)?);
                }
            }
            let library = load_library(path)?;
            macro_rules! symbol {
                ($name:literal,$ty:ty) => {
                    *library
                        .get::<$ty>(concat!($name, "\0").as_bytes())
                        .map_err(error)?
                };
            }
            let init = symbol!("GEOS_init_r", Init);
            let finish = symbol!("GEOS_finish_r", Finish);
            let create = symbol!("GEOSCoordSeq_create_r", Create);
            let set_x = symbol!("GEOSCoordSeq_setX_r", Set);
            let set_y = symbol!("GEOSCoordSeq_setY_r", Set);
            let ring = symbol!("GEOSGeom_createLinearRing_r", Ring);
            let polygon = symbol!("GEOSGeom_createPolygon_r", Polygon);
            let destroy = symbol!("GEOSGeom_destroy_r", Destroy);
            let destroy_seq = symbol!("GEOSCoordSeq_destroy_r", Destroy);
            let valid = symbol!("GEOSisValid_r", Valid);
            let buffer = symbol!("GEOSBuffer_r", Buffer);
            let intersection = symbol!("GEOSIntersection_r", Binary);
            let union = symbol!("GEOSUnion_r", Binary);
            let area = symbol!("GEOSArea_r", Area);
            libraries.push(library);
            let context = init();
            if context.is_null() {
                return Err(error("Could not initialize context"));
            }
            Ok(Self {
                context,
                finish,
                create,
                set_x,
                set_y,
                ring,
                polygon,
                destroy,
                destroy_seq,
                valid,
                buffer,
                intersection,
                union,
                area,
                _libraries: libraries,
            })
        }
    }
    fn polygon(&self, points: &[[f64; 2]]) -> Result<Geometry<'_>, Error> {
        self.make_polygon(points, true)
    }
    fn make_polygon(&self, points: &[[f64; 2]], repair: bool) -> Result<Geometry<'_>, Error> {
        unsafe {
            if points.len() < 3 || points.iter().flatten().any(|x| !x.is_finite()) {
                return Err(error("Invalid polygon coordinates"));
            }
            let close = points.first() != points.last();
            let count = points.len() + usize::from(close);
            let count = c_uint::try_from(count).map_err(error)?;
            let seq = (self.create)(self.context, count, 2);
            if seq.is_null() {
                return Err(error("Coordinate allocation failed"));
            }
            for i in 0..count as usize {
                let p = points[i % points.len()];
                if (self.set_x)(self.context, seq, i as _, p[0]) == 0
                    || (self.set_y)(self.context, seq, i as _, p[1]) == 0
                {
                    (self.destroy_seq)(self.context, seq);
                    return Err(error("Coordinate assignment failed"));
                }
            }
            // Both constructors take ownership of their input, including error paths.
            let ring = (self.ring)(self.context, seq);
            if ring.is_null() {
                return Err(error("Invalid ring"));
            }
            let p = (self.polygon)(self.context, ring, std::ptr::null_mut(), 0);
            if p.is_null() {
                return Err(error("Polygon construction failed"));
            }
            let mut geometry = Geometry {
                ptr: p,
                context: self,
            };
            match (self.valid)(self.context, p) {
                1 => {}
                0 if !repair => {}
                0 => {
                    let repaired = (self.buffer)(self.context, p, 0., 16);
                    if repaired.is_null() {
                        return Err(error("Polygon repair failed"));
                    }
                    (self.destroy)(self.context, p);
                    geometry.ptr = repaired;
                }
                _ => return Err(error("Validity check failed")),
            }
            Ok(geometry)
        }
    }
    fn area(&self, ptr: Ptr) -> Result<f64, Error> {
        unsafe {
            let mut area = 0.;
            if (self.area)(self.context, ptr, &mut area) == 0 {
                Err(error("Area computation failed"))
            } else {
                Ok(area)
            }
        }
    }
    /// Seal crops deliberately retain the original invalid-polygon rectangle fallback.
    pub fn seal_overlap(&self, a: &[[f64; 2]], b: &[[f64; 2]]) -> Result<Option<f64>, Error> {
        unsafe {
            let a = self.make_polygon(a, false)?;
            if (self.valid)(self.context, a.ptr) == 0 {
                return Ok(None);
            }
            let b = self.make_polygon(b, false)?;
            let intersection = (self.intersection)(self.context, a.ptr, b.ptr);
            if intersection.is_null() {
                return Err(error("Seal intersection failed"));
            }
            let intersection = Geometry {
                ptr: intersection,
                context: self,
            };
            let union = (self.union)(self.context, a.ptr, b.ptr);
            if union.is_null() {
                return Err(error("Seal union failed"));
            }
            let union = Geometry {
                ptr: union,
                context: self,
            };
            Ok(Some(
                self.area(intersection.ptr)? / (self.area(union.ptr)? + 1e-10),
            ))
        }
    }
    pub fn overlap(&self, a: &[[f64; 2]], b: &[[f64; 2]], mode: &str) -> Result<f64, Error> {
        unsafe {
            let a = self.polygon(a)?;
            let b = self.polygon(b)?;
            let intersection = (self.intersection)(self.context, a.ptr, b.ptr);
            if intersection.is_null() {
                return Err(error("Intersection failed"));
            }
            let intersection = Geometry {
                ptr: intersection,
                context: self,
            };
            let numerator = self.area(intersection.ptr)?;
            // Original code evaluates the union even for the small/large modes.
            let union = (self.union)(self.context, a.ptr, b.ptr);
            if union.is_null() {
                return Err(error("Union failed"));
            }
            let union = Geometry {
                ptr: union,
                context: self,
            };
            let union_area = self.area(union.ptr)?;
            let denominator = match mode {
                "union" => union_area,
                "small" => self.area(a.ptr)?.min(self.area(b.ptr)?),
                "large" => self.area(a.ptr)?.max(self.area(b.ptr)?),
                _ => return Err(error("Unknown overlap mode")),
            };
            if denominator == 0. {
                return Err(Error {
                    code: ErrorCode::InvalidInput,
                    message: "Polygon overlap division by zero".into(),
                });
            }
            Ok(numerator / denominator)
        }
    }
}
struct Geometry<'a> {
    ptr: Ptr,
    context: &'a Context,
}
impl Drop for Geometry<'_> {
    fn drop(&mut self) {
        unsafe { (self.context.destroy)(self.context.context, self.ptr) }
    }
}
impl Drop for Context {
    fn drop(&mut self) {
        unsafe { (self.finish)(self.context) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn packaged_geometry_runtime_with_source_archive() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../bindings/python/python/rapid_doc_rust/runtime/geos");
        assert!(root.join("geos-3.13.1.tar.bz2").exists());
        let library = std::fs::read_dir(root)
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .find(|p| {
                p.file_name()
                    .unwrap()
                    .to_string_lossy()
                    .starts_with("geos_c")
                    && p.extension().is_some_and(|e| e == "dll")
            })
            .unwrap();
        let context = Context::load(&library).unwrap();
        let square = [[0., 0.], [10., 0.], [10., 10.], [0., 10.]];
        assert_eq!(context.overlap(&square, &square, "union").unwrap(), 1.);
    }
}
