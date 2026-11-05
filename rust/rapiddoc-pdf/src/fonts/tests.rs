use super::*;
use std::ffi::{c_char, c_int, c_ulong, CStr};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
#[repr(C)]
struct Info {
    version: c_int,
    release: Option<unsafe extern "C" fn(*mut Info)>,
    enumerate: Option<unsafe extern "C" fn(*mut Info, *mut c_void)>,
    map: Option<
        unsafe extern "C" fn(
            *mut Info,
            c_int,
            c_int,
            c_int,
            c_int,
            *const c_char,
            *mut c_int,
        ) -> *mut c_void,
    >,
    get: Option<unsafe extern "C" fn(*mut Info, *const c_char) -> *mut c_void>,
    data: Option<unsafe extern "C" fn(*mut Info, *mut c_void, u32, *mut u8, c_ulong) -> c_ulong>,
    name: Option<unsafe extern "C" fn(*mut Info, *mut c_void, *mut c_char, c_ulong) -> c_ulong>,
    charset: Option<unsafe extern "C" fn(*mut Info, *mut c_void) -> c_int>,
    delete: Option<unsafe extern "C" fn(*mut Info, *mut c_void)>,
}
static INSTALLED: AtomicPtr<Info> = AtomicPtr::new(std::ptr::null_mut());
static DEFAULT: AtomicPtr<Info> = AtomicPtr::new(std::ptr::null_mut());
static FREED: AtomicUsize = AtomicUsize::new(0);
static DELETED: AtomicUsize = AtomicUsize::new(0);
static ENUM: AtomicUsize = AtomicUsize::new(0);
unsafe extern "system" fn get_default() -> *mut Info {
    DEFAULT.load(Ordering::SeqCst)
}
unsafe extern "system" fn free_default(_: *mut Info) {
    FREED.fetch_add(1, Ordering::SeqCst);
}
unsafe extern "system" fn set(info: *mut Info) {
    INSTALLED.store(info, Ordering::SeqCst);
}
unsafe extern "C" fn get(_: *mut Info, _: *const c_char) -> *mut c_void {
    0xabcdusize as *mut c_void
}
unsafe extern "C" fn map(
    _: *mut Info,
    _: c_int,
    _: c_int,
    _: c_int,
    _: c_int,
    _: *const c_char,
    exact: *mut c_int,
) -> *mut c_void {
    if !exact.is_null() {
        unsafe {
            *exact = 1;
        }
    }
    0xabcdusize as *mut c_void
}
unsafe extern "C" fn name(
    _: *mut Info,
    _: *mut c_void,
    out: *mut c_char,
    size: c_ulong,
) -> c_ulong {
    let value = b"SystemFont\0";
    if !out.is_null() && size as usize >= value.len() {
        unsafe {
            std::ptr::copy_nonoverlapping(value.as_ptr(), out.cast(), value.len());
        }
    }
    value.len() as c_ulong
}
unsafe extern "C" fn delete(_: *mut Info, _: *mut c_void) {
    DELETED.fetch_add(1, Ordering::SeqCst);
}
unsafe extern "C" fn enumerate(_: *mut Info, _: *mut c_void) {
    unsafe {
        let info = INSTALLED.load(Ordering::SeqCst);
        let handle = ((*info).get.unwrap())(info, c"SimSun".as_ptr());
        let mut buffer = [0i8; 64];
        ((*info).name.unwrap())(info, handle, buffer.as_mut_ptr(), 64);
        if CStr::from_ptr(buffer.as_ptr()).to_bytes() == b"SystemFont" {
            ENUM.fetch_add(1, Ordering::SeqCst);
        }
        ((*info).delete.unwrap())(info, handle);
    }
}

#[test]
fn independent_python_font_and_decoder_reference() {
    let resources = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("resources/fonts");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(resources.join("SOURCES.json")).unwrap()).unwrap();
    for (file, hash) in manifest["files"].as_object().unwrap() {
        assert_eq!(
            format!(
                "{:x}",
                Sha256::digest(std::fs::read(resources.join(file)).unwrap())
            ),
            hash.as_str().unwrap()
        );
    }
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(
                std::fs::read(
                    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/fonts/provider.rs")
                )
                .unwrap()
            )
        ),
        manifest["provider_sha256"].as_str().unwrap()
    );
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../build/migration/native-fixed-font-runtime.json");
    let f: serde_json::Value = serde_json::from_slice(&std::fs::read(root).unwrap()).unwrap();
    for c in f["cases"].as_array().unwrap() {
        let bytes: Vec<_> = c["face"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b.as_u64().unwrap() as u8)
            .collect();
        assert_eq!(
            serde_json::json!(classify(&bytes, c["charset"].as_i64().unwrap() as i32)),
            c["expected"],
            "{c}"
        );
    }
    for c in f["decodes"].as_array().unwrap() {
        let bytes: Vec<_> = c["face"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b.as_u64().unwrap() as u8)
            .collect();
        assert_eq!(
            serde_json::json!(decode(&bytes, c["codec"].as_u64().unwrap() as usize)),
            c["expected"],
            "{c}"
        );
    }
}
#[test]
fn same_abi_reentry_buffers_ownership_and_permanent_errors() {
    unsafe {
        let mut default = Box::new(Info {
            version: 1,
            release: None,
            enumerate: Some(enumerate),
            map: Some(map),
            get: Some(get),
            data: None,
            name: Some(name),
            charset: None,
            delete: Some(delete),
        });
        DEFAULT.store(default.as_mut(), Ordering::SeqCst);
        let make = || {
            FontProvider::new(
                [
                    get_default as *const () as usize,
                    free_default as *const () as usize,
                    set as *const () as usize,
                ],
                (0u8..16).collect(),
                HashMap::from([(0x123, (4, 4))]),
                b"fixed-cache".to_vec(),
                policy().aliases.clone(),
                policy().suffixes.clone(),
                legacy_classify as *const () as usize,
            )
            .unwrap()
        };
        let p = make();
        let info = p.interface_address() as *mut Info;
        assert_eq!((*info).version, 1);
        p.install().unwrap();
        p.install().unwrap();
        assert_eq!(Arc::strong_count(&p), 2);
        ((*info).enumerate.unwrap())(info, std::ptr::null_mut());
        assert_eq!(ENUM.load(Ordering::SeqCst), 1);
        let mut exact = 0;
        let system = ((*info).map.unwrap())(info, 400, 0, 2, 0, c"SimSun".as_ptr(), &mut exact);
        assert_eq!(exact, 1);
        let second = ((*info).get.unwrap())(info, c"SimSun".as_ptr());
        let fixed = ((*info).map.unwrap())(
            info,
            400,
            0,
            134,
            0,
            c"STSong".as_ptr(),
            std::ptr::null_mut(),
        );
        assert_eq!(((*info).charset.unwrap())(info, fixed), 134);
        assert_ne!(system, second);
        let mut small = [99u8; 3];
        assert_eq!(
            ((*info).data.unwrap())(info, fixed, 0x123, small.as_mut_ptr(), 3),
            4
        );
        assert_eq!(small, [99; 3]);
        let mut table = [99u8; 4];
        assert_eq!(
            ((*info).data.unwrap())(info, fixed, 0x123, table.as_mut_ptr(), 4),
            4
        );
        assert_eq!(table, [4, 5, 6, 7]);
        let mut data = [99u8; 16];
        assert_eq!(
            ((*info).data.unwrap())(info, fixed, 0, data.as_mut_ptr(), 16),
            16
        );
        assert_eq!(data, std::array::from_fn(|i| i as u8));
        let mut buf = [0i8; 64];
        assert_eq!(
            ((*info).name.unwrap())(info, fixed, buf.as_mut_ptr(), 64),
            12
        );
        assert_eq!(CStr::from_ptr(buf.as_ptr()).to_bytes(), b"fixed-cache");
        assert_eq!(p.stats(), (false, 1, 1));
        ((*info).delete.unwrap())(info, fixed);
        p.check().unwrap();
        ((*info).release.unwrap())(info);
        ((*info).release.unwrap())(info);
        assert_eq!(Arc::strong_count(&p), 1);
        assert_eq!(FREED.load(Ordering::SeqCst), 1);
        assert_eq!(DELETED.load(Ordering::SeqCst), 3);
        assert!(p.check().is_err());
        drop(p);
        assert_eq!(FREED.load(Ordering::SeqCst), 1);
        let p = make();
        let info = p.interface_address() as *mut Info;
        assert_eq!(
            ((*info).data.unwrap())(info, 999usize as *mut c_void, 0, std::ptr::null_mut(), 0),
            0
        );
        assert!(p.check().is_err());
        assert!(p.check().is_err());
        drop(p);
        assert_eq!(FREED.load(Ordering::SeqCst), 2);
    }
}
