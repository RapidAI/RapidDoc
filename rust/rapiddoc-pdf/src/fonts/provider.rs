//! 固定字体回调的原生实现；仅调用宿主传入的同库 PDFium 地址。
use parking_lot::ReentrantMutex;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_ulong, c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

// PDFium 字体接口字段使用 C 调用约定，导出的库函数单独保留 system ABI。
#[repr(C)]
pub struct FontInfo {
    version: c_int,
    release: Option<unsafe extern "C" fn(*mut FontInfo)>,
    enum_fonts: Option<unsafe extern "C" fn(*mut FontInfo, *mut c_void)>,
    map_font: Option<
        unsafe extern "C" fn(
            *mut FontInfo,
            c_int,
            c_int,
            c_int,
            c_int,
            *const c_char,
            *mut c_int,
        ) -> *mut c_void,
    >,
    get_font: Option<unsafe extern "C" fn(*mut FontInfo, *const c_char) -> *mut c_void>,
    get_data:
        Option<unsafe extern "C" fn(*mut FontInfo, *mut c_void, u32, *mut u8, c_ulong) -> c_ulong>,
    get_name:
        Option<unsafe extern "C" fn(*mut FontInfo, *mut c_void, *mut c_char, c_ulong) -> c_ulong>,
    get_charset: Option<unsafe extern "C" fn(*mut FontInfo, *mut c_void) -> c_int>,
    delete_font: Option<unsafe extern "C" fn(*mut FontInfo, *mut c_void)>,
}
#[derive(Clone, Copy)]
enum Handle {
    Bundled(i32),
    System(usize),
}

/// 回调表必须位于首字段；注册后的 Arc 强引用由 Release 回调归还。
#[repr(C)]
pub struct FontProvider {
    interface: FontInfo,
    lock: ReentrantMutex<()>,
    default: Cell<*mut FontInfo>,
    free: unsafe extern "system" fn(*mut FontInfo),
    set: unsafe extern "system" fn(*mut FontInfo),
    data: Vec<u8>,
    tables: HashMap<u32, (usize, usize)>,
    cache_name: Vec<u8>,
    aliases: HashMap<String, i32>,
    suffixes: Vec<String>,
    classify_legacy: unsafe extern "C" fn(*const u8, usize, i32) -> i32,
    handles: RefCell<HashMap<usize, Handle>>,
    requests: RefCell<HashMap<Vec<u8>, i32>>,
    next: Cell<usize>,
    depth: Cell<usize>,
    installed: Cell<bool>,
    released: Cell<bool>,
    failure: RefCell<Option<String>>,
    bundled_requests: Cell<usize>,
    full_font_copies: Cell<usize>,
}

// 安全约束：所有可变状态访问均持有可重入锁，借用在外部回调前结束；默认接口操作另受宿主 PDFium 全局锁保护。
unsafe impl Send for FontProvider {}
unsafe impl Sync for FontProvider {}

impl FontProvider {
    /// 验证资源边界并借用已初始化运行库的三个入口，绝不自行加载或初始化 PDFium。
    /// # Safety
    /// 地址须具有匹配 ABI、保持存活，全部方法和回调须由宿主运行时锁串行化。
    pub unsafe fn new(
        addresses: [usize; 3],
        data: Vec<u8>,
        tables: HashMap<u32, (usize, usize)>,
        cache_name: Vec<u8>,
        aliases: HashMap<String, i32>,
        suffixes: Vec<String>,
        classify_legacy: usize,
    ) -> Result<Arc<Self>, String> {
        if classify_legacy == 0
            || addresses.contains(&0)
            || tables
                .values()
                .any(|(offset, size)| offset.checked_add(*size).is_none_or(|end| end > data.len()))
        {
            return Err("Invalid font runtime arguments".into());
        }
        let get: unsafe extern "system" fn() -> *mut FontInfo =
            unsafe { std::mem::transmute(addresses[0]) };
        let default = unsafe { get() };
        if default.is_null() {
            return Err("PDFium did not provide its default system font interface".into());
        }
        Ok(Arc::new(Self {
            interface: FontInfo {
                version: 1,
                release: Some(release),
                enum_fonts: Some(enumerate),
                map_font: Some(map_font),
                get_font: Some(get_font),
                get_data: Some(get_data),
                get_name: Some(get_name),
                get_charset: Some(get_charset),
                delete_font: Some(delete_font),
            },
            lock: ReentrantMutex::new(()),
            default: Cell::new(default),
            free: unsafe {
                std::mem::transmute::<usize, unsafe extern "system" fn(*mut FontInfo)>(addresses[1])
            },
            set: unsafe {
                std::mem::transmute::<usize, unsafe extern "system" fn(*mut FontInfo)>(addresses[2])
            },
            data,
            tables,
            cache_name,
            aliases,
            suffixes,
            classify_legacy: unsafe {
                std::mem::transmute::<usize, unsafe extern "C" fn(*const u8, usize, i32) -> i32>(
                    classify_legacy,
                )
            },
            handles: RefCell::new(HashMap::new()),
            requests: RefCell::new(HashMap::new()),
            next: Cell::new(0),
            depth: Cell::new(0),
            installed: Cell::new(false),
            released: Cell::new(false),
            failure: RefCell::new(None),
            bundled_requests: Cell::new(0),
            full_font_copies: Cell::new(0),
        }))
    }
    /// 安装时增加原生所有权，使 Python 包装器提前回收也不会产生悬空回调。
    pub fn install(self: &Arc<Self>) -> Result<(), String> {
        let _guard = self.lock.lock();
        if self.released.get() {
            return self.check();
        }
        if !self.installed.replace(true) {
            let pointer = Arc::into_raw(self.clone()) as *mut FontInfo;
            unsafe {
                (self.set)(pointer);
            }
        }
        self.check()
    }
    /// 在离开 C 边界后读取永久错误，不清除已经记录的故障。
    pub fn check(&self) -> Result<(), String> {
        let _guard = self.lock.lock();
        if let Some(error) = self.failure.borrow().as_ref() {
            return Err(error.clone());
        }
        if self.released.get() {
            return Err("The PDFium font runtime has already been released".into());
        }
        Ok(())
    }
    /// 暴露只供 ABI 差分测试调用的接口地址，调用方必须保持提供器存活。
    pub fn interface_address(&self) -> usize {
        &self.interface as *const FontInfo as usize
    }
    /// 返回不依赖 Python 回调的运行时诊断。
    pub fn stats(&self) -> (bool, usize, usize) {
        let _guard = self.lock.lock();
        (
            self.released.get(),
            self.bundled_requests.get(),
            self.full_font_copies.get(),
        )
    }
    /// 分配与系统地址无关的 opaque 句柄，溢出作为永久回调故障处理。
    fn allocate(&self, handle: Handle) -> *mut c_void {
        let id = self
            .next
            .get()
            .checked_add(1)
            .expect("font handle exhausted");
        self.next.set(id);
        self.handles.borrow_mut().insert(id, handle);
        id as *mut c_void
    }
    /// 取得自有句柄，非法句柄由最外层 panic 隔离转为 ABI 失败值。
    fn lookup(&self, handle: *mut c_void) -> Handle {
        *self
            .handles
            .borrow()
            .get(&(handle as usize))
            .expect("unknown font handle")
    }
    /// 使用固定字符集及显式族名识别规则，系统字体请求不做任意子串匹配。
    pub fn classify(&self, face: &[u8], charset: i32) -> Option<i32> {
        let _guard = self.lock.lock();
        if [128, 129, 134, 136].contains(&charset) {
            return Some(charset);
        }
        if charset != 0 && charset != 1 {
            return None;
        }
        // 非 ASCII 名称保留宿主五编码及 Unicode casefold 的精确语义，仅此低频边界回调 Python。
        if !face.is_ascii() {
            let value = unsafe { (self.classify_legacy)(face.as_ptr(), face.len(), charset) };
            return (value >= 0).then_some(value);
        }
        let mut name = std::str::from_utf8(face)
            .expect("ASCII font name")
            .trim_start_matches('@');
        if name.len() >= 7
            && name.as_bytes()[..6].iter().all(u8::is_ascii_uppercase)
            && name.as_bytes()[6] == b'+'
        {
            name = &name[7..];
        }
        let name: String = name
            .chars()
            .filter(|c| {
                !c.is_ascii_whitespace()
                    && ![
                        '\u{000b}', '\u{001c}', '\u{001d}', '\u{001e}', '\u{001f}', ',', '_', '-',
                    ]
                    .contains(c)
            })
            .map(|c| c.to_ascii_lowercase())
            .collect();
        let candidates = std::iter::once(name.as_str()).chain(
            self.suffixes
                .iter()
                .filter_map(|suffix| name.strip_suffix(suffix.as_str())),
        );
        for candidate in candidates {
            if let Some(value) = self.aliases.get(candidate) {
                return Some(*value);
            }
            for (marker, value) in [("gb2312", 134), ("gbk", 134), ("big5", 136)] {
                if candidate.ends_with(marker) {
                    return Some(value);
                }
            }
        }
        None
    }
    /// 清理全部系统句柄及默认接口；允许未注册对象析构和重复清理。
    fn cleanup(&self) {
        let _guard = self.lock.lock();
        if self.released.replace(true) {
            return;
        }
        let default = self.default.replace(std::ptr::null_mut());
        if default.is_null() {
            return;
        }
        let handles = std::mem::take(&mut *self.handles.borrow_mut());
        unsafe {
            for handle in handles.into_values() {
                if let Handle::System(value) = handle {
                    if let Some(delete) = (*default).delete_font {
                        delete(default, value as *mut c_void);
                    }
                }
            }
            (self.free)(default);
        }
    }
}
impl Drop for FontProvider {
    /// 未安装的对象仍负责释放取得的默认接口；已释放对象不重复调用 PDFium。
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// 对所有 C 回调统一隔离 Rust panic，并保留首次失败供宿主检查。
unsafe fn guarded<T>(
    info: *mut FontInfo,
    fallback: T,
    name: &str,
    action: impl FnOnce(&FontProvider) -> T,
) -> T {
    let provider = unsafe { &*(info as *const FontProvider) };
    let _guard = provider.lock.lock();
    match catch_unwind(AssertUnwindSafe(|| action(provider))) {
        Ok(value) => value,
        Err(_) => {
            if let Ok(mut failure) = provider.failure.try_borrow_mut() {
                if failure.is_none() {
                    *failure = Some(format!("PDFium font callback {name} failed"));
                }
            }
            fallback
        }
    }
}
/// 读取 PDFium 借出的零结尾字体名，仅在当前回调内使用。
unsafe fn face_bytes<'a>(face: *const c_char) -> &'a [u8] {
    if face.is_null() {
        b""
    } else {
        unsafe { CStr::from_ptr(face).to_bytes() }
    }
}
/// 释放默认提供器后归还安装时增加的强引用，最后一步才可能销毁对象。
unsafe extern "C" fn release(info: *mut FontInfo) {
    let p = unsafe { &*(info as *const FontProvider) };
    let was_installed = {
        let _guard = p.lock.lock();
        unsafe {
            guarded(info, (), "_release", |p| p.cleanup());
        }
        p.installed.replace(false)
    };
    // 先销毁锁 guard，再归还安装所有权，避免 guard 引用已释放对象。
    if was_installed {
        unsafe {
            drop(Arc::from_raw(info as *const FontProvider));
        }
    }
}
/// 枚举期间禁止替换原字体元数据，并允许默认提供器重入其它回调。
unsafe extern "C" fn enumerate(info: *mut FontInfo, mapper: *mut c_void) {
    unsafe {
        guarded(info, (), "_enum_fonts", |p| {
            let default = p.default.get();
            if default.is_null() {
                return;
            }
            if let Some(call) = (*default).enum_fonts {
                p.depth.set(p.depth.get() + 1);
                call(default, mapper);
                p.depth.set(p.depth.get() - 1);
            }
        });
    }
}
/// 按请求记录字符集；系统代理保留原权重、斜体、pitch 与 exact 指针。
unsafe extern "C" fn map_font(
    info: *mut FontInfo,
    weight: i32,
    italic: i32,
    charset: i32,
    pitch: i32,
    face: *const c_char,
    exact: *mut i32,
) -> *mut c_void {
    unsafe {
        guarded(info, std::ptr::null_mut(), "_map_font", |p| {
            let name = face_bytes(face);
            if p.depth.get() == 0 {
                p.requests.borrow_mut().insert(name.to_vec(), charset);
                if let Some(value) = p.classify(name, charset) {
                    p.bundled_requests.set(p.bundled_requests.get() + 1);
                    return p.allocate(Handle::Bundled(value));
                }
            }
            let default = p.default.get();
            if !default.is_null() {
                if let Some(call) = (*default).map_font {
                    let value = call(default, weight, italic, charset, pitch, face, exact);
                    if !value.is_null() {
                        return p.allocate(Handle::System(value as usize));
                    }
                }
            }
            std::ptr::null_mut()
        })
    }
}
/// 名称查询复用最近请求的字符集，保留稳定字库缓存身份。
unsafe extern "C" fn get_font(info: *mut FontInfo, face: *const c_char) -> *mut c_void {
    unsafe {
        guarded(info, std::ptr::null_mut(), "_get_font", |p| {
            let name = face_bytes(face);
            if p.depth.get() == 0 {
                let charset = p.requests.borrow().get(name).copied().unwrap_or(1);
                let selected = p
                    .classify(name, charset)
                    .or_else(|| (name == p.cache_name).then_some(134));
                if let Some(value) = selected {
                    p.bundled_requests.set(p.bundled_requests.get() + 1);
                    return p.allocate(Handle::Bundled(value));
                }
            }
            let default = p.default.get();
            if !default.is_null() {
                if let Some(call) = (*default).get_font {
                    let value = call(default, face);
                    if !value.is_null() {
                        return p.allocate(Handle::System(value as usize));
                    }
                }
            }
            std::ptr::null_mut()
        })
    }
}
/// 返回所需字节数，小缓冲区完全不写入；固定字库直接复制原切片。
unsafe extern "C" fn get_data(
    info: *mut FontInfo,
    handle: *mut c_void,
    table: u32,
    buffer: *mut u8,
    size: c_ulong,
) -> c_ulong {
    unsafe {
        guarded(info, 0, "_get_font_data", |p| match p.lookup(handle) {
            Handle::System(value) => {
                let default = p.default.get();
                ((*default).get_data.expect("missing GetFontData"))(
                    default,
                    value as *mut c_void,
                    table,
                    buffer,
                    size,
                )
            }
            Handle::Bundled(_) => {
                let (offset, length) = if table == 0 {
                    (0, p.data.len())
                } else {
                    p.tables.get(&table).copied().unwrap_or((0, 0))
                };
                if !buffer.is_null() && size as usize >= length && length > 0 {
                    std::ptr::copy_nonoverlapping(p.data.as_ptr().add(offset), buffer, length);
                    if table == 0 {
                        p.full_font_copies.set(p.full_font_copies.get() + 1);
                    }
                }
                length as c_ulong
            }
        })
    }
}
/// 固定字库使用带哈希名称，系统句柄则返回原始提供器名称。
unsafe extern "C" fn get_name(
    info: *mut FontInfo,
    handle: *mut c_void,
    buffer: *mut c_char,
    size: c_ulong,
) -> c_ulong {
    unsafe {
        guarded(info, 0, "_get_face_name", |p| match p.lookup(handle) {
            Handle::System(value) => {
                let default = p.default.get();
                (*default)
                    .get_name
                    .map_or(0, |call| call(default, value as *mut c_void, buffer, size))
            }
            Handle::Bundled(_) => {
                let length = p.cache_name.len() + 1;
                if !buffer.is_null() && size as usize >= length {
                    std::ptr::copy_nonoverlapping(p.cache_name.as_ptr(), buffer.cast(), length - 1);
                    *buffer.add(length - 1) = 0;
                }
                length as c_ulong
            }
        })
    }
}
/// 保持每个固定句柄的请求字符集及系统接口缺省值。
unsafe extern "C" fn get_charset(info: *mut FontInfo, handle: *mut c_void) -> i32 {
    unsafe {
        guarded(info, 1, "_get_font_charset", |p| match p.lookup(handle) {
            Handle::Bundled(value) => value,
            Handle::System(value) => {
                let default = p.default.get();
                (*default)
                    .get_charset
                    .map_or(1, |call| call(default, value as *mut c_void))
            }
        })
    }
}
/// 先移除自有 ID，再释放系统句柄，避免重入或重复释放同一资源。
unsafe extern "C" fn delete_font(info: *mut FontInfo, handle: *mut c_void) {
    unsafe {
        guarded(info, (), "_delete_font", |p| {
            let value = p
                .handles
                .borrow_mut()
                .remove(&(handle as usize))
                .expect("unknown font handle");
            if let Handle::System(value) = value {
                let default = p.default.get();
                if !default.is_null() {
                    if let Some(call) = (*default).delete_font {
                        call(default, value as *mut c_void);
                    }
                }
            }
        });
    }
}
