# 本地 lopdf 精度补丁

此目录基于已发布的 `lopdf 0.42.0` crate 保留源码，通过 path 依赖编译；不是另写 PDF 引擎。用途是补充 PDFium 的声明级解析：Form XObject 的资源、调用、变换矩阵、BBox，以及文字可见性资源。页面渲染与字符读取仍由 PDFium 承担。

原版把 PDF Real 解析并保存为 `f32`，Python 路径使用双精度。先损失精度再转回 `f64` 无法恢复原值，会使几何边界和严格对照出现差异。因此仅修改三个文件：

- `src/object.rs`：Real 保存为 f64，保留原 f32 访问器的接口。
- `src/parser/mod.rs`：小数直接解析为 f64。
- `src/xobject.rs`：适配已有 f32 参数到新的 Real 存储类型。

上游文件清单及原始 hash 见 `UPSTREAM.json`，修改后 hash 见 `PATCHES.json`，MIT 许可证保留。可执行 `python scripts/migration/verify_lopdf_vendor.py` 核对。当前共核对 41 个上游文件，仅上述三个源文件有记录内改动；发布包的名称、源码范围等 Cargo.toml 元数据改动单独记录在 packaging 中；此前的空白、格式化及 use 声明排序差异单独冻结在 formatting 中，校验仍逐文件核对 SHA256。

直接换回 crates.io 原版会丢失此修补。后续若上游提供完整 f64 存储/解析能力，再经 Form/可见性与整篇回归后改为发布依赖；仅添加返回 f64 的访问器不足以解决解析阶段的精度损失。

Publishing: this local subset/fork uses the independent crates.io package name `rapiddoc-lopdf`. The Rust library/dependency alias and parser source remain unchanged.
