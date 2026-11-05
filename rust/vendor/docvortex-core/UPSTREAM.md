# 固定来源的 DocVortex 数值内核

src/ 与 tests/ 原样复制自本机已检查的 DocVortex Rust core。精确来源提交及逐文件 SHA256 见 UPSTREAM.json；Cargo.toml 仅展开了原 workspace 继承的元数据。未复制依赖 Python/ctypes 的绑定 crate，也不依赖本机绝对路径。

保留原 `MIT AND Apache-2.0 AND BSD-3-Clause` 元数据及 DocVortex 根许可证声明。不要覆盖本地修改；升级须重新做契约和 RapidDoc 差分。此 crate 纳入发布 workspace，主项目通过 rapiddoc-native 的输入校验边界使用它。


2026-10-30: selected upstream `a4c8b2e` updates only `src/text_content.rs` and its unit test to support all three Unicode private-use ranges. `UPSTREAM.json` preserves original hashes in `base_files`, records current hashes in `files`, and identifies the selected patch. Other source files remain pinned.

Publishing: this local subset/fork uses the independent crates.io package name `rapiddoc-docvortex-core`. The Rust library/dependency alias and parser source remain unchanged.
