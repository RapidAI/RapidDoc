# 发布声明内容筛选

本轮在前一轮 300 → 7 个文件的基础上继续删除不适用于发布物的正文。不是再次把相同内容换目录或压缩隐藏。

## 实测结果

Windows wheel 内三个许可证目录的文本和索引总量：1,637,574 → 669,751 字节，减少 967,823 字节（59.1%）；仍是 7 个文件。移除了 149 条原始路径对应的声明/元数据。计数包含重复来源路径；新运行时组件条目使路径数不能简单相减。

- 移除源码哈希、测试状态、Office 迁移记录等 4 份开发 JSON，源码中的审计记录仍保留。
- 替换约 580 KB 的旧下载依赖声明、约 49 KB 的字体锁依赖声明；重新读取锁定 Cargo 源码，只给实际运行时组件选择适用条款，不重复携带未选的替代许可。
- 清除不在 Windows/Linux 运行时图内的构建工具、proc-macro 辅助库、Wasm/其他平台组件声明。
- 对明确的 MIT OR Apache-2.0 等许可择一保留 MIT；AND 条款、多重必需条件、NOTICE、COPYRIGHT 继续保留。不能确认的条目不靠文件名直接删除。
- fontTools 实际只复制了 AGL 数据及对应算法，删除未分发的测试字体、cu2qu 等外部声明，保留 Adobe AGL 原始 BSD 条款与版权及 fontTools 自身声明。
- Windows/manylinux 不携带 Android、musl 专用 PDFium 声明；保留 all_except_musl/llvm-libc 和 FreeType 等共享内容，未凭“未启用某功能”猜测删除 V8/XFA 条款。

## 构建约束

`scripts/migration/generate_release_notice_policy.py` 使用 Cargo 官方 normal/no-proc-macro 依赖图，记录 Windows x86_64 MSVC 和 Linux x86_64 glibc 的 CPU/CUDA运行时许可决策。`packaging/release-notice-policy.json` 锁定 Cargo.lock 及参与解析的本地 Cargo.toml 哈希；依赖配置变更时要求重新生成，避免继续使用旧筛选结果。

`scripts/migration/release_notice_policy.py` 应用于暂存目录、重打 wheel、Linux 发布工具及 Office 声明收集。实际构建前注入原始大声明和开发 JSON 后再暂存，已确认它们不会回到发布包。未知文件保留，未知/musl 构建不套用 glibc 专用删除规则；当前验收范围仍为 Windows MSVC、Linux glibc。

Cargo 原条款和已选择条款分开记录。只允许择一的 OR 情况作单选，AND 情况保留全部必需条件。依据：

- https://doc.rust-lang.org/cargo/commands/cargo-tree.html
- https://opensource.org/license/mit
- https://www.apache.org/licenses/LICENSE-2.0

## 验证

15 项检查通过，包含原文/版权保护、AND/OR 选择、平台过滤、保留未知 NOTICE、重复暂存、wheel 原代码不变和 DOCX 图片公开调用。

`build/minimal-notices-wheels/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl` 已安装项目 `.venv`。Python/Pyd/DLL/SO 与前一版本逐字节一致；只变更许可证子树、许可证哈希清单和 RECORD。RECORD 与安装二进制校验通过，迁移声明 120 条原始路径校验通过。

未重编译 Rust，未同步服务器，未跑全量解析/CUDA。完整必要许可正文仍需保留，不能把每个许可证简写成一行或只放一个网址。

报告：`build/minimal-notices-wheel.json` 记录每条删除理由和原 SHA；`build/license-content-filter-20261104.json` 记录源码/暂存筛选。
