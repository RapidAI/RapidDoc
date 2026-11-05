# PPT 二进制记录、文字、组合形状与表格阶段

按项目当前 Python `office/ppt` 原生实现移植，不借助 PowerPoint/LibreOffice 转换，也不使用纯文本简化算法。当前是 Rust 内部解析阶段，公开 PPT 入口尚未启用；`.venv` 保持已交付的 PDF/DOCX/XLSX/XLS/PPTX。

## 实现范围

- `ppt/records.rs`：MS-PPT/OfficeArt 8 字节头、容器内边界、首条严格校验和后续尾部容错；借用 payload，避免逐记录复制。显式栈遍历，保留 round-trip opaque 1053～1064 不递归的规则。
- 原 Python 的累计 1,600 万记录、64 层容器、100 次 UserEdit 保存链预算原样保留，没有额外添加 PPT 字符数限制。记录访问数在形状和本地文档上下文参考中同时核对。
- `ppt/document.rs`：Current User 版本/密码标志、完整记录流预检、CryptSession 检测、最新 UserEdit/persist 映射、新版本优先、损坏/循环保存链恢复、顶层完整 Document 回退。定位幻灯片/母版/备注列表、页面尺寸、母版逐文本类型样式和外置文本记录。
- `ppt/styles.rs`：StyleTextPropAtom、TextMasterStyleAtom 的段落/字符异常字段、三态继承、删除线、baseline、pp9rt 和截断边界。
- `ppt/text.rs`：UTF-16LE 替换解码、cp1252 字节文本、UTF-16 单元范围匹配、字符/母版继承、段落分隔、软换行和链接范围；按 Python Unicode category C 过滤控制/私用/未分配字符，保留 tab。Unicode 类别表固定生成，不依赖 Rust 的不同分类规则。
- `ppt/shapes.rs`：OfficeArt 简单/复杂 FOPT、组合空间变换、Client/Child anchor 优先级、无父空间的 EMU 换算、稳定路径和 z-order、就近组合组成员。连接 ClientTextbox 和 OutlineTextRefAtom 的外置文本索引回退；PPT9 编号槽位覆盖及起始编号。
- `ppt/tables.rs`：原几何表格证明：矩形单元格/线条区分、边界聚类、最近边界匹配、行列合并、重叠/缺格拒绝、稳定单元格顺序及被表格占用的 shape key。覆盖使用单份位图，避免字典元组的额外内存；阈值、效果和拒绝规则保持原实现。
- `ppt/notes.rs`：隐藏页标记、备注正文提取、特殊 `*` 字段排除、非 textbox 备注回退、NotesAtom.slideIdRef 绑定及无主备注顺序。与本地原 Python 上下文核对。

上述语义结果直接保留 Rust 类型和借用记录，不将中间结果编码成 JSON 再交回 Python。JSON 只用于固定参考和检查点序列化。

## 验证

参考生成脚本：`export_ppt_primitives_reference.py`、`export_ppt_shapes_reference.py`。固定输入/输出在 `tests/fixtures/office_rust_reference_068/ppt_primitives.json.gz` 和 `ppt_shapes.json.gz`；Rust 测试不运行 Python。

- 320 组记录流，含截断、深度边界及 opaque 容器。
- 1,630 组样式/母版输入，覆盖字段掩码及截断位置。
- 400 组 Unicode/UTF-16 样式与链接段落，301 组 UTF-16 字节解码。
- 12 组保存版本布局，包括本地 3 份 PPT、损坏引用和 99/100/101 次保存链边界；另有 Current User/加密/完整预检参考。
- 191 个幻灯片/组合/表格/编号形状检查点，含 14 个本地页面及合并/缺格/重叠/非表格组/线条/微小误差；320 组 FOPT 和 576 组 PPT9 编号输入。
- 本地母版、外置文本、链接、备注绑定与原累计记录访问数一致。
- `ppt_primitives` 6 项、`ppt_shapes` 2 项通过；最后补充复杂 FOPT/编号形状后复查 `ppt_shapes` 2 项通过。初次接入模块另复查容器 foundation 5 项通过。

未统一构建 wheel/重装环境，没有全量 Office/PDF/CUDA 回归，也没有 PPT 速度或内存验收结论。本阶段没有改动已交付格式的算法。

## 尚待完整 PPT 交付

1. BStore/Pictures 的 PPT 绑定、BLIP 与公式预览、嵌入 OLE 存储解压及对象 id 映射，接已有原生图片、MTEF、XLS 嵌入图表引擎。
2. 表格候选优先级/占用、文本/图片/公式/图表的整页编排，缺页/顶层 slide/单页文本恢复，以及备注消费状态。已有底层表格结果不等于整页表格仲裁已完成。
3. 原 PPT 标题、嵌套列表、表格 HTML、阅读顺序及统一 Office 输出。
4. 公开 Python 路由、最终安装和效果/速度/峰值内存验收。随后移植 DOC。

公开 PPT 入口保持未连接，避免用尚未集成资源的文字检查点替代完整解析。后续从这些内部类型继续接线，无需重新移植本阶段内容。
