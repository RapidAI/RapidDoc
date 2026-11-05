# 表题注释与完整表格合并修复

用户已确认：定位首次差异，同步修复 Python/Rust，保留文字覆盖和占格检查。

根因：上游合并重复候选时保留表题注释；本地完整网格优先保护直接跳过弱候选，
连弱候选里已位于完整候选范围内的表题也丢弃。初始字符、文字行和完整核心框一致。

实施：
1. 冻结单页和合并输入，建立旋转、表体归属、越界、缺少行框的负例。
2. 保留完整候选的框、核心、成员、分数；仅合并有行框证明、位于既有范围且中心在核心外的注释，复用原有表体优先消歧。
3. 将同一处理移入共享 Rust 候选合并器，做完整输入/输出与所有权差分。
4. 受影响测试通过后，统一构建安装，解除 328 页整文档测试的 expected failure；阶段末回归既有 PDF 集合。

不删除弱候选保护，不放宽覆盖阈值，不按文件名/页码改生产算法，不改旋转文字排序。

## 实现与验证

Python `table_rules._merge_enclosed_annotations` 和共享 Rust `table_candidate_merge`
使用同一规则：弱候选已有注释必须有明确行框，完整行框已在获胜候选范围内，
来源不属于已确认表体，且中心在核心表体外，才能进入原来的表体优先注释合并器。
获胜候选的框、核心、成员、分数和网格属性保持不变。不能因为文字框下降部略过表格上沿，
就把已经由其它候选证明的表题丢掉。

- Python native PDF **437 项回归通过**，包含原上游 **328 页整文档测试**；已移除 strict xfail。
- 14 组合并差分覆盖四个方向、表题/表注、表体归属、核心内文字、越界、缺少核心/行框，以及真实宿主候选。Rust 单元测试和安装后 SDK 都通过，弱候选输入未被修改。
- 原有 **132 组**表格/行内样式/HTML 安装差分继续通过。
- 正式 `rapid_doc_rust.RapidDoc` 分段解析原 PDF，保留相邻页上下文，实际读取 **37 页**，原上游断言覆盖的 **31 个目标页全部通过**，包括第 36、53、54 页以及第 185–212 页。
- 生成器 `scripts/migration/export_caption_grid_merge.py` 能精确重现持久参考 `tests/fixtures/upstream_068/caption_grid_merge.json.gz` 的每个输入与输出。

安装包：`build/caption-grid-wheels-20261030/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl`，已装入原 `.venv`。
安装后重放：`.venv/Scripts/python.exe bindings/python/tests/caption_grid_install.py . --catalog-ranges`。

阶段末既有 **12 份、101 页** PDF 回归全部通过，Markdown、内容列表、中间结构和图片像素/名称均与原 Python 基线一致，Rust 图片编码字节与修复前也一致。148 个安装包文件与最终 wheel 完全一致，项目内 12 个 ONNX 文件 SHA 未变。回归记录在 `build/caption-grid-pdf-20261030`，最终交付记录为
`build/caption-grid-final-receipt-20261030.json`。

## 单独记录的全文诊断内存问题

旧 `analyze_native_pdf` 诊断入口一次处理 328 页时内存持续增长，观察到 13.08 GiB，
最后进程报告 `memory allocation of 632 bytes failed`。这次运行**未通过**，不能将分段正式入口的
37 页验证写成“Rust 328 页整篇通过”。已退出的诊断进程不再占用资源。
该入口的全文中间状态内存问题另行优化；本次没有通过截断内容、降低识别要求或新增字符/页数限制掩盖它。
