# 真实 UNET 倾斜格框与异常日志修复（2026-11-02）

## Python 实际行为

只复测用户报错的完整单页 PDF：
`docstructbench_llm-raw-scihub-o.O-j.eswa.2008.01.070.pdf_5.pdf`。
输入 SHA256：1e0e018f5c9345090b87634455ec7a6769640a347875e7a7fbb30a0cf03ee19d。
服务器原 Python GPU 解析成功，输出 6,274 个 Markdown 字符和两张 HTML
表格；第一张 53 行，第二张 7 行。不是空结果，也不是仅靠异常吞掉失败。
捕获到两次 Python TableRecover 调用，225 个格框及 1 个格框均恢复成功。

第一组第 22、47 个框触发 Rust 检查。它们四个连续叉积均为负，是正常
凸倾斜框；不是退化、自交或模型 NaN。左上到右下对角线在 Y 方向反序，
Rust 额外要求 p0.y < p2.y 因而误拒绝。第 22 框 p0.y=227.400024、
p2.y=221.800018；第 47 框 p0.y=355.500031、p2.y=318.500031。
Python 原实现不作该检查，直接使用顺序顶点、边长和轨道恢复逻辑格。
有表格输出不等于已经完成 OmniDocBench 的真值精度评分。

## 修复范围

生产 recover_legacy 不再拒绝这种四角关系，不排序、不修形、不删除格框；
有限数值和真实数组安全检查保留。诊断 recover 严格接口保留原校验。
原 Python 冻结的 5 组合成边界加本页两组真实格框，逐项比较完整逻辑
坐标；所有现有表格恢复测试通过。本页 225 个真实格框不作为文档特判，
只是复现夹具。

原生 RapidDocError 的 __module__ 原来为不存在的顶层 `_native`，导致
Loguru enqueue=True 的异常对象无法 pickle。修正为实际可导入的
rapid_doc_rust._native；评测脚本同时使用同步文件 sink，避免日志保存
依赖异步异常序列化。终端进度、完整日志、ERROR 日志和失败 JSONL 保留。
模拟真正 RapidDocError 的日志测试通过。

## 验证状态

已通过：5 项表格恢复测试、1 项评测日志测试、Python 绑定编译检查。
本地原始 Python 表格输出、225+1 格框及原逻辑坐标保存在
build/unet-polygon-fix/ 和 tests/fixtures/upstream_068/unet_collapsed_polygons.json。
整页复测又发现同一 p0/p2 顺序检查也存在 OCR 匹配 rect() 中。
Python box_4_2_poly_to_box_4_1 是直接取这两个点，保留有符号面积，交集
宽高按零截断，不拒绝对角点反序。匹配路径也恢复原行为，不把几何框
重排或归一化；新增真实倾斜框加正常框的 OCR 匹配回归用例。

最终 5 项恢复测试和 4 项匹配测试均通过；修复包完整解析原报错单页，
与服务器原 Python 的完整规范化快照（middle/content/Markdown/图片内容）
和 Markdown 原始字节一致，双方均输出两张表。RapidDocError pickle
往返和实际 Loguru enqueue=True 文件日志测试通过。

Linux 包已安装原手动评测命令使用的 env-delivery，pip check 通过；
服务器评测脚本同步日志 sink 已更新，未启动整套评测。Windows 修复包已构建并安装项目 `.venv`；原生异常 pickle / Loguru
异步写入、本次评测日志回归及既有 46 组 Magika 语言、414 组输出回归
通过。仅阶段结束构建并安装，不跑全量。
本轮未启动 1651 份整套评测，不修改原 Python 算法或识别阈值。


Linux 修复包及完整原始对照已下载：`build/unet-polygon-fix/linux-wheels/`；归档 `build/unet-polygon-fix/unet-tilted-polygon-fix-results.tar.gz`，35,643,391 字节，SHA256 `2865c389a1e2e498892fbb05146bb98886743c377bfcaeea564ef017fa66b553`。

当前 Windows wheel：`build/unet-polygon-v2-windows-wheels/`。两端最新安装包均含本次恢复、匹配及异常模块名修复。


## 后续：移除严格入口分叉

按用户要求删除 recover_legacy / recover_impl / legacy 参数，统一只保留
recover，并让生产单元格恢复和 Python 独立拓扑接口调用同一实现。
额外的四角方向/非退化拒绝、非正轨道拒绝和对应测试断言已删除；保留
Python 的源索引、有符号轨道、空逻辑区间行为，以及 NaN/Inf、数组形状、
真实索引越界和资源检查。不再存在“严格接口”和“Python 兼容接口”两套
恢复语义。五项恢复、四项匹配测试及 Python 绑定编译检查通过，包含
225+1 个真实单元格的原 Python 逻辑坐标对照。

这次是源码入口合并：已安装的修复版默认 PDF 之前就在使用相同的 Python
处理规则，行为不变；本次没有再次构建/替换 wheel，不需要为此重跑当前
整套评测。独立 Python 拓扑接口的新统一行为需下次构建后进入安装包。
