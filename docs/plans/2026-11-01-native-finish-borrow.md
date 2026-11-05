# 原生文字收尾、阅读列与脚注的只读借用

基线为已交付 `native-table-borrow-20261101`。该版细分诊断中，129 页几何/角色收尾约 2.59 秒、容器准备约 1.98 秒、来源收集约 2.07 秒；嵌套阶段不能相加。原始记录 `build/native-table-borrow-detail-{summary-,}20261101` 对应 JSON/日志。

## 改动

- 阅读列推断新增内部借用几何入口，直接读取原文字行和 bbox；公开 JSON 入口与内部入口共用原解析、验证、列推断及 Work 逻辑。文字簇和脚注列布局不再把整组带字符数据的行序列化进临时 `line_geometry`。原 512/4096 行诊断边界和数值兼容保留。
- 脚注几何行使用 `Cow<Value>` 借用原行，排序和分组只复制借用/坐标；真正合并行才生成自有行。上标引用扫描直接遍历原行，图表脚注直接借用参考行。
- 标题/展示文字角色的参考列表借用原数据。别名参考只保存展示判断确实读取的 text、angle、effective_height、font_signature 和原别名索引；保留缺失字段与 null 的区别。常规正文成员选择、初始字体/文本与后续实时别名几何/语义分开保留，防止简化别名影响判断。
- 全原生文字簇规范化时，已验证的字段直接移交到完整原生行，额外原生字段只复制一次；不再复制后立即被覆盖的字符数组。保留字段选择、默认值、未知字段处理和后续聚类算法。

Python 的传对象引用行为不能机械对应 Rust 的 Value 深复制收益。本轮不改 Python 生产解析、识别模型、精度或规则。

## 针对性验证

- 既有固定原 Python 参考包括 92 个文字角色、143 个脚注、32 个文字簇和 4 个脚本场景。输出继续匹配。
- 新增 3 个独立原 Python 对象别名/分离对象/稳定排序参考，`tests/fixtures/native_role_snapshot_reference.json.gz` 记录生成脚本与 Python 来源 SHA，生成脚本 `scripts/migration/export_native_role_snapshot_reference.py` 不读取历史 PDF fixture。
- 阅读列的借用/JSON 路径比较上述 roles/notes/clusters 输入，在两种行范围及两种 interval 设置下比较输出、错误类型和 Work。预算错误的代码位置随内部入口改变，不要求路径/行号字符串一致。
- 角色参考完整行快照/紧凑快照在有限与生产宿主作用域比较输出、错误及 Work，并与 Python 参考一致；文字角色/脚注引用检查和已有自有行/完整页/表格参考合计 10 项受影响 Rust 测试通过。记录 `build/native-finish-borrow-focused-20261101.log`。

## 安装验收与独立计时

- 最终包 `build/native-finish-borrow-20261101/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl` 已安装并保留项目 `.venv`。安装二进制与 wheel、公开回归 manifest 和计时记录指纹一致，现有 demo 直接使用。
- 16、65、129 页及密集长表格完整快照与上版严格一致；默认模型扫描件/论文/印章 6 组公开对照全部通过。记录 `build/native-finish-borrow-{window-verify,dense-verify,public}-20261101` 及 `build/native-finish-borrow-parity-20261101.json`。
- 构建、测试和回归结束后，独立进程按上版/新版/Python 顺序各测一次首次、两次复用。同一 129 页原生规则 PDF，64 页窗口、txt、关闭公式/表格模型；有轻量阶段日志，没有采样/cProfile或同期构建回归。

| 调用 | 上版 Rust 秒 | 新版 Rust 秒 | Python 秒 |
| --- | ---: | ---: | ---: |
| 首次 | 11.662 | 10.172 | 11.700 |
| 复用 1 | 11.502 | 10.098 | 10.249 |
| 复用 2 | 11.503 | 10.047 | 10.211 |
| 复用中位数 | 11.502 | 10.072 | 10.230 |

复用耗时下降约 12.4%，原生分析从 8.220/8.334 秒降到 6.891/6.955 秒；渲染/组装无实质改造。本轮整体略快于 Python 约 1.5%，只能称该输入基本追平，样本量和差距不足以证明稳定领先，更不能推广到全部 PDF 或扫描件。

源码、Python 参考、wheel、安装指纹及验收收据：`build/native-finish-borrow-delivery-receipt-20261101.json`。未跑全 workspace、全部 PDF 或 CUDA，也未新增内存基准。
