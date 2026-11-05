# 原生表格整页后处理

按旧 `table_detection.py` 的顺序迁移：稀疏横线链与表题分组、无列线表格正文尾段裁剪、Form 排除、续表标题外置、异常 CJK 字框短标题绑定及稳定排序。

保留此前未收尾的候选 API，新增完整候选检测入口。所有阶段共享工作量预算；源行、绘图规则及候选均设上限。以独立旧版参考语料验证辅助函数及完整候选检测，再构建一次预览 wheel 并运行受影响安装回归。

本阶段不切换默认 PDF 入口，不实现固定 CJK 字体运行时、diagram 或 enhanced 阅读顺序；候选检测完成不代表完整 PDF 迁移完成。

## 实现与证据

- 完整入口保持产生器原序，在最终排序之前执行后处理；旧 pre-policy 接口输出不变。
- 表题分组沿 CPython set 删除/重建遍历；核心边界和注释外框分开比较，合并时保留第一候选其他元数据。
- 续表标题只在唯一匹配时外置，保留第一 caption 注释及有序行框覆盖；四方向及 Python Unicode IGNORECASE/空白行为对照通过。
- 异常 CJK 字框页缺失 tight_bbox 时，沿旧 ValueError 转为 invalid_input，未用 fallback 掩盖异常。
- 页末续表轨道使用 prepared 块与剩余行证明最后物理行，不注入已选轨道；此接口尚未自动接入默认整页准备。
- 103 后处理＋1404 完整检测＋26 末表轨道，合计1533结果及3异常。24受影响Rust测试通过，新增28边界事实定向通过。
- 58页实时PDF重建：既有55页保持输出，新增3页独立后处理正例，其中2页完整检测输出变化。独立后处理的候选框是明确投影入参，完整检测直接消费实时原始源。
- Form 源级排除已验证；实际58页 form_bboxes=[]，不宣称验证真实 PDF Form 提取。CJK inflated 短标题仅源级正例；未嵌入 CID 字体已知失败不计通过。
- 111模块兼容性清单及固定36＋3＋1上游哈希通过；runner登记129组安装套件及独立PDF渲染验证，本阶段不全跑。

安装验收：一次release wheel及隔离重装后，新整页、旧全产生器、compatibility、smoke共4组套件通过。构建及安装日志 `build/native-table-page-policies-wheel.txt`、`build/native-table-page-policies-install-test.txt`、`build/native-table-page-policies-regression.txt`。Rust日志 `build/native-page-policies-related-rust.txt`、`build/native-page-policy-boundaries-rust.txt`、`build/native-complete-candidates-pdf-rust.txt`。

下一阶段从固定 CJK 字体运行时、diagram 缺口和正式整页准备接线推进；本阶段不再保留“最终表题/尾段/续表处理待完成”的当前待办。后续完整文档差分通过前，默认 PDF 和兼容性完成标志保持现状。
