# 独立矢量及小栅格证据

沿旧isolated_graphics.py迁移无文字矢量分组/图体证明和小栅格判定。保留增量合并“新路径在组首，其余命中组按旧组序追加”、正文/栅格覆盖拒绝、多色页角例外和字号尺度策略。复用已验证CPython hypot执行临界距离，接口无跨调用缓存。

冻结旧字段/顺序/阈值事实及真实PDF读取链，开发只跑定向测试，阶段末一次wheel/相关回归，不跑全量/CUDA。返回独立位置/框证据，尚非完整强图形/表格仲裁或默认PDF编排。

已完成229组旧完整输出，137矢量/7小栅格正例。覆盖路径过滤/面积/段数/尺度、合并次序及桥接、临界距离、重复几何、修复字号和普通中位数、多色页角（含None颜色）、正文0.1/小栅格0.01覆盖及栅格0.2覆盖阈值。41页既有实际PDF实时字符/路径/图片/视觉行输入通过；另8页专用四方向真实链含3矢量/1小栅格正例。没有注入行、路径、图片或分组结果。默认视觉行不含180度文字保持旧支持域，因此相关页的覆盖行为不能据此提升为完整旋转解析等价。

接口 `detect_native_isolated_graphics(source)` 返回em、path_groups（原path位置）、vector_bboxes、meaningful_raster_indices（原image_bboxes位置）。源数据不修改，union保留原值赢家及JSON类型；无行时矢量em为10，小栅格必须有正向文字行的尺度证据。没有跨调用缓存，复用CPython hypot直接执行旧Python距离条件，未使用vendor平台hypot临界回退接口。

普通JSON，最多4096行/5000路径/5000图片框，每个路径矩形容器最多20k，RGBA为0..255，finite double point数值abs<=1e9。沿既有JSON/10M累计Work/8MB结果限制，距离匹配、合并时复制旧组成员、颜色/尺度排序及正文/栅格覆盖扫描累计计费，接受上层已有预算；大量不相交和大量重合路径分别验证扫描/复制预算，资源异常明确传播。

两个定向Rust单元测试和一个实际PDF集成测试通过；一次release wheel构建并隔离安装。native_isolated_graphics、native_drawing_graphics、compatibility、smoke共四组安装套件通过，覆盖精确字段/类型/顺序、所有权/并发、输入/扫描/组复制限额及错误恢复。fmt、90模块来源审查及固定36 core＋3 paths＋1 script哈希保持；无全workspace、全部108组runner、未受影响渲染、全量clippy或CUDA，完成标志False。

日志：build/native-isolated-graphics-rust.txt、build/pdf-isolated-graphics-rust.txt、build/native-isolated-graphics-wheel.txt、build/native-isolated-graphics-install-test.txt、build/native-isolated-graphics-regression.txt。完整强图形/表格仲裁、跨页准备及默认PDF编排仍待，未作整文档质量或性能结论。
