# 图形/有字网格仲裁

迁移旧native_multicolumn_grid_bboxes及强图形候选末尾裁决。实时从页源绘图线/文字组装连通网格、竖轨覆盖、实际文字占列和跨候选带，覆盖候选至少70%才以表格排除；之后按旧稳定面积顺序做包含拒绝和重叠合并。候选产生器仍是独立输入，接口不称完整强图形入口。

均值用已缓存num-rational精确二进制比率计算，fmean先精确求和舍入再除样本数；最终行中心与覆盖和沿已有CPython sum。冻结独立旧网格/最终输出及真实PDF链，开发只跑定向测试，阶段末一次wheel和相关安装回归，无全量/CUDA。

已完成178组旧参考（164基础＋14数值边界），120组网格排除正例。最终框参考通过AST抽取原 `_detect_strong_graphic_bboxes` 的output初始化到return尾段执行；候选仅作为明确阶段输入，不修改尾段代码或替换网格证明。覆盖长横线分组/去重、贯穿竖轨/分段覆盖/重复轨均值、visual_row_id种子与几何聚行、有字物理列及候选内带、Form排除、多表并集70%覆盖、候选包含/重叠阈值、稳定顺序和大整数/混合float面积排序。末尾合并框保留union赢家原数值类型。

8页专用真实PDF四方向链：实时PDFium文字/绘图→视觉行→绘图组件图形候选→网格保护/最终裁决，8页均产生候选、1页用多列有字网格排除图形。先核对完整实时页源与候选，再检查最终输出；不注入路径、候选或网格结论。网格只消费angle0文字，旋转页保留旧支持范围，不把该样本称为完整旋转表格解析等价。

接口 `arbitrate_native_graphic_candidates(source, candidate_bboxes, *, median_height)` 返回graphic_bboxes、grid_rejections（面积排序后的候选位置与证据框）；`inspect_native_graphic_grids(source, regions, *, median_height)` 返回原区域顺序的grid_bboxes。source提供page_size、lines（bbox/angle/visual_row_id）、drawing_lines、form_bboxes。候选集合/median_height仍显式输入，不是完整强图形产生器入口。

普通JSON、有限double point几何abs<=1e9，最多4096行/5000绘图线/5000Form/5000候选或区域，合计最多20k输出网格证据框，沿既有JSON/10M累计Work/8MB结果。均值精确求和、排序/去重/连接、竖轨、行列占用、覆盖扫描及候选合并累计计费并接受调用方已有预算；输出证据数在构造时检查，资源错误传播。新增直接依赖num-rational=0.4.2、num-traits=0.2.19均已在原锁文件/本地缓存，相关MIT/Apache许可随wheel打包；无网络依赖下载。fmean先把精确总和转float再除count，statistics.mean先除count再转float；最终视觉行中心和区间sum复用既有CPython补偿和，覆盖面积外层累计按旧+=顺序。

两个定向Rust单元测试和一个实际PDF集成测试通过；阶段末一次release wheel构建并隔离安装。native_graphic_arbitration、native_drawing_graphics、compatibility、smoke共四组安装套件通过，覆盖精确字段/类型/顺序、独立网格证据接口、所有权/并发、输入/扫描限额和错误恢复。fmt、91模块来源审查和固定36 core＋3 paths＋1 script哈希保持。未跑全workspace、全部109组runner、未受影响渲染、全量clippy或CUDA，两个legacy完成标志False。

日志：build/native-graphic-arbitration-rust.txt、build/pdf-graphic-arbitration-rust.txt、build/native-graphic-arbitration-wheel.txt、build/native-graphic-arbitration-install-test.txt及build/native-graphic-arbitration-regression.txt。完整强图形产生器集合、跨页准备/候选检测/默认PDF编排仍待，未作整文档质量或性能结论。
