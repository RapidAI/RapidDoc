# 完整表格候选替换仲裁

迁移一般候选编排中的 `_prefer_complete_table_replacements` 完整规则：core优先、90%较小框重叠准入、横坐标扫描交集并面积、覆盖不足时累计拒绝替代候选，最后以全局通过的替代候选删除旧base/recovered。保留候选原序、完整元数据和数组位置身份；不把输入候选称候选检测已迁移。

冻结旧完整函数输出，覆盖重复片段/边界/多个旧表共享替代候选；补实际PDF实时路径→已有填充表格检测→显式替换查询，所有候选来自新检测而不注入旧结论。开发只跑受影响测试，阶段末一次wheel及相关回归，无全量/CUDA。一般候选产生器、diagram和默认PDF仍待。

## 实现及验证范围

- 新接口返回独立base/recovered/replacements，候选需要bbox及可选core_bbox，其余字段原样保留；数组原序和重复值保持。空core与None回退原bbox。
- 完整保留原函数的两阶段过程：每个旧表使用所有原替代候选计算覆盖并累计拒绝，不在中途过滤已拒绝项；全局拒绝确定后，再用剩余候选删除旧表。并面积逐横带合并纵区间，原序float累加和阈值乘法顺序保持。
- 冻结157旧完整函数参考，153结果变化、78替代拒绝。覆盖90%边界、重复半表/双半表、多个旧表共享候选的全局拒绝、core选择、整数/混合float坐标、完整注释/来源元数据、稳定排列。
- 实际28页PDF提取字符/路径/绘图并构造新视觉行，实时调用已有外框填充行带或相接填充检测，再构造44组self/partial/halves查询；24非空且结果变化。查询拆分只是测试输入，不宣称年份/数值/指标表产生器已迁移，不注入旧表格/覆盖结论。
- 普通JSON，各5000候选、finite abs<=1e9，既有JSON/10M累计Work/8MB输出；候选配对、边缘/区间排序、扫描覆盖和保留操作计费并接受已有预算。无跨调用缓存，输出所有权独立，错误传播。支持域为普通double几何，非任意Python数值对象。

## 验收完成

- 两个定向Rust单元测试、一个44组实际PDF实时候选查询集成测试通过。
- 阶段末仅一次release wheel与一次隔离安装；native_table_replacements、native_filled_grid、native_tiled_tables、compatibility、smoke五组安装套件通过，涵盖完整元数据/数值类型/顺序、所有权、并发、扫描预算/资源错误与异常恢复，安装环境不依赖旧Python解析器/PDFium。
- fmt、101模块审查、固定36＋3＋1上游哈希通过；全量runner登记119组，本轮未全跑，未执行CUDA。
- 日志build/native-table-replacements-rust.txt、build/pdf-table-replacements-rust.txt、build/native-table-replacements-wheel.txt、build/native-table-replacements-install-test.txt、build/native-table-replacements-regression.txt。
- 后续继续一般候选的产生器/合并与排除规则，再接diagram及强图形统一入口、跨页准备和默认PDF；本阶段仅完成完整覆盖替换仲裁，不作整文档质量或性能结论。
