# 绘图组件图形候选

迁移旧graphics.py绘图线连通分量、统计和复杂路径证明图形候选。沿原左缘稳定扫描、并查集连接和原行首次遇到的分量顺序，复用已验证CPython hypot及原union数值类型。返回分量原数组位置、框/横纵数目及图形框，不替代完整强图形/表格仲裁。

冻结旧边界/顺序事实及实际PDF读取链；开发定向测试，阶段末一次wheel和相关安装回归，无全量/CUDA。既有输入不修改，累计预算传播。

已完成239组旧组件/统计/候选输出（134图形正例），覆盖稳定左缘顺序、hypot临界舍入/欧氏间隙、桥接连接、原始行首次遇到的组件顺序、重复线/反向框、横纵计数/组件尺寸/页面比例、二维路径段数/深度/重叠证明及union类型。CPython hypot既有算术未改，只将可见性扩大供适配层复用；固定来源内核未变。

41页既有实际PDF绘图线/路径实时读取与候选参考一致；另8页专用四方向PDF组件链含2图形正例。实时链先核对完整绘图线/路径输入，再检查组件和输出，不注入路径、组件或候选结果。

接口 `detect_native_drawing_graphics(drawing_lines, path_infos, *, page_size, median_height)` 返回tolerance、complex_path_indices、components（line_indices/bbox/horizontal_count/vertical_count）及graphic_bboxes。原数组位置用于身份，候选框保留原union赢家数值类型，源数据不修改。普通JSON，最多各5000绘图线/路径，有限double point数值abs<=1e9，既有JSON/10M累计Work/8MB结果限制；稳定排序/配对扫描/复杂匹配预算累计，接受上层已有预算，资源错误传播。接口没有跨调用缓存，median_height仍显式传入。

两个定向Rust单元测试和一个实际PDF集成测试通过；阶段末一次release wheel构建并隔离安装。native_drawing_graphics、native_path_graphics、compatibility、smoke共四组安装套件通过，覆盖精确字段/类型/顺序、所有权/并发、输入/配对扫描限额和错误恢复。fmt、89模块来源审查及固定36 core＋3 paths＋1 script哈希保持，未跑全workspace、全部107组runner、未受影响渲染、全量clippy或CUDA，完成标志False。

日志：build/native-drawing-graphics-rust.txt、build/pdf-drawing-graphics-rust.txt、build/native-drawing-graphics-wheel.txt、build/native-drawing-graphics-install-test.txt和build/native-drawing-graphics-regression.txt。完整强图形/表格仲裁、跨页准备及默认PDF编排仍待，未作整文档质量或性能结论。
