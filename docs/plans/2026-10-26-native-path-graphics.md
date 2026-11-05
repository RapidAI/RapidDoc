# 复杂路径容器与坐标轴图形

沿旧graphics.py迁移二维复杂轮廓判定、填充容器内部路径证明和相交坐标轴图形候选。返回原数组位置和独立bbox，保留候选顺序、重复来源号排除及union数值赢家；不替代完整强图形/表格仲裁。

先冻结旧阈值/次序/实际PDF读取参考，再实现资源预算和SDK桥；阶段末一次wheel/相关安装回归，不跑全量/CUDA。横纵轴原序遍历加旧最终距离筛选，不跨锁保存原生对象。

已完成263组独立旧输出（60容器/73轴正例），覆盖根层/填色/段数、内部路径数量与来源号排除、面积/宽高/重叠阈值、轴长度/厚度/页内比例/端点容差及ULP边界、候选顺序和重复、union首项数值赢家及JSON类型。41页既有实际PDF实时路径输入也对照通过；另8页四方向专用实际PDF链有4页容器正例、2页轴正例，SDK读取路径/页面尺寸后直接检测，无路径/候选结果注入。已有PDF及冻结JSON未重写。

接口 `detect_native_path_graphics(path_infos, *, page_size, median_height)` 返回complex_path_indices（原数组位置）、container_bboxes、axis_bboxes。支持普通JSON和有限double point算术；最多5000路径/合计5000输出框、数值abs<=1e9，沿既有JSON/10M累计Work/8MB结果限制。内部路径、轴对和复杂轮廓匹配扫描逐步计费，接受上层已有预算；输出数量在构造时检查，避免重复轴产生巨大中间数组。median_height显式传入，完整图形尺度编排尚待接入。

两个定向Rust单元测试和一个真实PDF集成测试通过；阶段末一次release wheel构建并隔离安装。native_path_graphics、native_stacked_graphics、compatibility、smoke共四组安装套件通过，覆盖精确字段/类型/顺序、所有权/并发、输入/扫描/输出限额及错误恢复。fmt、88模块来源审查及固定36 core＋3 paths＋1 script哈希保持。未跑全workspace、全部106组runner、未受影响渲染、全量clippy或CUDA，两个legacy完成标志False。

日志：build/native-path-graphics-rust.txt、build/pdf-path-graphics-rust.txt、build/native-path-graphics-wheel.txt、build/native-path-graphics-install-test.txt和build/native-path-graphics-regression.txt。完整强图形/表格/重叠仲裁、跨页准备及默认PDF编排仍待，未作整文档质量或性能结论。
