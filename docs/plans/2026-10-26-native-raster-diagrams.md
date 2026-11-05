# 栅格标签连接图与文字框证据

迁移旧graphics.py的栅格连接图候选及普通文字框判断，复用已迁移绘图组件统计和真实有字网格保护。原顺序、12线/4图/横纵数量、内部短横线及图片覆盖门槛保持；文字框按双向覆盖、内部图像/复杂路径和文字证据决定。

先冻结旧输出和真实PDF输入链，再做定向测试；阶段末一次wheel和相关安装回归，无全量/CUDA。返回阶段候选，不代表完整强图形候选集合或默认PDF编排。

已完成144组旧输出，63连接图/46普通文字框正例。覆盖4图/12线/横纵数量/组件尺度面积、95%图片覆盖、先真实网格保护再内部3条短横线、Form排除对网格的影响、普通四线框双向98%覆盖/90%图片或复杂路径拒绝/95%文字证明。复用绘图组件原顺序及bbox类型，不改变源数据。

20页实际PDF全实时读取字符/图片/路径/绘图，生成视觉行和组件查询框后检测：5连接图/1文字框正例，另有1实际有字网格排除正例。主16页包括连接图、图片表格、普通和空文字框的四方向样本；另4页使用不污染外网格跨度分组的内部短横线，独立证明正向文字网格保护。已有16页和阶段JSON未覆盖；试验性pdf-raster-diagrams-grid.json四页保留原样，不计入20页安装链。无组件/框/网格结论注入。180度默认视觉行缺少文字，按旧支持域处理，不能称任意旋转方向表格保护已完成。

接口 `detect_native_raster_diagrams(source, *, em, frame_bboxes=())` 返回raster_diagram_bboxes及与frame查询原序一致的plain_text_frame_flags。source提供page_size、drawing_lines、image_bboxes、lines、path_infos、form_bboxes。少于4图或12线时保留旧候选早退；可选frame证据独立计算。普通JSON，消费drawing/image/frames各最多5000，文字框消费path最多5000/lines最多4096，finite double point数值abs<=1e9；沿既有JSON/10M累计Work/8MB结果。组件和网格调用传递同一Work，输入/组件/覆盖/内部线和网格扫描累计计费，资源错误传播，无跨调用缓存。

两个定向Rust单元测试和一个20页实际PDF集成测试通过；阶段末一次release wheel构建并隔离安装。native_raster_diagrams、native_drawing_graphics、native_graphic_arbitration、compatibility、smoke共五组安装套件通过，覆盖字段/类型/顺序、完整实时页源和网格证据、所有权/并发、输入/扫描限制及错误恢复。fmt、93模块来源审查和固定36 core＋3 paths＋1 script哈希保持；无全workspace、全部111组runner、未受影响渲染、全量clippy或CUDA，完成标志False。

日志：build/native-raster-diagrams-rust.txt、build/pdf-raster-diagrams-rust.txt、build/native-raster-diagrams-wheel.txt、build/native-raster-diagrams-install-test.txt及build/native-raster-diagrams-regression.txt。完整原生diagram/bar产生器、跨页准备和默认PDF编排仍待，未作整文档质量或性能结论。
