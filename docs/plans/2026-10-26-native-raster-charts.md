# 栅格图说明归组

复用固定rule_graphics刻度分组内核，迁移旧graphics.py的数字语法、图体/标签/图例扩框及说明续行分组。先冻结旧完整输出与中间刻度/图体事实，再实现普通JSON适配和累计预算，最后验证实际PDF字符/图片读取到归组链。保留稳定顺序、最大图面积首项平局、原分组号和同一core相邻链覆盖行为。

开发只跑定向测试；阶段末一次wheel构建及相关安装回归，无CUDA或全量测试。完整强图形检测、跨页页源准备、候选检测和默认PDF编排仍待。

已完成：269基础＋3大整数/混合浮点面积边界，共272组旧完整输出及刻度/图体事实；223图体、151归组正例。包括数字语法、刻度数目/等差/间隔/字号、图像邻接/比例、标签/图例扩框、列表/句末/字体/短尾拒绝、已有和负分组、原数组位置身份及同一core相邻链覆盖。最大面积比较保留Python整数精度和整数/浮点比较，等面积保持首项。

16页真实专用PDF包含横纵刻度、四方向及字体/列表/句末/宽尾拒绝；全链实时读取字符和图片→视觉行/局部页源→全文图片背景/水印/白边过滤→噪声→图体/续行归组，10图体/2归组正例。每页图片原始流独立，图像过滤按旧全文规则执行；不注入图片、刻度、图体、角色或颜色结果。源字符和语义保持，只在独立返回的source.lines修改paragraph_group。

接口 `group_native_raster_chart_descriptions(source)` 返回source、graphic_bboxes和axis_groups（vertical/line_indices，身份按原行数组位置）。source提供lines、可选image_bboxes；普通JSON输入保留未消费字段和未修改行的原始表示。最多4096行/每行20000码点/5000图片框；字体签名最多32项str（每项256字节）/i64/bool/null，输入group绝对值<=1e9，有限数值abs<=1e9。沿既有JSON/10M累计Work/8MB结果预算，重复中位数排序预付保守上界，图体和说明扫描、字体比较累计计费，并接受调用方已有预算；大规模刻度可能明确返回资源错误，不静默更换算法。

两个定向Rust单元测试和一个实际PDF集成测试通过；一次release wheel构建并隔离安装。native_raster_charts、native_text_noise、native_source_lines、compatibility、smoke共五组安装套件通过，覆盖字段/数值类型/顺序、所有权/并发、输入/扫描限制、错误恢复和默认完成标志False。fmt、86模块来源审查及固定36 core＋3 paths＋1 script哈希保持。未跑全workspace、全部104组runner、未受影响渲染、全量clippy或CUDA。

日志：build/native-raster-charts-rust.txt、build/pdf-raster-charts-rust.txt、build/native-raster-charts-wheel.txt、build/native-raster-charts-install-test.txt和build/native-raster-charts-regression.txt。完整强图形检测、跨页页源准备、候选检测及默认PDF编排仍待，未作整文档质量/时间/峰值内存结论。
