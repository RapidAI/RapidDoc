# 有图题的同心路径图形

迁移旧graphics.py图题/百分比/同心轮廓/引线证据产生器，保留根层、描边、段数、宽高/中心/图题距离、百分比距离和原顺序包含去重；图题按Python Unicode数字/空白/忽略大小写规则处理。先冻结旧边界和实际PDF链，再做定向差分，阶段末一次wheel和相关回归。候选仍接已有有字网格仲裁，尚非完整强图形入口；无全量/CUDA。

已完成196组旧完整输出，145正例。覆盖fig/figure编号英文图题、Unicode decimal/space、Python IGNORECASE的I/İ/ı变体、全角/阿拉伯数字百分比和逗号小数、根层/描边/段数/宽高比/直径、同心中心阈值、图题/百分比距离和交叠、引线尺寸扩框、重复来源号/位置身份、原序包含去重及bbox原值/类型。中心坐标分别求和除二后相减，保持旧运算次序，未用合并减式近似。

8页专用实际PDF四方向链，1正例；实时读取字符/路径并生成视觉行→同心图候选→既有有字网格/重叠仲裁，输入页源、候选和最终输出均逐项对照。没有路径/文字/候选结果注入。图题仍只消费angle0，默认视觉行不包括180度文字，其他角度/中文图题未自行增加新策略。

接口 `detect_native_concentric_graphics(source, *, em)` 返回concentric_bboxes，source包含page_size、lines（text/bbox/angle）、path_infos，输入不修改。em显式传入；普通JSON，最多4096行/每行20000码点/5000路径，finite double point数值abs<=1e9，沿既有JSON/10M累计Work/8MB结果限制。文本、peer筛选/合并、百分比/图题/引线及包含扫描累计计费，接受调用方原预算，资源错误传播。输出仍须交共用图形/有字网格仲裁，不是完整强图形候选集合入口。

两个定向Rust单元测试和一个实际PDF集成测试通过；阶段末一次release wheel构建并隔离安装。native_concentric_graphics、native_graphic_arbitration、compatibility、smoke共四组安装套件通过，覆盖精确字段/类型/顺序、Unicode语法、所有权/并发、输入/候选扫描限额及错误恢复。fmt、92模块来源审查及固定36 core＋3 paths＋1 script哈希保持，无全workspace/全部110组runner/未受影响渲染/全量clippy/CUDA，完成标志False。

日志：build/native-concentric-graphics-rust.txt、build/pdf-concentric-graphics-rust.txt、build/native-concentric-graphics-wheel.txt、build/native-concentric-graphics-install-test.txt和build/native-concentric-graphics-regression.txt。完整强图形产生器集合、跨页准备和默认PDF编排仍待，未作整文档质量或性能结论。
