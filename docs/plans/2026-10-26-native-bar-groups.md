# 原生柱组和数字标签证据

连接已迁移的相接填充表格检测及柱体准备，在同一累计Work中迁移共同零轴选择、CPython集合删除顺序、稳定分类方向排序、间隔切簇、round(...,1)位置区分、长度差/真实间隙门槛和数字标签几何准入。返回原准入柱体位置、合格组的核心/零轴补边框、数字行位置及label_anchor。

不执行后续图题保护、两轮轮廓/栅格/标签扩框和最终有字网格排除，不能称为完整柱图检测。冻结原函数AST至label_anchor完整前段，源表格及柱体结论不注入。只跑受影响Rust测试，阶段末一次wheel及相关安装回归，无全量/CUDA。

## 验收

- `prepare_native_bar_groups(source, *, em)` 自动组合真实相接填充表格排除与柱体准备，所有子阶段传递同一累计Work。返回table_bboxes、admitted_bars、groups；组内bar_indices指向原准入数组，numeric_line_indices指向原行数组，不按source_index去重。
- 原函数AST完整保留至label_anchor，仅在此处截断并投影证据；上游表格、基线、准入、分簇和数字结论未替换。168基础参考+19临界参考共187组，152柱组正例、2真实上游表框排除。另30个独立旧round(x,1)事实覆盖二进制临界点两侧。
- 20页实际PDF全实时字符/路径/绘图/视觉行→新接口完整前段，6正例页/10柱组、4表框排除；同时涵盖横竖柱、复合与独立路径、四方向及既有失真CJK字体表格。混合方向实测发现并修复了簇成员原准入位置再次误作方向局部位置的问题。实际180度默认无文字支持域沿旧保留，不把20页都称为柱组正例。
- 2个柱组单元测试、1个复用CPython集合轨迹单元测试、1个20页PDF集成测试通过。阶段末一次release wheel/一次安装，native_bar_groups、native_bar_geometry、native_tiled_tables、compatibility、smoke五组安装套件通过，涵盖字段/类型/顺序、原身份、输入所有权、并发、累计预算/资源错误和恢复。
- 正且有限em、输入finite abs<=1e9；各5000路径/绘图、每方向5000柱体、最多5000组，沿子阶段20k字符/准入/零轴与4096行、每行20k码点、JSON/10M累计Work/8MB输出。集合槽扫描、支持计数/排序、数字语法和成员几何扫描计费，无跨调用缓存。只是普通JSON/PDF point支持域，不支持任意Python对象或monkeypatch。
- fmt、96模块审查、固定36＋3＋1上游哈希通过，全量runner登记114组，本轮未全跑，无CUDA。日志build/native-bar-groups-rust.txt、build/bar-groups-pending-rust.txt、build/pdf-bar-groups-rust.txt、build/native-bar-groups-wheel.txt、build/native-bar-groups-install-test.txt、build/native-bar-groups-regression.txt。

## 后续

保留图题文字的完整识别/归属规则、两轮接触柱体/轮廓/栅格/标签扩框、最终真实有字网格排除仍待迁移，然后接入完整强图形候选集合、跨页准备和默认PDF。当前未证明整文档质量/速度/内存全面优于旧Python。
