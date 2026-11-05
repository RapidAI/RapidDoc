# 原生柱图几何准备

迁移旧复合矩形共同基线事实、柱体准入/方向门槛和零轴扩边。复用原样vendor compound_baselines内核，按原path/矩形位置返回独立证据。填充表格框table_bboxes是必须明确给出的上游阶段输入，当前不声称已迁移tiled-fill表格产生器或完整柱图检测。

冻结旧AST准入及原基线/零轴规则，补实际PDF实时路径/绘图链；定向测试，阶段末一次wheel/相关安装回归，无全量/CUDA。后续柱组聚类、标签/栅格/轮廓扩展和完整强图形入口仍待。

## 完成及验证

- 复用固定原样 compound_baselines；迁移共同基线、颜色/段数/尺寸/方向门槛、95%表格覆盖排除及零轴半线宽补边，保留路径/矩形位置身份、原序和 JSON 数值类型。
- 冻结170组旧AST/原零轴参考，123准入正例；8页专用实际PDF覆盖四方向及复合/独立矩形，8准入和零轴正例。实际链实时读取路径/绘图，经新接口准入后重建查询，不注入旧准入或基线结论。表格框仍是显式输入，旧实际样本的上游填充表格检测为空。
- 定向Rust两个单元测试、一个8页PDF集成测试通过。阶段末仅一次release wheel、一次安装；native_bar_geometry、native_stacked_graphics、compatibility、smoke四组安装套件通过，包含字段/类型/顺序、输入所有权、并发、资源边界和异常后恢复。
- 各5000路径/绘图/表格/查询、合计20k矩形/准入及20k零轴输出；有限数值abs<=1e9，JSON/10M累计Work/8MB输出。复合基线平方扫描预付，共享已有Work并传播资源错误。
- fmt、94模块审查及固定36＋3＋1上游文件哈希通过；完整runner登记112组，本轮未全跑，未运行CUDA。
- 日志：build/native-bar-geometry-rust.txt、build/pdf-bar-geometry-rust.txt、build/native-bar-geometry-wheel.txt、build/native-bar-geometry-install-test.txt、build/native-bar-geometry-regression.txt。

## 后续边界

填充表格产生器尚未迁移；table_bboxes是必填阶段输入。共同基线/准入不等于完整柱图检测：还需柱组聚类、数字标签/图题保护、轮廓/栅格扩框和真实网格排除，再接完整强图形入口、跨页准备与默认PDF。当前测试不证明整文档质量、速度或内存全面优于旧版。
