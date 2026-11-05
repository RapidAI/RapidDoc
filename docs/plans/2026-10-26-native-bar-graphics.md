# 完整原生柱图候选及图题保护

迁移旧图题保护 helper（含同字体续行、强图题、caption语义/paragraph_group副作用）与原生柱图候选剩余两轮扩框：接触柱体、复杂轮廓、混合栅格和普通标签，最后执行真实有字网格排除。前段复用已迁移柱组准备，全部使用同一累计Work。对照旧完整函数及可见行副作用；不改旧Python策略，不把候选检测称默认PDF完整编排。

开发只跑受影响Rust/差分测试，阶段末一次wheel及相关安装回归，无全量/CUDA；固定上游不改。图题识别只按旧英文fig/figure种子，强图题判定仅需该种子支持域，不扩展为新的中文图题策略。

## 实现范围与参考

- `protect_native_graphic_captions`：原序输出行与排序source ID保护集。沿旧英文fig/figure数字种子、angle0排序、子集字体族/字重/缩进/行距/宽度续行、种子域内强图题判断、父图窗口/inset tail及句末屏障；完整保留caption/title_suppressed/paragraph_group修改。NFKC和identifier/tail解析只覆盖已通过旧英文种子的支持域，不宣称完整共享annotation_text中英文API已迁移。
- `detect_native_bar_graphics`：复用真实相接填充表格/柱体/柱组，逐组调用图题保护，然后两轮按旧原序union接触柱体、复杂轮廓、栅格、普通标签、数字行，最后真实有字网格排除。全程共用已有累计Work，源行几何不变，网格准备仅本次调用复用，无跨调用结果缓存。返回独立source和原序bar_bboxes。
- 冻结151旧完整图题保护参考（146保护正例、107行语义修改）和122旧完整柱图参考（121候选正例、1source修改、1网格排除）。参考直接调用旧完整函数，不替换图题/柱组/图像/网格结论。
- 40页实际PDF实时读取字符/图片/路径/绘图及完整视觉行元数据，再调用完整新检测。10候选正例、1图题行修改、1实际有字网格排除；网格排除样本另确认前段存在合格柱组。包含混合栅格、复杂轮廓、来源屏障、四方向与既有填充表格排除。旋转正例分布沿旧保留，不把40页都称正例；自制字体样本仍不代表客户文档质量。
- 普通JSON，positive finite尺度、finite输入abs<=1e9；继承各5000路径/绘图/图片/Form/候选、4096行每行20k码点和子阶段20k字符/准入/零轴限制。字体签名两项，名称4096字节，已有group整数绝对值<=1e9；JSON/10M累计Work/8MB输出。文字、字体、排序、种子/续行/候选扫描、轮廓来源配对和union/网格扫描计费，资源错误传播；输入不修改。

## 后续边界

该候选产生器完成后仍需原生diagram组件等其他强图形产生器、完整产生器收集/仲裁入口、跨页页源准备和默认PDF编排。不据此宣称整文档解析质量、速度或峰值内存全面优于旧版。

## 验收完成

- 图题保护2个Rust单元测试、完整柱图2个Rust单元测试、1个40页实际PDF集成测试通过，含已有累计Work耗尽后不得重置及恢复验证。
- 阶段末仅一次release wheel和一次隔离安装；native_graphic_captions、native_bar_graphics、native_bar_groups、native_graphic_arbitration、compatibility、smoke六组安装套件通过。验证完整字段/数值类型/顺序/原身份、图题修改、输入所有权、并发、资源边界与异常恢复；安装环境未导入旧Python解析器/PDFium/PIL。
- fmt、98模块审查、固定36＋3＋1上游文件哈希通过；全量runner登记116组，本轮未全跑，未执行CUDA。
- 日志build/native-graphic-captions-rust.txt、build/native-bar-graphics-rust.txt、build/pdf-bar-graphics-rust.txt、build/native-bar-graphics-wheel.txt、build/native-graphic-captions-install-test.txt、build/native-bar-graphics-install-test.txt、build/native-bar-graphics-regression.txt。
- 下一步原生diagram组件和强图形统一入口；默认PDF完整编排、跨页准备和真实文档质量/性能验收仍待。
