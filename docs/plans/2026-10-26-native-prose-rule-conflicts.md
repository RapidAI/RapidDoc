# 跨正文规则草稿排除

迁移旧 `_exclude_prose_spanning_rule_drafts_python` 完整判断：有足够下方网格尺寸/距离、贯穿竖轨与上下横边界，并且草稿成员中至少三排普通长英文正文时排除草稿。保留caption、非0方向、无有效尺度/边界等跳过条件；源ID、正文词边界、字重和行距阈值保持。

输入为明确上游投影：候选kind=table或draft；draft含context尺度/角度、是否有caption、边界框及从rows.fragments提取的member_source_indices，不注入冲突结论，不执行draft产生或物化。返回原序候选与拒绝原位置。冻结旧完整判断，补实际PDF实时文字/绘图构造显式查询。只跑受影响测试，阶段末一次wheel/相关回归，无全量/CUDA。

## 实现与验证范围

- 完整迁移旧Python判断，不调用旧native加速分支，不以外部conflict标志替代规则。物化网格与草稿身份用kind明确投影；保留原序完整元数据，返回rejected_indices为原数组位置，输入不修改。
- angle不为0或存在caption的草稿跳过；正em且有边界才检查。网格至少8em宽/4em高、顶边距首边至少5em、末边在grid顶-em到底+em内、边界与核心横向交叠90%；贯穿竖轨按>.3em分轨至少3条，grid上下须有.5em内横边界。
- 正文必须是草稿成员source ID、angle0、六个以上ASCII英文单词且满足Python Unicode词边界、无caption/formula/semantic标签，weight默认400且<600；正文宽65%网格且横覆盖90%，中心严格位于首边+.5em到grid顶-em间。按中心排序并以>.6em分排，至少3排才冲突，重复source ID仍按各原行几何处理。
- 151旧完整参考（112拒绝）覆盖跳过路径、词边界、字重、行排/竖轨阈值、上下边界、重复及有符号成员ID、原序/重复草稿；20页四方向实际PDF实时提取字符/路径/绘图与完整视觉行资格字段，再重建显式draft/grid查询，2拒绝。em10与caption存在是查询输入，不能称自动草稿或caption检测。
- 普通JSON、finite abs<=1e9、5000候选/绘图、4096源行各20k码点、消费合计20k边界及20k成员IDs；既有JSON/10M累计Work/8MB结果，词法/集合/草稿网格配对/轨道行排序/覆盖扫描累计计费，接受上层Work，无跨调用缓存。

## 子任务进度

已完成：两套填充表格候选、给定候选闭合网格证明、完整覆盖替换仲裁、已物化候选合并及注释冲突消解、跨正文规则草稿排除。

后续进展：核心边界/成员、表题续行及注释记录准备已单独完成，见 `2026-10-26-native-rule-annotation-preparation.md`。下一项是自动表注行收集，随后接规则候选扩框；草稿产生、补充候选产生器的页面级编排和最终题名/正文尾修整仍待。完成这些后继续diagram与强图形统一入口、跨页准备和默认PDF。不作整文档质量/性能已达标的结论。

## 验收完成

- 两个新Rust单元测试、两个复用英文词边界/图题回归测试、一个20页PDF集成测试通过。
- 阶段末一次release wheel和一次隔离安装；native_prose_rule_conflicts、native_table_candidate_merge、native_graphic_captions、compatibility、smoke五组安装套件通过，验证完整元数据/原序/数值类型、所有权、并发、累计预算/资源错误与恢复；运行时未导入旧Python解析器/PDFium。
- fmt、103模块审查、固定36＋3＋1上游哈希通过；全量runner登记121组，本轮未全跑，无CUDA。
- 日志build/native-prose-rule-conflicts-rust.txt、build/prose-rule-caption-regression-rust.txt、build/pdf-prose-rule-conflicts-rust.txt、build/native-prose-rule-conflicts-wheel.txt、build/native-prose-rule-conflicts-install-test.txt、build/native-prose-rule-conflicts-regression.txt。
