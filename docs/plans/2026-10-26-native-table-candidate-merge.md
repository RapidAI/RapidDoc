# 已物化表格候选合并

迁移 `_merge_table_candidates` 对已物化 `_TableCandidate` 的完整行为：按score稳定逆序、同角度首个20%重叠目标、100分强网格拒绝低于100分弱候选扩张、bbox/local/core合并、来源集合并集、同类型注释合并及表体成员优先消解冲突，最终空间稳定排序。候选draft的惰性materialize和规则扩框仍是上游阶段，不声称已完成。

Python API保持普通候选字典；桥接把annotation.line_bboxes字典转成有序键框对，Rust不丢失原插入顺序，返回再恢复字典。冻结旧完整函数在已物化支持域的输出和输入副作用投影；新实现返回独立候选，不修改用户输入。开发只跑受影响测试，阶段末一次wheel/相关回归，无全量/CUDA。

## 实现与验证范围

- API接收已物化候选的bbox/local_bbox/core_bbox、angle/score、line_indices、annotations和其他既有字段；返回原最终空间序的owned candidates，输入不修改。低分候选不得扩张强网格，多个匹配目标只选首项；动态目标扩框会影响后续输入，但不反向合并已有目标。
- 注释按kind首个已有注释合并，逐ID合并行框；先并表体成员再去除caption/footnote冲突成员。未发生候选合并时不额外清理原注释。保留未合并目标的inferred_grid等元数据，不自行合并推断线或样式标志。
- Python字典插入顺序通过内部有序[id,bbox]数组桥接保留，返回再恢复整数键字典；顺序会影响数值平局时bbox原类型，不能仅将字典按键排序。Rust调用此JSON接口也须传有序行框对。注释集合成员以排序数组输出，shared-set的存储优化不暴露为Python对象。
- 158旧已物化完整函数参考，142输出变化；包含20%阈值、100分屏障、首项目标/动态扩框、core传播、同类注释、表体冲突、字典逆序及int/float平局、非合并情况下不清理冲突、目标额外元数据保持。
- 28页实际PDF实时提取字/路径/绘图/视觉行，实际填充候选检测后构造44组重复/部分/双半表查询再合并，24变化；查询片段是测试构造，不代表一般规则候选产生器或draft物化。
- 普通JSON，finite几何/score/angle abs<=1e9，各5000候选、每候选2000注释、消费合计20k表体/注释成员及行框；既有JSON/10M累计Work/8MB结果。排序/配对/集合/注释归并和冲突处理计费，接受已有Work，无跨调用缓存。

## 一般表格编排的具体进度

已完成：相接填充表格、外框填充行带表格、给定候选闭合网格证明、完整覆盖替换仲裁、已物化候选合并与注释冲突消解。

后续进展：`_exclude_prose_spanning_rule_drafts` 的跨正文排除已单独完成，见 `2026-10-26-native-prose-rule-conflicts.md`。当前下一子任务为规则草稿物化所需的边界/成员/注释准备。仍缺规则/闭合网格draft产生与惰性物化、补充候选产生器的检测编排，以及最终题名绑定/正文尾修整等；已有区域表格结构恢复不等于这些页面级候选产生器已经接通。上述阶段之后再接diagram与强图形统一入口、跨页准备和默认PDF。

## 验收完成

- 两个定向Rust单元测试和一个44组PDF实时检测候选查询集成测试通过。
- 阶段末仅一次release wheel和一次隔离安装；native_table_candidate_merge、native_table_replacements、native_tiled_tables、compatibility、smoke五组安装套件通过，含注释字典插入顺序、冲突移除、原类型/其他元数据、深层输出所有权、并发、累计预算/错误恢复；未导入旧Python解析器/PDFium。
- fmt、102模块审查、固定36＋3＋1上游哈希通过，全量runner登记120组，本轮未全跑，无CUDA。
- 日志build/native-table-candidate-merge-rust.txt、build/pdf-table-candidate-merge-rust.txt、build/native-table-candidate-merge-wheel.txt、build/native-table-candidate-merge-install-test.txt、build/native-table-candidate-merge-regression.txt。
