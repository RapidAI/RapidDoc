# 原生闭合网格证明

迁移旧 `has_native_closed_grid`：候选角度/core回退、较大Form包含排除、source ID成员、正effective_height中位数、完整贯穿横纵边界、round(...,2)轨道去重/端点容差和文字占用行列。保留默认至少3行、显式allow_two_by_two至少2行；不添加候选内文字或全格占用要求。

API对显式上游候选逐项生成闭合网格标志，不注入网格结论。冻结旧完整函数与异常行为，补实际PDF实时字符/路径/绘图/视觉行→候选查询→证明链。缺少正行高而需要计算中位数时按invalid_input传播，不静默返回False。阶段末一次wheel/相关安装回归，无全量/CUDA。一般表格候选编排和diagram整体仍待，真实Form提取不包含在本阶段。

## 完成及验收

- 新增 `inspect_native_closed_grids(source, candidates, *, allow_two_by_two=False)` 返回原候选顺序的closed_grid_flags。候选bbox/core_bbox/angle/line_indices、源行bbox/source_index/effective_height及实际绘图/Form数据均由普通JSON明确提供；不把候选元数据当闭合网格结论。
- 完整迁移旧单候选证明，保留默认6成员/3纵轨/4横轨/2列3行占用，显式2x2为4成员/3纵轨3横轨/2列2行。成员按原source ID选取，重复源行仍计数，不额外要求文字在框内或每个格子有字。core优先、空core回退、非0候选角度拒绝、Form95%包含且面积严格大于1.5、仅正行高中位数、端点/起点窗口和两位舍入按旧保留。
- 冻结176旧完整证明参考，117正例；2个旧StatisticsError在中位数消费时映射invalid_input，不静默返回False；24个旧round(x,2)事实覆盖半值附近相邻double。Form面积比较/覆盖复用既有有界int/float适配，未修改原填充网格算法。
- 16页实际四方向PDF、32组默认/2x2模式查询，7正例。实时提取字符/路径/绘图和视觉行，首路径矩形/原框及新行source IDs重建显式查询，未注入网格判断；不把这一查询构造称完整一般表格候选产生器。实际Form输入为空，Form排除正例来自规则参考，真实Form提取和实际Form正例另待。
- 两个新Rust单元测试、两个填充网格面积适配回归单元测试、一个32查询PDF集成测试通过。阶段末一次release wheel/一次隔离安装，native_closed_grids、native_filled_grid、compatibility、smoke四组安装套件通过；验证bool结果及原序、默认模式、source ID语义、输入所有权、并发、累计预算/资源错误与恢复，安装环境不导入旧Python解析器/PDFium。
- 4096源行、各5000规则/Form/候选、消费合计20k候选IDs、finite abs<=1e9、严格boolean模式；既有JSON/10M累计Work，flag数量保证输出小于8MB。源解析/集合成员/中位数及轨道排序/边界和占用扫描逐步计费，接受上层Work，无跨调用缓存。
- fmt、100模块审查、固定36＋3＋1上游哈希通过；全量runner登记118组，本轮未全跑，无CUDA。日志build/native-closed-grids-rust.txt、build/closed-grid-filled-regression-rust.txt、build/pdf-closed-grids-rust.txt、build/native-closed-grids-wheel.txt、build/native-closed-grids-install-test.txt、build/native-closed-grids-regression.txt。

## 后续

继续一般表格候选编排，随后diagram路径/栅格组件、正文屏障和时间轴扩框、强图形统一入口、跨页准备与默认PDF。当前接口不能替代候选产生器，也不证明整文档质量/性能全面优于旧版。
