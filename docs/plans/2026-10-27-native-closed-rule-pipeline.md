# 闭合物理网格候选完整链

本轮连续实现实际连通网格构造、正式候选扩框、页面行→局部片段/表格行、表题自动选择、稀疏闭合网格候选/相邻续表证明，以及与已有正式组装的衔接。保留旧算法，不注入网格/表题/核心成员结论；一般横线区间候选分支仍待，不把闭合分支叫完整页面编排。

尽量复用既有graphic_grid精确均值、轨道和分组内核，旧helper与整条闭合分支独立冻结对照；开发只跑受影响测试。本轮全部完成后一次wheel及相关回归，无全量/CUDA，不每个子helper单独构建。

## 实现范围

已接通四个SDK入口：`inspect_native_rule_table_geometry`、`detect_native_closed_rule_tables`、`expand_native_connected_table_grids`、`detect_native_physical_table_candidates`。源行先按查询方向筛选和旋转，复用旧式精确均值/CPython求和、聚行和轨道内核；自动表题支持旧Unicode词界、数字/罗马/中文编号及分离编号邻行。闭合网格按原走廊裁剪、中心门控、排除框和Form包含判定选核心；两横边分支保留稀疏占列及相邻页完整列轨证明，随后自动表题/表注和正式组装共享预算。扩框保留旧候选最佳网格排名及注释成员排除。

物理页面入口复用filled/tiled检测，按旧90%优先、按角度构造排除区（普通2.5em、tiled最多页宽1%）、闭合候选生成、强网格合并、20%填充覆盖排除、稳定y/x排序。扩大的排除区不能误作续表前方障碍；续表障碍仅原排除框/图像/签章/Form。

## 冻结证据

- 129自动局部行/表题/连通轨道事实，129完整闭合候选事实（114正例），129已有候选网格扩框事实，132物理分支编排事实（117正例），共519；并非519个不同文档。
- 24页实际PDF样本（本轮8页生成fixture＋此前16页闭合网格fixture），实时原字符/字形/绘图路径/视觉行重建source后比较四个入口。8页闭合候选正例、16页物理候选正例，闭合与物理输出各含2个实际表题/2个表注正例。自动检测没有注入行分组/core/表题/表注结论；只有“扩已有候选”入口消费明确种子区域。
- 6个当前单元测试、1个实际PDF集成测试，以及核心/注释准备/图形仲裁共6个受影响单元测试通过。累计预算不足时不能在子调用重置；扩框含无core分支也累计成员和注释引用。

## 边界与下一项

下一项是普通横线区间候选自动产生与核心选择。开放/阴影/数值/标题等其余候选分支、惰性draft物化/存储优化、最终整页caption/tail策略、diagram及默认PDF接入仍待。上一页末表网格提取尚待，当前传入既有previous_table_grid元数据；Form/图像/签章排除有旧规则事实，24页实际PDF输入为空，不冒称真实Form提取验证。保持原算法支持域，包括既有180度文字限制；不作整文档等价或性能更优结论。

## 验收完成

阶段末一次release wheel、一次隔离安装；新增整链安装测试以及正式组装、图形仲裁、候选合并、compatibility、smoke共6组安装套件通过。严格比较数值类型、成员和注释顺序、源所有权、并发、资源上限与异常恢复；安装进程没有导入旧Python解析器或旧Python PDFium。12个受影响Rust单元测试及1个24页实时PDF集成测试通过。fmt、107模块审查和固定36＋3＋1上游哈希通过；全量runner登记125组，本轮未全跑，无CUDA。

日志：build/closed-rule-pipeline-rust.txt、build/closed-rule-related-rust.txt、build/closed-rule-pipeline-pdf-rust.txt、build/native-closed-rule-pipeline-wheel.txt、build/native-closed-rule-pipeline-install-test.txt、build/native-closed-rule-pipeline-regression.txt。

