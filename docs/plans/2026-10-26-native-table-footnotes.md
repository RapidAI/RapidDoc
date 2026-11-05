# 自动表注收集

迁移完整旧 `_collect_footnote_rows` 与走廊行准备、辅助标记NFKC/casefold及L/N/S分类、紧凑token/上标字形表内引用、正文最高四分位字号参照。复用已迁移行裁剪及注释构造，返回自动选择的表注行和注释记录；不让调用方提供显式/辅助表注结论。

Unicode分类冻结为Python Unicode16，其他属性及casefold复用现有固定数据。仅普通JSON/有限几何，所有操作传同一累计Work，无跨调用缓存。冻结旧完整函数与标记事实，补实际PDF实时几何/文字查询，阶段末一次wheel/相关回归，无全量/CUDA。临时候选自动产生/转正式仍待接入。

## 实现与验证范围

- `collect_native_table_footnotes` 自动完成Note/Source/注等显式标记、1..3个通用Unicode辅助标记以及表体引用判定。辅助标记NFKC/casefold后必须全为Unicode L/N/S类别；数据冻结自CPython Unicode16并记录哈希，运行时不依赖宿主Python Unicode。
- 辅助引用同时检查核心所有源行的紧凑单元格token和小且上移的连续字形：最多12个非空白字符/4个token，或标记字高<=普通字高85%且中心上移>=12%。字形使用原char bbox归一化及局部方向转换；缺失或零面积字框不当作证据。
- 按旧走廊裁剪实际rows，用source ID最后出现的line提供字号/字体（核心引用仍看所有成员行）。辅助首行还需与正文最高四分位字号比较，样本不足回退1.25em；显式表注不提前扫描正文参照。首行gap、x覆盖/缩进/字高、续行字高比例/字体交集/缩进、10em链限和已选成员推进bottom均保持旧规则，不增加启发式。
- 返回自动选择的footnote_rows及footnote annotation（无则None），其原坐标bbox与有序整数键line_bboxes复用已迁移构造。没有外部explicit/auxiliary/reference布尔值或预选表注行输入，核心框/成员和视觉行仍为明确上游事实。
- 冻结155旧完整自动收集参考（107正例）与22额外角度/链边界参考（22正例），共177参考/129正例；另75辅助标记语法事实。覆盖显式/辅助/缺引用、紧凑token/上标字形、正文字号、字体/缩进/间隔、重复来源样式映射、原序、局部0/90/180/270/其他方向、链限及首行不施加续行长度帽的旧行为。
- 实际20页PDF全实时字符/字形/路径/视觉行查询，6表注正例，含近距离辅助紧凑引用及真实上标字形引用；缺引用负例保持。原16页中辅助样本超过旧.75em首间距仍为负例，另独立4页近距语料补正例，未覆写原参考。页面旋转0/90/270、查询angle0；每个视觉行投影为单片段local=原框，核心按首填充框取成员，未声称完整表格聚行/候选产生。
- 普通JSON、positive有限median_height/finite abs<=1e9、4096行/rows、合计20k字形与20k片段/20k核心IDs、每行20k码点，可靠字体签名两项且名称<=4096字节；JSON/10M累计Work/8MB输出。词法归一化、字形/上标、字体比较、集合、排序和注释扫描累计计费，接受已有Work，无跨调用缓存；输入不修改。

## 下一项

自动表注行选择已消掉。后续进展：核心、表题与自动表注组装为正式候选及方向还原已完成，见 `2026-10-27-native-rule-candidate-expansion.md`。下一项为连通网格扩框；页面级候选自动产生、补充产生器和默认PDF编排仍待。

## 2026-10-27 验收完成

- 两个新Rust单元测试、两个复用行准备回归测试、一个20页PDF集成测试通过；额外75标记事实由Rust逐项核对。
- 阶段末一次release wheel、一次隔离安装；native_table_footnotes、native_rule_annotation_preparation、native_table_candidate_merge、compatibility、smoke五组安装套件通过，验证字体/Unicode/字形证明、原序行框字典/类型、深层所有权、并发、累计预算与异常恢复；运行时不导入旧Python解析器/PDFium。
- fmt、105模块审查、固定36＋3＋1上游哈希通过；全量runner登记123组，本轮未全跑，无CUDA。
- 日志build/native-table-footnotes-rust.txt、build/table-footnotes-preparation-regression-rust.txt、build/pdf-table-footnotes-rust.txt、build/native-table-footnotes-wheel.txt、build/native-table-footnotes-install-test.txt、build/native-table-footnotes-regression.txt。
