# 目录预认领与 Form 选择接线

沿旧 index_blocks.py 完整提取器迁移目录行归并、标题、页码/点线、无页码编号和语义副作用；同一入口支持旧的两个标题要求选项。Form 迁移 graphics.py 的完整选择及收紧路径，保留字符来源成员限制、保留 Form 的面积例外、收紧后的来源映射别名。

用独立 page_containers 模块接现有代码/表格页事务：先明确目录预认领，再选择 Form，规则代码排除 Form，候选经 Form 吞并过滤，完整表格提交后移除被表格覆盖的 Form，底色代码只排除活跃 Form。目录块按旧顺序放在已完成容器阶段末尾，仍不声称 Form/图片图形物化、后续文本公式和默认 PDF 已完成。

来源映射以 bbox/source_indices 条目列表跨 JSON 边界传递，恢复旧 tuple-key 字典后导出参考；SDK 只返回独立副本。旧 table-only 与 code/table 接口不扩范围。共享 Work、唯一来源号、4096行/20k字形/5000框路径、有限数值及8MB输出上限；每个完整旧入口及实际 PDF 源进行对照，只跑受影响测试，阶段末构建安装一次，无全量/CUDA。

## 当前实现与验证

- 三入口及连续顺序已接通；目录固定块在本阶段末尾，Form吞并候选与表格覆盖后的活跃Form剔除保持旧0.9/0.5阈值。不是完整_prepare_page_source。
- 242目录（213＋27边界＋2长整数有效输入）、104Form（88canonical＋10边界＋6字典键类型）、70连续阶段，共416结果；另2个旧4301位编号ValueError保留invalid_input。Python默认4300位限制固定记录，Unicode数字也验证。
- Python字典重复键保留首项键类型/位置而使用最后值，映射frozenset按来源号排序；新增别名不改既有整数键为浮点，SDK副本不修改输入。
- 初轮FormInfo.bbox未还原tuple、编排输入未显式提供原提交器要求的angle：原冻结文件保留，不计有效证据，另canonical正确类型对照，不修改旧解析器结果。
- 75不同实际PDF页（8新增＋67既有），4页目录/41页表格/3页代码，旋转0/90/180/270的接受与拒绝均保留。Form框/归属为空，不声称实际Form正例或完整Form提取。
- 8个受影响Rust单元（新2＋旧表格页4＋旧代码2）及1个75页集成通过，含共享10M Work/恢复；旧接口限额范围保持。审查117模块、runner135套件登记，未全跑/CUDA。

阶段末1次release wheel、1次隔离安装及5组套件已通过：native_page_containers、native_code_containers、native_table_page_commit、compatibility、smoke。安装日志build/native-page-containers-*-install-test.txt。Rust日志build/native-page-containers-final-rust.txt、native-page-containers-final-pdf-rust.txt、native-page-containers-related-rust.txt。剩余Form完整结构采集/页面角色归一化、Form/图像图形物化及后续几何公式文本分类接线、默认PDF和全文差分。
