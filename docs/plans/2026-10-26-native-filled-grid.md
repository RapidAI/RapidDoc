# 原生外框填充行带表格候选

原生diagram组件依赖两套填充表格排除；相接单元格阶段已有，本轮补 `_detect_filled_grid_table_candidates` 完整阶段。保留外框页面比例、最大单元格/半行底纹排除、行带数量/覆盖/间隙与端点、垂直覆盖/空带及按行带/覆盖/面积稳定选优和重叠去重。不同于相接填充检测，本函数不要求rectangle_bboxes长度为1，也不依赖CJK尺度。

冻结完整旧函数及关键单元格/行带参考，覆盖整数/混合float排序及临界比例；实际PDF读取路径后检测，无候选/行带结论注入。开发只跑受影响测试，阶段末一次wheel及相关安装回归，无全量/CUDA。diagram还需要一般表格检测/闭合网格证明，当前不声称该产生器或默认PDF已完成。

## 完成及验收

- 新增 `detect_native_filled_grid_tables(path_infos, *, page_size, excluded_bboxes=())`，返回完整独立candidates：bbox/local_bbox/core_bbox、angle、float score、空来源/注释/推断线及默认标志。源路径不修改，框保留原数值类型和坐标；None排除框与空排除框一致。
- 完整迁移原 `_detect_filled_grid_table_candidates` 及其最大单元格、行带和覆盖 helper，不依赖CJK尺度、不额外要求rectangle_bboxes数量。按旧根层fill/segments5和正面积过滤，外框页面面积0.005..0.25、宽0.12/高0.03；半行底纹1.08、95%格去重、2%行带容差、4带/2格/90%宽覆盖、1%横缝/2%端点、75%垂直覆盖/5%空带，以及行带数/覆盖/面积稳定选优和90%候选去重。
- 冻结165旧完整函数参考，101候选正例；6独立旧大整数/混合float面积比较、除法和较小框覆盖事实。原参考未覆写，整数/float类型差异用局部有界N适配，不先把大整数面积舍入为double；复用既有BigRational与CPython补偿sum，横向覆盖仍按旧逐项+=。
- 28页实际PDF实时路径→新检测精确对照，4正例。覆盖四方向、外框缺失/三行/重复半行底纹和既有相接填充表格负例；90/270后只有两条水平带的样本沿旧拒绝，没有旋转归一化扩展策略，无旧候选/行带结果注入。
- 两个定向Rust单元测试、一个28页PDF集成测试通过；阶段末仅一次release wheel与一次隔离安装，native_filled_grid、native_tiled_tables、compatibility、smoke四组安装套件通过。验证完整字段/类型/顺序、输入所有权、并发、预算/资源错误与恢复，运行时未导入旧Python解析器/PDFium。
- 各5000路径/排除框、finite abs<=1e9、既有JSON/10M累计Work/8MB输出，外框/嵌套格/排序/成带/覆盖和去重计费并接受已有Work；无跨调用缓存。fmt、99模块审查、固定36＋3＋1文件哈希通过，已核对wheel打包99模块审查；全量runner登记117组，本轮未全跑，无CUDA。
- 日志build/native-filled-grid-rust.txt、build/pdf-filled-grid-rust.txt、build/native-filled-grid-wheel.txt、build/native-filled-grid-install-test.txt、build/native-filled-grid-regression.txt。

## 后续

diagram还依赖一般表格候选与 `has_native_closed_grid` 证明；继续补齐后迁移路径/栅格组件、正文屏障及时间轴扩框，再接强图形统一入口、跨页准备及默认PDF编排。本轮只完成该前置函数，不声称完整diagram或整文档质量/性能已完成。
