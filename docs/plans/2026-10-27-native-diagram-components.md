# 原生 diagram 组件

迁移旧 `_detect_native_diagram_components` 完整链：异常 CJK 字框门控，铺片/填充及完整表格检测与闭合占格证明，路径筛选与增量连接，正文屏障、两轮图片/路径扩框、重叠合并，最后时间轴图标/彩色横轨/窄列说明认领。全部步骤共享工作量预算，不要求调用方注入表格判定。

保留旧路径/图体顺序、整数浮点坐标赢家及异常行为。冻结旧完整函数参考和真实 PDF 实时输入；阶段末一次 wheel 及受影响安装回归。默认整页接线不在单组件完成时提升完成标志，enhanced 排序/CUDA继续后置。

## 实现与证据

- 完整表格检测经 `detect_complete_with_work` 复用，闭合占格仍用旧 `closed_grid::inspect_with_work`，allow_two_by_two=True，流程不接受注入表格判定。
- 路径连接保持新成员在前、命中组依次追加、未命中组原序保留；两轮扩框和最终时间轴第一次命中即停止均沿旧顺序。
- 原始坐标Value保留union首项数值赢家；距离复用已验证CPython hypot。
- 226旧完整函数结果＋11旧异常；缺tight旧异常保留，单独补26组字框支持的时间轴、正文、标签长度/深度/横轨颜色分支。
- 2受影响单元测试及55实际PDF源集成通过。实际55页全为inflated门控负例，无实际diagram正例声明；图体正例与表格屏障证明为源级合成。
- SDK新增独立入口/capability；默认完成标志False。字体及完整表格候选缺口此前已消除，当前下一项是正式整页准备和图形总编排接入。

阶段末一次release wheel及6组隔离安装套件已通过。日志 `build/native-diagram-components-rust.txt`、`build/native-diagram-components-pdf-rust.txt`；兼容性清单113模块。无全量/CUDA。

安装包：新组件226结果/11异常、55实际PDF负例以及旧整页候选、闭合占格、固定字体、compatibility、smoke均已通过；构建及安装日志 `build/native-diagram-components-wheel.txt`、`build/native-diagram-components-install-test.txt`、`build/native-diagram-components-regression.txt`。
