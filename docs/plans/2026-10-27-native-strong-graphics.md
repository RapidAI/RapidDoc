# 强图形总编排

按旧 `_detect_strong_graphic_bboxes` 顺序调用十类已迁移图体产生器，再统一执行有字多列网格排除、面积稳定排序、包含拒绝、重叠合并及最终位置排序。字号来自旧行字体尺度中位数和 CJK 字宽校准，调用方不传入预选候选。

坐标图产生器仅检测图框，不执行另一个描述分组函数；柱图产生器的行语义/标题副作用须继续传给后续检测和最终源投影。复用函数共享Work，不改变旧独立入口输出。完整图形总编排差分通过后再推进正式整页准备；不提升默认PDF完成标志。

## 实现与证据

- 新strong_graphics模块复用十类已验证产生器；所有实际表格网格与包含/重叠裁定都由源计算，不接受预选候选结论。
- stacked_graphics的尺度读取提为共享measure helper，独立入口算术不变；raster_charts新core-only路径在描述分组之前返回，原分组入口保持原结果。
- 柱图source语义副作用延续到后续同心轮廓等分支，最终只把真实变化投影回原source字段；不为缺省输入补出额外公开字段。
- 493完整结果（335正例、1标题副作用）＋37真实旧异常。首次投影错误的冻结文件保留，但其122个constructor错误不算旧函数证据，canonical语料正确重建路径默认值。
- 8个受影响Rust单元测试：strong/stacked/raster-charts/graphic-arbitration各2；59页实时PDF集成通过。55既有页无图体，新增4方向复杂轮廓页2接受/2拒绝。
- 114模块清单与固定36＋3＋1上游核哈希保持；只跑受影响验证，无全量/CUDA。

阶段末一次release wheel及7组隔离安装套件已通过。日志 `build/native-strong-graphics-related-rust.txt`、`build/native-strong-graphics-pdf-rust.txt`。下一步正式整页准备/图表提交，随后默认PDF及全文差分，原完成标志False。

安装结果：新强图形入口、堆叠柱、栅格坐标图、图形仲裁、diagram、compatibility、smoke共7组通过。构建/安装日志 `build/native-strong-graphics-wheel.txt`、`build/native-strong-graphics-install-test.txt`、`build/native-strong-graphics-regression.txt`。
