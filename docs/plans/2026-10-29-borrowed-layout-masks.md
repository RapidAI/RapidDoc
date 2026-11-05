# 版面输出借用与按需掩码转换

状态：上一阶段交付记录；当时已安装 .venv / build/wheels，现已由公式批量2包替代，见2026-10-29-formula-batching-table-borrowing.md。旧 09:31 包保留于 build/pre-layout-render-owned-wheels-20261029。CPU arena 关闭实验已撤回，恢复默认配置；不修改线程、模型、批量、图优化、算术精度及版面判定规则。

确认真实 PP-DocLayoutV3 输出 fetch_name_2 为 int32 [300,200,200]，每页 48,000,000 字节（约45.8 MiB）。原正式链先由 run_named 深拷贝所有整数掩码，再将全部转换为同长度 FP32 Vec，之后按实际 count 截取，最后才筛选框/裁取区域。

改动：

- `named.rs` 增加内部同步消费的借用视图，仍逐个检查完整输出的类型、形状、总元素预算和全部 FP32 非有限值。公开 run_named 仍返回自有数组，保留整数位精度。
- 正式 RGB 与兼容 PNG 版面入口在 ORT 输出有效期内完成后处理，不复制大张量。
- 掩码后处理按原顺序筛框，在原循环的读取位置将 int32/int64 转为 FP32。保持原来 FP32 转换/求和/截断/取模、几何处理与错误规则。Float32 直接借用。移除每页两个约45.8 MiB 的临时数组，未跳过模型输出验证或改变模型输出。
- raw layout/run_named API 的输出结构和拥有数据的语义不变。

验证通过：24组原/非对称彩色PNG的rect/auto/poly/quad完整输出与修改前包完全一致；named安装测试覆盖FP32/int32/int64、超2^53整数位精度、完整输出总预算及错误恢复；15组版面规则及3张实际PNG参考通过。七篇60页的Markdown/content与原Python一致，middle与既有Rust参考一致（保留small_ocr两处已知旧分数差异），92图片像素与原Python一致。源文件119模块审计通过；已安装包144个非字节码缓存文件与已测wheel逐字节一致，pip check通过。未跑全量套件/CUDA。

同条件整篇对照：各后台独立串行进程，相同三篇顺序、导出与JSON回读，seal关闭，包含首次模型加载；Rust两轮，新旧顺序反转，Python一轮。

| 文档 | 旧Rust两轮平均秒 | 新Rust两轮平均秒 | Python本轮秒 |
| --- | ---: | ---: | ---: |
| 比亚迪财报 | 11.812 | 10.756 | 18.438 |
| small_ocr | 16.228 | 15.607 | 20.019 |
| 03 | 51.820 | 53.127 | 56.003 |
| 合计 | 79.861 | 79.490 | 94.460 |

新Rust相对本轮Python耗时少15.85%，但相对旧Rust整体只有0.46%，应视为基本持平，不能说本次改动让整条链路快了15.85%。03首轮新Rust54.422秒、旧51.882秒；换序复测新51.833秒、旧51.759秒。该文件两轮平均仍较旧慢，完整保留不利数据，不用较快一次替换前次结果。两轮版面阶段合计均改善（旧18.097/17.637秒，新15.593/16.103秒），但公式/OCR模型耗时与波动占主导。

七篇本轮一次总耗时138.171秒，只验证本轮绝对耗时与输出，不拿历史不同条件Python日志拼接比较。证据：build/runtime-fair-{borrow,installed}-v{3,4}-20261029/summary.json，以及build/runtime-fair-python-v3-20261029/summary.json和对应txt；每篇记录进程CPU时间和Windows内存。

内存限制：去掉临时数组不等于进程峰值同幅下降。三篇峰值工作集新旧Rust均约3089MiB，Python约2609MiB；ORT arena保留分配仍占较大比例，不宣称Rust整体内存更低。关闭arena实验已证实本轮会降低吞吐，未交付。只有两个Rust轮次与一个Python轮次，不构成统计意义的长期基准，也不能保证任意文件/机器快于Python。

当前wheel SHA256：15f16901fcfbeedae16908aeecb525dfc32b22fafedb7f60acfdc7e0e172ef17；安装和对照记录见build/borrowed-layout-release-state.json。
