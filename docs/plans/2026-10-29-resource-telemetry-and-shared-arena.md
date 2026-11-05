# 资源采样与多模型共享 CPU 内存池

## 本轮结论

已观察到系统资源状态明显变化，也已实现、验证共享 CPU arena。共享池稳定减少项目自己的内存保留，但本轮没有证明解析速度提高，因此默认保持原独立池、公式批量1。不能用降低内存的百分比代替速度提升。

## 只读采样

`scripts/migration/runtime_telemetry.py` 通过Windows PDH英文计数器每秒记录CPU占用、Processor Performance、Pages Input/sec、Page Reads/sec、提交内存百分比，以及GlobalMemoryStatusEx的可用物理内存。仅当比较工具设置 `RAPIDDOC_BENCH_TELEMETRY=1` 时启用，与正常解析无关。

同一进程连续解析财报三次，输出均与现存参考一致：

| 后端 | 首次秒 | 重复1秒 | 重复2秒 | 采样最少可用物理内存 | 最高系统提交比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原独立池 Rust | 22.567 | 19.158 | 21.423 | 1.31GiB | 97.1% |
| 原 Python | 26.931 | 13.822 | 12.793 | 3.27GiB | 96.2% |

Rust采样期间系统Pages Input/sec中位约2345、范围92–96674；Processor Performance约110–187。后者是相对标称性能的系统级指标，不能直接当作温度或降频证据。Pages Input也包含映射文件/模型读入，不能全部称为pagefile换页。首次加载之外仍存在波动；观察证实资源压力存在，不足以唯一解释此前所有慢例。

原始记录：build/monitored-byd-{rust,python}-auto-20261029/telemetry.jsonl 和 summary.json。

## 共享 arena 实现

沿用[ORT官方共享内存池机制](https://onnxruntime.ai/docs/performance/tune-performance/memory.html)：同一Rust环境中注册一个CPU arena，启用的模型复用临时内存，不共享权重或计算状态，不修改模型、FP32、图优化、线程数、裁图和解码。

- 原生Descriptor、缓存Key、Python裸会话和LazySession均加入可选 `cpu_shared_arena`；默认None，行为保持独立池。高层仅通过原有engine_cfg传递，不要求日常用户调参。
- CPU专用；CUDA请求不能携带CPU配置，原CUDA路径不改。
- 环境锁串行注册，原子租约计数保护存活模型；Session先析构，最后租约才注销arena。新的模型在注销前获取租约时不会误释放。构造失败也释放租约。注销失败时保留注册状态，不在析构中panic。
- 原预算/模型SHA/类型/非有限数验证保持；不同池策略不误复用同一缓存会话。

原生缓存测试通过。SDK验证不同权重模型共享池、独立池同时存活、多线程调用、释放一模型后其他模型继续执行、最后模型释放后重新注册，以及配置校验；公式排序/重复输入/跨页回填测试通过。

## 原生实测：共享→独立→独立→共享

同一候选包、每组独立串行进程，顺序财报→03→财报。含首次模型加载、相同导出和JSON回读；公式batch1。每次完整比较Markdown、content、原Rust middle和图片像素。

| 池策略 | 首次财报秒 | 03秒 | 重复财报秒 | 合计秒 | 峰值工作集 | 最后私有提交 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 共享A | 21.921 | 107.970 | 19.556 | 149.447 | 2.34GiB | 2.28GiB |
| 独立A | 22.939 | 97.958 | 18.352 | 139.249 | 3.06GiB | 3.48GiB |
| 独立B | 18.116 | 104.709 | 18.579 | 141.404 | 3.06GiB | 3.48GiB |
| 共享B | 20.553 | 96.685 | 21.392 | 138.630 | 2.35GiB | 2.29GiB |

共享池少保留约1.2GiB私有提交（约34%），峰值工作集少约0.71GiB（约23%）。但两组共享平均144.039秒，独立140.327秒，共享平均慢约2.6%。没有稳定速度收益，不默认启用。

原始证据在build/native-arena-{shared,separate}-{a,b}-20261029。共享/独立Python原型在相同三篇上峰值3.01/4.14GiB，速度125.10/117.31秒；仅为实施前探路，不拿来宣称Rust收益。

## 验证范围

基础原生缓存隔离、SDK并发与生命周期、公式合同已通过。四组12次/112页原生完整输出核对通过。七篇参考从现存用户Python output和17:41 Rust输出单独保存为build/shared-arena-reference-{python,rust}-20261029.json，未重写已删除参考的旧路径。两后端Markdown/content及92图片像素一致，middle保留small_ocr既有两处置信度差异。

共享池七篇/60页阶段回归通过，全部Markdown/content/既有Rust middle及92图片像素一致；总耗时299.225秒，峰值工作集2.593GiB。这是共享池实验运行，不能当作当前默认独立池耗时，也不能与不同资源状态的旧217秒直接作因果比较；本次未得到稳定速度收益。

新版已安装到项目.venv并同步build/wheels，144个非pyc文件与wheel一致、pip check及安装后共享池并发/生命周期测试与43组规则合同通过。共享池默认关闭、公式批量1，不要求用户修改demo或参数。wheel SHA256为2865f56ec85f359001cb16b0e7b51c80d9204b5916457e6998b88e67a04e57c3，状态见build/shared-arena-install-state-20261029.json，旧包保留在build/pre-shared-arena-install-*。

没有全量测试、CUDA测试或修改系统电源/进程设置。本轮速度目标尚未达成。
