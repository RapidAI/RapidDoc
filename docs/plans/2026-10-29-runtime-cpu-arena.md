# CPU 会话配置与当前环境的 Python 对照

本轮开始时稳定安装为 09:31 对应回退 wheel；arena 关闭实验仅在开发包验证并撤回。后续版面掩码借用包恢复默认 arena，已独立验证安装，见 2026-10-29-borrowed-layout-masks.md。

确认配置差异：原 Python layout/formula/table 的 engine_cfg、RapidOCR config.yaml 以及方向分类均设置 enable_cpu_mem_arena=False；Rust 没有显式 CPU EP 注册，使用 ORT C API 默认 arena。CPU arena 会跨动态形状复用/保留分配，本轮尝试显式关闭以对齐原实现。不变更模型、精度、图优化、线程数、批量或算法。CUDA 路径不动。

该差异在早期快版本中也存在，**不是已经确认的近期回退根因**。需要同时验证实际内存占用、原结果及整篇耗时，不能只根据配置推断提速。

对照脚本 `scripts/migration/compare_runtime_fair.py` 在独立、串行进程运行原 Python、已安装回退 Rust 和开发 Rust。相同文档顺序（财报、small_ocr、03），相同模型/默认 CPU 线程、seal 关闭，均导出 Markdown/middle/content/images 并回读 JSON；排除可视化叠加和原 PDF 额外备份。首次模型加载计入，记录进程峰值工作集和提交内存。原 Python PDFium 与 Rust PDFium 不混入同一进程；不在编译期间运行对照。

第一版诊断在财报输出检查时将 images 下文件按根路径查找，已更正；其失败输出保留，不能用作通过的基准。使用带 v2 的新目录重新采集，不覆盖历史数据与 `build/migration/*.json`。

本轮结果（同顺序前三篇：财报、small_ocr、03，含首次加载及相同导出）：

| 配置 | 财报秒 | small_ocr秒 | 03秒 | 合计秒 | 三篇进程峰值工作集 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原 Python | 36.722 | 37.302 | 81.747 | 155.770 | 2551 MiB |
| 已安装回退 Rust（arena 开） | 18.674 | 26.393 | 85.178 | 130.246 | 3087 MiB |
| 开发实验（arena 关） | 21.486 | 42.958 | 133.794 | 198.239 | 1589 MiB |

证据在 `build/runtime-fair-{python,installed,dev}-v2-20261029/summary.json` 及同名 txt。关闭 arena 虽然显著降低内存，但整篇变慢，不作为速度优化交付，源码恢复默认 arena。单轮串行对照仍受系统状态影响，不能推导固定比例或所有文件性能。

开发实验五组模型安装测试通过，七篇 60 页 Markdown/content/图片与原 Python 参考一致，middle 与既有 Rust 参考一致（保留 small_ocr 已知两处旧分数差异），92 图片像素一致；119 模块源文件审计通过。这些只证明本轮已测结果，不证明提速。arena 关闭实验未安装到 .venv；后续交付的借用优化使用默认 arena。

后续独立优化聚焦版面输出：在 ORT 输出仍存活时直接后处理，保留完整输出预算/类型/非有限值检查。公开 raw API 仍返回自有数组。只有同条件整篇对照支持时才考虑重新交付。
