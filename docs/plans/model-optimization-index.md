# 模型优化归档索引

**当前加载方式：** [Rust 初始化时内存优化](2026-11-02-native-memory-models.md)。已移除磁盘优化图缓存、清单安装及旧缓存兼容。原模型直接复用，算子库随运行库分发；下文旧阶段的离线准备流程仅作历史记录。

这些文档、Python 开发脚本和图变换测试均已保存到本地项目。当前 Git 状态下，多数新增文件仍为 untracked，**不代表已经提交 Git 或上传远程备份**。本索引不执行提交。

模型、冻结输入、算子跟踪与性能报告主要在 `build/` 中，原始模型主要在项目模型目录及 `D:/CodeProjects/doc/RapidAI/models/`。清理整个 `build/` 会丢失这些实验资产；如需长期保存，应一并保留对应候选、清单、输入和报告，或通过脚本重新生成。脚本不会覆盖原始模型。

## 公式 M：最终保留方案

- [最初原模型与保留候选的直接对照](2026-10-29-formula-original-vs-final.md)：大小、同条件速度、内存与实际 PDF 的最终汇总。
- [注意力融合](2026-10-29-formula-attention-fusion.md)：当前保留候选路径、试用配置、语义与数值边界。
- [FP16 存储及量化对照](2026-10-29-formula-compression-candidates.md)：压缩方法，以及未采用的 INT8 和混合候选。
- [原始计算图分析](2026-10-29-formula-model-analysis.md)、[投影外提](2026-10-29-formula-projection-candidate.md)：重复计算证据与第一次改图。

当前保留模型：`build/formula-mha-half-20261029/formula-attention.onnx`。

SHA256：`ee215a4db50b85bbfa3478ab81120bdb436beb3e86fd6f721b77effce53e871a`。

原始模型 SHA256：`52afa60ff51ad66cd7840a74188a058fe99081e617d3bd780117a2d67d94f50d`。

默认路径和安装包尚未切换；显式传入模型路径才使用保留候选。

## 开发脚本

| 脚本 | 用途 |
| --- | --- |
| `scripts/migration/optimize_formula_projection.py` | 校验 M 原模型哈希，外提重复的图片特征投影，另存候选 |
| `scripts/migration/build_formula_compression.py` | FP16 存储、INT8、混合精度实验；没有将失败量化方案作为默认 |
| `scripts/migration/fuse_formula_attention.py` | 当前保留的 12 组注意力融合，可选缓存、投影拼接实验 |
| `scripts/migration/profile_formula_model.py` | 显式指定模型、哈希和独立输出目录，对代表输入做算子分析 |
| `scripts/migration/validate_formula_projection.py` | 冻结输入原始 token 校验、串行 ABBA、内存、动态 batch 测试 |
| `scripts/migration/compare_runtime_fair.py` | 实际 Python/Rust PDF 对照，可用 `RAPIDDOC_BENCH_FORMULA_MODEL` 指定候选 |
| `scripts/migration/capture_formula_tensors.py`、`replay_formula_tokens.py` | 捕获和回放真实公式输入，避免以随机输入判断识别结果 |
| `scripts/migration/inspect_layout_models.py` | 新增：只读检查版面模型结构和体积，支持真实页面算子跟踪 |

`tests/migration/test_formula_graph_candidates.py` 保存 8 项图变换语义测试。实际模型校验及完整 PDF 证据仍应一起保留，不能只跑小型语义测试就宣称精度一致。

## 未采用的后续方案

- [缓存更新、形状简化和线程实验](2026-10-29-formula-cache-and-shape-experiments.md)。
- [融合后重新分析及交叉缓存外提](2026-10-29-fused-formula-profile-and-cross-cache.md)。

这些实验未测出可采用的新增收益。保留记录是为了避免后续重复投入，不表示它们已进入最终模型。

## 重建当前 M 候选

在源模型存在、开发环境已安装 onnx/numpy 的新工作副本中：

```text
.venv/Scripts/python.exe scripts/migration/optimize_formula_projection.py --output build/formula-projection-candidate-20261029
.venv/Scripts/python.exe scripts/migration/fuse_formula_attention.py --output build/formula-mha-half-20261029 --half-storage
```

已存在目录时工具会拒绝覆盖。上述第二步目前读取固定的投影外提路径；若另用目录，应先明确修改输入绑定与清单，不能混用旧文件。验证还依赖此前冻结的 74 个真实输入及原始 token，位于 `build/formula-tensor-audit-v3-20261029/native-model/`；这部分不能由模型文件自行重建。

## 后续适配公式 S、L

现有图改写匹配和源哈希保护针对 M。不能仅更改模型文件名或绕过哈希检查后直接运行。

1. 分别记录 S/L 原始模型哈希、输入输出、注意力维度、缓存结构和生成规则。
2. 按实际节点模式适配投影外提与注意力融合；不能假定都有相同重复投影或 12 组注意力。
3. 单独生成 FP16 存储候选。融合与 FP16 都需要各自数值验证。
4. 候选 S 与原始 S 比，候选 L 与原始 L 比。不能要求 S 的输出与 M 参考相同，或把 M 的 74 个期望 token 当成 S/L 的参考。
5. 保留动态 batch、短长混合和实际 PDF 结果验证，再做关闭 profiling 的同条件速度比较。

核查时发现当前 Rust 兼容封装只接受 `pp_formulanet_plus_m`；版面只接受 `pp_doclayoutv3` 或规则模式。因此目前不能只修改 `model_type` 就使用 S/L 或版面 V2，仍需独立接入和验收。此前对模型切换的泛化说明不适用于尚未接入的这些型号。

## 版面 V2、V3

前期结构及代表页分析见 [版面 V2/V3 优化可行性检查](2026-10-29-layout-v2-v3-optimization-inspection.md)。

后续已完成 [V2/V3 压缩与速度实验](2026-10-29-layout-v2-v3-compression-experiments.md)：生成了 10 个独立候选并检查 13 个真实页面。没有切换默认模型；V3 全量 FP16 存储出现检测数量变化，不采用。新增脚本为 `experiment_layout_compression.py` 与 `fuse_layout_attention.py`。

## CUDA 表格执行与缩放缓存

[原生产 SLANet 执行计划与精确回归](2026-11-02-cuda-table-execution.md)：离线准备 CUDA encoder / CPU decoder，Rust 持有双 session；原权重不变。提供独立准备脚本及 `prepare_cuda_models.py --partition-tables`。UNET CachedConv 因掩码差异未采用。

## UNET CUDA 生产图对齐

[UNET 原生产执行图和 CPU 复用](2026-11-02-cuda-unet-production.md)：新增 `prepare_cuda_unet.py`，离线延伸已有模型清单；31 组真实输入与原 Python 缓存图严格一致。此结论纠正了早先只对比标准整图 CUDA 而拒绝 UNET 缓存的口径。CPU 分类行缓存和形态学优化也已通过原始张量及完整 PDF 回归，原始模型文件未修改。

## CUDA 公式生产图与 cubic SIMD

[公式与版面性能审计](2026-11-02-cuda-formula-vector-audit.md)：`prepare_cuda_formula.py` 复用原 Python MaxWithIndex 优化，延伸当前 UNET 模型清单；Rust cubic 原计算向量化，保留普通 CPU 路径。整篇输出和真实 UNET 张量已严格回归。需选用新的清单 `cuda-models-formula/manifest.json` 才能使用服务器公式图收益，旧清单不会因 wheel 重装自动改变。

## 一次准备与自动加载

[优化包自动发现、安装及真实 GPU 验收](2026-11-02-cuda-auto-bundles.md)：`prepare_cuda_bundle.py` 一次准备并安装全部已验收图；已有包通过 `python -m rapid_doc_rust.cuda install 清单` 导入。普通 `RapidDoc` 之后自动发现、校验并加载，无需每次设置清单变量。`cuda status` 可检查是否具备优化包。缺少兼容包时保持原模型，不在首次解析动态改图/编译。
