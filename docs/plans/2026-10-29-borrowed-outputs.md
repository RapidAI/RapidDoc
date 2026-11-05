# 表格无变换 PNG 与 OCR 输出借用

用户要求继续无损提速。只消除确定的图片编码、概率张量复制及重复扫描；不改模型、线程、批量、检测阈值、图像插值、置信度运算、表格判定或文档专用分支。

- 原生表格尺寸查询执行原 RGB 完整解码及像素预算校验，但不再逐像素复制和编码 PNG。
- 无正交旋转的内部图像用 Cow 借用原 PNG；90/270 度继续原旋转入口。旧 180 度身份变换行为保留，公共 image() 输出接口不变。内部后续仅使用解码像素，不依赖 PNG 压缩字节。返回有所有权的图像时仍需要一次压缩字节复制，不能称全部表格图像零复制。
- 公共 FP32 run() 仍返回复制且独立拥有的结果；CTC 使用相同输入/输出数量、预算、FP32 类型和有限值检查后，在 ORT 输出生存期内借用概率切片。仅返回拥有所有权的文字、token、时间步和置信度，不允许输出引用越过下一次运行。
- CTC 的内部有限值已验证入口只供模型运行器使用，仍检查形状和字典；公共解码入口保留 NaN/Infinity 检查。消除原 runner → Vec → decoder 中的概率复制和第二次有限值遍历，argmax 的严格大于、首项同分规则、blank/repeat、五位小数舍入及累加顺序不变。

整页 PNG、表格遮罩/旋转模型通路及原生几何 JSON 复制仍存在，本次不宣称已经全部消除。

## 验证

- Rust：原 CTC blank/repeat/同分规则和非法字典通过；16 正文/4 表格裁图像素 SHA 保留；12 原批识别张量位模式保留，每个实际批次额外将公共复制输出解码与新借用输出解码比较，文字/token/时间步/置信度完全一致。
- 表格原角度与门控、原旋转/BGR 像素、借用指针、原/新逐像素等价、截断及损坏 PNG 拒绝通过。尺寸入口继续完整解码，不只信任 PNG 头。
- 同 7,864,320 个 FP32 概率、六轮交替的组件测试：旧复制及重复扫描后解码中位数 0.018449 秒，借用及一次有限值扫描后解码 0.009138 秒。输出完全一致；不包含 ORT 推理，不能当成整篇翻倍。记录 `build/borrowed-output-ctc-component-test.txt`。
- 六组安装测试：5 CTC 原参考及并发；15 版面规则/3 实际布局；131 OCR 框/32 计划/16 裁图/112 策略/5 实际 det1-rec6 链；131 表格框/75 重试/209 策略/64 HTML/8 实际 CPU 链；320 表格方向门控/18 像素/76 工作流/28 实际 CPU 检测与方向链；73 后置 OCR/12 批识别张量/11 CPU 后处理链及 33 输出。来源审计 119 模块通过。
- 三份文档共 14 页、17 图片，Markdown/content list/图片尺寸模式像素与冻结原 Python 一致，完整 middle 与修改前 Rust 一致。small_ocr 两处原有置信度差异继续显式保留；另两份 middle 与原 Python 一致。所有原参考保持不变。没有全量、七篇整套或 CUDA 回归。

记录：`build/borrowed-output-inference-test.txt`、`borrowed-output-table-direction-test.txt`、`borrowed-output-{ctc,layout,model_ocr_execution,table_text,table_direction,legacy_post_ocr}-test.txt`。

## 相邻旧新文档对照

独立进程、相同模型、默认线程、文档顺序和首次模型加载范围；编译结束后执行，未与 Cargo 并行。脚本 `scripts/migration/compare_borrowed_outputs.py`，原始完整记录 `build/borrowed-outputs-control-{old,new}.json` 和 `build/borrowed-output-{old,new}-documents-test.txt`。

| 文档 | 修改前 | 修改后 |
| --- | ---: | ---: |
| small_ocr，8 页 | 18.398 秒 | 17.203 秒 |
| 密集长表格，1 页 | 3.693 秒 | 3.625 秒 |
| 研究报告，5 页 | 14.312 秒 | 10.261 秒 |
| 本次三份合计 | 36.403 秒 | 31.089 秒 |

small_ocr 识别阶段 5.803→4.513 秒；研究报告表格文字阶段 5.560→2.366 秒。但未修改的密集表格恢复也从 1.496 波动到 1.666 秒，研究报告原生恢复从 1.436 到 0.897 秒，说明单次整篇对照有运行波动，不能把全部差值归因于本次代码，更不能承诺固定比例或整体比 Python 快很多。此次三份子集也不能替换此前七份总耗时。

## 构建与交付

只用 release/offline/-j1。表格 Rust 测试编译 5 分 28 秒，wheel 因不同特性组合再次编译 5 分 50 秒；后续同类包装优化优先用一次 wheel 构建直接覆盖实际安装链路，避免无必要的重复核心编译。

新 wheel 已安装至现有 `.venv`，旧包保存在 `build/pre-borrowed-output-wheels-20261029/`。正式 wheel 与测试 wheel 一致；全部 wheel RECORD 哈希/长度验证通过，144 个安装原生/源码/数据文件与测试 wheel 逐字节一致（排除 pip/import 按安装位置重新生成的 pyc）。`pip check`、119 模块审计通过；不对同一二进制再跑一遍相同文档。日志 `build/borrowed-output-{build-20261029,install,pip-check}.txt`。
