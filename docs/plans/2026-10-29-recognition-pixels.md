# 识别裁图内存传递与公共坐标优化

依据用户最新计时，small_ocr 的识别裁图准备占4.434秒，检测模型执行占2.156秒。这里是已经观察到的热点，不根据文件名或内容添加生产分支。

## 实现

- 正文透视裁图直接保留BGR像素，跨页批识别按原顺序接收拥有像素的图像，不再对内部识别裁图编码PNG、复制PNG、解码PNG或进行两次RGB/BGR交换。
- 表格文字行裁图接入同一内存识别入口。来源仍是原始未遮罩表图，透视/高宽比2旋转、原NumPy排序、batch 6及共同padding不变。
- 公共识别PNG接口保留原输入校验，解码后委托同一排序/预处理/CTC内核。后置补识别仍兼容PNG资产入口，未宣称全部图片通路已经消除PNG。
- 透视裁图将相同采样点的行/列边界钳位和整数地址计算移出颜色通道循环。每通道的四行/四列浮点tap、首项乘法、后续FMA、ties-even舍入、边界复制及旋转均不变，没有改成近似插值或矩形裁图。
- 默认可移植编译目标的 `f32::mul_add` 生成 `fmaf` 函数调用，而支持FMA的函数生成硬件 `vfmadd`，本机生成代码记录在 `build/fma-codegen-probe.asm`。裁图及版面浮点缩放增加CPU/OS运行时FMA检测，支持时进入只含同一算术实现的硬件版本，不支持时保留通用版本。不设置全局 `target-cpu=native`、fast-math或降低精度，wheel仍包含回退代码。
- 当前窗口累计识别图像最多1000张、32M像素（BGR约96MB），准备阶段即限制累计缓存，不随16页窗口无限累积原始图。已验证的输入按原计划完整处理，超预算返回错误，不截断文字或图像；当前裁图和输入来源的原像素预算保留。识别完成后释放像素。

## 验证

16个原正文透视裁图、4个原表格裁图的像素SHA和新旧通路逐字节一致。12组原CPU批识别（含等宽高比跨批排序）张量位模式与原参考SHA一致；原PNG和新BGR入口的文字、置信度完全相同；原文字/分数参考、空批、无效BGR、batch范围及取消回调通过。裁图退化/自交/越界预算/旋转坐标测试通过。

专项代码 `rust/rapiddoc-inference/tests/raw_recognition.rs`，初步证据 `build/recognition-pixels-{component,kernel}-test.txt`。

硬件与通用路径另有64组裁图、50组版面边界/缩放像素严格相等；本机硬件分支实际执行。32,768像素的同内核裁图六轮交替中位数为通用0.0024809秒、硬件0.00115165秒；硬件耗时约46.4%。另在同一个当前内核、同24个合成裁图下比较PNG往返和内存传递，六轮交替中位数0.03332095秒和0.02531455秒，耗时下降约24.0%，输出摘要相同。这两个组件数字不能相加或直接当作整篇收益，PNG对照也不包含模型推理；校验摘要开销在两边均存在。

证据：`build/recognition-hardware-fma-test.txt`、`build/fma-crop-kernel-timing.json`、`build/recognition-pixels-final-component-test.txt`、`build/recognition-crop-transfer-timing.json`。非本机CPU、任意浮点输入和其他平台未作普遍等价/性能声明；只对已经验证的模型及图像合同保持原算术。

## 最终验证及实际耗时

- 15组版面预处理FP32张量逐位一致，15组布局规则和3张实际模型布局通过。
- 131框规则、32识别计划/16精确裁图、112识别策略及5条实际CPU det1→rec6→模型链通过，包括1280/2048输入。131表格框规则、75重试、209 OCR策略、64 HTML回填及8条实际表格CPU链通过，分数最大误差0。
- 73后OCR合同、12原CPU批识别的共享padding张量位模式、11条实际后处理链/33输出通过。119模块来源审计通过。
- 七份60页完整输出与修改前用户Rust完全一致；Markdown、content list、92图片模式/尺寸/像素与原Python一致。6份middle JSON与原Python完全一致；small_ocr两处0.001置信度既有差异继续显式保留。没有改写旧参考。只跑受影响测试和阶段回归，没有全量或CUDA。

第一轮最终七篇计时为论文18.823秒、demo1 34.226秒、密集表格3.432秒、03 78.252秒、small_ocr 24.056秒、财报10.320秒、研究报告10.598秒，合计179.706秒。这轮03公式推理50.544秒，明显高于此前记录，不能因裁图变快就宣称整篇全部加速；完整原始记录保留于 `build/recognition-fma-seven-{test.txt,summary.json}`，没有用后续较快单篇结果替换本轮合计。

为调查该异常，只重复两个相关文档的独立进程相邻旧/新对照，均从首次模型加载开始计时，使用相同模型、默认线程和输入；完整输出精确一致：

| 相邻对照 | 旧包 | 新包 | 观察 |
| --- | ---: | ---: | --- |
| 03整篇 | 55.642秒 | 53.704秒 | 本次耗时下降约3.5% |
| 03公式推理 | 30.339秒 | 30.183秒 | 此次未复现50秒回退 |
| 03识别裁图 | 0.693秒 | 0.335秒 | 本次约减半 |
| small_ocr整篇 | 19.965秒 | 18.487秒 | 本次耗时下降约7.4% |
| small_ocr识别裁图 | 2.915秒 | 1.460秒 | 本次约减半 |

相邻对照也不是多轮受控正式基准，不能承诺固定百分比，更不能声称整体比Python快很多。它说明本次明确的裁图收益与公式阶段的较大运行波动应分别评估。脚本 `scripts/migration/compare_recognition_execution.py`，数据 `build/recognition-execution-control-{old,new}.json`、`build/recognition-execution-control-{old,new}-small_ocr.json`。首轮small_ocr裁图1.431秒，用户此前日志4.434秒；该跨时刻差值不当作严格收益比例。

先完成内存版构建，期间进一步定位FMA生成代码后又完成最终构建；中间wheel没有执行七篇完整回归。最终构建 `build/recognition-fma-build-20261029.txt` 耗时7分46秒；其余构建/专项编译完成后才做整篇计时。安装验证日志 `build/recognition-fma-{layout-preprocess,layout,model-ocr,table-text,post-ocr,audit}-test.txt`，实际输出 `build/recognition-fma-seven/`。

最终wheel已安装到现有 `.venv`。安装后的原生二进制、适配器和审计与测试wheel逐字节相同，全部RECORD哈希/长度校验及 `pip check` 正常；不再对相同二进制重复完整回归。旧wheel保留在 `build/pre-recognition-pixels-wheels-20261029/`，安装日志 `build/recognition-fma-install.txt`、`build/recognition-fma-pip-check.txt`。用户原脚本无需改参数，重新启动解析进程即可使用更新。
