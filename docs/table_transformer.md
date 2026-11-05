# Python Table Transformer（可选模型）

已接入 Microsoft `table-transformer-structure-recognition-v1.1-all` 的 FP32 ONNX。
模型只处理版面流程裁出的单表结构，文字复用现有 OCR 或 PDF 文字，不另装 OCR。
现有默认模型、原生表格规则和 Rust 接口均未切换到 TATR。

## 直接测试

使用当前项目 `.venv`，修改示例中的文件和页码后运行：

```powershell
.venv/Scripts/python.exe demo/table_transformer_demo.py
```

已有 Python 调用只需传入：

```python
from rapid_doc import RapidDoc

engine = RapidDoc(output_dir="output/table-transformer", table_config={
    "model_type": "table_transformer",  # 也支持 ModelType.TABLE_TRANSFORMER
    "use_rule_table": False,
    # "model_dir_or_path": r"D:\models\table_transformer_v1_1_all.onnx",
    # "engine_cfg": {"intra_op_num_threads": 4},
})
result = engine("demo/pdfs/49－长表格-密集型.pdf", end_page_id=0)
```

`use_rule_table=False` 用于直接比较模型效果。不配置时，原有规则仍可能优先处理电子表格。
`use_word_box` 沿用原项目默认值 False，不根据模型或规则引擎自动改变。
只有显式设置 True 才使用字词框；更细的框并不一定更准，错误的合并格可能将完整句子拆进不同列。
方向矫正、公式与图片占位沿用原流程。

如果要先尝试原生/扫描框线规则、失败后再用 TATR，设置
`use_rule_table=True, rule_table_engine="docvortex_ocr"`，见 [OCR 规则说明](docvortex_ocr_table.md)。

单独测试表格裁图：

```powershell
.venv/Scripts/python.exe scripts/table_transformer_image_to_html.py tests/fixtures/table_transformer/basic.png --words-json tests/fixtures/table_transformer/basic.words.json
```

不传 `--words-json` 时复用本机 resources 中的 PP-OCRv6 检测/识别模型。
输出 HTML、单元格框图、文字及逻辑坐标；指定 `--output-dir` 可隔离输出。

## 模型与运行依赖

本机已复用并校验另一个项目导出的模型：115,828,480 字节，约 110.5 MiB。
ONNX SHA256：`2c361306169b92b8fdf1ae677b0bbcc5111503729b598c388e5ef20237dc9085`。

默认查找 `RAPID_MODELS_DIR/table_transformer_v1_1_all.onnx`；未设置该环境变量时，
使用 `rapid_doc/model/table/rapid_table_self/models/`。也可显式配置文件路径。
模型目录受 Git 忽略，权重不打入 Python wheel。本地文件存在且 SHA256 正确时直接复用；
缺少或校验失败时，复用原有下载器从下面的 ModelScope 地址下载并校验。
仍支持显式指定自定义 ONNX 路径。MIT 许可证和来源说明保留在适配器目录。

[ModelScope ONNX 下载地址](https://www.modelscope.cn/models/RapidAI/RapidDoc/resolve/master/table/table_transformer_v1_1_all.onnx)

上传后的远端文件已完整读取并核对：115,828,480 字节，SHA256 与上面已验证模型一致。
`.validation.json` 是本地导出验证记录，运行和自动下载均不需要它。

运行复用现有 ONNX Runtime、NumPy、Pillow，无新增 torch/transformers/PyMuPDF 运行依赖。
CPU 已验证；CUDA 会沿用现有 ORT 设备配置，本次未测试 CUDA。

## 与参考项目相比的修正

- OCR 批次不匹配时报错，避免静默丢掉整批文字。
- 高层文字只还原一次 HTML 转义，再由 XML 序列化，避免 `&amp;amp;`。
- 公式/图片占位回填不修改调用方传入的 OCR 数据。
- 保留被 rowspan 完全覆盖的空行，使 HTML 行号与逻辑坐标一致。
- 单表选择前应用已声明的 0.5 整表置信度阈值，排除低置信度表格假设。
- 无 OCR 的底层调用保留结构网格，空 OCR 与图像保持逐张对齐。

## 验证与限制

已检查权重 SHA256，并与原 PyTorch、独立 Microsoft 官方后处理对照三个不同裁图：
预处理逐元素相同，ONNX logits 最大绝对误差约 9.64e-5、框约 1.41e-5，所测单元格解码一致。
另通过长表格 PDF 第一页的公共 `RapidDoc` 调用，产出一张 HTML 表格。
这些是集成验证，不能证明 TATR 比 UNET/SLANet 更准确；复杂合并格、竖排、超密长表仍需实际评测。
本次 `49－长表格-密集型.pdf` 第一页已经观察到明显漏行、错列和文字串格，
不能把“公共调用成功”理解为该页解析效果合格，因此不建议直接替换默认模型。
该模型按官方方式将长边缩放到 1000，且行、列等结构共享 125 个预测 query；
超密长表需要另行研究通用分段等方案，本次不通过定点补丁或静默切换其他模型掩盖问题。

相关回归：

```powershell
.venv/Scripts/python.exe -m pytest tests/test_table_transformer.py tests/test_rule_table_routing.py tests/test_rule_table.py tests/test_native_table_defaults_and_edges.py -q
```

需要重新导出时，额外安装 torch、torchvision、transformers、safetensors、onnx、requests 和 PyMuPDF：

```powershell
.venv/Scripts/python.exe scripts/export_table_transformer.py
# 仅校验已有 ONNX，不重导出或覆盖权重：
.venv/Scripts/python.exe scripts/export_table_transformer.py --verify-only
```

导出器固定 HF revision `7587a7ef111d9dcbf8ac695f1376ab7014340a0c`，
官方后处理 revision `16d124f616109746b7785f03085100f1f6247575`；验证结果保存在模型旁的 JSON 中。
本次参考对照及 PDF 输出位于 `build/table-transformer-validation/` 和 `build/table-transformer-*.log`。

来源：[Microsoft 源码](https://github.com/microsoft/table-transformer)、
[模型说明](https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-all)。

TATR 漏失 OCR 文字时，可显式配置备用模型或文字回退，见 [表格回退](table-fallback.md)。该功能同时支持独立 Python 和 Rust，默认关闭。
