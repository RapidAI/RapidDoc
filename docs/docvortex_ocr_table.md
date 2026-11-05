# 可选扫描表格规则：docvortex_ocr

适合先尝试恢复框线明确的扫描表格，复用已有 OCR 和原生表格恢复核心。
此项不会自动开启，不替换当前默认 `docvortex`，也没有修改 Rust 接口。

## 使用

```python
from rapid_doc import RapidDoc

engine = RapidDoc(output_dir="output/ocr-rules", table_config={
    "rule_table_engine": "docvortex_ocr",
    "use_rule_table": True,
    # 默认仍以 UNET_SLANET_PLUS 为模型回退；也可选择其他已有模型：
    # "model_type": "table_transformer",
    # "rule_table_debug_dir": "output/ocr-rule-debug",
})
result = engine("demo/pdfs/49－长表格-密集型.pdf", end_page_id=0)
```

如果从 `demo/table_transformer_demo.py` 修改，请把其中的 `use_rule_table=False`
改为 True；仅增加 `rule_table_engine` 不会覆盖明确关闭规则的选项。

路由顺序：**原生 PDF 规则 → OCR 字词框与图像框线规则 → 所选表格模型**。
`force_ocr=True` 跳过原生文字规则，但仍允许 OCR 规则。
`use_word_box` 在所有模式下均沿用原默认值 False，不因启用此引擎或 TATR 而变化。
只有显式设置 `use_word_box=True` 才启用字词框。
OCR 和方向模型照常运行；规则成功才免去表格结构模型初始化/推理。
已获得的 OCR 结果继续传给回退模型，不重新构建另一套识别链路。

## 本次采用的保守边界

- 复用参考项目的光栅线提取、小角度纠偏及现有 DocVortex 表格核心。
- 与参考实现不同：只接受 `vector_grid` 物理网格候选，并检查四条外边都有实测框线覆盖。
  线段容忍度随笔画宽度变化，只允许小缺口；不靠纯文字对齐补造外边框。
- 每个输入 OCR 字词保持完整，不按字数伪造字符框；转换过程中不能遗漏字词。
- 每个字词至少 80% 面积落入唯一单元格，低置信度、跨格、结构歧义或异常均回退。
- 已知公式、勾选框、表内图片交给现有模型流程，即使相关输出开关关闭也不强行接管。
- 原生规则、既有文字质量和嵌套单元格保护保留，不通过降低阈值增加命中率。

因此，有明显框线不代表一定接管。无线、缺边、模糊、扭曲或复杂合并的表格，
当前可能拒绝规则候选，这比贸然替换模型结果更符合本次保持精度的目标。
几何置信分不是 OCR 正确率；OCR 漏检或错字仍可能传入结果。

## 诊断与独立测试

设置 `rule_table_debug_dir` 后，每次尝试会保存唯一目录下的 JSON 和框线/文字证据图。
它记录采用/拒绝原因、纠偏角度、框线、字词和候选网格。坐标属于纠偏后的裁图，
不会写成原始 PDF 页面坐标。诊断写入失败只警告，不丢弃成功结果或阻断模型回退。

```powershell
.venv/Scripts/python.exe scripts/evaluate_docvortex_ocr.py tests/fixtures/table_transformer/dense.png --word-box --output-dir build/ocr-rule-check
```

脚本默认复用 resources 中现有 PP-OCRv6 文件；自定义模型目录也支持 det.onnx、rec.onnx、dict.txt。
可用 `--words-json` 重放一张图的固定 OCR。固定文字需未转义；没有 score 的样本默认 1.0，
不能据此宣称实际 OCR 置信度。独立脚本只诊断规则，不执行回退模型。

## 已验证的范围

- 171 项相关回归通过：规则、TATR、原生表格恢复、合并/空白格、OpenCV 4/5 返回格式、
  字词遗漏拒绝、缺外框拒绝、原生优先、懒加载、公式/图片排除和异常/低分回退。
- 5 张真实裁图重新执行 OCR：table_04 接受，恢复 32 行、5 列、144 个逻辑格；
  table_01、table_02、table_10、table_12 均拒绝，由生产链路回退模型。
  本次边界比参考项目更保守，不能沿用其其他样本的成功结论。
  上述裁图试验显式使用 `--word-box`；它不代表默认整行框模式的命中率。
- 公共 `RapidDoc` 调用实测：电子长表格第一页走原生规则（121 行含表头）；
  table_04 的无损扫描 PDF 走 OCR 规则；复杂图片回退 TATR。
  扫描规则命中记录是在字词框生效的条件下取得的；恢复原默认 False 后，
  如需复用该验证条件，应显式传入 `use_word_box=True`，默认模式的命中结果可能不同。

以上证明接入与所测路由可用，不是全量精度提升证明。没有修改模型权重、运行全语料或验证 CUDA。
输出与日志：`build/docvortex-ocr-validation/`、`build/docvortex-ocr-*.log`。

```powershell
.venv/Scripts/python.exe -m pytest tests/test_docvortex_ocr_table.py tests/test_table_transformer.py tests/test_rule_table_routing.py tests/test_rule_table.py tests/test_native_table_defaults_and_edges.py tests/native_pdf/test_native_pdf_table.py -q
```
