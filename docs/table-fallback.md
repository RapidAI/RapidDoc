# 表格模型与文字回退

独立 Python 包 `rapid_doc` 和 Rust 包 `rapid_doc_rust` 的 `RapidDoc(table_config=...)` 支持相同配置。两种回退均默认关闭，原有主模型 `unet_slanet_plus` 和规则优先顺序不变。

```python
table_config = {
    "model_type": "table_transformer",
    "rule_table_engine": "docvortex_ocr",
    "fallback_model_type": "unet_slanet_plus",  # None：关闭备用模型
    "fallback_text_loss_threshold": 0.2,
    "fallback_to_text": True,  # False：关闭文字回退，可独立开关
}
engine = RapidDoc(table_config=table_config)
```

保留原主模型时，可将 `model_type` 留空，显式设置 `fallback_model_type="table_transformer"`。未做新一轮完整评测前，不更改全局默认模型。

## 配置

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `fallback_model_type` | `None` | 备用模型：`table_transformer`、`unet_slanet_plus`、`unet`、`slanet_plus`；相同主备模型不会重复执行 |
| `fallback_to_text` | `False` | 候选仍不可靠时，把当前表格区域已有 OCR 输出为普通文字 |
| `fallback_text_loss_threshold` | `0.2` | 可靠 OCR 文字缺失比例严格超过此值时触发；范围 `[0, 1]`；空表始终触发 |
| `fallback_model_dir_or_path` | `None` | 单一备用模型的自定义路径；组合模型沿用 `unet.model_dir_or_path` 和 `slanet_plus.model_dir_or_path` |

只配置 `fallback_to_text=True` 时，不加载备用模型，直接对低保留率的主模型结果进行文字回退。只配置备用模型时，失败则保留主模型结果。

## 判定规则

1. 已接受的原生规则、OCR 规则表格保持原有流程；本功能检查模型结果。
2. 复用同一裁图、同一 OCR，以及公式和图片提示，不重复检测或识别文字。
3. OCR 分数不低于 `0.5` 的普通文字作为依据。HTML 实体解码一次，忽略空白，按字符及出现次数计算保留率；公式、复选框和图片占位符不计入普通文字。
4. 有可靠文字时，空结果、非闭合/不规则行列结构，或文字缺失超阈值才触发。
5. 备用结果须结构有效、文字保留数高于主模型，且缺失比例不超阈值，才会替换主结果。无法仅凭输出文字更多获选。
6. 文字回退保留已有 OCR 和公式、图片占位符。先根据四角坐标的垂直重叠和中心距离归并同一行，再按横坐标从左到右输出；框间使用两个普通空格。纯文字在 Markdown 和 NLP Markdown 中使用 `text` 代码块，保留空格和换行；含公式分隔符或图片/富文本标签的区域沿用原渲染。OCR 文字自带反引号时自动延长围栏，避免内容截断。content list 中仍为无围栏的 `text`，历史中间结果里的 U+2003 宽空格也会转换成普通空格。行归并使用固定锚点，避免连续重叠把相邻两行串起来。不伪造单元格，也不声称恢复了复杂跨行表头的网格关系。中间 JSON 仍保留区域和 `text_fallback` 标记，便于追查。

备用模型在首次实际触发时才加载，之后复用。Rust 的判断、调度和选择均在 Rust 内完成；Python 绑定仅传配置和按需提供模型。独立 Python 实现不依赖 Rust 包。

## 验证与边界

覆盖空表、缺失重复字符、HTML 实体、低置信文字、合并单元格、非法候选、独立开关、输入不变性、OCR 不重复执行，以及 Markdown/content list 输出。

真实样例 `jiaocaineedrop_jiaocai_needrop_en_3146.pdf` 的两张评分表用于验证 TATR → UNET+SLANet 回退。此功能避免明显文字丢失，不能证明所有单元格的对应关系正确，也不能纠正 OCR 原本识别错的文字。启用备用模型会增加触发页面的推理和模型内存开销；没有据此宣称整体评测分数提升。

2026-11-03 本机验证：

- 两张真实裁图均恢复两行两列结构及全部五个表头字符。
- 完整 PDF 的独立 Python / Rust 备用模型结果 Markdown 完全一致，均为 690 字符、两张表。
- Rust 只开文字回退时，两组表头均保留，不输出伪表格。
- 两种回退关闭时，Rust Markdown 与改动前保存的输出逐字一致（572 字符）。
- 67 项表格、绑定配置及安装后输出测试通过；另有 7 项运行时配置测试和 2 项 Rust 保留率/结构检查通过。未跑全量评测或 CUDA。
- 最新 release wheel 已安装到项目 `.venv`。验证日志和输出位于 `build/table-fallback-*`；这些计时包含不同程度的首次模型加载，不能作为速度对比。

后续同日修正文字回退顺序：`page-14a6b411-9097-4eec-86da-92075868d243.pdf` 的 463 个文字框归并为 41 行，Freecell、GED、Grid、Hiking、Logistics 等行的行名和数字均按原始左右次序输出。只调整文字回退排布，不修改模型和表格结果仲裁。

顺序修正验证：14 项相关 Python 测试通过；安装后 Rust 完整解析该 PDF，上述五行均与独立 Python 坐标归并结果完全一致，每行保留 11 个数字。最新 wheel 已重新安装到项目 `.venv`；原 `output7` 中的历史输出需重新解析后更新。

代码块输出修正验证：21 项独立 Python 测试、3 项安装后 Rust 输出测试通过。使用该真实 PDF 的中间结果复渲染，Python / Rust Markdown 完全一致；已确认代码块、普通空格、反引号保护及公式/图片保留行为。预览为 `build/table-fallback-fenced-preview.md`；新 wheel 已安装到 `.venv`，未重复模型推理或全量评测。
