# RapidDoc 与 Firecrawl 文档解析对比

本页比较已发布的 RapidDoc Rust SDK 0.1.1、AnyDoc 0.2.4 和 pdf-inspector 0.2.6。Office 和文本 PDF 分开统计。既有 OmniDocBench 结果见[项目 README](../../README.md)。

## Office：AnyDoc 公开夹具

样本来自 [AnyDoc 测试夹具](https://github.com/firecrawl/anydoc/tree/e7178e099b6bf9171e2c52b832cbdb4134b69112/tests/fixtures)。按 SHA-256 去重，六种共同支持的格式共 41 份，其中正常组 28 份，异常组 13 份。夹具用于回归和边界覆盖，样本数量少且各格式不均衡。本次结果与 [AnyDoc 网站使用的 100 份私有文档](https://github.com/firecrawl/anydoc/blob/main/bench/README.md)分别报告。

在 Windows x64、Intel Core Ultra 9 185H、Python 3.12 上通过各自的 Python SDK 调用 Rust 引擎。顺序处理，首份文档预热一次，正常组每份测量 5 次。计时范围从已读入的字节到 Markdown 返回，包含 SDK 自身的处理与输出构建。

| 格式 | 样本 | AnyDoc 非空输出 | RapidDoc 非空输出 | AnyDoc 耗时（ms） | RapidDoc 耗时（ms） |
| --- | ---: | ---: | ---: | ---: | ---: |
| DOC | 4 | 4 | 4 | 0.020 | 0.117 |
| DOCX | 11 | 11 | 11 | 0.087 | 0.300 |
| XLS | 1 | 1 | 1 | 0.100 | 0.272 |
| XLSX | 2 | 2 | 2 | 0.244 | 0.317 |
| PPT | 3 | 3 | 3 | 0.017 | 0.163 |
| PPTX | 7 | 7 | 7 | 0.093 | 0.192 |

每格耗时是该格式下逐文件中位耗时的中位数。小文件耗时处于亚毫秒量级，适合用来描述这组夹具上的表现。

对照原始文件结构核验的例子：

- **RapidDoc 保留 Excel 合并关系。** 在 `xlsx/handmade-merged.xlsx` 中，原工作表声明 `A1:B1` 和 `A2:A3` 两个合并范围。RapidDoc 导出的 HTML 表格保留 `colspan="2"` 和 `rowspan="2"`，AnyDoc 导出的 Markdown 网格保留文字但未表达合并关系。
- **AnyDoc 保留更多 Word 脚注与公式内容。** 在 `docx/handmade-altpath.docx` 中，AnyDoc 输出了重定位后的脚注，RapidDoc 输出了正文和标题。在 `docx/handmade-math.docx` 中，AnyDoc 保留了源 XML 中的求和公式，RapidDoc 输出遗漏该公式。
- **AnyDoc 保留部分演示文稿附加文本。** 在 `pptx/handmade-order.pptx` 中，两个工具都保留了标题与正文的相对顺序，AnyDoc 额外输出了源幻灯片中的 `Quarterly numbers`。

异常组单列。5 个 `abuse/` 输入中 AnyDoc 全部拒绝，RapidDoc 有 3 个返回非空 Markdown。8 个 `malformed/` 输入中 AnyDoc 有 5 个、RapidDoc 有 2 个返回非空 Markdown。这里记录输出和错误，异常文件返回内容本身并不能证明正确恢复。

逐文件记录见 [Office CSV](firecrawl-office.csv)。

### Office 源文件检查项得分

基于 28 份正常夹具的原文件结构及 [AnyDoc 夹具生成代码](https://github.com/firecrawl/anydoc/blob/e7178e099b6bf9171e2c52b832cbdb4134b69112/tests/gen_fixtures.py)，人工列出 178 项可复查的内容、结构和格式检查。两边使用同一清单，符合检查项得 1 分。语义相同的 HTML 和 Markdown 表示均可得分。先求每份文档的通过率，再在各格式内平均，最后六种格式等权平均。检查项在查看两边输出后编制，因此属于这组夹具的定向核验，和官网私有语料的模型评分口径不同。

| 格式 | 文档数 | RapidDoc | AnyDoc |
| --- | ---: | ---: | ---: |
| DOC | 4 | 87.06 | 95.59 |
| DOCX | 11 | 65.34 | 93.25 |
| XLS | 1 | 92.86 | 92.86 |
| XLSX | 2 | 75.00 | 85.32 |
| PPT | 3 | 82.83 | 96.97 |
| PPTX | 7 | 85.02 | 100.00 |
| 六种格式等权 | 28 | **81.35** | **94.00** |

按检查项直接计数，RapidDoc 通过 **136/178**，AnyDoc 通过 **166/178**。RapidDoc 在 Excel 的四处受检合并关系上保留了 **4/4**，AnyDoc 为 **0/4**。RapidDoc 的主要失分项包括 DOCX 标题层级、列表编号和脚注，以及 PPT/PPTX 的备注与部分对象文本。XLSX 还存在显示格式差异，例如原表的百分比和货币被输出为底层数值。

[检查清单](firecrawl-office-checks.tsv)包含每份输入的 SHA-256 和判定条件。[逐项得分](firecrawl-office-scores.csv)包含双方的通过状态。用同一批转换输出复算：

```powershell
python scripts/bench/score_firecrawl_office.py build/anydoc-benchmark docs/benchmarks/firecrawl-office-scores.csv
```

### Office 四维盲评复现

[评审脚本](../../scripts/bench/judge_firecrawl_office.py)将同一份文件的两份 Markdown 匿名配对，以 LibreOffice 渲染的前六页作为参考。每份交换 A/B 顺序评两次，分别记录完整性、结构、格式、整洁度的 1～5 分，再转为百分制并按格式汇总。沿用 AnyDoc 的 GFM 口径，只有 HTML 能表达的结构会受该口径影响。原文件中合并单元格的保留情况仍按上面的结构核验单独报告。

本次准备阶段覆盖 28 份正常夹具，LibreOffice 25.2.7.2 渲染出 15 份参考 PDF。其余 13 份未生成参考 PDF，未进入评分。评审服务暂未返回有效评分，文档中的 Office 表仍是非空输出率和耗时。

运行前将两个 Office 转换结果目录命名为 `formal-anydoc` 和 `formal-rapid-office`，放在同一个工作目录。用包含 PyMuPDF 的 Python 环境执行：

```powershell
python scripts/bench/judge_firecrawl_office.py prepare --root build/anydoc-benchmark --fixtures <AnyDoc夹具目录> --soffice <LibreOffice程序路径>
python scripts/bench/judge_firecrawl_office.py judge --root build/anydoc-benchmark --model claude-sonnet-5
```

评审默认读取环境变量 `ANTHROPIC_API_KEY`。材料、渲染状态、逐轮评分和按格式汇总分别保存在工作目录的 `office-judge/` 下。重跑会复用同一模型、同一输入的已完成评分。

## PDF：OpenDataLoader Bench 全量 200 份

使用 [Firecrawl 公开结果分支固定提交](https://github.com/firecrawl/opendataloader-bench/tree/340f25d70f5b2dbc4bc1cc6b154769f74b1fc745)中的 200 份 PDF、标注和官方评分脚本。三个引擎均在同一台 Windows x64 / Intel Core Ultra 9 185H / 32 GB 机器上串行执行。每组先用首份文件预热一次，再对完整语料计时一遍。表中的总耗时是 200 次 SDK 调用之和，排除文件读取、启动、预热和评分。单次全量计时用于同机观察，与 Firecrawl 原报告的五轮中位数口径有别。

RapidDoc 两组均用 CPU、`parse_method="txt"`、`formula_enable=False`、`table_enable=True`，保留其余 SDK 默认值，唯一的配置差异是版面策略。两组均优先使用原生 PDF 文本，仍可能走局部 OCR 和表格模型。pdf-inspector 使用本地 `process_pdf_bytes` 文本提取入口，关闭 OCR。

| 组别 | 综合 ↑ | 阅读顺序 NID ↑ | 表格 TEDS ↑ | 标题层级 MHS ↑ | 非空输出 | 200 份耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pdf-inspector 0.2.6 | 0.8753 | **0.9147** | 0.8141 | 0.7879 | 199/200 | **0.689 s** |
| RapidDoc 0.1.1 · 默认版面 `pp_doclayoutv3` | 0.8646 | 0.8873 | 0.8977 | 0.7913 | **200/200** | 286.576 s |
| RapidDoc 0.1.1 · 规则版面 `docvortex_rule` | **0.9035** | 0.9123 | **0.9198** | **0.8734** | **200/200** | 158.316 s |

阅读顺序、表格、标题分别对 200、42、107 份有相应标注的文件计分。综合分采用[上游评测器](https://github.com/firecrawl/opendataloader-bench/blob/340f25d70f5b2dbc4bc1cc6b154769f74b1fc745/src/evaluator.py)的逐文件平均方式。pdf-inspector 的本机输出与其公开预测一致（忽略首尾空白）。`01030000000141.pdf` 被它归类为图像 PDF，输出为空，评分时保留空预测。仅在两边都有非空输出的 199 份上，规则版面和 pdf-inspector 的综合分分别是 **0.9078** 和 **0.8797**。

- **规则版面的优势主要在表格和标题。** 对比 pdf-inspector，TEDS 提升 10.56 个百分点，MHS 提升 8.54 个百分点，综合分提升 2.81 个百分点。`01030000000182.pdf` 中，规则版面输出了标注中的标题和 HTML 表格，TEDS 从 0.453 升至 0.995。
- **pdf-inspector 在阅读顺序上略领先，且速度优势很大。** 同机完整语料耗时为 0.689 秒，RapidDoc 规则版面为 158.316 秒，约为其 230 倍。`01030000000018.pdf` 中，pdf-inspector 保留了标注标题，规则版面漏判了该标题。
- **默认版面也提高了表格分，但综合分低于 pdf-inspector。** 这组语料下，RapidDoc 的规则版面比默认版面更快，综合分也更高。

[PDF 逐文件 CSV](firecrawl-pdf.csv)记录每组的输入 SHA-256、输出状态、耗时和四项评分。

## 复现

以下命令使用 PowerShell。准备 RapidDoc 模型目录，并按 SDK 文档设置 `RAPID_MODELS_DIR`。实测 Python 3.12.11，ONNX Runtime 1.31.1。公开基准仓库包含参考预测、PDF 与标注。

```powershell
$root = 'build/firecrawl-bench'
git clone https://github.com/firecrawl/anydoc "$root/anydoc"
git -C "$root/anydoc" checkout e7178e099b6bf9171e2c52b832cbdb4134b69112
git clone --branch abi/pdf-parser-benchmark-results https://github.com/firecrawl/opendataloader-bench "$root/odl"
git -C "$root/odl" checkout 340f25d70f5b2dbc4bc1cc6b154769f74b1fc745
uv venv "$root/venv" --python 3.12.11
$py = (Resolve-Path "$root/venv/Scripts/python.exe").Path
uv pip install --python $py "rapid-doc-rust[cpu]==0.1.1" firecrawl-anydoc==0.2.4 pdf-inspector==0.2.6 onnxruntime==1.31.1 apted==1.0.3 beautifulsoup4==4.15.0 lxml==6.1.3 rapidfuzz==3.14.6
foreach ($engine in 'anydoc','rapid-office') {
    & $py scripts/bench/compare_firecrawl.py $engine "$root/anydoc/tests/fixtures" "$root/$engine"
    & $py scripts/bench/compare_firecrawl.py $engine "$root/anydoc/tests/fixtures" "$root/$engine-exceptional" --exceptional --repeat 1
}
foreach ($engine in 'pdf-inspector','rapid-rule','rapid-default') {
    & $py scripts/bench/compare_firecrawl.py $engine "$root/odl/pdfs" "$root/odl/prediction/local-$engine" --repeat 1
}
Push-Location "$root/odl"
foreach ($engine in 'pdf-inspector','rapid-rule','rapid-default') {
    & $py src/evaluator.py --engine "local-$engine"
}
Pop-Location
```
