"""Audit local benchmark references and write a date-free report."""
from pathlib import Path
import json
import unicodedata
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
def norm(s):
    return ''.join(unicodedata.normalize('NFKC', s).split())

base = ROOT / 'output-ocr-benchmark'
raw = json.loads((base / 'line_accuracy.json').read_text(encoding='utf-8'))
reviewed = []
for row in raw:
    if row['file'] == '05.pdf':
        continue
    row['reference'] = row['reference'].replace('q 2005 Elsevier', '© 2005 Elsevier')
    row['errors'] = {k:Levenshtein.distance(norm(row['reference']), norm(row[k]))
                     for k in ['indra','rapidocr_small']}
    reviewed.append(row)
chars = sum(r['chars'] for r in reviewed)
cer = {k:sum(r['errors'][k] for r in reviewed)/chars for k in ['indra','rapidocr_small']}
comparison = {'reference_lines':len(reviewed), 'reference_chars':chars, 'recognition_only_cer':cer,
              'reference_audit':'Exclude broken 05.pdf text layer; correct visually checked copyright glyphs in demo1.pdf.',
              'runs':{}}
table = []
for folder, label in [('output-ocr-benchmark','RapidOCR 自动检测分辨率'),
                       ('output-ocr-benchmark-fixed','RapidDoc 固定 960')]:
    summary = json.loads((ROOT/folder/'summary.json').read_text(encoding='utf-8'))
    pages = json.loads((ROOT/folder/'pages.json').read_text(encoding='utf-8'))
    comparison['runs'][label] = summary
    table.append(f"| {label} | {summary['page_latency_seconds']['indra']:.3f} | "
                 f"{summary['page_latency_seconds']['rapidocr_small']:.3f} | {summary['speedup']:.2f} 倍 |")
    for p in pages:
        (ROOT/folder/f"{Path(p['file']).stem}-p{p['page']}-indra.txt").write_text(p['outputs']['indra'],encoding='utf-8')
        (ROOT/folder/f"{Path(p['file']).stem}-p{p['page']}-rapidocr.txt").write_text(p['outputs']['rapidocr_small'],encoding='utf-8')
    dense = next(p for p in pages if p['file'].startswith('49'))
    comparison['runs'][label]['dense_page_nonspace_characters'] = {
        k:len(norm(dense['outputs'][k])) for k in ['indra','rapidocr_small']}
(base/'reviewed_line_accuracy.json').write_text(json.dumps(reviewed,ensure_ascii=False,indent=2),encoding='utf-8')
(base/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2),encoding='utf-8')
report = f"""# Indra 与 RapidOCR small 本地测试

Indra 在这批整页测试上更快；PP-OCRv6 small 在文字行识别测试上更准。固定较低检测分辨率会漏掉密集小字，不能只比较耗时。

## 条件

- Indra OCR 0.3.1，模型 v3.1；RapidOCR 3.9.2，项目内 PP-OCRv6 det/rec small ONNX；ONNX Runtime 1.22.0。
- Windows CPU，Intel Family 6 Model 170，22 个逻辑处理器；每个 ONNX 会话 4 个内部线程、1 个会话间线程。
- 同一页面渲染为 200 DPI，图像已准备好后开始计时。模型加载、PDF 渲染不计入速度。预热一次，串行交替运行，每页每引擎 3 次，取各页中位数再求平均。
- 强制使用栅格图像，不读取文本层。Indra 关闭公式，包含方向、纠偏、检测、识别、阅读顺序和版面步骤；RapidOCR 只做检测识别，关闭文字方向分类。
- RapidOCR 使用 RapidDoc 的检测阈值、归一化设置。第一组保留 RapidOCR 原生自动分辨率；第二组使用 RapidDoc 的固定 960 补丁。两组分别运行，Indra 的绝对时间有波动，应比较组内比值。

## 速度

8 页，包含 03.pdf 两页、small_ocr.pdf、05.pdf、demo1.pdf、比亚迪财报、密集长表格和研究报告。

| 配置 | Indra 秒/页 | small 秒/页 | Indra 加速比 |
|---|---:|---:|---:|
{chr(10).join(table)}

密集表格页：原生自动分辨率的 small 输出 7,583 个字符，固定 960 输出 45 个字符；Indra 输出 7,586 个字符。这里包含空格和换行，字符数量用于展示漏检规模，不是准确率。

## 文字识别精度

对 3 份文档各均匀抽取 20 个文字行区域，共 {len(reviewed)} 行、{chars} 个参考字符。使用相同裁剪区域，分别调用两者识别器，不运行检测；用 PDF 文本层作为参考，NFKC 归一化并忽略空白，计算编辑距离/参考字符数。

| 指标 | Indra | PP-OCRv6 small |
|---|---:|---:|
| 字符错误率 CER（越低越好） | {cer['indra']:.2%} | {cer['rapidocr_small']:.2%} |

05.pdf 文本层有明显错误，排除出精度统计。demo1.pdf 的两处版权字符文本层提取为 q，经查看渲染页面后修正为 ©。原始 80 行及未审核统计保留在 line_accuracy.json/summary.json，审核后的结论在 reviewed_line_accuracy.json/comparison.json。

Indra 的具体错误包括上标作者标记、表格边框识别成竖线、203,606 识别成 205.606，以及小字号中文乱码。扫描页 03.pdf 正文人工查看较接近，图中标签 small 提取更多，但没有完整人工标注，不计入定量精度结论。

## 边界与后续

这不是全量 OCR 准确率：识别精度不含整页检测召回率、阅读顺序、表格结构和公式；60 行没有错误也不代表 small 在其他文档没有错误。测试仅覆盖 CPU，未测试 GPU。整页速度使用各引擎自己的检测策略，未强行统一网络输入尺寸。建议当前保留 small 作为精度优先方案；如接入 Indra，应作为可选速度方案，并扩大中文小字、表格和扫描质量测试。

## 复现

```powershell
python tests/benchmark_indra_rapidocr.py
python tests/benchmark_indra_rapidocr.py --fixed-det
python tests/summarize_ocr_benchmark.py
```

只新增测试和本地结果，未修改生产 OCR 配置。输出目录内有每页两引擎的文本及原始计时。
"""
(base/'report.md').write_text(report,encoding='utf-8')
print(json.dumps({'recognition_only_cer':cer,'reference_lines':len(reviewed),'reference_chars':chars},ensure_ascii=False))
