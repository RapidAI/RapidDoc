"""Write manually audited, date-free results for the fixed formula fixtures."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output-formula-benchmark'
CORRECT={
    'indra':{'energy','root'},
    'S':{'energy','fraction','root','integral','derivative','trig','real_vehicle'},
    'M':{'energy','fraction','root','quadratic','integral','derivative','trig',
         'real_vehicle','zh_area','zh_variance','zh_speed'},
}
NOTES={
    'sum':'S adds spurious punctuation/factorial; M changes image variable i to j. Variable fidelity is required.',
    'quadratic':'S includes malformed array/end tokens; Indra changes variables and denominator.',
    'integral':'Indra changes x to chi; confidence passes but transcription is wrong.',
    'zh_average':'M preserves Chinese but outputs x_{^{1}} etc rather than x_1 etc.',
    'real_vehicle':'S and M preserve the scanned equation; Indra changes denominator and loses equality/subscripts.',
}
summary={'samples':13,'synthetic_samples':12,'real_scan_samples':1,
         'criterion':'Manual content transcription: ignore spacing/font/style wrappers; preserve variables and formula structure. Not a LaTeX renderer test.',
         'engines':{}}
table=[]
for name in ['indra','S','M']:
    data=json.loads((OUT/(name+'.json')).read_text(encoding='utf-8'))
    rows=data['samples']
    assert len(rows)==13 and len({r['id'] for r in rows})==13
    assert data['providers']==['CPUExecutionProvider']
    checks=[]
    for row in rows:
        stable=len(set(row['predictions']))==1
        assert stable, (name,row['id'])
        pred=row['predictions'][0]
        expected=row['chinese']
        chinese_kept=all(c in pred for c in expected) if expected else None
        checks.append({'id':row['id'],'reference':row['reference'],'prediction':pred,
                       'content_correct':row['id'] in CORRECT[name],
                       'chinese_preserved':chinese_kept,'note':NOTES.get(row['id'],'')})
    pure=sum(c['content_correct'] for c in checks if not c['id'].startswith('zh_'))
    chinese=sum(c['content_correct'] for c in checks if c['id'].startswith('zh_'))
    keep=sum(c['chinese_preserved'] is True for c in checks)
    accepted=sum(r['confidence']>=0.9 for r in rows) if name=='indra' else None
    summary['engines'][name]={'mean_sample_median_ms':data['mean_sample_median_seconds']*1000,
        'non_chinese_correct':pure,'chinese_formula_correct':chinese,
        'chinese_words_preserved':keep,'passes_indra_confidence_gate':accepted,'checks':checks}
    table.append(f"| {name} | {data['mean_sample_median_seconds']*1000:.1f} ms | {pure}/9 | {chinese}/4 | {keep}/4 |")
(OUT/'reviewed_results.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
report=f"""# 公式模型本地比较

## 方法

CPU、ONNX Runtime 1.22.0，每个会话内部 4 线程、会话间 1 线程；确认三者只使用 CPUExecutionProvider。单个公式一个批次，预热后每个公式重复 3 次，取每个样本中位数再求平均。不计模型加载、图片生成、PDF 渲染、公式区域检测和公式分类。Indra 直接调用 MathReader.read，S/M 调用项目 RapidFormula，使用 D:/CodeProjects/doc/RapidAI/models 下的 PP-FormulaNet_plus-S/M ONNX，未修改生产配置。

共 13 个样本：12 张生成图片（8 个无中文、4 个含中文），另有 demo/pdfs/03.pdf 第 2 页真实扫描公式 (2-1)。生成图片有已知公式参考，扫描公式由查看图片后人工转写。所有原图、参考答案及每次输出保存在此目录。Indra 使用文字墨迹的紧边界作为识别框，避免把外围留白当成文字行高度。

内容判断忽略空白、字体与字号包装，但要求变量符号和上下标、分数等结构保持原图一致。因此将求和变量 i 改成 j 仍算转写错误，尽管该表达式可能数学等价。本次没有验证每种 LaTeX 输出在实际渲染器中都能正确显示。

## 结果

| 模型 | 平均公式耗时 | 无中文内容正确数 | 中文公式完整正确数 | 中文文字保留数 |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

Indra 按默认 0.9 置信度阈值，13 个输出中只有 3 个达到接受阈值，其中 2 个内容正确。其默认低置信度提前终止解码机制参与上述计时；因此其更低耗时不能理解为正确完成同等公式的固定加速倍数。实际整页管线还需要公式分类，并可能保留普通 OCR 文本。

## 中文与错误实例

- Indra v3.1 的公式词表无汉字，本次面积、平均值、方差及“速度=路程/时间”都没有保留中文。
- S 同样未保留这四组中文。它能正确识别多个简单公式和真实扫描公式，但二次方程样本有异常 array/end 内容，求和样本增加了错误字符。
- M 四组中文文字都保留。速度分数中，分子“路程”、分母“时间”都正确；平均值样本却把 x_1 等写成 x_{{^{{1}}}}，所以不能算完整正确。
- Indra 的积分样本达到置信度阈值，却将 x 改成 chi；真实扫描公式出现等号遗漏、下标和分母错误。这说明置信度通过不保证公式转写正确。

## 限制与使用判断

这是 13 个简单样本的功能与性能检查，不能替代真实论文大样本评测。没有测试复杂矩阵、多行长公式、手写公式、模糊扫描或 GPU。S/M 使用项目本地 ONNX 导出模型，Indra 使用 v3.1 MathRec；各引擎保留自己的预处理和解码方式，耗时可能随 CPU 状态和公式长度波动。

当前结果支持：中文混合公式优先 M，简单纯数学公式可使用 S。Indra 在这些样本上很快，但错误和低置信度拒绝较多，暂不适合直接替换现有公式识别。

## 复现

```powershell
python tests/benchmark_formula_engines.py indra
python tests/benchmark_formula_engines.py S
python tests/benchmark_formula_engines.py M
python tests/summarize_formula_benchmark.py
```

reviewed_results.json 保存人工内容核对结果；indra.json、S.json、M.json 保存计时、输出、置信度和后端。
"""
(OUT/'report.md').write_text(report,encoding='utf-8')
print(json.dumps({k:{a:b for a,b in v.items() if a!='checks'} for k,v in summary['engines'].items()},ensure_ascii=False,indent=2))
