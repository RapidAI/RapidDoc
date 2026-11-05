# 整页方向矫正

补齐原 Python `BatchAnalyze` 中的 `USE_DOC_ORIENTATION_CLASSIFY`。
此前 Rust 只有表格内部方向分类，未消费此整页开关。

```python
import os
os.environ['USE_DOC_ORIENTATION_CLASSIFY'] = 'true'
from rapid_doc_rust import RapidDoc
engine = RapidDoc()
result = engine('input.pdf')
```

默认关闭；`true`、`1`、`yes`、`on` 忽略大小写与首尾空格。Python 仅
转交布尔配置，原生 `pipeline::Request.doc_orientation_classify` 控制
主流程，其他语言绑定也可使用。预加载模型也遵循此开关。

保持原 Python 的规则：高宽比大于 1.2 才调用已有方向模型；仅 90°
与 270°旋转，0°和 180°保持原图。未另行增加 180°旋转。方向模型
仍为已有 `rapid_orientation.onnx`，无新增权重或优化缓存。

版面/公式/OCR/表格/印章统一使用矫正图。原页面事实写入方向标签，
供正文 OCR 与表格流程判断是否可复用 PDF 原生文字。所有识别完成后
按原 `restore_poly` 还原顶层 poly，并用原图组装最终页面与资产；
不把中途坐标错配到原始 PDF。原始 PNG 按窗口释放。

## 验证与交付

- 两页真实旋转图片：原 Python 与 Rust 实际方向模型均输出 270°、90°；
  完整 snapshot（Markdown、中间结构、内容列表、资产）精确一致。
- 两页旋转表格：默认公式/表格/印章阶段开启，实际方向为 270°、90°；
  两张表及两项资产包含在完整 snapshot 中，对照精确一致。
- 默认关闭及受控 0°、180°响应分支，三组各两页的完整输出均精确一致。
- Windows 项目 `.venv` CPU 实际解析两页，输出 1333 字符并还原正文；
  Linux 独立 SDK 使用 CUDA 验证。原生坐标还原测试通过。
- 原始证据：`build/doc-orientation/`。可复用的定向对照脚本为
  `scripts/migration/accept_doc_orientation.py`，`--label` 仅用于测试所有
  分类分支，生成独立测试权重，不改原模型；不传则使用实际方向模型。

最新 Windows wheel 在 `build/doc-orientation-windows-wheels/`；Linux
CUDA wheel 在 `build/doc-orientation/linux-wheels/`。两端均包含之前的
UNET、Magika、异常日志及退化裁图修复。已安装项目 `.venv` 和服务器
`release-acceptance/linux-cuda/env-delivery`，依赖检查通过。

未重跑整套评测。上述结果证明已测分支一致，不代表全部文档精度验收；
前一份科学文档的长公式差异仍记录于退化裁图修复文档，未宣称解决。
