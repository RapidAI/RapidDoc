# PPTX 公开接入与交付验收

已连接 `rapid_doc_rust.RapidDoc` 的原生 PPTX 分发。`demo/rust_pptx_demo.py` 使用项目 `.venv`，调用方式与 DOCX/XLSX/XLS 一致，输出到 `demo/output/rust-pptx/<文件名>/office`。不调用 Python Office 解析器，不加载识别模型。

## 本次变化

- 失败重试接入 Strict URI 转换、缺失 w 命名空间、contentPart 备用图形、坏媒体移除与对应关系删除、沿关系图修复内容类型。通过 ZIP 覆盖层修改，不重压缩或复制整个包。正常解析不提前规范化；资源预算异常不重试。
- 与原 Python 一致，流式校验所有 ZIP 成员的压缩/CRC，包括未引用的成员；不将未引用的坏文件静默忽略。坏媒体恢复只适用于原规则允许的 `ppt/media/`。
- `contentPart` 正常路径明确触发备用图形重试，避免未支持的分支产生空内容却不进入修复。
- DOCX 的内容类型修复抽出共享参数入口，原 DOCX 策略不变；PPTX 使用自身关系尾段到内容类型映射。
- 已连接公开 Python 路由；保留旧 `parse_docx_bytes` 格式边界。DOC/PPT 尚未接入。
- 安装验收发现超长列表起始编号在 JSON 中间值被转成浮点数，启用 serde_json arbitrary_precision 保持公开接口整数精度；这也修复共享 Office 导出边界。重新构建并复查共享导出及安装结果。

## 已有检查

- 280 份 PPTX 完整输出参考：279 成功一致，1 份原始坏跨度异常保留。含 SVG/位图伴随回退、空 blip 和缺失关系补充。
- 26 份包修复成员语义检查，以及对应原始包到 pages 的完整失败重试检查。2 份原 normalizer 异常保留；成员 XML 按 namespace、属性、文字和节点顺序比较，二进制按 SHA256 比较。
- 原始 Strict/非标准路径/阅读顺序三份手工文件同时对照完整中间结构、资产及四种导出，已不依赖 Python 预修复输入交付。
- 1,151 组第二套公式、240 几何、2,077 字体与 800 标题参考沿用前阶段。

公开安装与本地性能验收结果以本文后续补充及 `build/pptx-delivery-receipt-20261031.json` 为准；未完成前不把构建中的 wheel 当作 `.venv` 已更新。

脚本：`export_pptx_package_reference.py`、`check_pptx_install.py`；性能复用 `accept_xlsx_delivery.py --format pptx`，按两轮交替顺序隔离进程运行 Python/Rust，严格比较内容及资产像素。没有模型 PDF/CUDA 全量测试。

## 最终交付结果

新版 wheel 已安装项目 `.venv`，390 个安装文件逐字节匹配 wheel。279 组公开成功结果一致、1 组原异常保留；bytes/bytearray/可复位与不可复位流、路径、batch、data URI、自定义 writer、四种导出及 DOCX/XLSX/XLS 兼容通过。禁用 Python Office 包和模型加载的检查通过。8 组最终 PDFium 同进程共存、pip check 通过。79 项 Office 阶段测试通过；最后整数精度修复后，DOCX/XLS/XLSX/PPTX 导出与包重试 5 项受影响测试通过。公开 demo 已实际运行。

wheel SHA256：`5a96be4ec50d129ea48b8ae39f7dd19fe3fc977230f55a312734f0a5b52fb841`。最终收据：`build/pptx-delivery-receipt-20261031.json`。wheel 内 provenance 保留构建时快照，最终状态以此收据和源文档为准。

最终性能结果采用独立的 `build/pptx-acceptance-clean-20261031`：9 份输入、两轮交替、36 个独立进程，全部成功且完整输出一致。首次与 PDFium 检查重叠的测量已中止弃用，未混入结果。

| 输入 | Python ms | Rust ms | 提速 | Python 峰值 MiB | Rust 峰值 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| demo/pptx/powerpoint_sample.pptx | 46.69 | 22.19 | 2.10x | 335.7 | 62.3 |
| tests/fixtures/docvortex_migration/pptx_01.pptx | 65.11 | 23.94 | 2.72x | 336.8 | 63.8 |
| tests/fixtures/pptx/handmade-altpath.pptx | 2.78 | 0.55 | 5.02x | 313.6 | 35.6 |
| tests/fixtures/pptx/handmade-inherit.pptx | 5.54 | 0.59 | 9.40x | 313.4 | 35.9 |
| tests/fixtures/pptx/handmade-links.pptx | 3.31 | 0.62 | 5.30x | 313.7 | 35.8 |
| tests/fixtures/pptx/handmade-math.pptx | 4.13 | 0.76 | 5.45x | 313.9 | 36.6 |
| tests/fixtures/pptx/handmade-order.pptx | 4.54 | 0.60 | 7.51x | 313.5 | 36.0 |
| tests/fixtures/pptx/handmade-strict.pptx | 3.80 | 0.60 | 6.33x | 313.7 | 35.6 |
| tests/migration/inputs/synthetic.pptx | 4.44 | 0.81 | 5.51x | 314.6 | 36.3 |

峰值含导入基线差异，小样例不是完整客户文档；不推广为全部 PPTX 的速度或内存保证。未运行 CUDA/全量模型 PDF。后续为 PPT → DOC。
