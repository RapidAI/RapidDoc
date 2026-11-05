# XLS 资源、完整导出与公开入口

## 实现

- `officeart.rs` 移植当前 Python `legacy/officeart.py`：借用记录载荷、显式栈遍历、FSP/FOPT/tertiary properties、Excel anchor、隐藏标志、BSE/BLIP、裸 DIB 文件头与延迟 BLIP。图片保持一基 BSE 序号，坏图片不会使后续资源编号错位。
- WMF/EMF BLIP 保留 render_size_emu、raw/zlib 两种解压与完成判定。解压以固定缓冲区累计，图像局部超原版预算跳过该 BSE，全局素材/记录/嵌套预算仍报原资源异常；没有新增文档字符上限。
- `xls/resources.rs` 按 drawing/OBJ 顺序关联形状，解析 FtCmo、FtCbls、FtPictFlags/FtPictFmla、TXO 与其 CONTINUE。保留文本框富文本追加偏移、控件文字、隐藏/无 anchor 对象、对象数量失配和图表对象索引。
- MBD Equation Native 使用已有 MTEF v3/v5，引入原生公式→图片 comment 公式→图片回退优先级。原生公式生效时不再输出预览图。保留链接/ActiveX/camera 等 FtPictFlags 排除；不会因为 CompObj ProgID 不同而漏掉原版可读的 Equation Native。
- 图片/公式进入共用 Cell IR 之后做表格发现。表格、嵌入 chart 和独立 equation 按原锚点/顺序稳定排序，已吸收的资源不再独立输出，普通独立图仍在最后。图表保留源选区 HTML 和可用预览；不支持的引用保留原预览/诊断。
- 独立 chart sheet 保持原目录顺序和当前 Python 最后一个已转换 sheet 的资源上下文。无待接资源的早期诊断入口仍可用；新 `Workbook::open_complete/from_ole` 完成绑定后允许全文分页。
- 统一 Office 中间结果、Markdown、NLP Markdown、content_list/v2、图片落盘复用已验收的导出器。Python `parse_office_bytes`/`RapidDoc` 新增 XLS OLE 路由，复用已打开 OLE，不经 Python Office 包或先转 XLSX/PDF。

## 通用 PNG 兼容修复

XLS 坏公式回退暴露一张 Pillow 能 load 的 PNG，其 IDAT CRC/末尾 chunk 不满足 Rust PNG reader 的默认检查。原版 load 与 verify 不是同一行为：load 校验 IDAT 前的头部 CRC，但略过图像数据 CRC，完整像素可能在 IEND/Adler 尾部之前就已解出。

原生普通 Office PNG 读取现在保留 IDAT 前 CRC 校验，略过 IDAT CRC，并继续拒绝**完整但错误的 Adler checksum**。仅普通读取失败时，流式验证完整 deflate 数据，在像素数据完整、但 IDAT CRC/IEND 或 Adler 尾部不完整时补齐容器尾部；不更改像素、不重压缩图像数据，也不为某份文件写条件。解压过程只保存固定验证缓冲，不保存第二份原始像素。元文件内部的严格 PNG 路径没有更改。

229 组位图参考包含 RGB/RGBA/调色板的坏 IHDR/PLTE/tRNS/IDAT/IEND CRC、缺尾部、短 IDAT、坏 Adler 等对照；能恢复和必须失败的结果分别与当前 Pillow 核对。坏头部携带巨大尺寸仍按原版拒绝，不误报为额外的尺寸预算故障。

## 验证

- XLS 资源/分页：**161 份完整输入**，150 份成功结果与当前 Python 一致、11 份原错误保留。比较资源绑定后的单元格、图片/公式/图表语义模型和全文分页顺序。
- 完整导出：上述 161 份对应的 middle、四种导出和资产全部核对。图片先核对 MIME/尺寸/完整 RGBA 像素（SVG 核对内容），再允许编码差异导致的文件名别名；不放宽文字、结构、顺序。
- 28 组独立 OfficeArt 对照：raw/zlib/截断 metafile、声明尺寸超限后的后一张图片、一/双 UID、延迟资源、嵌套上限和记录计费；严格比较解码载荷字节、元数据、shape 与记录访问次数。
- PNG 修复的受影响位图、XLS 资源和完整导出检查通过。阶段结束统一运行 Office crate 回归；未跑全量 PDF/CUDA。
- 参考生成脚本 `export_xls_resources_reference.py`、`export_xls_output_reference.py`；测试日志 `build/xls-resource-final-tests.log`、`build/xls-png-final-tests.log`、`build/xls-office-stage-regression.log`。

## 安装与验收

正式 wheel 目录 `build/office-xls-wheels`，安装检查 `scripts/migration/check_xls_install.py`：冻结完整导出、公用 RapidDoc IO、data URI、writer、DOCX/XLSX 兼容，以及禁止 Python Office 包/识别模型加载。

运行示例：`.venv/Scripts/python.exe demo/rust_xls_demo.py`，修改 `XLS_FILES` 即可。输出为 `demo/output/rust-xls/<文件名>/office`。

速度与内存使用 `scripts/migration/accept_xlsx_delivery.py --format xls --output build/xls-acceptance-20261031`；复用已验收的独立进程计时和图片/结构严格核对，两轮交替 Python/Rust。计时排除构建、验证和落盘；内存峰值包含导入基线，2ms 采样可能遗漏短峰值。具体安装、性能和回归结论以最终收据为准，不根据语言直接声称一定更快。

后续为 PPTX → PPT → DOC。非常规未知文档、少见图片和跨平台仍需持续扩充覆盖；增强 PDF 阅读顺序/CUDA 继续按用户要求延后。

## 最终安装与本地性能结果

71 项 Office 阶段测试通过，最后补充坏头部巨大尺寸边界后复查位图测试通过。新版 wheel 已安装原 `.venv`，389 个安装文件逐字节匹配 wheel；150 组公开成功结果与冻结原 Python 输出一致、11 组原异常保留，四种 IO、batch/data URI/writer/四种导出和 DOCX/XLSX 兼容通过，禁止 Python Office 包及识别模型加载检查通过。最终 8 组 PDFium 同进程共存、`pip check` 和格式检查通过。

wheel SHA256：`5f5e526cab5768ab933c2f7e0188fe3b246ce6eb98be926711b2f5fc46e34b97`；最终收据 `build/xls-delivery-receipt-20261031.json`。

两份本地输入、两轮交替顺序共 8 个独立进程，全部成功且完整输出一致。热运行中位数与整个进程峰值如下：

| 输入 | Python ms | Rust ms | 提速倍数 | Python 峰值 MiB | Rust 峰值 MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| demo/ms_office_docs/xlsx_01.xls | 23.62 | 8.45 | 2.80 | 310.2 | 46.5 |
| tests/migration/inputs/synthetic.xls | 2.12 | 0.61 | 3.45 | 305.8 | 39.0 |

这里仅有一份真实文档和一份合成基线，不能推广为所有 XLS 文件的提速倍数；峰值包含导入基线差异。原始报告/快照在 `build/xls-acceptance-20261031`。XLS 可以直接运行示例测试，下一格式为 PPTX。
