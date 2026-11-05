# XLSX 原生资源与分页编排

本阶段将单元格读取、共享表格投影和 XLSX 图片/图表/公式编排接成 Rust 内部 `Workbook::into_pages()`。没有调用 Python、openpyxl 或宿主回调；没有新增模型或依赖。当前公开 Python 包仍为已交付的 PDF/DOCX，尚未开放 XLSX。

## 实现

- 同一 Package 读取 DrawingML 图片和锚点，按锚点与原始载荷 SHA256 去重；相同图不同位置、同位置不同图和绝对定位均保留。复用已有 Office 位图/矢量序列化器和图片 MathType 解码器。
- OMML 保留原版首个 drawing、先 twoCellAnchor 后 oneCellAnchor 的扫描规则；OLE 复用原生 Equation Native/MTEF v3/v5 解码，保留 linked/icon、objectPr/DrawingML/VML 锚点恢复、预览回退和 OMML 优先级。
- 原生处理 WPS DISPIMG 名称/关系映射、成功位图缓存与图片公式。单元格图片恢复公式后直接返回公式，不再转换预览；该路径的异常回退边界沿原版保留。
- 图片/公式先补入 Cell IR，参与语义判定、自动 gap 与表格投影；已被表格吸收的资源不再独立输出。表格、图表和独立公式按锚点与原 order 稳定排序，普通独立图片仍在这些块之后输出。
- 图表保持 XLSX 原有的“引用单元格选区投影成表格”，不改成通用图表缓存导出。保留 cat/val/xVal/yVal/bubbleSize/系列标题的引用顺序、同表限制、非连续行列集合、首个组合图 plot 的系列规则，以及 openpyxl 的 anchor 类别顺序。关系存在而 worksheet 未写 drawing 标签时，仍按原 reader 保留资源。
- `into_pages()` 消费工作簿，资源与文档同生命周期，不克隆整个工作簿，不重复打开 ZIP。原 `convert_cells()` 保留为明确拒绝资源的单元格诊断入口，避免误用后丢图。

## 验证

参考生成器为 `scripts/migration/export_xlsx_resources_reference.py`，直接执行现有 Python `XlsxConverter.convert()`；运行时不依赖该脚本。

- **95 份工作簿、190 组完整文件到 pages 对照**通过：默认与包含隐藏表/显式 gap=0/关闭 singleton 两组配置。
- 覆盖本地真实文件、PNG/BMP/TIFF/JPEG/GIF/WebP、表内/表外/同位置/绝对位置/合并格图片、相同像素不同锚点、16 类图表、跨表与无效引用、非连续引用、组合图、未写 drawing 标签的关系、OMML、图片公式、OLE 原生/坏对象/icon/link/图片恢复/OMML 抑制/DrawingML/VML，以及 DISPIMG 成功/缺失/绝对目标/重复引用。
- 图片不是只比数量：对解码后的尺寸、完整 RGBA 像素 SHA256 和输出 MIME 核对后，才替换数据 URI 比较全文块/HTML/顺序；SVG 比较源字节 SHA256。**不声称 PNG/JPEG 编码字节一致。**
- 运行 8 个受影响测试目标（共 9 项测试）：XLSX 资源、单元格、共享投影、图表、图片 MTEF、OLE/MTEF、OMML 和位图。日志 `build/xlsx-resources-affected.log`，收据 `build/xlsx-resource-receipt.json`。
- 新资源入口另通过既有 **123 组纯单元格完整分页**对照，检查日期、富文本、样式、隐藏表与标题等规则没有因资源接线改变；日志 `build/xlsx-resources-cell-compat.log`。
- `cargo fmt -p rapiddoc-office --check`、参考脚本编译检查通过。未跑全量 PDF/CUDA，无本阶段性能结论。

## 剩余交付

1. 非规范 XLSX 包的规范化/失败重试、原始工作表映射失配时的图片备用来源及损坏 DrawingML/图表边界验收。这些尚未作为完整兼容性承诺，本阶段参考是已正常进入资源转换的包。
2. 资源分页转统一 Office 输出、公开 Rust/Python XLSX 路由、构建并更新 `.venv`，提供可直接运行示例。
3. XLSX 效果、速度和峰值内存验收，阶段完成后统一回归，再进入 XLS。

保留 DocVortex/openpyxl 许可证与来源哈希，记于 `packaging/office-native-provenance.json`。
