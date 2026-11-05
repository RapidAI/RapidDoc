# 印章识别交付（第 4 项）

状态：第 4 项已完成本机实现、验收，安装到原 `.venv`；按约定停下供用户测试。

## 实现范围

正式入口为 `rapid_doc_rust.RapidDoc` 及共享 `rapiddoc::pipeline::Engine`。默认 `ocr_config.seal_enable=True`；设置为 `False` 跳过印章模型和处理。没有印章候选的页面不加载印章检测模型、不额外解码页图。旧 `parse_pdf_models` 开发预览入口继续保留原用途，本轮没有给它追加 Python 版印章调度。

保持原 Python `_run_seal_ocr`：选取 `original_label=seal` 的版面区域，按 `crop_img` 裁图及多边形白底遮罩；专用 PP-OCRv4 mobile seal DB 检测；多边形扩框；弯曲文字展开/矩形回退；PP-OCRv6 small 识别；按原 0.5 置信度过滤，将文字列表写回原版面，沿现有图片/印章输出生成 Markdown 和 JSON。没有存活识别结果时保留原版面文字。

检测参数保持 `limit_side_len=736`、`limit_type=min`、`thresh=0.2`、`box_thresh=0.6`、`unclip_ratio=0.5`、`box_type=poly`、关闭 dilation。原 Python 补丁实际上没有使用 `max_side_limit=4000`；Rust 同样不据此缩小图像，保留通用分配/张量预算，超出时明确报错。

检测、裁剪、识别和回填均在 Rust；Python 仅增加参数映射、模型清单和底层检测调用封装。识别器不被正文的自定义 `Rec.model_path` 替换，延续原 Python 独立印章 OCR 的默认权重。印章与正文使用相同默认权重及会话配置时，共享现有 Rust 会话缓存。

保留 OpenCV 5 线性透视采样的 FP32、FMA/标量尾部计算顺序，以及竖排分支的 NumPy FP32 归约。矩形、无效多边形、采样不足、展开异常的回退边界来自原实现，没有用普通四角 OCR 替代曲线展开。原代码的相机标定、可视化分支未被 `mode=homography` 调用，没有引入。

多边形扩框使用与 `pyclipper` 相同的 Clipper 6.4.2。现有 Rust 四边形扩框只支持凸四角，不能代替此处的任意多边形，因此新增约 158 KB 未修改上游 C++ 源及薄 C ABI，静态编入 Rust，不要求安装 `pyclipper`/OpenCV。版本、SHA、来源和 Boost 许可记录在 `rust/rapiddoc-inference/vendor/clipper6/UPSTREAM.json`，随 wheel 分发许可；其他移植署名见 `licenses/PADDLE-SEAL.txt`、`licenses/OPENCV-SEAL-WARP.txt`。

## 验证与复现

- 原 Python 裁剪基准：`scripts/migration/freeze_seal_crops.py --output-dir build/seal-crop-reference-20261030`。记录原源文件 SHA、本机 OpenCV/NumPy 版本、输入、原裁图和逐条 Homography。
- 受影响 Rust 检查：设置 `RAPIDDOC_SEAL_REFERENCE_DIR`、`RAPIDDOC_SEAL_GEOS_LIBRARY`，运行 `cargo test -p rapiddoc-inference --lib --release --offline -j 1 seal::crop::tests::original_polygon_crops -- --ignored --nocapture`。
- 检测、扩框、裁剪及真实印章模型：`scripts/migration/verify_seal_runtime.py --sdk-root <解包目录> --output-dir <新报告目录> --models`。
- 正式 PDF：`scripts/migration/verify_seal_pdf.py --sdk-root <解包目录> --output-dir <新报告目录>`。使用现有 `demo/pdfs/含印章.pdf`，比较原 Python/Rust 印章文字和默认/禁用开关，记录实际解析过的模型；这不是第 5 项完整效果、速度、峰值内存验收。

已通过 105 组原 Python 差分：78 组 Clipper 扩框、14 组检测概率图、12 组逐像素裁图、1 组真实印章模型。真实图片检出的 4 段文字及多边形一致。12 组裁图覆盖不同曲率、旋转、竖排、无效多边形矩形回退。

现有 4 页 `含印章.pdf` 中，原 Python/Rust 正式 API 识别结果均为“中国证券投资基金业协会”“170000079”。版面中实际一处印章；中间 JSON 在预处理块和段落块各保留同一 span，因此报告有两个一致条目。关闭开关后没有解析 `seal_det` 模型，也没有印章文字。禁用验收最初把 `[None]` 当作文字，已修正验收脚本，复用保存的推理结果验证，不重复跑模型；初次日志保留为 `build/seal-pdf-20261030-r2-initial-validation.log`。

安装后以默认 API、默认参数单独解析第 4 页，同样得到上述印章文字。4 项惰性加载/权重配置单测、6 项 PDFium 隔离单测、安装后的 8 组 PDFium 共存检查及 `pip check` 均通过。

## 安装与证据

- wheel：`build/seal-wheel-20261030-r2/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl`，已同步 `build/wheels/`，SHA256 为 `65697f4d71e15c64ab584b469fb33f08ac48792ae42104af39e97b984ad513b9`。
- 原 `.venv` 的 148 个包文件与该 wheel 逐项一致；安装收据 `build/seal-installed-acceptance-20261030.json`。wheel 不包含生成字节码。
- 原安装包和文件备份：`build/pre-seal-install-20261030/`。
- 105 组差分：`build/seal-runtime-20261030-r2/summary.json`。
- 正式 PDF/开关验收：`build/seal-pdf-20261030-r2/summary.json`。
- 安装后默认入口：`build/seal-installed-smoke-20261030/summary.json`。
- 安装后共存：`build/seal-installed-pdfium-20261030/summary.json`。

可以继续用原 `.venv` 直接运行现有 demo；常驻 Python/IDE 控制台应先结束旧进程。默认启用印章，无须额外配置。显式关闭方式为 `RapidDoc(ocr_config={"seal_enable": False})`。

随后修正 IDE 导航：包根原先只经动态 `__getattr__` 导出 `RapidDoc`，现增加 `TYPE_CHECKING` 下的显式同名导出，便于 IDE 定位 `_compat.py:56` 的真实构造签名，同时保留运行时延迟加载。已同步原 `.venv` 和 `build/wheels/`；当前 wheel SHA256 为 `b7f2e140fef05cf900fcec576660260028de0a8ccf05f532cbe075544427126a`，收据 `build/ide-api-installed-20261030.json`。148 个安装文件核对通过，构造参数和延迟加载验证通过，无 Rust 重编译；上述印章交付 wheel/收据保留作为原始验收记录。

本轮只做受影响验证及阶段 release 构建；未跑全量或 CUDA，未改模型权重。实机验收限本机 Windows x64。第 5 项完整输出效果、速度和峰值内存对比仍未开始，不将本轮印章差分当作整条 PDF 质量/性能验收。
