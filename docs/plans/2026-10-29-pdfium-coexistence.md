# PDFium 与 pypdfium2 同进程共存交付

## 状态

第 3 项已实现并安装到项目原有 `.venv`，Windows x64 同进程验证通过；按用户此前约定停在本项，供用户测试。印章接入和最终 Python/Rust 效果、速度、峰值内存验收仍分别属于第 4、5 项，不将本轮共存验证替代最终验收。

## 原因与处理

SDK 原本已打包独立文件 `runtime/rapiddoc_pdfium.dll`，但 `_prepare_pdf` 仍使用过早的统一限制：只要 `pypdfium2` 或 `pypdfium2_raw` 出现在 `sys.modules` 就报错。这也挡住了本来使用不同 DLL 实例的正常情况。

移除该限制，保留独立运行库实例和独立所有权：

- 默认继续使用包内独立命名、独立文件的 PDFium，不借用 pypdfium2 的初始化状态。
- 显式 `pdfium_library` 或 `RAPIDDOC_PDFIUM_LIBRARY` 指向外部 DLL 时，首次加载前复制到内容哈希命名的私有文件，不论此时是否已导入 pypdfium2。这样 Rust 先运行、之后再导入 Python 包也不会误用同一实例。
- Windows 自定义副本位于 `%LOCALAPPDATA%/rapid_doc_rust/pdfium/<sha256>/`；同一内容复用缓存，不每页复制。默认包内库不需要这一步。
- 写入使用临时文件、哈希核验、原子发布及线程锁；损坏缓存或指向源文件的链接会被替换为真实副本。
- 不修改 pypdfium2 的初始化、销毁、字体回调或已打开对象；两个库分别管理自己的文档和句柄。

没有改 PDF 解析、OCR、版面、公式、表格、阅读顺序或模型参数。新增的 Python 模块只处理宿主运行库路径和隔离，不增加算法编排或第三方依赖。

Rust 内部原有全调用锁和进程级 PDFium 所有者保持不变；同一进程的 Rust 运行库初始化后仍不允许切换成另一份不同运行库。外部库覆盖应保持一致，或统一使用环境变量。不要跨两个实例传递原生文档/页面指针。

## 验证

### 路径与缓存单元测试

`tests/migration/test_pdfium_runtime_isolation.py` 六项通过：默认路径、自定义复制与相同内容复用、环境变量覆盖、并发首次准备、损坏缓存与硬链接恢复、缺失输入不发布缓存。

### 八组真实进程内测试

`scripts/migration/verify_pdfium_coexistence.py` 在开发包和最终安装包中通过：

1. Python 初始化在先，Rust 在后。
2. Rust 初始化在先，Python 在后。
3. 只导入 raw 绑定后调用 Rust，再初始化 Python 高层包。
4. Python 在先，Rust 显式指定 pypdfium2 的 DLL 路径。
5. Rust 先显式指定该 DLL，之后导入 Python 包。
6. 环境变量指定外部 DLL。
7. Rust 与 Python 两线程交错重复读取及渲染。
8. Python 侧正常销毁 PDFium 后继续调用 Rust，再初始化 Python 并复测两边。

最终安装测试包括原生文本、扫描页及中文论文三份文档的第一页。保持 Python 文档、页面、文字页、位图句柄存活期间调用 Rust，再核对原对象的文字和渲染。每个引擎的重复输出保持一致。

通过函数地址比较确认两边实际映射的是不同 PDFium 实例，不只比较文件名或复制文件的哈希。Rust 固定字体策略仍为 `droid-cjk-v1`。Python 原 DLL 哈希保持不变。

### 正式 API 混合进程测试

`scripts/migration/verify_pdfium_full_api.py`：

- 对 `示例1-论文模板.pdf` 和 `small_ocr.pdf` 各解析第一页，保留版面、OCR、公式、表格和正常输出。
- 分别取得 Rust、原 Python 的独立进程输出。
- 同一进程同时引入原 Python 解析器和 Rust 包，保持 Python 文字句柄打开，先运行 Rust，再运行 Python，再运行 Rust。
- 混合进程中 Rust 前后输出均与 Rust 独立输出一致；Python 输出与 Python 独立输出一致。
- 比较完整 Markdown、中间 JSON、content list，以及图片名称、模式、尺寸和像素哈希。

这是对**各自独立输出与混合输出**的一致性验证，不声称所有 Python 与 Rust 结果都完全等价，也不作为速度或内存基准。印章在本测试中显式关闭，未提前纳入第 4 项。

测试工具的首次原 Python 运行使用了字符串模型枚举，触发原 Python 接口类型错误；已改为原接口要求的枚举类型后重新完成独立和混合测试，不将该测试配置错误当成 PDFium 问题。

## 安装与可测试用法

已安装 `rapid-doc-rust 0.1.0` 到原项目解释器，无需更换解释器或修改模型配置。版本号未变，内容 SHA256 为：

`e15a83c33d3ffc99b129c8e9e977e86ba6ea3fc402736bed34c2aca7853d4b0c`

`build/wheels/rapid_doc_rust-0.1.0-cp310-abi3-win_amd64.whl` 已同步。当前 wheel 移除生成的 `.pyc`，RECORD 已重建核验；本次只有 `__init__.py` 和新模块 `_pdfium_runtime.py` 的运行时代码变化，其余源码、原生二进制、模型清单和资源字节不变。安装后的 145 个包文件与最终 wheel 一致，`pip check` 通过。

可直接在原 `.venv` 中试用：

```python
import pypdfium2 as pdfium
from rapid_doc_rust import RapidDoc

source = "demo/pdfs/示例1-论文模板.pdf"
engine = RapidDoc(ocr_config={"seal_enable": False})  # 印章待第4项接入
with pdfium.PdfDocument(source) as document:
    result = engine(source, start_page_id=0, end_page_id=0)
    print(len(document), result.markdown)
```

先运行 Rust、后导入 pypdfium2 也已测试。无须手工绑定、替换或卸载对方 PDFium。已运行的 IDE Python 进程需要重新启动才能载入更新后的 Python 模块。

## 证据与回退

- 最终安装确认：`build/pdfium-coexist-installed-acceptance-20261029.json`。
- 八组最终测试：`build/pdfium-coexist-installed-final-r2-tests-20261029/summary.json`，每组有独立报告和标准输出/错误日志，包含保持打开的 Python 位图像素验证。
- 独立及混合正式输出：`build/pdfium-coexist-full-api-r2-20261029/`；安装后混合输出：`build/pdfium-coexist-installed-mixed-20261029/`。最终包与已测混合包的源码、原生文件、资源逐项一致，只清理旧生成字节码；最终安装包又通过八组共存测试。
- 构建清单：`build/pdfium-coexist-wheels-r2-20261029/build-manifest.json`；仅更新 Python 边界及 RECORD，无 Cargo 编译。
- 安装和依赖日志：`build/pdfium-coexist-install-r2-20261029.txt`、`build/pdfium-coexist-pip-check-r2-20261029.txt`。
- 旧 wheel 与原 Python DLL 哈希备份：`build/pre-pdfium-coexist-install-20261029/`。
- 原生进度专项仍通过；未跑全量或 CUDA，也没有修改已试验的优化模型和默认权重。

复现工具：`build_pdfium_coexist_wheel.py`、`verify_pdfium_coexistence.py`、`verify_pdfium_full_api.py` 均在 `scripts/migration/`。本轮完成性声明限于本机 Windows x64 当前依赖；Linux/macOS 代码路径未做实机验收。
