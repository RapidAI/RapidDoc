# 固定 CJK 字体运行时

移植旧字体提供器，在唯一 PDFium 运行库初始化后、打开文档前安装。保留原 DroidSansFallbackFull 字体、SHA256、缓存名、C 回调与 system 库入口 ABI、系统字体代理及永久错误传播。字体和资源编入 Rust 运行库，Python 发行包不依赖旧 rapid_doc 安装。

原生提供器沿本地 DocVortex 现有 Rust 实现；非 ASCII 名称改由独立 Rust 执行旧五编码解码与 Unicode casefold。编码表由旧 Python codecs 冻结，不使用近似编码替代。

先验证字体资源/分类/回调，再重新运行已知未嵌入 CID 失败和受影响实时 PDF 字框、表格与渲染。阶段末构建一次 wheel 和相关安装回归。不改变默认 PDF 编排完成标志；不运行 CUDA 或全量套件。

## 实现与证据

- `rapiddoc-pdf/src/fonts/provider.rs` 保持旧Rust源码字节不变，来源SHA记录在资源SOURCES.json并实际验证；适配器不初始化第二个PDFium。
- 固定字体、四组单/双字节映射和207个GB18030四字节范围、旧Unicode casefold编入运行库。资源来自记录的Python版本与旧字体策略，源码/字库/编码资源哈希验证通过。
- 4771分类/3209解码对照，ABI模拟覆盖C回调与system库入口、默认接口重入、系统/固定句柄区分、小缓冲区不写、表数据/全字体复制、单次清理和永久故障。
- 新增8页未嵌入CID PDF（简中/繁中/日文/韩文×0/90文字方向），原始字符元数据/字框和72/144 DPI的16个旧RGBA像素参考通过。
- 原CID失败常规回归通过，原始冻结文件及旧失败日志未覆盖。
- 6库单元及5集成测试通过：新增字体、既有58字符查询、47数值候选查询及58完整候选页；仅受影响测试，无全量/CUDA。
- 固定上游36＋3＋1未修改；另新增字体/编码资源及provider来源哈希，不能混算为原始vendored内核文件。

安装验收：一次release wheel及隔离重装，7组相关安装套件通过（新字体/字符/数值/全产生器/整页/compatibility/smoke）；32个RGB/RGBA像素比较及原CID实时源、候选回归通过。字体身份True不代表默认PDF已整合。构建及安装日志 `build/native-fixed-font-wheel.txt`、`build/native-fixed-font-install-test.txt`、`build/native-fixed-font-regression.txt`。Rust日志 `build/native-fixed-font-related-rust.txt`、`build/native-fixed-font-numeric-rust.txt`、`build/native-fixed-font-complete-rust.txt`。字体身份True不代表完整PDF/整文档等价，diagram和默认接线仍待。
