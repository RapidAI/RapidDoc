# RapidDoc Java SDK

Java 8 语法、API 和字节码兼容，直接调用共享 Rust 引擎，支持 PDF、DOC、DOCX、XLS、XLSX、PPT、PPTX，无须 Python 环境。模型推理仍要求 ORT 能在所用 JDK 进程中正常加载。

## 安装

当前尚未发布到 Maven Central。本地构建和安装：

```powershell
.venv/Scripts/python.exe scripts/migration/build_java_sdk.py --ort-dir target/release/runtime
```

构建工具需要 Rust、JDK、Maven 和已准备的 RapidDoc PDFium/GEOS 运行库。脚本默认复用本项目的运行库目录；也可使用 `--runtime-dir` 指定独立目录。构建脚本使用 Python，最终 Java 应用不需要 Python。没有 ORT 时先通过 Rust 的 `download-runtime` 功能构建获取，或提供已有 ORT 目录。

安装到本地 Maven 仓库后，在项目中添加：

```xml
<dependency>
  <groupId>io.github.rapidai</groupId>
  <artifactId>rapid-doc</artifactId>
  <version>0.1.0</version>
</dependency>
```

Maven 根据系统自动引入 Windows/Linux x64 的引擎包 `rapid-doc-native` 和默认 CPU 运行库包 `rapid-doc-ort`。两者独立校验和提取，更换 ORT 不需要重编译 Rust。原生库首次使用时提取并校验至 `~/.cache/rapid_doc/native`，模型由 Rust 自动下载到共享模型目录。Java 字节码目标为 8。全部源码已用本机 Oracle JDK 8u421 编译，Oracle 8u421 / Corretto 8u422 已通过 DOCX 解析；Windows CPU 完整集成回归使用 Microsoft JDK 21 验证。Linux/CUDA 构建入口已提供，尚待独立验证。

部分旧 Windows JDK 自带的 MSVC 运行库会遮蔽系统新版本，导致 ORT 报 `LoadLibraryExW failed`。Windows 平台包现在默认携带私有 CRT 14.51，由共享 Rust 加载层在 ORT 初始化和创建会话时自动隔离使用，不改动 JDK、不安装系统运行库，也不在运行时下载 CRT。该机制目前验证 ORT 1.23.0/1.23.2 CPU；Windows CUDA 尚需实际显卡环境验收。详见[运行库排查报告](../../docs/plans/2026-11-04-windows-ort-runtime-compatibility.md)。

## 调用

```java
import io.github.rapidai.rapiddoc.RapidDoc;
import io.github.rapidai.rapiddoc.ParseOptions;
import io.github.rapidai.rapiddoc.DocumentResult;
import java.nio.file.Paths;

try (RapidDoc engine = RapidDoc.builder().build()) {
    DocumentResult result = engine.parse(Paths.get("input.pdf"));
    System.out.println(result.getMarkdown());
    result.writeTo(Paths.get("output"));
}
```

复用一个实例可复用模型会话；使用 `close()` 或 try-with-resources 释放资源。同一个实例支持多线程并发解析，Office 请求独立执行；PDF 保留 PDFium 和各模型会话的锁，模型权重共享。

### Spring Boot 中复用实例

将引擎注册为单例 Bean，应用停止时由 Spring 调用 `close()`：

```java
@Configuration
public class RapidDocConfiguration {
    @Bean(destroyMethod = "close")
    public RapidDoc rapidDoc() {
        return RapidDoc.builder().build();
    }
}
```

在服务中注入该实例，并串行调用。以下服务通过同步方法管理单例的解析请求：

```java
@Service
public class DocumentParser {
    private final RapidDoc engine;

    public DocumentParser(RapidDoc engine) {
        this.engine = engine;
    }

    public DocumentResult parse(Path path) {
        return engine.parse(path);
    }
}
```

`Configuration` / `Bean` 来自 `org.springframework.context.annotation`，`Service`
来自 `org.springframework.stereotype`。使用者在应用中提供 Spring 依赖。
并发请求可共用该引擎；应用停止时统一关闭。进度回调不要同步重入同一引擎解析；多个请求写文件时，应使用不同的输出路径。

### 单次调用选项

```java
try (RapidDoc engine = RapidDoc.builder()
        .layoutModel("pp_doclayoutv3")
        .formulaModel("pp_formulanet_plus_m")
        .tableModel("unet_slanet_plus")
        .outputDirectory(Paths.get("output"))
        .build()) {
    ParseOptions options = ParseOptions.builder()
        .startPage(0).endPage(2) // 从 0 开始，结束页不包含：共前两页
        .drawLayout(true).drawSpans(true)
        .build();
    DocumentResult result = engine.parse(Paths.get("input.pdf"), options, event -> {
        System.out.printf("%s %d/%d%n", event.getStage(), event.getCompleted(), event.getTotal());
    });
}
```

默认模型、参数和回退规则来自共享 Rust SDK。`formulaEnabled`/`tableEnabled` 默认开启；可通过各 `*Config(Map)` 方法设置高级配置。`ParseOptions` 提供单次调用覆盖；`option` 传入共享 Rust 字段，未知字段会明确报错。

结果提供 Markdown、middle JSON、内容列表、图片和绘图数据。`writeTo(outputRoot)` 保存到 `outputRoot/源文件名/auto|txt|ocr|office/`，文件加源文件名前缀，例如 `示例1.md`、`示例1_middle.json`、`示例1_layout.pdf`；图片路径仍遵循 `imageDirectory`。`getOutputDirectory(outputRoot)` 返回最终目录。路径输入设置 `outputDirectory` 后自动按同一规则保存；字节/流输入使用 `document` 作为名称，需要保存时调用结果的 `writeTo`。

### 输入重载

以下类型均支持 `parse(input)`、`parse(input, options)`、`parse(input, options, listener)`：

| 输入 | 返回 |
| --- | --- |
| `String` 本地文件路径或 HTTP/HTTPS URL | `DocumentResult` |
| `Path` 本地文件路径 | `DocumentResult` |
| `byte[]` 文档内容 | `DocumentResult` |
| `InputStream` 二进制文档流 | `DocumentResult`，读取失败抛 `IOException` |
| `List<Path>` / `List<String>` 本地路径或 URL 列表 | 按输入顺序返回 `List<DocumentResult>` |

```java
DocumentResult one = engine.parse(Paths.get("input.pdf"));
DocumentResult memory = engine.parse(Files.readAllBytes(Paths.get("input.docx")));
try (InputStream input = Files.newInputStream(Paths.get("input.xlsx"))) {
    DocumentResult streamed = engine.parse(input);
    streamed.writeTo(Paths.get("output/xlsx"));
}
List<DocumentResult> documents = engine.parse(Arrays.asList(
    Paths.get("input.pdf"), Paths.get("input.docx")));
```

示例的 `Files` / `Paths` 位于 `java.nio.file`，`InputStream` 位于 `java.io`，`Arrays` / `List` 位于 `java.util`。

输入流从当前位置读取，不回退、不自动关闭；当前会完整读入内存后交给原生字节入口，大文件优先传 `Path`。字符串支持本地路径或 HTTP/HTTPS URL，URL 由共享 Rust 层自动下载。支持格式仍是 PDF、DOC、DOCX、XLS、XLSX、PPT、PPTX。

列表是顺序处理，会复用同一引擎的模型会话并保留所有结果；某个文件失败会停止并抛出异常，大量文件建议逐个解析、保存和释放。`parseAll` 保留为 `parse(List<Path>)` 的兼容别名，同样支持 options/listener。`parsePath` 是内部实现，不是用户入口。Java 泛型擦除不允许再声明同名 `parse(List<byte[]>)`，当前批量输入统一为路径列表。

进度回调在当前解析线程同步执行；未提供的计数为 `-1`，页索引从 0 开始。回调抛出异常会中止本次解析并原样传回。`RapidDocException.getCode()` 返回稳定的原生错误码。

不传回调时默认显示解析开始、页数、模型阶段进度、后处理和总耗时，阶段名称与 Python 对齐，例如 `Layout Predict` / `OCR-rec Predict` / `Table Predict`。交互终端及 IDEA Run 控制台使用回车单行刷新，无 ANSI 清屏序列、时间/类名前缀。进度条写 stderr，开始/完成等状态写 stdout，与 Python 的流区分一致；IDEA 中进度显示红色不代表错误。其他重定向环境默认按行输出并节流。过滤无计数及重复计数的内部事件，不显示 `-1/-1` 或反复打印相同的 `Table Predict 1/1`，新批次会重新计数。`processing` 时间为原生阶段实际处理时间，可能不包含模型加载及交错执行的其他阶段；最终总耗时使用真实经过时间。

可用 `.progressLogging(false)` 关闭默认解析展示。传入自定义 `ProgressListener` 时由用户控制解析展示，下载日志仍独立保留。JVM 属性 `-Drapid_doc.progress.mode=bar` / `plain` 可强制进度条/逐行输出，默认 `auto`。IDEA 启动后另行重定向输出到文件时可显式选 `plain`。

缺少模型时默认通过 `java.util.logging` 的 `io.github.rapidai.rapiddoc.download` logger 显示下载 URL、保存路径、大小进度、SHA-256 校验及完成/失败日志，无须传入进度回调。进度日志约每 5 秒输出一次；等待服务器响应时已显示开始日志，大小未知则不显示百分比。原始回调同时提供 `download_started` / `download_progress` / `download_verifying` / `download_completed` / `download_failed` 事件，其 completed/total 单位为字节；回调抛异常可取消下载。

默认下载目录是 `~/.cache/rapid_doc/models`，Windows 本机对应 `C:/Users/huazhen/.cache/rapid_doc/models`。可用 `.modelDirectory(Paths.get("D:/models"))` 或 `RAPID_MODELS_DIR` / `RAPIDDOC_MODEL_DIR` 配置；已有模型校验通过后直接复用。下载期间先写临时 `.part`，校验成功后才发布为正式文件。

## 运行库与 GPU

无需安装 Java ONNX Runtime 绑定。`rapid-doc-native` 包含 JNI 引擎、PDFium 和 GEOS；`rapid-doc-ort` 单独包含 ORT、私有 CRT，以及 CUDA 包所需的算子库。推理由 Rust 执行。也可通过 Builder 的 `runtimeDirectory` / `resource` 指定运行库、`modelDirectory` 指定模型目录，`allowDownload(false)` 禁止模型下载。

**使用者不需要另装 `com.microsoft.onnxruntime:onnxruntime` 或 Python 的 `onnxruntime`。** 安装 RapidDoc 的 Maven 依赖时，独立的 `rapid-doc-ort` 依赖自动提供 ORT；运行时从包中提取，不会另行联网下载 ORT。模型下载是独立行为。

当前默认构建下载的 CPU/GPU 原生 ORT 基线为 **1.23.2**。共享 Rust 引擎使用 C API 23，外部 ORT 最低为 **1.23.0**；最低版本不等于全部后续版本都经过验证。构建脚本的 `--ort-dir` 可以传入其他兼容版本，因此自行构建时实际版本以提供的动态库为准。Rust `ort` crate 的 `2.0.0-rc.15` 是绑定版本，与 ORT 运行库版本不同。

只有自行构建发布包的人需要准备 ORT。使用默认官方 1.23.2 CPU 运行库时，可先执行：

```powershell
cargo build --release -p rapiddoc --features download-runtime
.venv/Scripts/python.exe scripts/migration/build_java_sdk.py --ort-dir target/release/runtime
```

这一步构建下载需要联网；安装已构建的平台包的用户无需执行。Windows 打包时从 Visual Studio 的可再分发目录自动发现 CRT 14.51，也可由发布者使用 `--crt-dir` / `RAPIDDOC_CRT_REDIST_DIR` 指定，不使用 System32 作为分发来源。

打包会调整 ORT 私有副本的嵌入清单，保持 `.text` 执行代码不变；原始 ORT 和 JDK 均不修改。派生 ORT 不再带有效的微软签名，来源及哈希记录于包内 `provenance.json`。发布者可用 `--ort-sign-command-json` 传入自己的签名命令参数数组，以 `{dll}` 作为待签名 DLL 占位符；签名在生成包校验清单之前进行。强制签名的企业环境需要发布者提供可信签名。CRT 文件自身不修改。

私有 CRT 打包目前支持已验证的 1.23.0/1.23.2。自行打包其他 ORT 版本可用 `--external-ort` 保留原始 DLL，不启用私有 CRT 隔离；这类运行库仍要求与宿主 JDK 的 CRT 兼容。应用的显式 `resource("ort_library", ...)` 设置同样优先于包内运行库。

原生库可通过 JVM 属性 `rapid_doc.native.library` 指定 JNI 文件，或用 `rapid_doc.native.directory` 指定完整独立运行库目录。缓存目录可由 `rapid_doc.native.cache` 覆盖。

### 在应用 POM 中选择 ORT 版本和 CPU/GPU

默认只依赖 `rapid-doc` 即可使用 CPU ORT 1.23.2。需要自己选择时，排除默认 ORT，再明确声明一个运行库依赖：

```xml
<properties>
  <rapid-doc.ort.version>1.23.2</rapid-doc.ort.version>
  <rapid-doc.ort.classifier>windows-x86_64-cuda</rapid-doc.ort.classifier>
</properties>
<dependencies>
  <dependency>
    <groupId>io.github.rapidai</groupId>
    <artifactId>rapid-doc</artifactId>
    <version>0.1.0</version>
    <exclusions>
      <exclusion>
        <groupId>io.github.rapidai</groupId>
        <artifactId>rapid-doc-ort</artifactId>
      </exclusion>
    </exclusions>
  </dependency>
  <dependency>
    <groupId>io.github.rapidai</groupId>
    <artifactId>rapid-doc-ort</artifactId>
    <version>${rapid-doc.ort.version}</version>
    <classifier>${rapid-doc.ort.classifier}</classifier>
    <scope>runtime</scope>
  </dependency>
</dependencies>
```

CPU 使用 `windows-x86_64`，GPU 使用 `windows-x86_64-cuda`；Linux 对应 `linux-x86_64` / `linux-x86_64-cuda`。GPU 调用设置 `.device("cuda")`，还需要匹配的 NVIDIA 驱动和 CUDA/cuDNN。Java JNI 引擎统一支持两种运行库，更换 ORT 包不需要重新编译引擎。

只切换相同 classifier 的 ORT 版本，也可用应用的 `dependencyManagement` 覆盖该依赖。仅定义一个同名 POM 属性不会修改依赖 SDK 内部的属性；上例的属性由应用自己的直接依赖使用，因此会生效。classifiers 不同的依赖不会相互替换，选择 GPU 时必须排除默认 CPU 包；同时出现多个 ORT 包会明确报错。

只能选择仓库实际提供的 `rapid-doc-ort` 版本，不会自动转换任意官方 ORT 下载。目前尚未发布到 Maven Central；本地已构建 1.23.0 CPU、1.23.2 CPU/CUDA。其他版本仍需发布者打包和验证，最低要求为 ORT 1.23。

修改依赖后重新启动应用；同一进程不支持热切换 ORT 版本。

### 发布者单独制作 ORT 包

使用现有官方 ORT 目录即可打包另一个版本，不重编译 Rust 或 Java：

```powershell
.venv/Scripts/python.exe scripts/migration/build_java_sdk.py --ort-only --ort-version 1.23.0 --ort-dir <ORT目录>
```

GPU 运行库加 `--cuda`。包的 Maven 版本会与动态库实际版本核对，防止错标；私有 CRT 未验证的版本需要显式 `--external-ort`，并单独检查宿主 CRT 兼容性。该命令只安装到本地 Maven 仓库，不会对外发布。

Windows CUDA 包已在无 GPU 的本机完成包校验和 CPU OCR 验证；明确请求 CUDA 时会返回 `device_unavailable` 并说明缺失依赖，不会静默改用 CPU。实际 Windows GPU 推理、自定义算子及其 CRT 共存仍待显卡环境验收。

`parse(String)` 自动识别 HTTP/HTTPS 地址并由 Rust 下载：`engine.parse("https://example.com/report.pdf")`。请直接传字符串，勿使用 `Paths.get(url)`。带查询参数的下载地址也支持；输出名称优先使用 `filename` 查询参数，或 URL 路径末段。URL 内容在内存中读取，不创建下载缓存；网络连接/响应/读取超时为 60 秒。
