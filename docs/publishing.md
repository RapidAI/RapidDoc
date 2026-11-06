# 原生 SDK 发布

本次准备发布 SDK `0.1.1`，上一公开版本为 `0.1.0`。原 Python `rapid-doc` 保持独立版本及现有
`publish_whl.yml`，本页仅描述 Rust 和语言绑定。

## 当前发布状态（2026-11-06）

| 渠道 | `0.1.0` 状态 | 使用入口 |
| --- | --- | --- |
| Python / PyPI | 四个平台的 CPU wheel 已发布 | `pip install "rapid-doc-rust[cpu]==0.1.0"` |
| Java / Maven Central | 主包、四个平台 JNI 和 ORT 1.23.2 CPU JAR 已发布 | `io.github.rapidai:rapid-doc:0.1.0` |
| Go / GitHub | 模块标签和四个平台 CPU 原生库 ZIP 已发布 | `go get github.com/RapidAI/RapidDoc/bindings/go@v0.1.0` |
| Rust / crates.io | 11 个 crate 均已发布，包括主包与 CLI | `cargo add rapiddoc@0.1.0` |
| Node.js / npm | Linux x64、macOS Intel/ARM64 平台包已发布；Windows 包等待 npm 审核，主包暂未发布 | 当前使用本地构建的 standalone `.tgz` |

四个平台构建、安装、CPU ONNX 推理及七格式跨绑定一致性均通过，见[验证任务](https://github.com/RapidAI/RapidDoc/actions/runs/42139545416)。
[SDK Release](https://github.com/RapidAI/RapidDoc/releases/tag/sdk-v0.1.0) 已公开，Go 模块代理可读取 `v0.1.0`。
Maven 首次发布任务曾等待超时，但服务端随后完成发布，公共仓库已可下载；不要因此重复上传。
CUDA 产物和实机验收不包含在本次四平台 CPU 发布中。

## 下一版本 ARM64 适配（尚未发布）

已在源码和构建流程中补充 Linux ARM64（glibc）与 Windows ARM64 CPU 目标，
覆盖 Python、Java、Go、Node.js 及 Rust 运行库下载。加上现有四个平台，共六个 CPU 组合。
这两个新增目标仍需提交后在 GitHub ARM64 runner 上完成编译、安装、ONNX 推理和七格式一致性验收。
Windows ARM64 使用原生 MSVC 构建 Rust/GEOS，Go cgo 单独使用 LLVM-MinGW；不提供 ARM64 CUDA 包。
Windows ARM64 的私有 CRT 接受 runner 自带的 VS 2022 14.44 或指定的 14.51，实际版本写入产物来源记录，兼容性仍待上述原生验收。
当前已发布的 `0.1.0` 平台范围保持上表所述。

下一版本的 Unix ORT 打包只提取上游真实库文件，避免把软链接展开成重复库；
Java、Go、Node 打包也会过滤旧目录中内容完全相同的 ORT 别名，保留 provider 和许可证。
本地 Linux x64 已验证单库加载、测试/OCR 检测/OCR 识别模型输出、七格式 Python/Node 输出一致性，
以及 npm 离线安装与自动平台加载。Linux Node 平台包实测从 61.9 MB 降到 45.3 MB（约 27%）。
macOS 已检查官方归档的真实文件提取，动态加载及新增 ARM64 整包仍需原生 CI 验收。
Java 瘦身包使用新的 Maven 打包版本 `1.23.2-rapiddoc.1`，依赖已同步更新；其中原生 ORT 仍为 `1.23.2`，已发布的旧 Maven 坐标不覆盖上传。

## 工作流入口

| Actions 名称 | 文件 | 默认行为 |
| --- | --- | --- |
| Build native SDK packages | `sdk-build.yml` | Windows/Linux/macOS 的 x64 和 ARM64 CPU 构建、安装和七格式跨绑定对照；不发布 |
| Publish native SDK (manual) | `publish-sdk.yml` | `artifacts-only`、`publish=false`，仅生成验证产物 |
| Check or publish Rust crates (manual) | `publish-crates.yml` | 生成 11 个源码包并验证解包编译，不上传 |

新版本的源码、资源、脚本和工作流提交推送后，先运行 **Build native SDK packages**，
不需要注册发布账号。工作流读取仓库文件，不会使用开发机 `build/` 和 `.venv`。
构建成功后产物保存在 Actions Artifacts，保留 14 天。

正式发布必须同时满足：所有版本一致、工作流运行的 Git ref 为 `sdk-v0.1.0`、
选择一个发布渠道，并明确勾选 `publish`。`sdk-v*` 与原 Python 的 `v*` 标签分离，
不会触发现有 Python 发布流程。此处没有自动创建版本标签，也没有 push/PR 发布触发器。

每次选择一个目标：`pypi`、`npm`、`maven` 或 `github`。凭据错误时，可重跑失败的
发布 job，保留同次运行已成功构建的 artifacts；不要改代码后覆盖已发布版本。
不同包仓库不支持跨仓库原子发布，某个平台成功不会因另一平台失败而撤销。

`publish-sdk.yml` 的 `source_run_id` 可复用成功的 `sdk-build.yml` 产物；留空才重新构建。
所选任务必须包含完整、未过期的平台产物，发布源码必须与构建源码一致（仅允许发布工作流自身不同）。
首次 CPU 产物对应任务 `42139545416`；过期后需重新构建。

`publish-crates.yml` 始终检出所选版本的 `sdk-vVERSION` 标签，工作流自身可从 `main` 运行。
`start_at` 可指定第一个尚未成功的 crate，避免重传已发布版本。
新建 crate 遇到 `429` 时会等待服务端返回的时间再试，短暂连接重置最多重试三次。
不要把仍在自动等待的任务取消后反复重开。

## 账号和 Secrets

在仓库 Settings → Secrets and variables → Actions 配置以下名称，也可放在对应
GitHub Environment。创建环境后建议限制可发布的 `sdk-v*` 标签，并配置审核人。

| 目标 | Environment | Secrets / 外部设置 |
| --- | --- | --- |
| PyPI | `sdk-pypi` | 优先使用专用 `PYPI_RUST_TOKEN`，未配置时复用 `RAPID_DOC`；两者均未配置时使用 PyPI Trusted Publisher：仓库 `RapidAI/RapidDoc`，workflow `publish-sdk.yml`，environment `sdk-pypi` |
| npm | `sdk-npm` | 各包配置同一 workflow/environment 的 Trusted Publisher；首次创建包可使用具有相应包权限的 `NPM_TOKEN` |
| Maven Central | `sdk-maven` | `MAVEN_CENTRAL_USERNAME`、`MAVEN_CENTRAL_PASSWORD`（Portal Token 生成的凭据）、`MAVEN_GPG_PRIVATE_KEY`（ASCII armored）、`MAVEN_GPG_PASSPHRASE` |
| crates.io | `sdk-crates` | `CARGO_REGISTRY_TOKEN`（新建及更新 crate 权限） |
| GitHub Releases / Go | `sdk-github` | 自动提供的 `GITHUB_TOKEN`，不手填 |

crates.io 发布账号还需完成邮箱验证。npm 首次 token 发布需要相应包的写入权限及 Bypass 2FA，或使用已配置的 Trusted Publisher。
Maven 还需在 Central Portal 验证 `io.github.rapidai` 命名空间，并公开 GPG 公钥。
保留原 Python 使用的 `RAPID_DOC` Secret。新工作流可复用其中的账号级 PyPI Token，无须重复配置；原 Python 发布流程不变。

凭据不进入构建 job，也不写入源码或 artifacts。构建不需要这些发布 Secrets。
参考：[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/using-a-publisher/)、
[npm Trusted Publishing](https://docs.npmjs.com/trusted-publishers/)、
[Central Portal API](https://central.sonatype.org/publish/publish-portal-api/)。

## 平台和运行库

默认构建 Windows x64 / Linux glibc x64 / macOS Intel x64 和 Apple Silicon ARM64 CPU 包，macOS 最低目标为 14。原生代码保留 CUDA 能力，但
CPU 产物不携带 GPU ORT 和自定义 CUDA 算子，不能当作完整 GPU 部署包。
Python wheel 不捆绑 ORT，仍由 `[cpu]`、`[gpu]` 或显式运行库提供；构建安装检查
只通过隔离运行库目录供给 ORT，不依赖 Python `onnxruntime` 模块。

ORT 固定为 1.23.2，下载官方归档并校验已记录 SHA256；PDFium 来自
`pypdfium2==5.10.1`，GEOS 来自 `shapely==2.1.2`，保留原始许可与匹配源码。
Linux wheel 使用 auditwheel 根据实际依赖确定兼容标签；本次为 `manylinux_2_35_x86_64`（glibc 2.35+）。
macOS wheel 使用 delocate 检查和修复依赖，发布标签为 `macosx_14_0_x86_64` / `macosx_14_0_arm64`；实际安装验证在 macOS 15 runner 上运行。
不能仅修改文件名降低运行库的系统要求。

Windows 私有 CRT 仍执行现有 **14.51 x64** 检查。托管 runner 若没有该版本，
需要提供合规的 Visual Studio redist 路径；可设置 Actions Variable
`WINDOWS_CRT_REDIST_DIR`，不接受 System32 作为分发来源，也不跳过兼容检查。

### CUDA 可选构建

默认关闭 `include_cuda`。启用后额外需要两台专用 self-hosted runner：

- Linux：`self-hosted, Linux, X64, rapiddoc-cuda`
- Windows：`self-hosted, Windows, X64, rapiddoc-cuda`

工作流不会自动注册或改动现有 GPU 业务服务器。应使用隔离 runner 工作目录，
不能把业务部署目录作为 Actions checkout。

配置 Variables：`CUDA_OPERATORS_LINUX`、`CUDA_OPERATORS_WINDOWS`，值是相应
runner 上审核过的自定义算子 bundle 目录。目录内必须有 `manifest.json`，例如：

```json
{
  "platform": "linux-x86_64-cuda",
  "files": {
    "libonnxruntime_ops.so": "文件的完整 SHA256",
    "licenses/THIRD_PARTY_NOTICES.txt": "文件的完整 SHA256",
    "licenses/third-party-notices.json": "文件的完整 SHA256"
  }
}
```

Windows 对应 DLL 名为 `onnxruntime_ops.dll`。本工作流校验、复制算子库，
不从未固定版本的网络位置下载算子，不把它冒充成新编译结果。
打包后的基本安装测试使用 CPU；实机 CUDA 推理、性能和精度验收是单独阶段。
未配置这些 runner 时不要勾选，否则相应任务会等待 runner。

## 各语言产物

- **Python**：`rapid-doc-rust`。每个 OS 只上传一个 wheel；启用 CUDA 构建时选择
  包含自定义算子的版本，不把两个同名 wheel 同时上传。PyPI 不上传当前缺少
  独立构建运行库准备流程的 sdist。
- **npm**：公开入口仍是 `rapid-doc`；CPU 包作为精确版本的 optionalDependencies：
  `@hzkitty/rapid-doc-win32-x64`、`rapid-doc-linux-x64-gnu`、`rapid-doc-darwin-x64`、`rapid-doc-darwin-arm64`、`rapid-doc-win32-arm64`、`rapid-doc-linux-arm64-gnu`。Windows x64/Linux x64 CUDA 包分别增加 `-cuda` 后缀（Windows x64 保留 `@hzkitty/` scope），
  用户显式安装并通过 `config.device='cuda'` 选择，不给所有 CPU 用户下载 GPU 库。
  standalone tarball 的原有 `runtime/` 查找方式保留。发布平台包后才发布入口包，
  正式版本 npm dist-tag 使用 `latest`；包名所有权需发布前确认。
- **Java**：一个 JVM SDK、平台 JNI classifier JAR、独立 ORT classifier JAR；
  生成 sources/javadoc、签名及校验文件后，按 Maven 仓库目录结构一次提交 Central。
  如果超时，先查看 receipt 中的 deployment ID，避免重复提交相同版本。
- **Go / C ABI**：工作流创建 GitHub draft prerelease，携带平台原生 SDK ZIP，并创建
  `bindings/go/v0.1.0` 标签。Go 源码版本可被模块代理读取；使用者仍需按 Go README
  提供对应原生库。确认资产和说明后发布该草稿；`0.1.0` 已完成此步骤。Go 包不会自动下载原生库。
- **Rust crates**：11 个包按依赖顺序发布，包括独立命名的
  `rapiddoc-docvortex-core`、`rapiddoc-docvortex-pdfium-paths`、`rapiddoc-lopdf`。
  本地修改仍保留，不替换成未修改的上游实现。内部依赖同时声明路径和精确版本，
  消费者安装时使用 crates.io 版本。
  `python scripts/release/crates.py` 默认只打包验证，使用 Cargo 多包临时 registry
  验证尚未发布的依赖；本地未提交时可加 `--allow-dirty`。
  显式 `--publish` 才会上传，并要求干净 checkout、匹配的 SDK tag 和 Token。
  包内保留运行所需 JSON、字体、模型及许可证，仓库中的回归夹具不作为生产依赖。
  `ready` 表示静态依赖检查通过，`verified` 表示解包编译通过；实际名称所有权和
  上传权限仍由 crates.io 发布时校验。

## 验证范围

本机 helper 测试及工作流语法检查通过；GitHub Actions 已完成四个平台的实际构建与安装验证。
Python、Maven、Go/GitHub、Rust 已公开发布，npm 的部分发布状态见本页顶部。
CI 七格式检查比较同平台 Python/Java/Go/Node 的 Markdown、middle JSON、
content list 和图片哈希，避免把 Windows/Linux 渲染差异误判为绑定差异。
它不替代完整 OCR/版面/公式/表格精度、速度和 CUDA 实机验收。

## 发布平台限制

- 原 npm 包名 `rapid-doc-win32-x64` 被包名反垃圾检测拒绝，申诉可继续保留。下一次构建改用 `@hzkitty/rapid-doc-win32-x64`，主包名和其他平台包名不变。发布前确认 Token 有该 scope 下包的写入权限；脚本已使用 `--access public`。改名需要重新构建平台包及主包，不能直接重用旧名称的 tarball；scope 仍可能受到反垃圾审核。
- crates.io 新建 crate 可能限流，按返回的重试时间等待；已发布版本不可覆盖。
- Maven Central 按组织统计月度发布大小、文件数和版本数。首次上传的 JAR 约 213 MB，其中原生解析库约 139 MB、ORT 约 73 MB。开源多平台项目可以申请提额或豁免，具体额度查看 Usage Center。后续应复用未变化的 ORT 版本，避免重复上传已发布坐标。参见[官方限制与豁免说明](https://central.sonatype.org/publish/maven-central-publishing-limits/)。
