# SDK HTTP 输入统一

HTTP/HTTPS 下载集中到 `rust/rapiddoc/src/sdk/input.rs`。Rust、JNI、C ABI、Node 和 Python 绑定共用下载实现。没有更改识别模型、阈值或解析算法。

## 调用

- Python：`engine(url)`，也支持路径、字节、二进制流及列表。
- Java：`engine.parse(url)`，直接传字符串；支持 Path、byte[]、InputStream 及路径/URL 列表。
- Go：`engine.Parse(url, options)`，沿用 ParseBytes、ParseReader 和 ParseAll。
- Node：`await engine.parse(url)`，也支持 URL 对象、文件路径、Buffer、Uint8Array 和列表。
- Rust：`engine.parse(url, &options)`，字节仍使用 parse_bytes。

上述公共文档格式为 PDF、DOC、DOCX、XLS、XLSX、PPT、PPTX。Python 已有图片转 PDF 适配继续保留，不能据此宣称所有语言支持完全相同的输入类型。URL 不要经过 Path/Paths.get/path.resolve，也不要拼接 ROOT。

## 行为

- 跟随重定向，非成功 HTTP 状态和网络异常向调用方报告。
- 文件名优先采用 filename 查询参数，否则采用 URL 路径末段；解码并移除目录及非法文件名字符，缺省 document。
- 连接、响应和每次读取超时各 60 秒。内容在内存读取，不写磁盘下载缓存，不新增默认文档大小上限。
- 每次下载独立客户端，请求 Connection: close，避免无正文重定向响应的连接复用异常。
- 下载开始、读取进度、完成分别产生 source_download_started/progress/completed 事件。读取事件最多每 100 毫秒一次；回调异常可取消后续下载。Python 兼容层当前显示开始/完成日志。
- 无 Content-Disposition 文件名、自定义请求头/鉴权配置；这些仍非本次公共输入 API 的能力。

## 验证范围

本机 CPU：七格式 URL/字节输出对照、重定向、查询文件名、404、取消、超过 10 MiB 下载；绑定生命周期和并发相关回归。用户提供的实际 HTTPS 地址另做下载和 Java 首页面解析检查。未做 CUDA 或全量文档效果回归。

本地交付必须同时更新 Python .venv、Java 两个 Maven 仓库及 Go/Node 示例的原生包；源码修改不等于已安装包自动更新。无远端发布。

实测用户 HTTPS PDF 首页面在 V3 流程成功（80 个 Markdown 字符）。同文件纯规则版面会触发 `processing/text/source_lines.rs` 的旧数组容量限制；这是独立遗留问题，本次没有通过换阈值或改解析算法掩盖。默认模型流程的通过不能视为纯规则流程也通过。
