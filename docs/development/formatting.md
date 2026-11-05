# 源码格式化

在项目根目录运行：

```powershell
.venv/Scripts/python.exe scripts/format_code.py
.venv/Scripts/python.exe scripts/format_code.py --check
```

第一条命令修改格式，第二条只检查。脚本调用现有格式化工具，不生成或重写解析实现。

| 范围 | 工具与规则 |
| --- | --- |
| 自有 Rust、两种绑定的 Rust、迁移 Rust 适配器 | rustfmt，100 列，不排序导入或模块 |
| Python 绑定及其测试 | Ruff formatter，100 列、4 空格、双引号 |
| Java 源码及测试 | google-java-format 1.24.0，AOSP 4 空格，保留导入和长字符串 |
| 手写 Cargo/pyproject/格式配置 | Taplo，100 列，不排序键 |
| Java POM | Python 标准库 XML 格式化，2 空格 |

配置保存在 `.editorconfig`、`.rustfmt.toml`、`.taplo.toml` 和 `bindings/python/ruff.toml`。脚本不运行 lint 自动修复，不修改依赖版本、模型或算法。

需要 rustfmt、Ruff 和运行格式化工具的 JDK 17+；Java SDK 的编译目标仍是 Java 8。可通过 `--java <java.exe>` 指定格式化工具使用的 JVM。Java 格式化工具和 Windows Taplo 首次使用时从官方分发地址下载到 `build/tools`，并校验固定 SHA-256。其他系统需安装 Taplo 0.10.0。Ruff 不作为解析器运行依赖安装。

第三方 vendor、运行库目录、生成的 Unicode 数据表、模型、许可归档和 Cargo.lock 不在格式化范围。XML 配置文本、Python 语法树、Java 有效标记及 Rust 字面量已在本次整理中与格式化前快照核对；手写 TOML 的解析结果也保持一致。

本次全工作区检查发现，未纳入格式化的 `experiments/onnx-probe/src/bin/unet_post_benchmark.rs` 仍调用已不存在的 `table_recover::recover_legacy`。正式 crates 的检查排除了实验程序；该旧测速脚本需另行维护，不在格式化中修改。
