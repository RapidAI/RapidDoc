# 迁移基线语料

当前清单包含 23 份文档：六种 Office 自生成烟雾样本、14 份恢复的历史 Office 样本，以及原生/扫描/混合 PDF。它们不代表完整兼容性。PDF 当前使用 `capture: pdf-facts` 导出字符、路径、线段、图片框与链接；扫描页只验证图片事实，不代表 OCR 已通过。历史样本的来源和哈希见 `tests/fixtures/provenance.json`，另有一份字符去重 JSON 片段由原单测覆盖。

在仓库根目录运行，使用现有 `.venv`（生成器额外需要 docx、pptx、openpyxl、Pillow；当前版本见 manifest 的 generator_packages，缺少发行元数据的项为 null）：

```powershell
.venv\Scripts\python.exe scripts/migration/generate_corpus.py
.venv\Scripts\python.exe scripts/migration/inventory.py --check
.venv\Scripts\python.exe scripts/migration/export_baseline.py --manifest tests/migration/corpus_manifest.json --output build/migration/new-baseline
.venv\Scripts\python.exe scripts/migration/compare_results.py --manifest tests/migration/corpus_manifest.json --expected tests/migration/golden --actual build/migration/new-baseline
.venv\Scripts\python.exe -m pytest tests/migration -q
.venv\Scripts\python.exe scripts/bench/parse_bench.py --engine python --manifest tests/migration/corpus_manifest.json --warmup 1 --repeat 5 --output build/bench/python.json
```

生成器会更新 inputs 与 manifest；内容相同的输入不重写，历史样本不重写，已有 golden 不自动重写。先审查差异再人工接受新基线。输出目录已有同名结果时导出器拒绝覆盖。比较器不做语义归一化；字符串枚举按 JSON 编码转换，PDF Bbox 明确保存为四个坐标。

清单以仓库根目录解析输入路径，检查 SHA256、唯一 ID 和输出名；缺文件直接失败。`inventory --check` 仅证明登记语料可用；`--require-complete` 还检查 UPSTREAM 历史引用与迁移目录，当前应失败。它不证明计划中的全部场景已覆盖。

基准记录 read/init/parse 三段、重复输出哈希和进程生命周期峰值 RSS。导入与 JSON 序列化不计入 parse，多个样本共享进程及缓存，不提供每样本独立峰值。尚无内部 PDF/模型阶段 profile 或 CUDA 数据。`--engine rust` 在实现前显式报错。

golden 资产使用 base64 保存完整字节，另带 SHA256/大小；environment.json 记录首批导出环境，expanded_environment.json 记录新增样本导出环境。模型目录清单记录配置声明，不代表模型已安装、许可证已审核或推理已验证。关键结果见 docs/plans/rust-migration-status.md。
