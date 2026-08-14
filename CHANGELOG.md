# CHANGELOG

## v0.9.9-oom-fix (2026-07-27)

### 已完成

- 修复大 Excel（33MB XML sheet）OOM 崩溃问题
- 修复轻量级解析路径下 sst 为空导致单元格值丢失
- 修复轻量级 sheet 表格不识别（输出空内容）
- 修复洪水填充在 18 万行合并区域上 OOM
- 修复轻量级 sheet 图片不输出（63 张图片丢失）
- 优化镜像构建：pip 改清华源、.dockerignore 排除重复权重
- 优化 docker-compose：添加 build 指令 + 内存限制（10g/swap 20g）
- 新增连续 10 行空行截断 + 合并区域按数据边界过滤

### 性能指标

- 大 Excel 处理：OOM 崩溃 -> 16.1s 完成
- 内存峰值：12GB+ (OOM) -> 2.2GB
- 图片输出：0 张 -> 63 张（完整）
- 镜像大小：5.21GB -> 3.72GB

### 修改文件

- `rapid_doc/model/xlsx/xlsx_converter.py` - 核心：轻量级 SAX 解析 + 6 个 bug 修复
- `rapid_doc/model/xlsx/main.py` - 透传 gap_tolerance
- `rapid_doc/backend/office/office_analyze.py` - 透传 gap_tolerance
- `rapid_doc/cli/common.py` - 透传 gap_tolerance
- `docker/app.py` - API 新增 gap_tolerance 参数
- `docker/Dockerfile` - pip 清华源 + COPY rapid_doc 覆盖
- `docker/docker-compose.yml` - build 指令 + 内存限制
- `.dockerignore` - 新增，排除重复权重

### 详细文档

见 `docs/大Excel_OOM修复案例.md`

## v0.9.9

- 基于 mineru 2.1.11 版本二开
- 移除 vlm，专注于 pipeline 产线下 CPU 高速识别
- ocr 模型更换为 rapid_ocr
- layout 模型更换为 PP-DocLayout 系列 onnx 模型
- 公式模型更换为 PP_FORMULANET_PLUS 系列 onnx 模型
