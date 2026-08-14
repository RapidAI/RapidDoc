# RapidDoc 大 Excel OOM 修复案例

## 背景

RapidDoc v0.9.9 在处理大 Excel 文件（含 33MB XML sheet）时频繁 OOM 崩溃。问题出现在生产环境（Docker 容器，内存限制 10GB），导致服务不可用。

## 问题现象

- 输入：9.55MB xlsx 文件，内部含 1 个 33MB XML 的 sheet（74 万个单元格，18 万行合并区域）
- 现象：Python 进程 RSS 达到 12GB，撞上 Docker 内存硬限制，被 cgroup OOM Kill
- 影响：容器崩溃重启，请求失败

## 根因分析

### 根因 1：openpyxl 全量加载百万级 Cell 对象

`load_workbook()` 为 sheet 中每个单元格（含空位）创建 openpyxl Cell 对象。33MB XML 含 74 万个单元格，单个 Cell 对象含样式、字体等元数据，百万级对象占用约 12GB。

### 根因 2：异常文件的合并区域定义到了 18 万行

该 Excel 文件在 18 万行处定义了大量合并单元格区域（18.6 万个），但实际数据只在 1-38 行。openpyxl 加载这些合并区域进一步放大内存占用。

### 根因 3：表格识别的洪水填充遍历过大范围

`_find_true_data_bounds` 将合并区域纳入边界计算，导致 `max_row=186365`，洪水填充算法在此范围内遍历 74 万个空单元格。

## 解决方案

### 核心思路：SAX 流式解析替代 openpyxl 全量加载

对大 sheet XML（>5MB）使用 `xml.etree.ElementTree.iterparse` 流式解析，构建轻量级数据结构替代 openpyxl Worksheet，内存从 O(百万级 Python 对象) 降至 O(非空单元格 dict)。

### 新增轻量级数据结构

```python
class _LightCell:
    """轻量级单元格，仅存储值，替代 openpyxl Cell。"""
    __slots__ = ("row", "column", "value", "font", "alignment", "fill", "hyperlink")

class _LightweightSheet:
    """轻量级工作表，用 dict 存储单元格值，替代 openpyxl Worksheet。"""
    # 仅存储非空单元格，不创建空 Cell 对象
```

### 修复的 6 个 Bug

#### Bug 1：`_parse_shared_strings` tag 比较未带命名空间

**根因**：`iterparse` 返回的 `el.tag` 是带命名空间的完整 tag（`{http://...}t`），但代码比较的是不带命名空间的 `"t"`，导致 sst 始终为空，所有共享字符串类型的单元格值丢失。

```python
# 修复前：永远不匹配
if tag == "t":
    sst.append(el.text or "")

# 修复后：正确匹配带命名空间的 tag
if el.tag != f"{{{NS}}}si":
    continue
texts = "".join(t.text or "" for t in el.findall(f".//{{{NS}}}t"))
sst.append(texts)
```

**注意**：不能对 `<t>` 等子元素调用 `el.clear()`，否则 `<si>` end 事件时 text 已丢失。

#### Bug 2：`_find_tables_in_sheet` 不支持轻量级 sheet

**根因**：`_find_tables_in_sheet` 依赖 `self.workbook is not None` 才执行表格识别，但轻量级路径没加载 workbook，导致表格识别被跳过，输出空内容。

```python
# 修复前：轻量级路径 workbook=None，跳过表格识别
if self.workbook is not None:

# 修复后：轻量级 sheet 也可识别表格
if isinstance(sheet, _LightweightSheet) or self.workbook is not None:
```

#### Bug 3：`_find_true_data_bounds` 合并区域拉大边界

**根因**：合并区域纳入整体边界计算，18 万行的合并区域导致 `max_row=186365`，洪水填充遍历 74 万空单元格。

```python
# 修复前：合并区域纳入边界，max_row=186365
for merged in sheet.merged_cells.ranges:
    max_row = max(max_row, merged.max_row)

# 修复后：合并区域不参与边界，max_row=38（实际数据行）
# 仅遍历有值的单元格确定边界
for cell in sheet._cells.values():
    if cell.value is not None:
        max_row = max(max_row, cell.row)
```

#### Bug 4：连续空行截断

**根因**：异常文件含 18 万行空行，SAX 全量遍历浪费时间。

```python
# 遇到连续 10 行无非空 cell 则停止解析
EMPTY_ROW_CUTOFF = 10
if current_row > 0 and not row_has_value:
    if current_row - last_nonempty_row >= EMPTY_ROW_CUTOFF:
        stop_parsing = True
```

#### Bug 5：合并区域按数据边界过滤

**根因**：18.6 万个合并区域全加载进内存，其中 18.6 万个在远离数据的 18 万行处。

```python
# 合并区域超过数据边界 + 10 行的直接丢弃
merged_row_limit = data_max_row + EMPTY_ROW_CUTOFF
if min_row > merged_row_limit:
    continue  # 18.6 万 -> 115 个
```

#### Bug 6：`_find_images_in_sheet` 不支持轻量级 sheet

**根因**：`_find_images_in_sheet` 依赖 `self.workbook is not None` 才输出图片，轻量级路径下图片被跳过，导致 63 张图片只输出 26 张（表格内图片通过 table_image_map 输出，独立图片丢失）。

```python
# 修复前：轻量级路径跳过图片输出
if self.workbook is not None:

# 修复后：检查 sheet_images 是否非空
if isinstance(self.sheet_images, list) and self.sheet_images:
```

### Docker 部署优化

#### 问题：镜像构建相关

1. **pip 源慢**：改用清华源 `https://pypi.tuna.tsinghua.edu.cn/simple`
2. **镜像内权重重复**：`COPY rapid_doc/` 把 809MB 权重拷进镜像，但 `download_models.py` 又下载一份到 `/app/models`。通过 `.dockerignore` 排除 `layout/table/formula` 的权重（这些会被 `download_models.py` 重新下载），保留 `resources/` 下的 OCR 权重和 magika 权重（不在下载列表）
3. **compose 未构建**：`docker-compose.yml` 只有 `image:` 没有 `build:`，导致 `docker-compose up -d` 直接用官方镜像，本地修改的代码没进容器。添加 `build:` 指令

#### 内存限制

```yaml
# docker-compose.yml
mem_limit: 10g        # 物理内存 10GB
memswap_limit: 30g    # 物理+swap 总上限 30GB（swap 上限 20GB）
```

## 修改文件清单

本案例基于官方仓库 [RapidAI/RapidDoc](https://github.com/RapidAI/RapidDoc) 的 v0.9.9 版本（commit `3402968`）二开。

### 修改的文件（7 个）

| 文件 | 改动 | 说明 |
|------|------|------|
| `rapid_doc/model/xlsx/xlsx_converter.py` | +631 -100 | 核心：新增轻量级解析路径，修复 6 个 bug |
| `rapid_doc/model/xlsx/main.py` | +2 -2 | 透传 gap_tolerance 参数 |
| `rapid_doc/backend/office/office_analyze.py` | +6 -1 | 透传 gap_tolerance 参数 |
| `rapid_doc/cli/common.py` | +4 -0 | 透传 gap_tolerance 参数 |
| `docker/app.py` | +2 -0 | API 新增 gap_tolerance 表单参数 |
| `docker/Dockerfile` | +4 -2 | pip 改清华源 + COPY rapid_doc 覆盖 |
| `docker/docker-compose.yml` | +6 -0 | 添加 build 指令 + 内存限制 |

### 新增的文件（3 个）

| 文件 | 说明 |
|------|------|
| `.dockerignore` | 排除权重文件避免镜像内重复（809MB -> 0） |
| `CHANGELOG.md` | 项目变更记录 |
| `docs/大Excel_OOM修复案例.md` | 本文档 |
| `docs/大Excel_OOM修复案例-代码diff.md` | 所有修改文件的完整 diff（相对官方 v0.9.9） |

### 完整代码 diff

所有文件的改前改后完整对比见 [`大Excel_OOM修复案例-代码diff.md`](./大Excel_OOM修复案例-代码diff.md)，包含 8 个文件的完整 diff，可直接用于 patch 或代码审查。

## 验证结果

### 测试环境

- 本地：macOS arm64，Docker Desktop，容器内存限制 10GB
- 服务器：Linux x86_64，容器内存限制 10GB

### 测试文件

6 个文件（PDF/PPTX/XLSX），含 1 个大 Excel（9.55MB，内部 33MB XML sheet）

### 性能对比

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| 大 Excel 处理 | OOM 崩溃 | 16.1s 完成 |
| 内存峰值 | 12GB+ (OOM) | 2.2GB |
| 大 Excel MD 输出 | 0B | 36,600B |
| 大 Excel 图片输出 | 0 张 | 63 张（完整） |
| 镜像大小 | 5.21GB | 3.72GB |
| 容器稳定性 | 崩溃重启 | 稳定存活 |

### 完整测试结果（服务器）

| 文件 | 大小 | 耗时 | MD | 图片 |
|------|------|------|-----|------|
| 0721Tims随行果冻包买赠活动信息.pdf | 0.29MB | 10.3s | 4,665B | 3 |
| 0721档期企划物料陈列标准.xlsx | 9.55MB | 16.1s | 36,600B | 63 |
| 2026年SUM Wave6 档期新品通知.pdf | 1.98MB | 20.8s | 8,492B | 11 |
| TIMS-新品物料清单-0804.xlsx | 0.01MB | 0.1s | 2,453B | 0 |
| Tims咖啡2026年热烤小食套餐新品上市通知.pptx | 8.58MB | 13.4s | 1,362B | 7 |
| 效期表20260804-SUM Wave6.xlsx | 0.04MB | 0.6s | 66,811B | 0 |

## 关键经验

### 1. XML 命名空间陷阱

`iterparse` 返回的 `el.tag` 是带命名空间的完整 tag（`{http://schemas...}t`），不是简单的 `"t"`。比较时必须带命名空间，否则永远不匹配。

### 2. `el.clear()` 的副作用

`iterparse` 的 `el.clear()` 会清空子元素的 text。不能对需要保留数据的子元素调用 clear，否则父元素 end 事件时子元素 text 已丢失。

### 3. 轻量级路径的兼容性

替换 openpyxl 时，所有依赖 `self.workbook is not None` 的判断都要检查是否需要兼容轻量级 sheet。本次修复了 `_find_tables_in_sheet` 和 `_find_images_in_sheet` 两处。

### 4. 异常文件的防御性处理

生产环境的 Excel 文件可能格式异常（如 18 万行空合并区域）。解析时需要：
- 按实际数据确定边界，不盲目信任文件的维度声明
- 连续空行截断，避免遍历无效区域
- 合并区域按数据边界过滤，丢弃远离数据的异常区域

### 5. Docker 部署的镜像构建

`docker-compose.yml` 里只有 `image:` 没有 `build:` 时，`docker-compose up -d` 会直接拉取官方镜像，不会构建本地 Dockerfile。必须显式添加 `build:` 指令或使用 `docker-compose up -d --build`。

## 部署步骤

```bash
# 1. 解压覆盖到项目根目录
cd /path/to/RapidDoc
tar xzf rapiddoc-patch.tar.gz

# 2. 重建镜像并启动
cd docker
docker-compose down
docker-compose build
docker-compose up -d

# 3. 验证容器内代码版本
docker exec rapid-doc-server grep -c "EMPTY_ROW_CUTOFF" /app/rapid_doc/model/xlsx/xlsx_converter.py
# 应返回 5

# 4. 健康检查
curl http://localhost:8888/health
```

## 调用方建议

OOM 修复后，轻量级路径能直接处理大文件，**无需调用方拆分 Excel**。直接传完整 xlsx 即可，避免拆分导致的图片丢失和额外开销。
