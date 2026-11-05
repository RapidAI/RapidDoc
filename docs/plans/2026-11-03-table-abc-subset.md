# 三组表格配置子集对照

## 固定输入和参数

- 三次既有评测低分页合并去重后的 385 份 PDF；来源清单及阈值明细在 `D:/Download/OmniDocBench/rust_low_score_union_20261103/`。
- `OmniDocBench_subset.json` 仅包含相同 385 页标注；最终评分必须使用此子集，不能搭配 1651 页全量标注。
- 运行原生 Rust CUDA 链路；V3 版面、Formula M，公式/表格开启，整页方向分类关闭；新表格区域方向纠正策略在全部组别相同。
- 表格文字回退关闭，缺字触发阈值 `0.2`；其他参数保持默认。

| 组 | 主模型 | 规则 | 备用模型 |
| --- | --- | --- | --- |
| A | unet_slanet_plus | docvortex | None |
| B | unet_slanet_plus | docvortex | table_transformer |
| C | unet_slanet_plus | docvortex_ocr | table_transformer |

A→B 检查模型备用收益；B→C 检查 OCR 规则收益。问题集是人为筛出的低分子集，分数不能和既有全量 Overall 直接比较。

## 实现与运行

复用现有服务器源码、Cargo target、运行库和模型目录，只同步发生变化的文件；不创建新 SSH 同步工具，不清空构建缓存，不改原全量评测脚本。

正式批测入口 `demo/RapidDoc_table_ab_eval.py`：固定清单，三组分别启动进程且串行执行，独立保存 Markdown、图片、控制台日志、错误日志、逐文件耗时与阶段汇总。完成的 Markdown 原子写入，原清单相同的任务可重跑续接。逐文件异常单独记录并继续下一份；组别非零退出仍会尝试后续组别。

服务器目录：

```text
/web/hzkitty/rapiddoc-cuda-validation/20261101/table-abc-20261103/
  run.sh                 # 构建、安装、CUDA 单页检查、启动三组解析
  driver.log             # 构建/安装/检查日志
  pdf_files.txt          # 固定 385 页清单
  results/
    manifest.json        # 清单哈希与固定配置
    A/ B/ C/             # 各组预测
    A.console.log ...    # 各组完整日志
    A.errors.log ...     # 错误日志
    A.files.jsonl ...    # 逐文件结果与耗时
    A.summary.json ...  # 最新进度
    status.json          # 已结束组别退出状态
```

环境沿用 `release-acceptance/linux-cuda/env-delivery/bin/python`，源码位于既有 `source/`。后台任务使用 `nohup` 脱离 SSH 会话，断开客户端不应终止解析。不停止服务器原有 GPU 服务。

服务器没有既有 OmniDocBench 评分容器；本机已有验证过的评分镜像。预测生成与评分是两步，生成完成后取回 A/B/C 的 Markdown，以固定 385 页标注按相同评分配置顺序评分，再统计逐页差异和耗时。没有宣称启动解析等于完成评分。
