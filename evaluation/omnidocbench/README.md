# OmniDocBench 评测

从仓库根目录启动脚本。

| 脚本 | 用途 |
| --- | --- |
| `RapidDoc_end2end.py` | 用原生 Python 生成预测 |
| `RapidDoc_rust_end2end.py` | 用 Rust Python 绑定生成预测 |
| `RapidDoc_table_ab_eval.py` | 在固定 PDF 子集上比较表格配置 |
| `omnidocbench-eval.sh` | 按脚本内的目录配置运行 OmniDocBench 评分 |
| `generate_result_tables.py` | 从评分 JSON 汇总指标并生成 CSV |
| `review/review_eval.py` | 查看并对比历史评测结果 |

运行全量预测前，在对应脚本内手动修改输入、输出目录和模型配置。运行评分前，在 `omnidocbench-eval.sh` 内修改标注、预测和评分输出的挂载路径。运行汇总前，修改 `generate_result_tables.py` 的评分目录和 CSV 文件名。

已有的指标汇总 CSV 和新生成的汇总都放在 `results/`。原始 PDF、预测 Markdown、图片、评分明细 JSON 和日志放在外部评测数据目录。核对工具沿用 `pdfs`、`<预测目录名>` 和 `result_<预测目录名>` 的目录约定，详见[评测核对页面](../../docs/evaluation-review.md)。
