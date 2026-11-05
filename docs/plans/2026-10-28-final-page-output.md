# 完整最终页接续

已统一构建安装：release 7 分 43 秒，安装后的 211 组最终页、35 条原始页源链、1,458 规则及 3 页实时 PDFium 完整输出通过；源码 298 XYCut 与 7 项相关测试通过。日志为 `build/native-final-page-output-*.txt`。来源审计和相关回归见主链最新记录。

`prepare_native_final_page_blocks(..., finalize_output=True)` 与全文同名开关，继续执行现有 Python `_finalize_prepared_page` 的全部剩余语句，额外返回归一化 `model_list` 和已有 `refined_text_blocks`。旧默认检查点保留。

| Python 逻辑 | Rust 对应 |
| --- | --- |
| `recover_image_annotation_bands`：分离图注、物理行、多栏符号图例及父图身份 | `native_visual_annotations/bands.rs` |
| 候选、图像组件父区域、表格面板、方向与父区域仲裁 | `candidates.rs`、`relations.rs` |
| 双语、跨栏续注与表格来源注续块 | `continuations.rs` |
| 区域成员扩展及内部顺序 | `regions.rs` |
| 完整原 XYCut++：交叠项、密度、间隙、窄项重试和重插 | `xycut.rs` |
| 视觉区域、行内图片、拆分行与页边分组 | `native_grouped_order.rs` |
| 编号子图、平行标题面板及编号步骤 | `native_grouped_order/panels.rs` |
| 同栏轻微重叠稳定、页首标题/共享流、公式编号 | `tail.rs` |
| tight、窄图形栏与文献上方正文、负参考组、归一化 | `native_final_output.rs`、`native_finalization.rs` |

顺序为合流→图注带→tight→视觉绑定→完整分组排序→窄图形栏→文献上方正文→负参考组稳定→输出归一化。保留原值相等删除与对象身份区别、稳定同分、CPython 集合轨迹、图例父图映射、旋转及千分位舍入。未另写简化算法。

参考来源是未修改的当前 vendored Python。冻结参考包括 686 视觉候选/绑定、363 图注带、384 完整分组排序、298 完整 XYCut、25 Unicode/非字符串内容合同；211 完整最终页及 35 原始 source/geometry 文档链执行整个原 `_finalize_prepared_page`。样本复用，不累加为独立文档数。安装结果及日志以主链最新记录为准。

此节点完成的是逐页最终输出。外层仍需按 `_materialize_document_inline` 顺序接入 spacing→链接/样式/上下标证据，完成图像/表格媒体物化接线，再接默认 PDF 文件入口，进行用户整篇质量与同条件耗时/峰值内存验收；不据局部通过提升这些标志。行内及媒体已有独立组件，接线仍必须保留原空格/投影规则和调用顺序。无全量或 CUDA；enhanced 阅读顺序继续后置。
