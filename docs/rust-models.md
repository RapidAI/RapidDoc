# Rust 模型下载与目录

模型清单位于 `rust/rapiddoc-inference/models.json`，编译进 Rust 公共推理库。
模型查找、流式下载、SHA256 校验及失败清理由 Rust 完成，Python 仅传递参数。

默认原始模型目录为 `~/.cache/rapid_doc/models`。需要共享已有模型时设置：

```python
import os
os.environ['RAPID_MODELS_DIR'] = r'D:\CodeProjects\doc\RapidAI\models'
from rapid_doc_rust import RapidDoc
```

也可设置 `RAPIDDOC_MODEL_DIR`；前者优先。查找范围仅限指定目录，支持其下原有模型子目录。
不再扫描 Python 包、项目源码及构建目录。显式传入的模型路径仍优先，支持自定义权重。
已有同名文件必须通过校验才会复用；下载到独立临时文件，校验成功后发布，失败清理临时文件。
不生成优化后的 ONNX 磁盘缓存。

清单采用 `schema_version / defaults / models`：

- `models` 按具体模型 ID 区分模型，每项只保留 `filename`、`url` 和 `sha256`。
- `defaults` 将推理角色映射到模型 ID；默认仍为 V3 版面、M 公式及原有表格组合。
- 原有 10 个默认模型的文件名、ModelScope 地址、SHA256 均保持不变。
- V2、V3 版面，公式 S/M/L，以及 TATR 表格结构识别支持通过兼容参数切换。Plus-L 支持下载和独立检测，完整 PDF 流程按用户要求等待 XY-cut 阅读顺序接入。支持情况由实际代码决定，不在下载清单中另设状态字段。
- TATR 使用已上传的 `table_transformer_v1_1_all.onnx`；不下载其验证报告 JSON。

Python 安装包可通过 `rapid_doc_rust._runtime.CATALOG` 查看同一份嵌入清单。
`MANIFEST` 保留默认角色视图，供现有诊断及验收脚本使用。

首次使用需联网下载缺失的原始模型；已有模型只需将共享目录指向正确位置。
下载依赖的许可证随 wheel 收录于 `licenses/MODEL-DOWNLOAD-NOTICES.txt`。

## 模型切换

```python
from rapid_doc_rust import RapidDoc

engine = RapidDoc(
    layout_config={"model_type": "pp_doclayoutv2"},
    formula_config={"model_type": "pp_formulanet_plus_s"},
    table_config={
        "model_type": "table_transformer",
        "rule_table_engine": "docvortex_ocr",
    },
)
result = engine("example.pdf")
```

完整 PDF 版面可选 `pp_doclayoutv2`、`pp_doclayoutv3`，以及原有 `docvortex_rule`。
`pp_doclayout_plus_l` 可通过模型加载器与独立版面检测接口使用；其模型没有阅读顺序，Python 对应的 XY-cut 分支此前暂缓迁移，本次确认继续暂缓，因此不会拿检测框顺序替代阅读顺序。PDF 入口会明确提示这一限制。
公式可选 `pp_formulanet_plus_s`、`pp_formulanet_plus_m`、`pp_formulanet_plus_l`。
表格增加 `table_transformer`，保留 `unet_slanet_plus`、`unet`、`slanet_plus`。
各模型原有 `model_dir_or_path` 显式路径仍可使用。

`docvortex_ocr` 使用原生 PDF 规则优先；未接纳时尝试 OCR 整词和图像线段。
保留倾斜矫正、低置信度拒绝、物理外框与整词唯一占格校验；含已知公式、复选框、嵌入图片时不走 OCR 规则。
未接纳或规则恢复异常时回退所选表格模型。`use_rule_table=False`、`force_ocr=True` 仍可关闭规则捷径。
`use_word_box` 默认行为保持原状，未自动启用拆字。Rust 共享配置接受显式
`use_word_box=False`（以及空值），与省略参数一致；当前生产表格链路尚未接入
`True` 所需的原生及 OCR 字符框拆分，不会把启用请求静默当作关闭处理。

默认仍为 V3、M、`unet_slanet_plus` 和 `docvortex`，不会因接入新模型而改变。
现有 CUDA 图优化继续按已验证的默认权重哈希启用；新增模型先使用原始 ONNX 图，不套用其他规格的图改写。

可选的备用模型与文字回退配置见 [表格回退](table-fallback.md)。独立 Python 和 Rust 接口配置一致，两个开关均默认关闭。
