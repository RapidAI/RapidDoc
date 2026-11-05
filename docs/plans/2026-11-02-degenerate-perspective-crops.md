# 表格 OCR 退化裁图兼容

## 原因

OmniDocBench 的 `docstructbench_llm-raw-scihub-o.O-s001280000131.pdf_5.pdf`
在表格文字检测中产生两个零宽框，横坐标均为 508，纵坐标分别为
324–370、362–394，来源图为 1500×610。原 Python 成功解析；Rust 在
裁图函数中额外要求严格凸四边形，提前抛出“裁图四边形退化或自交”。
这不是 UNET 单元格恢复的同一个检查，而是迁移时另一处额外限制。

## 修改

- 移除裁图的严格凸形拒绝；普通凸框保留既有快速矩阵/插值路径。
- 非凸、共线及零尺寸框使用 OpenCV 的四点方程 LU 求解与必要的
  Jacobi SVD 回退；同步其 FP32 坐标乘积、矩阵求逆和数值运算顺序。
- `warpPerspective` 的目标宽或高为零时，输出尺寸遵循原图尺寸；
  输出方向继续采用 Python 的高宽比 2 旋转规则。
- 透视极点不再额外抛错，保持对应采样规则；极端投影坐标先处理
  BORDER_REPLICATE，避免整数 tap 下标溢出。
- 保留真实输入长度、NaN/Inf、尺寸及显式资源检查。没有跳过框、
  表格或页面，没有调整识别阈值，也不按文件名分支。

矩阵/SVD 来源：OpenCV 4.13.0 `imgwarp.cpp`、`lapack.cpp`，归属与
许可记录于 `licenses/OPENCV-WARP.txt`，随 Python 包带入。

## 验证

`scripts/migration/export_degenerate_crop_reference.py` 调用原 Python
裁图函数生成 OpenCV 4.13.0/5.0.0 参考。八类框（零宽、零高、全部重合、
共线、自交、凹形、透视极点、普通凸框）共 16 组尺寸和完整像素摘要一致。
原 OpenCV4 随机几何/边界及真实倾斜表格回归通过。真实 PDF 捕获的两个
零宽框逐字节像素一致，捕获数据在 `build/crop-degenerate/`。

定向测试：`crop`、`crop_degenerate`、`crop_opencv4`。捕获测试显式
使用 `RAPIDDOC_CROP_REFERENCE` 并加 `--ignored` 执行。
既有 `model_pipeline_crop` 依赖的本地生成文件
`build/migration/model-ocr-execution.json` 缺失，未将它记为通过。

服务器该页已完成 CUDA 解析。与 Python 比较，两端表格中的 14 个公式
有 1 个产生重复 LaTeX 的长序列差异（Python 5264 字符、Rust 5017
字符），不能记录为全文精度一致；除此长公式外，完整结构/内容一致。
原始结果与明确的差异记录见 `build/crop-degenerate/parity.json`，没有
通过改写输出隐藏差异。此次裁图变更未调整公式模型/参数。

该修复已包含在后续 `doc-orientation` wheel，项目 `.venv` 和服务器
手动评测的 `env-delivery` 均安装最新包。未启动 1651 份全量评测。
