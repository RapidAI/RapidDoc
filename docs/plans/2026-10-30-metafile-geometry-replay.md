# WMF/EMF 文件、几何与 GDI 回放层

本轮移植本机 `metafile-render 0.3.0` 的文件校验和几何回放规则，未用其他转换库或简化 SVG 算法替换原实现。**完整图像渲染尚未完成，未切换 DOCX 默认资源入口。**

## 已实现

- `rust/rapiddoc-office/src/metafile/binary.rs`、`header.rs`：有界借用记录、绝对偏移、WMF placeable 校验和、EMF 头/声明范围、物理尺寸与 DPI、EMF+ 子记录和 Header/EOF 模式预扫描。保留原源码对提前 EOF、声明记录数、损坏/资源错误的区别，不擅自严格化历史行为。EMF+ 逐条迭代，不先收集全部子记录。
- `model.rs`、`geometry.rs`：仿射变换、路径构造/闭合、矩形/圆角矩形/椭圆/弧线、包围盒、自适应三次曲线离散及原点数预算。保留 Python 的 ties-even 尺寸取整、默认误差/递归深度与角度运算顺序。
- `playback.rs`：GDI 坐标模式、world/window/viewport 变换、SaveDC/RestoreDC、画笔/画刷/字体对象与 stock handle、当前点、路径 bracket、裁剪操作序列、图元来源、诊断截断及画布推导。路径、裁剪、样式使用 Arc 共享，状态保存和多条命令不深复制大路径。
- `wmf.rs`、`emf.rs`：接入原有状态、对象和几何记录，覆盖多边形、折线、Bezier、矩形、椭圆、弧线、区域裁剪、扩展画笔及路径填充/描边语义。保留原 ROP2/图案画刷等已有诊断与回退，不把未实现功能当作新成功结果。

字体名称读取和字体对象选择已移植；字体查找、字形测量和实际绘制仍未接入。裁剪操作顺序已经保留，最终几何布尔/像素裁剪仍属于后端工作。当前 `GeometryDocument` 只保存路径命令，不能当作完整的矢量渲染结果。

## 验证

生成器 `scripts/migration/export_office_metafile_reference.py` 直接调用安装版本的 prepare、记录读取、GDI handler、几何函数和 finalize，保存于 `tests/fixtures/office_rust_reference_068/metafile_geometry.json.gz`。

- 166份构造的 WMF/EMF 文件，包含损坏、边界和 EMF+ 模式案例；760个逐记录快照比较状态、路径、当前点、资源计费、新命令、诊断和 partial 标志。
- 40组几何/离散/变换与7组画布约束；浮点数按 `1e-12 * max(1, abs(expected))` 容差比较，结构、点数和非数值字段严格比较。**这不是渲染像素或编码字节一致性验收。**
- Office crate 的28组测试通过；之后的角度运算改动只重跑受影响矢量测试。Rust格式和参考脚本语法检查通过。
- 暂无独立真实 WMF/EMF 文件效果验收；未跑 PDF 全量、CUDA、Office整体性能或新的 wheel 安装。

当前保留的数量/画布/坐标限制来自 `metafile-render/limits.py` 及原回放规则，未新增 PDF 字符/公式限制。MIT声明保存于 `rust/rapiddoc-office/licenses/METAFILE-RENDER.txt`，原文件来源/SHA记录在 `packaging/office-native-provenance.json`。

## 下一步

继续移植文字/字体测量、DIB及位图合成、ROP/裁剪后端和 EMF+ 绘图，再接 SVG 输出及其栅格 fallback，验证实际图像后接回 DOCX 资源流程。当前遇到尚未移植的原受支持文字/位图记录或 EMF+ only 绘图会明确返回未接入，不新增占位图或静默跳过来冒充完成。

完整 DOCX 正文/公开入口仍待贯通；本轮未修改 Python生产解析器、模型或 `.venv`。上一阶段：[MTEF与DOCX资源](2026-10-30-office-mtef-and-docx-embedded.md)。
