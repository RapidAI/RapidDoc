# PDF 图片证据自动读取

沿旧 native_objects.py 读取裁剪/Form/旋转后图片框、受限原始流 SHA-256 指纹、平滑背景和白边证据。FFI 全程持现有唯一 PDFium 锁，所有句柄在锁内释放，只返回自有 JSON。复用固定对象遍历和已有 Pillow 重采样算术（提至 core，共用且保持原运算）。

冻结实际旧PDF图像结果，新增专用合成PDF覆盖正负像素/透明度/裁剪/嵌套/旋转，验证自动读取→页源准备→图片过滤链。开发仅定向测试，阶段末一次wheel构建/相关回归，CUDA和全量测试延后。完整噪声/页源/候选检测/默认集成仍待。

已完成：85页旧PDF事实逐值/类型/顺序一致，包含57个图片输入、13处平滑背景、20处白边。新增专用PDF保留RGB/L/alpha、白边比例边界、Form矩阵、裁剪/翻转、四方向、重复图片和极窄灰度图；既有PDF样本未重写。导出器拒绝覆盖JSON；`--existing-pdf`仅复用已生成PDF，仍禁止覆盖参考JSON。

10页完整实时字符/绘图/图片读取→视觉行→局部页源→全文图片过滤链与旧参考一致，无图片/字符/样式结果注入。该链只把最终裁决实际消费的行字段传入全文图片阶段，避免延长不需要的字符容器；不代表噪声、跨页准备和完整默认解析编排已完成。

资源支持域：固定深度15，最多5000图片、既有页/对象/全部裁剪段限制；每图指纹原始流16MiB以上不生成指纹，累计流读取64MiB；平滑背景解码最多16M像素、白边最多4M像素，每次调用累计64M像素；共享Pillow缩放256M工作预算。位图尺寸/步长另行验证，所有句柄/借用缓冲限制在全局锁内，损坏对象证据失败保留框，资源错误明确传播。未宣称任意资源规模下旧流程全等价或更低内存。

一个定向Rust实际PDF集成测试通过，含无效输入/页号/NUL/自定义对象限制；阶段末一次release wheel构建并隔离安装。新增pdf_image_infos和原native_image_regions、native_source_lines、formula_preprocess、compatibility、smoke共六组安装套件通过。公式88组既有输入像素/张量回归保持，图片精确指纹/证据、所有权/并发/错误恢复通过。fmt、84模块来源审查和固定36 core＋3 paths＋1 script哈希通过。未跑全workspace、全部102组runner、未受影响渲染、全量clippy或CUDA，两个legacy完成标志False。

日志：build/pdf-image-infos-rust.txt、build/pdf-image-infos-wheel.txt、build/pdf-image-infos-install-test.txt、build/pdf-image-infos-regression.txt。下一步：旧噪声/栅格图注归组及完整页源/候选检测与默认集成。
