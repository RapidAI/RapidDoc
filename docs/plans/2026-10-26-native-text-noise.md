# 原生文字噪声迁移

沿旧 text_noise.py 的短数字候选、正文/目录/图间空隙几何、按需填色读取和来源号清理。复用固定rule_text内核，只迁移Python文本特征和颜色裁决；默认保留证据不足的文字。准备接口给出行位置候选和颜色索引，应用接口返回独立source及删除来源号，重复来源号和缺失char_idx的清理按旧行为。

冻结旧候选/请求/完整副作用，另生成实际PDF灰字正负例，读取颜色后接实际字符→视觉行→页源→噪声链。开发定向测试，阶段末一次wheel构建及相关安装回归。强图形框仍为显式输入，完整图形检测/栅格图注归组/默认流程集成待后续；无CUDA或全量测试。

已完成：208基础＋28边界，共236组旧决策，含199候选正例、178删除正例。边界覆盖短Unicode数字/空白、英文词/CJK计数、正文/目录/图缝几何、同字体限制、浅灰范围/通道差/alpha、参考色120/245及80%比例、重复来源号和缺失char_idx的全局清理。以原始JSON表示保留未删除行和source额外字段，不补默认字段、不改变数组顺序。准备阶段返回数组位置候选及精确请求索引；应用阶段paint=None保留原文，已提供颜色映射时按旧请求/副作用执行。

32页实际专用PDF，纯填色/描边/填色描边/不可见/裁剪模式、灰度边界及深色/白色正文均与旧读取一致；32页全部实时字符→视觉行→局部页源→候选→按需颜色→清理链通过，4页删除正例，不注入候选/字体/颜色结果。高层filter无候选时不访问PDF。普通阶段不负责生成强图形框，真实链使用无需此证据的嵌套正文样本；图缝分支的框仍是独立阶段输入。

支持普通JSON，最多4096行/每行20000码点、每行chars及source.chars各20000、5000视觉框、20000颜色请求/映射；字体签名为最多32项的str（每项256字节）/i64/bool/null数组，来源号i64，发出颜色请求的索引为整数，PDF读取索引限制i32。缺失字符索引保留旧None清理语义。有限坐标abs<=1e9，RGBA为0..255。沿既有JSON/10M累计Work/8MB结果限制，预付候选的visual平方、行扫描/排序，并对重复字符可见性扫描按字节计费，接受已有预算；资源错误传播，不静默删字。

两个定向Rust单元测试和一个实际PDF集成测试通过；阶段末一次release wheel构建并隔离安装。native_text_noise、native_source_lines、pdf_character_primitives、compatibility、smoke共五组安装套件通过，覆盖精确字段/类型/顺序、所有权/并发、惰性读取、输入/扫描限额及错误恢复。fmt、85模块审查和固定36 core＋3 paths＋1 script哈希通过，无全workspace/全部103组runner/未受影响渲染/全量clippy/CUDA；legacy完成标志保持False。

日志：build/native-text-noise-rust.txt、build/pdf-text-noise-rust.txt、build/native-text-noise-wheel.txt、build/native-text-noise-install-test.txt和build/native-text-noise-regression.txt。下一步继续旧栅格图注归组、强图形/完整页源及跨页准备、候选检测和默认流程集成。
