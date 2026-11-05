# Office共享底座实施记录（O00）

目标：开始生产Rust Office迁移，建立可独立测试的ZIP/XML/CFB和语义数据模型，不以容器可打开代替格式解析成功。

架构：`rust/rapiddoc-office` 成为workspace成员，直接依赖core和已验证底层库，不依赖Python、PDFium或ONNX Runtime。Rust facade/绑定的Office公开入口在格式转换完成后接入。

## 已实现

- `package.rs`：ZIP按需读取、共享不可变部件补丁、根关系/主part查找、部件关系、Content_Types索引、Strict/Transitional主根识别。外部关系保留为外部引用，不执行文件或网络读取。
- 原Python OPC路径函数及嵌入对象路径函数分别移植，保留二者的反斜杠、空白、根路径及越界处理差异，没有混成一套“更严格”的新规则。
- `xml.rs`：UTF-8/BOM、UTF-16LE/BE和声明代码页解码；小型关系/样式上下文可借用只读树；大XML提供流式结构检查。识别格式时只读根事件，不为判断格式建整份主文档DOM。
- `ole.rs`：大小写无关流索引、Python Unicode16 casefold、流读取、可选/必选缺失流及累计读取计费。保留Python原有128MiB单流、512MiB输入/累计上限；ZIP侧没有擅自加入这些二进制Office专属上限。
- `ole_directory.rs`：CFB库拒绝部分Python能读的非规范目录。只在普通打开失败时，校验FAT/目录链和对象归属，临时重排无序目录树后交回cfb校验。仅修改内存副本的目录链接/颜色，文件、名称、流和FAT不改。循环、重复归属和重名不进行修复。无需fork整个cfb库。
- `error.rs`：malformed/missing_part/encrypted/resource_limit等Office错误类别，并提供到core错误的适配；未做不支持的解密。
- `model.rs`：段落、富文本、列表、公式、表格/跨度、图像资产和逻辑页/sheet/slide的结构体契约。样式和媒体通过Arc共享；不采用整文档JSON中间请求。

底层版本：zip9.1.3、quick-xml0.42.0、roxmltree0.21.2、cfb0.15.0、encoding_rs0.8.42。calamine仍仅为准备探针候选，未引入生产Office主链。

## 实际差分发现与修正

首次测试暴露了CFB目录排序容错、根路径 `//_rels/...`、Transitional URI中的 `/2006/` 与Strict URI差异。均按现有Python参考和真实输入修正，不按文件名打补丁。

CFB兼容层用通用构造的无序目录验证流字节不变；额外验证目录环、重复对象和重名不能通过修复。修改范围限定为目录中的链接和颜色字段。

## 验证

`cargo test -p rapiddoc-office`：3组单元测试、5组集成测试通过，覆盖：

- 78组OPC Target、12组嵌入Target、7组relationship成员路径、6组source目录和4组Unicode大小写参考。
- 当前20个Python基线输入的容器识别：14个OOXML包含Strict与非标准路径；6个OLE文件的所有已枚举流逐一对照Python SHA256及大小写查找。
- UTF-8/UTF-16/Windows-1252、损坏XML/ZIP、缺失部件、外部关系不读取、Arc补丁共享、OLE预算和失败计费、目录归一化的边界行为。

参考：`tests/fixtures/office_rust_reference_068/foundation.json`；生成器：`scripts/migration/export_office_foundation_reference.py`。Unicode数据通过 `generate_office_unicode.py --check` 与当前Python逐码位生成结果核对。

只运行本crate受影响测试与格式检查，不跑PDF全量回归、不构建/重装Python wheel。现有 `.venv` PDF包保持上一轮可用版本，新Office转换尚未对外开放。

## 来源与当前边界

OPC及OLE包装行为参考项目内DocVortex来源文件，保留MIT声明；Unicode casefold源于当前CPython3.14/Unicode16；CFB比较所需异常大写数据取自cfb0.15.0，保留其MIT许可证。来源、SHA和本地变更范围记录在 `packaging/office-native-provenance.json`。

尚未完成：DOCX段落/样式/编号/字段/表格/公式转换，完整包损坏恢复/内容类型修复，OLE属性元数据与旧二进制格式的内部加密标志，其他五种格式转换，公开RapidDoc入口及速度/内存验收。底座的格式识别不代表已经支持这些解析能力。

下一阶段先接DOCX：按原Python实现建立样式、关系与编号上下文，贯通段落、表格、图片和公式，再扩充后续格式共用模型。公式/矢量资源不会用新增占位或删除内容替代原能力。

后续更新：段落/样式/编号/字段及表格结构的本轮实现和验证见 [DOCX上下文与表格结构](2026-10-30-docx-context-and-table-structure.md)。原“尚未完成”列表为O00阶段记录，最新状态以上述记录为准。
