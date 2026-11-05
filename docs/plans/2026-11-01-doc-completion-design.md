# DOC 收尾实施规则

用户已授权一并完成内存优化、两端共同缺口、阶段回归和文档。保持当前 Python/Rust 精度；使用原生二进制索引，不改用图片 OCR 或忽略复杂内容。

- 内存：用独立计数分配器示例测量正文解析、中间模型和绑定序列化，优先删除可证实的复制。开发期检查受影响测试；阶段完成才构建安装并重测。
- 代码：raw/middle 使用 code 类型和原始文本 span；保留缩进、制表符、软换行。Markdown 使用能容纳内容反引号的动态围栏；content_list 为 code/text/text_format，v2 为 code/code_content。不假设语言。
- 批注：按 PlcfandRef（30 字节 ATRDPre10）、PlcfandTxt 定位；正文标记 [comment N] 上标，注释段首 [comment N]，按引用 section 归入 page_footnote，保留原有富文本/字段处理。作者信息暂不猜测、不以不可靠字符串标记代替正文。
- 文本框：用 PlcftxbxTxt/PlcfHdrtxbxTxt 的 FTXBXS lid 绑定 PlcfSpaMom/PlcfSpaHdr 的 shape id；跳过 reusable 和最后备用记录。链式文本框 story 整段只读一次，避免按每个 shape 重复。正文文本框保持其内部表格、列表、样式和资源，作为独立子流放在锚点正文段之后；页眉文本框按 header CP 对应的节/页眉页脚 slot 归属。损坏或无锚点范围给出诊断，不默默贴到无关正文。
- 两端规则一致；已有独立语料做差分，新功能另用具有明确预期的二进制构造样本验证，不能只比较两个同源实现。
- 最后运行 Office 阶段回归、Python 新增/相关测试、wheel 构建安装、公开 API 验收及性能/内存对比。真实语料按 SHA256 去重；不足则如实记录，不声称全部 DOC 已保证等价。

格式参考：https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-doc/f87b3560-2c23-4d10-9751-ff141d307308 和 https://learn.microsoft.com/en-us/openspecs/office_file_formats/ms-doc/9d82ca25-ff21-488c-bee2-dd654935c65a 。
