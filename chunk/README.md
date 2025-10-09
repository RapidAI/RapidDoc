# 分块和定位工具

本项目提供文档分块（Chunk）和定位功能，可在 Chunk 中添加坐标与所属页面信息，并根据内容返回构成该 Chunk 的连续 block 的边界框（bbox）列表。  
匹配时采用 [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) 找出最相似的 block。

---

## 分块依赖
- tiktoken
- markdown
## 定位依赖
- rapidfuzz

推荐使用国内镜像加快安装速度：

```bash
pip install tiktoken markdown rapidfuzz -i https://mirrors.aliyun.com/pypi/simple/
```

## 使用示例
- [代码示例](./example_chunk.py)
- [分块示例](./text_splitters.py)
- [定位示例](./get_bbox_page_fast.py)



