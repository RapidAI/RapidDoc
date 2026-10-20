# RapidDoc ONNX CUDA 算子

Python 封装和原生库位于项目根目录下的 `rapid_doc/backend/onnx_ops/`，无需单独安装 `onnxruntime-ops` 包。

原生版本 0.2.0 支持 `conv_api="classic"`，Python 封装默认选择此路径，减少新尺寸的 cuDNN 准备开销。可显式选择 frontend；classic 的识别置信度可能变化，性能、精度和复现见[新尺寸优化](../../docs/others/ocr_cold_shape_optimization.md)。

在项目根目录执行构建：

```bash
python native/onnx_ops/tools/build.py --cudnn-root PATH --architectures "75;80;86;89"
```

`PATH` 替换为包含 cuDNN 头文件和链接库的目录。构建需要 CUDA Toolkit、cuDNN9、C++ 编译器和 CMake >=3.24；默认生成器为 Ninja，需安装 Ninja 或通过 `--generator` 指定其他生成器。编译不需要 NVIDIA GPU，但 GPU 推理验证需要实机。

通过 `--architectures` 显式指定目标 GPU 架构。构建工具会下载 cuDNN frontend v1.12.0 和 nlohmann/json v3.11.3 头文件，也可通过 `--frontend-include`、`--json-include` 指定本地目录。生成的 `.so` 或 `.dll` 会安装到 `rapid_doc/backend/onnx_ops/lib/`。

Linux CUDA12/cuDNN9 原生库已在 RTX4090、ORT1.23.0 和 ORT1.19.2 上验证；其他环境仍需验证。已提供 Windows x64 DLL（CUDA12.6/cuDNN9），无 GPU 的本机已通过加载和 ORT1.22.0 注册检查，Windows GPU 推理正确性和性能尚未验证。详见[Windows 构建记录](../../docs/others/onnx_ops_windows_build.md)及[接入说明](../../docs/others/ocr_onnx_ops.md)。

第三方许可证位于项目根目录下的 `rapid_doc/backend/onnx_ops/licenses/` 和本目录下的 `third_party/onnxruntime/LICENSE`。

原独立项目保留作历史参考，RapidDoc 运行时不再依赖它。
