"""
ocr_patch.py
------------
此模块用于优化和修正 RapidOCR 的推理逻辑。

包含：
1. 修复 TextDetector 的预处理逻辑。
2. 优化 PyTorch 推理性能，替换内部方法以支持更高效的推理。
"""

from rapidocr.ch_ppocr_det import TextDetector
from rapidocr.ch_ppocr_det.utils import DetPreProcess
from rapid_doc.utils.model_utils import import_package
from importlib.metadata import version
rapidocr_version = version("rapidocr")


def patch_text_detector():
    """修复 TextDetector 的 get_preprocess 方法"""
    def new_get_preprocess(self, max_wh: int) -> DetPreProcess:
        limit_side_len = self.limit_side_len
        return DetPreProcess(limit_side_len, self.limit_type, self.mean, self.std)

    # 覆盖原始方法
    TextDetector.get_preprocess = new_get_preprocess


def patch_torch_ocr():
    """优化 PyTorch OCR 推理性能"""
    torch_ = import_package("torch")
    if not torch_:
        return  # 未安装 PyTorch，跳过
    if rapidocr_version >= "3.4.3":
        from rapidocr.inference_engine.pytorch.networks.backbones.rec_lcnetv3 import LearnableRepLayer, ConvBNLayer
    else:
        from rapidocr.networks.backbones.rec_lcnetv3 import LearnableRepLayer, ConvBNLayer

    from torch import nn
    import torch

    def _fuse_bn_tensor2(self, branch):
        """替换 LearnableRepLayer._fuse_bn_tensor 方法"""
        if not branch:
            return 0, 0
        elif isinstance(branch, ConvBNLayer):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = torch.zeros(
                    (self.in_channels, input_dim, self.kernel_size, self.kernel_size),
                    dtype=branch.weight.dtype,
                    device=branch.weight.device,
                )
                for i in range(self.in_channels):
                    kernel_value[
                        i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
                    ] = 1
                self.id_tensor = kernel_value

            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps

        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std

    LearnableRepLayer._fuse_bn_tensor = _fuse_bn_tensor2

    # 替换 rapidocr 的 TorchInferSession
    if rapidocr_version >= "3.4.3":
        from rapidocr.inference_engine.pytorch import main as rapidocr_torch
    else:
        from rapidocr.inference_engine import torch as rapidocr_torch
    from rapid_doc.model.ocr.torch import TorchInferSession
    rapidocr_torch.TorchInferSession = TorchInferSession

def patch_openvino_ocr():
    """优化 openvino OCR 推理，使用异步推理替代同步，修复并发识别报错"""
    torch_ = import_package("openvino")
    if not torch_:
        return  # 未安装 PyTorch，跳过
    # 替换 rapidocr 的 OpenVINOInferSession
    from rapidocr.inference_engine import openvino as rapidocr_openvino
    from rapid_doc.model.ocr.openvino import OpenVINOInferSession
    rapidocr_openvino.OpenVINOInferSession = OpenVINOInferSession

def apply_ocr_patch():
    """统一入口：应用所有 OCR 相关补丁"""
    patch_text_detector()
    patch_torch_ocr()
    patch_openvino_ocr()
