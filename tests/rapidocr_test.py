# from rapidocr import RapidOCR
import cv2
import numpy as np
from rapidocr import RapidOCR, EngineType, OCRVersion, ModelType
from rapidocr.ch_ppocr_rec import TextRecInput, TextRecOutput
from rapidocr.ch_ppocr_det import TextDetector
from rapidocr.ch_ppocr_det.utils import DetPreProcess


# 定义新的方法实现
def new_get_preprocess(self, max_wh: int) -> DetPreProcess:
    limit_side_len = self.limit_side_len
    return DetPreProcess(limit_side_len, self.limit_type, self.mean, self.std)

# 绑定到类上，覆盖原方法
TextDetector.get_preprocess = new_get_preprocess

# 默认配置
default_params = {
    # "Global.use_cls": False,
    "Det.engine_type": EngineType.ONNXRUNTIME,
    "Rec.engine_type": EngineType.ONNXRUNTIME,
    "Det.ocr_version": OCRVersion.PPOCRV5,
    "Rec.ocr_version": OCRVersion.PPOCRV5,
    # "Det.model_type": ModelType.SERVER,
    # "Rec.model_type": ModelType.SERVER,
    "Det.limit_side_len": 960,
    "Det.limit_type": 'max',
    "Det.std": [0.229, 0.224, 0.225],
    "Det.mean": [0.485, 0.456, 0.406],
    "Det.box_thresh": 0.3,
    "Det.use_dilation": True,
    "Det.unclip_ratio": 1.6,
}

engine = RapidOCR(params=default_params)

img_url = "b896a7ebfc79e0a7916429bb58b7791c25e2c79f83817f9a94decc6cbd844de2.jpg"


def preprocess_image(image_path):
    """图像预处理的完整流程"""
    img = cv2.imread(image_path)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 去噪处理
    # denoised = cv2.medianBlur(gray, 5)

    # 二值化处理——这步很关键
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # # 形态学操作,连接断开的文字
    # kernel = np.ones((2, 2), np.uint8)
    # processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    return processed

# result = engine(preprocess_image(img_url))
result = engine(img_url, return_word_box=False)

result.vis("vis_result.jpg")