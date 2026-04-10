import cv2
from rapidocr.utils.load_image import LoadImage

from rapid_doc.model.ocr.rapid_ocr import RapidOcrModel
from rapidocr import EngineType as OCREngineType, OCRVersion, ModelType as OCRModelType

if __name__ == '__main__':
    ocr_config = {
        "Det.ocr_version": OCRVersion.PPOCRV4,
        "Rec.ocr_version": OCRVersion.PPOCRV4,
    }
    ocr_model = RapidOcrModel(ocr_config=ocr_config, is_seal=True)
    seal_img = cv2.imread('009d4472250f4f4f0c6ba5bda35a5a86.png')
    # seal_img1111 = cv2.imread('11111.png')
    # seal_imgseal_text_det = cv2.imread('seal_text_det.png')
    # seal_img = cv2.cvtColor(seal_img, cv2.COLOR_RGB2BGR)
    # load_img = LoadImage()
    # seal_img = load_img('3958b278ec0e5342b31030411deea1b9ae5157b5a2217000d83dd3839ea4ec41.jpg')

    results = ocr_model.ocr(seal_img, det=True, rec=True)[0]

    for result in results:
        bbox = result[0]
        print(bbox)
        txt, score = result[1]
        print(txt, score)




