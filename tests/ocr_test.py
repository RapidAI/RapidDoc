import cv2

from rapid_doc.model.ocr.rapid_ocr import RapidOcrModel

if __name__ == '__main__':
    rapid_ocr = RapidOcrModel()
    img0 = cv2.imread('table_01.jpg')

    img_list = [img0]
    for img in img_list:
        ocr_res = rapid_ocr.ocr(img=img)
        print(ocr_res)
