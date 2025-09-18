import cv2

from rapid_doc.model.ocr.rapid_ocr import RapidOcrModel
import time


if __name__ == '__main__':
    rapid_ocr = RapidOcrModel()
    img0 = cv2.imread('cc4e955d-7831-4147-8768-94e806581210.png')
    img1 = cv2.imread("checkbox_guo.png")
    img2 = cv2.imread("checkbox_Ticked.png")
    img3 = cv2.imread("checkout_Unticked.png")

    img_list = [img0, img1, img2, img3]
    for img in img_list:
        ocr_res = rapid_ocr.ocr(img=img, det=False, rec=True)
        print(ocr_res)
