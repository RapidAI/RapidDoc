import time

import cv2
import numpy as np

from rapid_doc.model.ocr.rapid_ocr import RapidOcrModel
from rapid_doc.utils.ocr_utils import get_rotate_crop_image


if __name__ == '__main__':
    ocr_model = RapidOcrModel()
    bgr_image = cv2.imread('reader_order_01.png')
    start_time = time.time()
    det_res = ocr_model.ocr(bgr_image, rec=False)[0]

    rec_img_list = []
    for dt_box in det_res:
        rec_img_list.append(
            {
                "cropped_img": get_rotate_crop_image(
                    bgr_image, np.asarray(dt_box, dtype=np.float32)
                ),
                "dt_box": np.asarray(dt_box, dtype=np.float32),
            }
        )
    cropped_img_list = [item["cropped_img"] for item in rec_img_list]
    ocr_res_list = ocr_model.ocr(cropped_img_list, det=False, tqdm_enable=False)[0]
    print(ocr_res_list)
    print(f"总运行时间: {time.time() - start_time}秒")
