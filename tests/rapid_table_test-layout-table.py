import time
import cv2
import numpy as np

from rapid_doc.cli.common import read_fn
from rapid_doc.model.table.rapid_table_self.table_cls import TableCls
from rapid_doc.model.layout.rapid_layout_self import RapidLayoutInput, RapidLayout, ModelType as LayoutModelType
from rapid_doc.model.table.rapid_table_self import ModelType, RapidTable, RapidTableInput
from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType, RapidLayout, RapidLayoutInput
from rapid_table import RapidTable
from rapidocr import RapidOCR

from rapid_doc.utils.pdf_image_tools import load_images_from_pdf


def crop_by_box(img, box, mode="xyxy"):
    """
    根据box裁剪图像区域

    :param img: 原始图像 (numpy.ndarray)
    :param box: 坐标，支持 [xmin, ymin, xmax, ymax] 或 [x, y, w, h]
    :param mode: "xyxy" 或 "xywh"
    :return: 裁剪后的图像 (numpy.ndarray)
    """
    h, w = img.shape[:2]

    if mode == "xyxy":
        xmin, ymin, xmax, ymax = map(int, box)
    elif mode == "xywh":
        x, y, bw, bh = map(int, box)
        xmin, ymin = x, y
        xmax, ymax = x + bw, y + bh
    else:
        raise ValueError("mode 必须是 'xyxy' 或 'xywh'")

    # 限制边界
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(w, xmax), min(h, ymax)

    if xmax <= xmin or ymax <= ymin:
        return None  # 无效框

    return img[ymin:ymax, xmin:xmax].copy()

# cfg = RapidLayoutInput(model_type=LayoutModelType.PP_DOCLAYOUT_PLUS_L, conf_thresh=0.5)
# model = RapidLayout(cfg=cfg)
# # img_path = r'D:\CodeProjects\doc\paddleocr-v3\doc\output9\imgs\img_in_table_box_215_99_1192_940.jpg'
# # img_path = r'C:\ocr\img\88b41949e122304fe0b98f45c08aaf14.jpg'
#
# path = r'D:\CodeProjects\doc\paddleocr-v3\doc\aaa.pdf'
# pdf_bytes = read_fn(path)
# images_list, pdf_doc = load_images_from_pdf(pdf_bytes)
# img_path = images_list[0]['img_pil']

table_crop = "img_in_table_box_0_0_2530_2179.jpg"

# all_results = model(img_contents=[img_path])[0]
# tables = [
#     (img, name, box, score)
#     for img, name, box, score in zip(all_results.img, all_results.class_names, all_results.boxes, all_results.scores)
#     if name == "table"
# ]
# # 取第一个table
# img, name, box, score = tables[0]
#
# # 如果 box 是 [xmin, ymin, xmax, ymax]
#
# img2 = np.array(img_path)
# # cv2.imread(img_path, flags=cv2.IMREAD_COLOR)
# table_crop = crop_by_box(img2, box, mode="xyxy")
# cv2.imwrite("table_crop.jpg", table_crop)
ocr_engine = RapidOCR()

table_cls = TableCls()
# img_path = "https://raw.githubusercontent.com/RapidAI/RapidTable/refs/heads/main/tests/test_files/table.jpg"
# img_path = r'C:\ocr\img\309d3419-570c-4398-b975-7d5c4231a601.png'


# img_path = "table_recognition.png"
cls, elasp = table_cls(table_crop)
print(cls, elasp)
if cls == "wired":
    cfg = RapidLayoutInput(model_type= LayoutModelType.RT_DETR_L_WIRED_TABLE_CELL_DET)
    input_args = RapidTableInput(model_type=ModelType.SLANEXT_WIRED)
else:
    cfg = RapidLayoutInput(model_type=LayoutModelType.RT_DETR_L_WIRELESS_TABLE_CELL_DET)
    input_args = RapidTableInput(model_type=ModelType.SLANEXT_WIRELESS)
layout_engine = RapidLayout(cfg=cfg)

# input_args = RapidTableInput(model_type=ModelType.SLANETPLUS)


table_engine = RapidTable(input_args)


ori_ocr_res = ocr_engine(table_crop)
ocr_results = [ori_ocr_res.boxes, ori_ocr_res.txts, ori_ocr_res.scores]

start_time = time.time()
cell_res = layout_engine([table_crop])
cell_results = (cell_res[0].boxes, cell_res[0].scores)
results = table_engine(table_crop, ocr_results=ocr_results, cell_results=cell_results)
# results = table_engine(img_path, ocr_results=ocr_results)
print(f"总运行时间: {time.time() - start_time}秒")
# print(results)
results.vis(save_dir="outputs_wireless_cells_lay111", save_name="vis")
