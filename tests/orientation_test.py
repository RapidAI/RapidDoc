import cv2

from rapid_orientation import RapidOrientation

orientation_engine = RapidOrientation()
img = cv2.imread("table_90.png")
cls_result, _ = orientation_engine(img)
print(cls_result)