import cv2

from rapid_orientation import RapidOrientation
print("1111111111111")
orientation_engine = RapidOrientation()
img = cv2.imread("img_1.png")
cls_result, _ = orientation_engine(img)
print(cls_result)
