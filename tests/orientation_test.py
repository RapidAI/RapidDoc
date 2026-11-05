import cv2
from dotenv import load_dotenv

load_dotenv()
from rapid_doc.model.orientation.rapid_orientation import RapidOrientation

orientation_engine = RapidOrientation(model_path=r"D:\CodeProjects\doc\RapidAI\models\rapid_orientation.onnx")
img = cv2.imread("5d09b95be1d642fae49f0faa6449a8f7265665ce7dd5635d0e1565eb54f37c58.png")
cls_result, _ = orientation_engine(img)
