import os
import cv2
from pathlib import Path
from rapid_orientation import RapidOrientation

class RapidOrientationModel(object):
    def __init__(self):
        rapid_doc_dir = Path(os.path.abspath(__file__)).parent.parent.parent
        model_path = os.path.join(rapid_doc_dir, 'resources', 'rapid_orientation.onnx')
        self.orientation_engine = RapidOrientation(model_path=model_path)

    def predict(self, input_img, det_res=None):
        rotate_label = "0"  # Default to 0 if no rotation detected or not portrait
        bgr_image = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
        # First check the overall image aspect ratio (height/width)
        img_height, img_width = bgr_image.shape[:2]
        img_aspect_ratio = img_height / img_width if img_width > 0 else 1.0
        img_is_portrait = img_aspect_ratio > 1.2

        if img_is_portrait:
            # Check if table is rotated by analyzing text box aspect ratios
            if det_res:
                vertical_count = 0
                is_rotated = False

                for box_ocr_res in det_res:
                    p1, p2, p3, p4 = box_ocr_res

                    # Calculate width and height
                    width = p3[0] - p1[0]
                    height = p3[1] - p1[1]

                    aspect_ratio = width / height if height > 0 else 1.0

                    # Count vertical vs horizontal text boxes
                    if aspect_ratio < 0.8:  # Taller than wide - vertical text
                        vertical_count += 1
                    # elif aspect_ratio > 1.2:  # Wider than tall - horizontal text
                    #     horizontal_count += 1

                if vertical_count >= len(det_res) * 0.28 and vertical_count >= 3:
                    is_rotated = True
                # logger.debug(f"Text orientation analysis: vertical={vertical_count}, det_res={len(det_res)}, rotated={is_rotated}")

                # If we have more vertical text boxes than horizontal ones,
                # and vertical ones are significant, table might be rotated
                if is_rotated:
                    rotate_label, _ = self.orientation_engine(input_img)
                    # logger.debug(f"Orientation classification result: {label}")
            else:
                rotate_label, _ = self.orientation_engine(input_img)

        return rotate_label
