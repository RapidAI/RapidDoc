# -*- coding: utf-8 -*-
import cv2
import json
import base64
import pandas as pd
import gradio as gr
from PIL import Image
import requests
from urllib.request import urlretrieve
from uuid import uuid4
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import torch
from transformers import DetrFeatureExtractor
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from paddleocr import PaddleOCR

image_processor = AutoImageProcessor.from_pretrained("./models/table-transformer-detection")
detect_model = TableTransformerForObjectDetection.from_pretrained("./models/table-transformer-detection")
structure_model = TableTransformerForObjectDetection.from_pretrained(
    "./models/table-transformer-structure-recognition-v1.1-all")
print(structure_model.config.id2label)
# {0: 'table', 1: 'table column', 2: 'table row', 3: 'table column header', 4: 'table projected row header', 5: 'table spanning cell'}
feature_extractor = DetrFeatureExtractor()

ocr = PaddleOCR(use_angle_cls=True, lang="ch")


def paddle_ocr(image_path):
    result = ocr.ocr(image_path, cls=True)
    ocr_result = []
    for idx in range(len(result)):
        res = result[idx]
        if res:
            for line in res:
                print(line)
                ocr_result.append(line[1][0])
    return "".join(ocr_result)


def table_detect(image_box, image_url):
    if not image_url:
        file_name = str(uuid4())
        image = Image.fromarray(image_box).convert('RGB')
    else:
        image_path = f"./images/{uuid4()}.png"
        file_name = image_path.split('/')[-1].split('.')[0]
        urlretrieve(image_url, image_path)
        image = Image.open(image_path).convert('RGB')
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = detect_model(**inputs)
    # convert outputs (bounding boxes and class logits) to COCO API
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    i = 0
    output_images = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {detect_model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
        )

        region = image.crop(box)  # 检测
        output_image_path = f'./table_images/{file_name}_{i}.jpg'
        region.save(output_image_path)
        output_images.append(output_image_path)
        i += 1
    return output_images


def table_ocr(output_images, image_index):
    output_image = output_images[int(image_index)][0]
    image = Image.open(output_image).convert("RGB")
    encoding = feature_extractor(image, return_tensors="pt")
    with torch.no_grad():
        outputs = structure_model(**encoding)

    target_sizes = [image.size[::-1]]
    results = feature_extractor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]
    print(results)
    # get column and row
    columns = []
    rows = []
    for i in range(len(results['boxes'])):
        _id = results['labels'][i].item()
        if _id == 1:
            columns.append(results['boxes'][i].tolist())
        elif _id == 2:
            rows.append(results['boxes'][i].tolist())

    sorted_columns = sorted(columns, key=lambda x: x[0])
    sorted_rows = sorted(rows, key=lambda x: x[1])
    # ocr by cell
    ocr_results = []
    for row in sorted_rows:
        row_result = []
        for col in sorted_columns:
            rect = [col[0], row[1], col[2], row[3]]
            crop_image = image.crop(rect)
            image_path = 'cell.png'
            crop_image.save(image_path)
            row_result.append(paddle_ocr(image_path=image_path))
        print(row_result)
        ocr_results.append(row_result)

    print(ocr_results)
    return ocr_results


if __name__ == '__main__':
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                image_box = gr.Image()
                image_urls = gr.TextArea(lines=1, placeholder="Enter image url", label="Images")
                image_index = gr.TextArea(lines=1, placeholder="Image Number", label="No")
            with gr.Column():
                gallery = gr.Gallery(label="Tables", show_label=False, elem_id="gallery", columns=[3], rows=[1],
                                     object_fit="contain", height="auto")
                detect = gr.Button("Table Detection")
                submit = gr.Button("Table OCR")
                ocr_outputs = gr.DataFrame(label='Table',
                                           interactive=True,
                                           wrap=True)
        detect.click(fn=table_detect,
                     inputs=[image_box, image_urls],
                     outputs=gallery)
        submit.click(fn=table_ocr,
                     inputs=[gallery, image_index],
                     outputs=ocr_outputs)
    demo.launch(server_name="0.0.0.0", server_port=50074)