from typing import List, Dict, Any

from rapid_doc.model.ocr.ocr_patch import apply_ocr_patch

# 应用所有 OCR 相关补丁
apply_ocr_patch()

import os
import cv2
import copy
import time
import warnings
import numpy as np
from pathlib import Path
from loguru import logger
from rapidocr import RapidOCR, EngineType, OCRVersion, ModelType
from rapidocr.ch_ppocr_rec import TextRecInput, TextRecOutput
from tqdm import tqdm

from rapid_doc.utils.config_reader import get_device
from rapid_doc.utils.model_utils import check_openvino
from rapid_doc.utils.ocr_utils import check_img, preprocess_image, sorted_boxes, merge_det_boxes, update_det_boxes, get_rotate_crop_image
from rapidocr.inference_engine.base import InferSession
models_dir = os.getenv('RAPID_MODELS_DIR', None)
if models_dir:
    # 从指定的文件夹内寻找模型文件
    InferSession.DEFAULT_MODEL_PATH = Path(models_dir)
    from rapidocr.ch_ppocr_rec import main as rec_main
    rec_main.DEFAULT_MODEL_PATH = Path(models_dir)

class RapidOcrModel(object):
    def __init__(self, det_db_box_thresh=0.3, lang=None, ocr_config=None, use_dilation=True, det_db_unclip_ratio=1.8, enable_merge_det_boxes=True):
        self.drop_score = 0.5
        self.enable_merge_det_boxes = enable_merge_det_boxes
        device = get_device()
        # 默认配置
        default_params = {
            "Global.use_cls": False,
            "engine_type": EngineType.ONNXRUNTIME,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Det.ocr_version": OCRVersion.PPOCRV5,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
            # "Det.model_type": ModelType.SERVER,
            # "Rec.model_type": ModelType.SERVER,
            "Det.limit_side_len": 960,
            "Det.limit_type": 'max',
            "Det.std": [0.229, 0.224, 0.225],
            "Det.mean": [0.485, 0.456, 0.406],
            "Det.box_thresh": det_db_box_thresh,
            "Det.use_dilation": use_dilation,
            "Det.unclip_ratio": det_db_unclip_ratio,
        }

        # 获取用户传入的 engine_type
        engine_type = ocr_config.get('engine_type') if ocr_config else None

        # CPU 上优先使用 OpenVINO（如果可用且用户未指定 engine_type）
        if device.startswith('cpu') and check_openvino() and not engine_type:
            default_params["Det.engine_type"] = EngineType.OPENVINO
            default_params["Rec.engine_type"] = EngineType.OPENVINO
        if engine_type == EngineType.TORCH:
            default_params["Det.engine_type"] = EngineType.TORCH
            default_params["Rec.engine_type"] = EngineType.TORCH

        # 如果传入了 ocr_config，则覆盖参数
        if ocr_config is not None:
            for key, value in ocr_config.items():
                default_params[key] = value

        if device.startswith('cuda'):
            if not engine_type:
                # cuda 环境默认使用 torch
                default_params["Det.engine_type"] = EngineType.TORCH
                default_params["Rec.engine_type"] = EngineType.TORCH
            gpu_id = int(device.split(':')[1]) if ':' in device else 0 # GPU 编号
            if default_params.get('Det.engine_type') == EngineType.TORCH:
                default_params['EngineConfig.torch.use_cuda'] = True
                default_params['EngineConfig.torch.gpu_id'] = gpu_id
        elif device.startswith('npu'):
            if not engine_type:
                # npu 环境默认使用 torch
                default_params["Det.engine_type"] = EngineType.TORCH
                default_params["Rec.engine_type"] = EngineType.TORCH
            npu_id = int(device.split(':')[1]) if ':' in device else 0  # npu 编号
            if default_params.get('Det.engine_type') == EngineType.TORCH:
                default_params['EngineConfig.torch.use_npu'] = True
                default_params['EngineConfig.torch.npu_id'] = npu_id
        default_params.pop('engine_type', None)
        default_params.pop('use_det_mode', None)
        self.ocr_engine = RapidOCR(params=default_params)
        self.text_detector = self.ocr_engine.text_det
        self.text_recognizer = self.ocr_engine.text_rec
        self.rec_batch_num = self.text_recognizer.rec_batch_num

    def ocr(self,
            img,
            det=True,
            rec=True,
            mfd_res=None,
            tqdm_enable=False,
            tqdm_desc="OCR-rec Predict",
            return_word_box=False,
            ori_img=None,
            dt_boxes=None,
            ):
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        img = check_img(img)
        imgs = [img]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if det and rec:
                ocr_res = []
                for img in imgs:
                    img = preprocess_image(img)
                    dt_boxes, rec_res = self.__call__(img, mfd_res=mfd_res)
                    if not dt_boxes and not rec_res:
                        ocr_res.append(None)
                        continue
                    tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]
                    ocr_res.append(tmp_res)
                # print(f"ocr===运行时间1: {time.time() - start_time}秒")
                return ocr_res
            elif det and not rec:
                ocr_res = []
                for img in imgs:
                    img = preprocess_image(img)
                    det_res = self.text_detector(img)
                    dt_boxes, elapse = det_res.boxes, det_res.elapse
                    # logger.debug("dt_boxes num : {}, elapse : {}".format(len(dt_boxes), elapse))
                    if dt_boxes is None:
                        ocr_res.append(None)
                        continue
                    dt_boxes = np.array(dt_boxes) # 转为np
                    dt_boxes = sorted_boxes(dt_boxes)
                    # merge_det_boxes 和 update_det_boxes 都会把poly转成bbox再转回poly，因此需要过滤所有倾斜程度较大的文本框
                    if self.enable_merge_det_boxes:
                        dt_boxes = merge_det_boxes(dt_boxes)
                    if mfd_res:
                        dt_boxes = update_det_boxes(dt_boxes, mfd_res)
                    tmp_res = [box.tolist() for box in dt_boxes]
                    ocr_res.append(tmp_res)
                # print(f"ocr===运行时间222: {time.time() - start_time}秒")
                return ocr_res
            elif not det and rec:
                ocr_res = []
                for img in imgs:
                    if not isinstance(img, list):
                        img = preprocess_image(img)
                        img = [img]
                    rec_input = TextRecInput(img=img, return_word_box=return_word_box)
                    rec_result = self.text_recognizer_call(rec_input, tqdm_enable=tqdm_enable, tqdm_desc=tqdm_desc)
                    if return_word_box and ori_img is not None and dt_boxes:
                        op_record = {'padding_1': {'left': 0, 'top': 0}, 'preprocess': {'ratio_h': 1.0, 'ratio_w': 1.0}}
                        raw_h, raw_w = ori_img.shape[:2]
                        dt_boxes_np = [np.array(box, dtype=np.float32) for box in dt_boxes]
                        word_results = self.calc_word_boxes(img, dt_boxes_np, rec_result, op_record, raw_h, raw_w)
                        rec_res = list(zip(rec_result.txts, rec_result.scores, word_results))
                    else:
                        rec_res = list(zip(rec_result.txts, rec_result.scores))
                    ocr_res.append(rec_res)
                return ocr_res

    def calc_word_boxes(
        self,
        img: List[np.ndarray],
        dt_boxes: np.ndarray,
        rec_res: TextRecOutput,
        op_record: Dict[str, Any],
        raw_h: int,
        raw_w: int,
    ) -> Any:
        rec_res = self.ocr_engine.cal_rec_boxes(
            img, dt_boxes, rec_res, self.ocr_engine.return_single_char_box
        )

        origin_words = []
        for word_line in rec_res.word_results:
            origin_words_item = []
            for txt, score, bbox in word_line:
                if bbox is None:
                    continue

                origin_words_points = self.map_boxes_to_original(
                    np.array([bbox]).astype(np.float64), op_record, raw_h, raw_w
                )
                origin_words_points = origin_words_points.astype(np.int32).tolist()[0]
                origin_words_item.append((txt, score, origin_words_points))

            if origin_words_item:
                origin_words.append(tuple(origin_words_item))
        return tuple(origin_words)

    def map_boxes_to_original(
            self, dt_boxes: np.ndarray, op_record: Dict[str, Any], ori_h: int, ori_w: int
    ) -> np.ndarray:
        for op in reversed(list(op_record.keys())):
            v = op_record[op]
            if "padding" in op:
                top, left = v.get("top"), v.get("left")
                dt_boxes[:, :, 0] -= left
                dt_boxes[:, :, 1] -= top
            elif "preprocess" in op:
                ratio_h = v.get("ratio_h")
                ratio_w = v.get("ratio_w")
                dt_boxes[:, :, 0] *= ratio_w
                dt_boxes[:, :, 1] *= ratio_h

        dt_boxes = np.where(dt_boxes < 0, 0, dt_boxes)
        dt_boxes[..., 0] = np.where(dt_boxes[..., 0] > ori_w, ori_w, dt_boxes[..., 0])
        dt_boxes[..., 1] = np.where(dt_boxes[..., 1] > ori_h, ori_h, dt_boxes[..., 1])
        return dt_boxes

    def __call__(self, img, mfd_res=None):

        if img is None:
            logger.debug("no valid image provided")
            return None, None

        ori_im = img.copy()
        det_res = self.text_detector(img)
        dt_boxes, elapse = det_res.boxes, det_res.elapse

        if dt_boxes is None:
            logger.debug("no dt_boxes found, elapsed : {}".format(elapse))
            return None, None
        else:
            pass
            # logger.debug("dt_boxes num : {}, elapsed : {}".format(len(dt_boxes), elapse))
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        # merge_det_boxes 和 update_det_boxes 都会把poly转成bbox再转回poly，因此需要过滤所有倾斜程度较大的文本框
        if self.enable_merge_det_boxes:
            dt_boxes = merge_det_boxes(dt_boxes)

        if mfd_res:
            dt_boxes = update_det_boxes(dt_boxes, mfd_res)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)

        rec_result = self.text_recognizer(TextRecInput(img_crop_list))
        rec_res = list(zip(rec_result.txts, rec_result.scores))
        elapse = rec_result.elapse
        # logger.debug("rec_res num  : {}, elapsed : {}".format(len(rec_res), elapse))

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        return filter_boxes, filter_rec_res


    def text_recognizer_call(self, args: TextRecInput, tqdm_enable=False, tqdm_desc="OCR-rec Predict") -> TextRecOutput:
        """
        复制 TextRecognizer 类 __call__方法，增加进度条显示
        """
        img_list = [args.img] if isinstance(args.img, np.ndarray) else args.img
        return_word_box = args.return_word_box

        width_list = [img.shape[1] / float(img.shape[0]) for img in img_list]

        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        img_num = len(img_list)
        rec_res = [("", 0.0)] * img_num

        batch_num = self.text_recognizer.rec_batch_num
        with tqdm(total=img_num, desc=tqdm_desc, disable=not tqdm_enable) as pbar:
            index = 0
            elapse = 0
            for beg_img_no in range(0, img_num, batch_num):
                end_img_no = min(img_num, beg_img_no + batch_num)

                # Parameter Alignment for PaddleOCR
                imgC, imgH, imgW = self.text_recognizer.rec_image_shape[:3]
                max_wh_ratio = imgW / imgH
                wh_ratio_list = []
                for ino in range(beg_img_no, end_img_no):
                    h, w = img_list[indices[ino]].shape[0:2]
                    wh_ratio = w * 1.0 / h
                    max_wh_ratio = max(max_wh_ratio, wh_ratio)
                    wh_ratio_list.append(wh_ratio)

                norm_img_batch = []
                for ino in range(beg_img_no, end_img_no):
                    norm_img = self.text_recognizer.resize_norm_img(img_list[indices[ino]], max_wh_ratio)
                    norm_img_batch.append(norm_img[np.newaxis, :])
                norm_img_batch = np.concatenate(norm_img_batch).astype(np.float32)

                start_time = time.perf_counter()
                preds = self.text_recognizer.session(norm_img_batch)
                line_results, word_results = self.text_recognizer.postprocess_op(
                    preds,
                    return_word_box,
                    wh_ratio_list=wh_ratio_list,
                    max_wh_ratio=max_wh_ratio,
                )

                for rno, one_res in enumerate(line_results):
                    if return_word_box:
                        rec_res[indices[beg_img_no + rno]] = (one_res, word_results[rno])
                        continue

                    rec_res[indices[beg_img_no + rno]] = (one_res, None)
                elapse += time.perf_counter() - start_time

                # 更新进度条，每次增加batch_size，但要注意最后一个batch可能不足batch_size
                current_batch_size = min(batch_num, img_num - index * batch_num)
                index += 1
                pbar.update(current_batch_size)

            all_line_results, all_word_results = list(zip(*rec_res))
            txts, scores = list(zip(*all_line_results))
        return TextRecOutput(
            img_list,
            txts,
            scores,
            all_word_results,
            elapse,
        )

    def det_batch_predict(self, img_list, max_batch_size=8):
        """
        批处理预测方法，支持多张图像同时检测

        Args:
            img_list: 图像列表
            max_batch_size: 最大批处理大小

        Returns:
            batch_results: 批处理结果列表，每个元素为(dt_boxes, elapse)
        """
        if not img_list:
            return []

        batch_results = []

        # 分批处理
        for i in range(0, len(img_list), max_batch_size):
            batch_imgs = img_list[i:i + max_batch_size]
            # assert尺寸一致
            batch_dt_boxes, batch_elapse = self._batch_process_same_size(batch_imgs)
            batch_results.extend(batch_dt_boxes)

        return batch_results


    def _batch_process_same_size(self, img_list):
        """
            对相同尺寸的图像进行批处理

            Args:
                img_list: 相同尺寸的图像列表

            Returns:
                batch_results: 批处理结果列表
                total_elapse: 总耗时
            """
        starttime = time.time()
        # 1、前置处理
        prepro_img_list = []

        for img in img_list:
            ori_img_shape = img.shape[0], img.shape[1]
            self.text_detector.preprocess_op = self.text_detector.get_preprocess(max(img.shape[0], img.shape[1]))
            prepro_img = self.text_detector.preprocess_op(img)
            if prepro_img is None:
                return [(None, 0) for _ in img_list], 0

            prepro_img_list.append(prepro_img)

        # 拼接 batch
        img_inputs = np.concatenate(prepro_img_list, axis=0)

        # 2、批处理推理
        batch_preds = self.text_detector.session(img_inputs)

        # 3、后处理每个图像的结果
        batch_results = []
        total_elapse = time.time() - starttime

        for i in range(len(img_list)):
            # 提取单个图像的预测结果
            single_preds = batch_preds[i:i+1]
            boxes, scores = self.text_detector.postprocess_op(single_preds, ori_img_shape)
            boxes = self.text_detector.sorted_boxes(boxes)
            batch_results.append((boxes, total_elapse / len(img_list)))
        return batch_results, total_elapse


if __name__ == '__main__':
    rapid_ocr = RapidOcrModel()
    img = cv2.imread("C:\\ocr\\img\\aaa.png")
    start_time1 = time.time()
    for i in range(10):
        start_time = time.time()
        ocr_res = rapid_ocr.ocr(img=img, det=True, rec=False)
        print(f"运行时间: {time.time() - start_time}秒")
    print(f"总运行时间: {time.time() - start_time1}秒")

    # rapid_ocr = RapidOcrModel()
    # img = cv2.imread("C:\\ocr\\img\\aaa.png")
    # start_time = time.time()
    # # rec 慢
    # ocr_res = rapid_ocr.ocr(img=img, det=True, rec=True)
    # print(ocr_res)
    # print(f"运行时间: {time.time() - start_time}秒")