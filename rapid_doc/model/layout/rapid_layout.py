from rapid_doc.model.layout.rapid_layout_self import ModelType, RapidLayout, RapidLayoutInput
from rapid_doc.model.layout.rapid_layout_self.utils.typings import PP_DOCLAYOUT_PLUS_L_Threshold, PP_DOCLAYOUT_L_Threshold
from rapid_doc.utils.config_reader import get_device
from rapid_doc.utils.enum_class import CategoryId

class RapidLayoutModel(object):
    def __init__(self, layout_config=None):
        cfg = RapidLayoutInput(model_type=ModelType.PP_DOCLAYOUT_PLUS_L, conf_thresh=0.4)

        device = get_device()
        if device.startswith('cuda'):
            device_id = int(device.split(':')[1]) if ':' in device else 0  # GPU 编号
            engine_cfg = {'use_cuda': True, "cuda_ep_cfg.device_id": device_id}
            cfg.engine_cfg = engine_cfg

        # 如果传入了 layout_config，则用传入配置覆盖默认配置
        if layout_config is not None:
            if not layout_config.get("conf_thresh"):
                if cfg.model_type == ModelType.PP_DOCLAYOUT_PLUS_L:
                    # PP-DocLayout_plus-L 默认阈值
                    cfg.conf_thresh = PP_DOCLAYOUT_PLUS_L_Threshold
                elif cfg.model_type == ModelType.PP_DOCLAYOUT_L:
                    # S可能存在部分漏检，自动调低阈值
                    cfg.conf_thresh = PP_DOCLAYOUT_L_Threshold
                elif cfg.model_type == ModelType.PP_DOCLAYOUT_S:
                    # S可能存在部分漏检，自动调低阈值
                    cfg.conf_thresh = 0.2
            # 遍历字典，把传入配置设置到 default_cfg 对象中
            for key, value in layout_config.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
                    setattr(cfg, key, value)
        self.model = RapidLayout(cfg=cfg)
        self.model_type = cfg.model_type
        self.doclayout_yolo_list = ['title', 'plain text', 'abandon', 'figure', 'figure_caption', 'table', 'table_caption', 'table_footnote', 'isolate_formula', 'formula_caption',
                          '10', '11', '12', 'inline_formula', 'isolated_formula', 'ocr_text']

        # PP-DocLayout-L、PP-DocLayout-M、PP-DocLayout-S 23个常见的类别
        self.category_dict = {
            "paragraph_title": CategoryId.Title,
            "image": CategoryId.ImageBody,
            "text": CategoryId.Text,
            "number": CategoryId.Abandon,
            "abstract": CategoryId.Text,
            "content": CategoryId.Text,
            "figure_title": CategoryId.Text,
            "formula": CategoryId.InterlineEquation_YOLO,
            "table": CategoryId.TableBody,
            "table_title": CategoryId.TableCaption,
            "reference": CategoryId.Text,
            "doc_title": CategoryId.Title,
            "footnote": CategoryId.Abandon,
            "header": CategoryId.Abandon,
            "algorithm": CategoryId.Text,
            "footer": CategoryId.Abandon,
            "seal": CategoryId.Abandon,
            "chart_title": CategoryId.ImageCaption,
            "chart": CategoryId.ImageBody,
            "formula_number": CategoryId.InterlineEquationNumber_Layout,
            "header_image": CategoryId.Abandon,
            "footer_image": CategoryId.Abandon,
            "aside_text": CategoryId.Text,
        }

        # PP-DocLayout_plus-L 20个常见的类别
        self.category_plus_mapping = {
            "paragraph_title": CategoryId.Title,
            "image": CategoryId.ImageBody,
            "text": CategoryId.Text,
            "number": CategoryId.Abandon,
            "abstract": CategoryId.Text,
            "content": CategoryId.Text,
            "figure_title": CategoryId.Text,
            "formula": CategoryId.InterlineEquation_YOLO,
            "table": CategoryId.TableBody,
            "reference": CategoryId.Text,
            "doc_title": CategoryId.Title,
            "footnote": CategoryId.Abandon,
            "header": CategoryId.Abandon,
            "algorithm": CategoryId.Text,
            "footer": CategoryId.Abandon,
            "seal": CategoryId.Abandon,
            "chart": CategoryId.ImageBody,
            "formula_number": CategoryId.InterlineEquationNumber_Layout,
            "aside_text": CategoryId.Text,
            "reference_content": CategoryId.Text,
        }

    def predict(self, image):
        return self.batch_predict(images=[image], batch_size=1)[0]

    def batch_predict(self, images: list, batch_size: int) -> list:
        images_layout_res = []

        all_results = self.model(img_contents=images, batch_size=batch_size, tqdm_enable=True)
        for results in all_results:
            layout_res = []
            img, boxes, scores, class_names, elapse = results.img, results.boxes, results.scores, results.class_names, results.elapse

            temp_results = []
            for xyxy, conf, cla in zip(boxes, scores, class_names):
                # xmin, ymin, xmax, ymax = [int(p) for p in xyxy]
                xmin, ymin, xmax, ymax = [p for p in xyxy]
                if self.model_type == ModelType.PP_DOCLAYOUT_PLUS_L:
                    category_id = self.category_plus_mapping[cla]
                else:
                    category_id = self.category_dict[cla]
                # 如果是表格/图片，边界适当扩展（DocLayout模型识别的边框坐标，稍微有一点不全）
                if category_id in [CategoryId.TableBody, CategoryId.ImageBody]:
                    xmax = min(img.shape[1], xmax + 3)
                    ymax = min(img.shape[0], ymax + 5)
                temp_results.append({
                    "category_id": category_id,
                    "original_label": cla,
                    "bbox": (xmin, ymin, xmax, ymax),
                    "score": round(float(conf), 3)
                })

            # 行内公式判断
            temp_results = self.check_inline_formula(temp_results)

            for item in temp_results:
                xmin, ymin, xmax, ymax = item["bbox"]
                layout_res.append({
                    "category_id": item["category_id"],
                    "original_label": item["original_label"],
                    "poly": [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax],
                    "score": item["score"],
                })
            images_layout_res.append(layout_res)
        return images_layout_res

    # def check_inline_formula(self, temp_results):
    #     """
    #     判断哪些公式是行内公式（被plain text框完全包含）
    #     """
    #     for item in temp_results:
    #         if item["category_id"] == CategoryId.InterlineEquation_YOLO:  # isolated_formula
    #             xmin, ymin, xmax, ymax = item["bbox"]
    #             for other in temp_results:
    #                 if other["category_id"] == CategoryId.Text:  # plain text
    #                     oxmin, oymin, oxmax, oymax = other["bbox"]
    #                     if xmin >= oxmin and ymin >= oymin and xmax <= oxmax and ymax <= oymax:
    #                         item["category_id"] = CategoryId.InlineEquation # inline_formula
    #                         break
    #     return temp_results

    def check_inline_formula(self, temp_results):
        """
        判断行内公式（公式框大部分被文字框包含）
        """
        for item in temp_results:
            if item["category_id"] == CategoryId.InterlineEquation_YOLO:  # 公式初始默认为行间公式
                for other in temp_results:
                    if other["category_id"] == CategoryId.Text:  # plain text
                        if is_contained(item["bbox"], other["bbox"]):
                            # 如果公式框大部分被文字框包含，则修改为行内公式
                            item["category_id"] = CategoryId.InlineEquation
                            break
        return temp_results

def is_contained(box1, box2, thresh=0.9):
    """
    判断 box1 是否被 box2 包含（重叠面积 / box1面积 >= 阈值）
    """
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2

    # 计算 box1 面积
    box1_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算交集区域
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersect_area = inter_width * inter_height

    # 计算覆盖率：交集面积 / box1面积
    ratio = intersect_area / box1_area if box1_area > 0 else 0

    return ratio >= thresh



if __name__ == '__main__':

    # pytorch_paddle_ocr = RapidLayoutModel()
    # lay = RapidLayoutModel()
    # img = cv2.imread("C:\ocr\img\page_6.png")
    # # img = cv2.imread("D:\\file\\text-pdf\\img\\defout1.png")
    # aa = lay.batch_predict([img], 1)
    # print(aa)

    # r"C:\ocr\models\ppmodel\layout\PP-DocLayout-M\openvino\pp_doclayout_m.xml"

    cfg = RapidLayoutInput(model_type=ModelType.PP_DOCLAYOUT_PLUS_L, conf_thresh=0.4)
    model = RapidLayout(cfg=cfg)


    all_results = model(img_contents=[r'f7e21fd7.png'])
    print(all_results)
    all_results[0].vis(r"layout_vis.png")