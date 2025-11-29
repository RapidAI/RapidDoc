import time

from rapid_doc.model.formula.rapid_formula_self import ModelType, RapidFormula, RapidFormulaInput, EngineType
from rapid_doc.utils.config_reader import get_device

class RapidFormulaModel(object):
    def __init__(self, formula_config=None):
        cfg = RapidFormulaInput(model_type=ModelType.PP_FORMULANET_PLUS_M)
        # TODO cuda环境，默认使用torch推理，公式模型onnx gpu推理会报错
        device = get_device()
        if device.startswith('cuda'):
            device_id = int(device.split(':')[1]) if ':' in device else 0  # GPU 编号
            engine_cfg = {'use_cuda': True, "gpu_id": device_id}
            cfg.engine_cfg = engine_cfg
            cfg.model_type = ModelType.PP_FORMULANET_PLUS_M
            cfg.engine_type = EngineType.TORCH
        elif device.startswith('npu'):
            device_id = int(device.split(':')[1]) if ':' in device else 0  # npu 编号
            engine_cfg = {'use_npu': True, "npu_id": device_id}
            cfg.engine_cfg = engine_cfg
            cfg.model_type = ModelType.PP_FORMULANET_PLUS_M
            cfg.engine_type = EngineType.TORCH
        # 如果传入了 formula_config，则用传入配置覆盖默认配置
        if formula_config is not None:
            # 遍历字典，把传入配置设置到 default_cfg 对象中
            for key, value in formula_config.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
        self.latex_engine = RapidFormula(cfg=cfg)

    def predict(self, image):
        return self.batch_predict(images=[image], batch_size=1)[0]

    def batch_predict(self, images: list, batch_size: int) -> list:
        images_formula_res = []
        all_results = self.latex_engine(img_contents=images, batch_size=batch_size, tqdm_enable=True)
        for results in all_results:
            # fixed_str = latex_rm_whitespace(results.rec_formula)
            # images_formula_res.append(fixed_str)
            images_formula_res.append(results.rec_formula)
        return images_formula_res


if __name__ == "__main__":
    # cfg = RapidFormulaInput(model_type=ModelType.PP_FORMULANET_PLUS_M, engine_type=EngineType.ONNXRUNTIME)
    cfg = RapidFormulaInput(model_type=ModelType.PP_FORMULANET_PLUS_M, engine_type=EngineType.TORCH)

    # engine_cfg = {'use_cuda': True, "cuda_ep_cfg.device_id": 0} #onnx
    engine_cfg = {'use_cuda': True, "gpu_id": 0} #torch
    cfg.engine_cfg = engine_cfg
    layout_engine = RapidFormula(cfg=cfg)

    img_path = r"C:\ocr\latex\6.png"
    img_paths = [img_path] * 10
    start_time = time.time()
    for path in img_paths:
        results = layout_engine([path], batch_size=4)
        print(results[0].rec_formula)
    print(f"总运行时间: {time.time() - start_time}秒")