import sys
from pathlib import Path
from typing import Union
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from download_file import DownloadFileInput, DownloadFile

def read_yaml(file_path: Union[str, Path]) -> DictConfig:
    return OmegaConf.load(file_path)

def default_download(models_pkg, configs_pkg):
    # 获取 models 模块的目录
    model_dir = Path(models_pkg.__path__[0])
    # 获取 configs 模块所在目录
    configs_dir = Path(configs_pkg.__file__).parent
    # 拼接 default_models.yaml 文件路径
    default_models_yaml = configs_dir / "default_models.yaml"
    model_map = read_yaml(default_models_yaml)

    for model_name, model_info in model_map.items():
        if model_name in ['unitable']:
            # multi_models
            model_root_dir = model_info["model_dir_or_path"]
            save_model_dir = model_dir / Path(model_root_dir).name
            for file_name, sha256 in model_info["SHA256"].items():
                save_path = save_model_dir / file_name

                download_params = DownloadFileInput(
                    file_url=f"{model_root_dir}/{file_name}",
                    sha256=sha256,
                    save_path=save_path,
                )
                DownloadFile.run(download_params)
        else:
            model_dir_or_path = model_info["model_dir_or_path"]
            sha256 = model_info["SHA256"]

            save_model_path = (
                    model_dir / Path(model_dir_or_path).name
            )
            download_params = DownloadFileInput(
                file_url=model_dir_or_path,
                sha256=sha256,
                save_path=save_model_path,
            )
            DownloadFile.run(download_params)

def ocr_download(models_pkg, configs_pkg):
    # 获取 models 模块的目录
    model_dir = Path(models_pkg.__path__[0])
    # 获取 configs 模块所在目录
    configs_dir = Path(configs_pkg.__file__).parent
    # 拼接 default_models.yaml 文件路径
    default_models_yaml = configs_dir / "default_models.yaml"
    model_map = read_yaml(default_models_yaml)

    for engin_name, engin_info in model_map.items(): # model_info为onnxruntime层级
        if engin_name in ['openvino', 'torch', 'fonts']:
            if engin_name == 'fonts':
                for lang, font_info in engin_info.items():
                    font_path = font_info["path"]
                    font_sha256 = font_info["SHA256"]

                    font_save_model_path = (
                            model_dir / Path(font_path).name
                    )
                    download_params = DownloadFileInput(
                        file_url=font_path,
                        sha256=font_sha256,
                        save_path=font_save_model_path,
                    )
                    DownloadFile.run(download_params)
            else:
                for version, ocr_info in engin_info.items(): # ocr_info为PP-OCRv4层级
                    for det, det_info in ocr_info.items(): # info为det层级
                        for model_name, model_info in det_info.items():
                            # 如果又字典文件，下载字典
                            dict_download_url = model_info.get("dict_url")
                            if dict_download_url:
                                dict_path = (model_dir / Path(dict_download_url).name)
                            if dict_download_url and not Path(dict_path).exists():
                                DownloadFile.run(
                                    DownloadFileInput(
                                        file_url=dict_download_url,
                                        sha256=None,
                                        save_path=dict_path,
                                    )
                                )
                            # 下载模型
                            model_path = model_dir / Path(model_info["model_dir"]).name
                            download_params = DownloadFileInput(
                                file_url=model_info["model_dir"],
                                sha256=model_info["SHA256"],
                                save_path=model_path,
                            )
                            DownloadFile.run(download_params)

def download_pipeline_models():
    """下载Pipeline模型"""
    try:
        # 下载版面识别模型
        logger.info('开始下载版面识别模型...')
        import rapid_doc.model.layout.rapid_layout_self.models as layout_models_pkg
        import rapid_doc.model.layout.rapid_layout_self.configs as layout_configs_pkg
        default_download(layout_models_pkg, layout_configs_pkg)

        # 下载公式识别模型
        logger.info('开始下载公式识别模型...')
        import rapid_doc.model.formula.rapid_formula_self.models as formula_models_pkg
        import rapid_doc.model.formula.rapid_formula_self.configs as formula_configs_pkg
        default_download(formula_models_pkg, formula_configs_pkg)

        # 下载表格识别模型
        logger.info('开始下载表格识别模型...')
        import rapid_doc.model.table.rapid_table_self.models as table_models_pkg
        import rapid_doc.model.table.rapid_table_self as table_configs_pkg
        default_download(table_models_pkg, table_configs_pkg)

        # 下载表格分类
        logger.info('开始下载表格分类模型...')
        from rapid_doc.model.table.rapid_table_self.table_cls.main import KEY_TO_MODEL_URL as TABLE_CLS_KEY_TO_MODEL_URL
        from rapid_doc.model.table.rapid_table_self.table_cls.utils.download_model import DownloadModel as TABLE_CLS_DownloadModel
        TABLE_CLS_DownloadModel.download(TABLE_CLS_KEY_TO_MODEL_URL.get('paddle'))

        # 下载有线表格
        logger.info('开始下载有线表格模型...')
        from rapid_doc.model.table.rapid_table_self.wired_table_rec.main import KEY_TO_MODEL_URL as UNET_KEY_TO_MODEL_URL
        from rapid_doc.model.table.rapid_table_self.wired_table_rec.utils.download_model import DownloadModel as UNET_DownloadModel
        UNET_DownloadModel.download(UNET_KEY_TO_MODEL_URL.get('unet'))

        # 下载OCR模型
        logger.info('开始下载OCR模型...')
        import rapidocr.models as ocr_models_pkg
        import rapidocr as ocr_configs_pkg
        ocr_download(ocr_models_pkg, ocr_configs_pkg)
        logger.info('所有模型下载完成: success download')
        return True
    except Exception as e:
        logger.error(f'模型下载失败: {e}')
    return True


if __name__ == '__main__':
    success = download_pipeline_models()
    sys.exit(0 if success else 1)
