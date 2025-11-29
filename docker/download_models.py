#!/usr/bin/env python3
"""
Model download script for Docker build
Downloads pipeline models for offline deployment
"""
import sys
from rapid_doc.utils.models_download_utils import download_pipeline_models

if __name__ == '__main__':
    # os.environ['RAPID_MODELS_DIR'] = r'D:\CodeProjects\doc\RapidAI\models' #模型文件存储目录
    # os.environ["MINERU_DEVICE_MODE"] = "cpu" # cpu、cuda、npu、all（all只是用来下载）
    success = download_pipeline_models()
    sys.exit(0 if success else 1)
