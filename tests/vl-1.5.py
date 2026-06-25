from paddleocr import PaddleOCRVL

# 英伟达 GPU
pipeline = PaddleOCRVL(pipeline_version="v1.5", vl_rec_backend="vllm-server", vl_rec_server_url="http://localhost:8118/v1")

path = r'D:\CodeProjects\doc\RapidAI\RapidDoc\demo\pdfs\示例1-论文模板1.pdf'
output = pipeline.predict(path, merge_layout_blocks=False)
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_json(save_path="57_origin1") ## 保存当前图像的结构化json结果
    res.save_to_markdown(save_path="57_origin1") ## 保存当前图像的markdown格式的结果


# PADDLEOCRVL_VERSION
# PADDLEOCRVL_VL_REC_BACKEND
# PADDLEOCRVL_VL_VL_REC_SERVER_URL

"""
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    --user root \
    -v /root/.cache/modelscope/hub/models/PaddlePaddle/PaddleOCR-VL-1.5:/home/paddleocr/.paddlex/official_models/PaddleOCR-VL-1.5 \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest-nvidia-gpu \
    paddleocr genai_server --model_name PaddleOCR-VL-1.5-0.9B --host 0.0.0.0 --port 8118 --backend vllm
"""
