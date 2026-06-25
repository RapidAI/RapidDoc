from paddleocr import PaddleOCRVL

# 英伟达 GPU
pipeline = PaddleOCRVL(pipeline_version="v1", vl_rec_backend="vllm-server", vl_rec_server_url="http://localhost:8118/v1")

output = pipeline.predict(r"D:\CodeProjects\doc\RapidAI\RapidDoc\demo\pdfs\demo1.pdf")
for res in output:
    res.print() ## 打印预测的结构化输出
    res.save_to_json(save_path="output8") ## 保存当前图像的结构化json结果
    res.save_to_markdown(save_path="output8") ## 保存当前图像的markdown格式的结果

"""
docker run \
    -it \
    --rm \
    --gpus all \
    --network host \
    --user root \
    -v /root/.cache/modelscope/hub/models/PaddlePaddle/PaddleOCR-VL:/home/paddleocr/.paddlex/official_models/PaddleOCR-VL \
    ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-vllm-server:latest \
    paddleocr genai_server \
        --model_name PaddleOCR-VL-0.9B \
        --host 0.0.0.0 \
        --port 8118 \
        --backend vllm
"""