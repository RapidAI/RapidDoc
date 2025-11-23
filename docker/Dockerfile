FROM python:3.10.16-slim

# Install libgl for opencv support & Noto fonts for Chinese characters
RUN apt-get update && \
    apt-get install -y \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        wget \
        curl \
        libgl1 && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 安装 Python 依赖
RUN python3 -m pip install 'rapid-doc[cpu]==0.4.0' -i https://pypi.org/simple --break-system-packages && \
    python3 -m pip install 'rapid-doc[api]==0.4.0' -i https://pypi.org/simple --break-system-packages && \
    python3 -m pip cache purge

# 复制配置文件和脚本（优先复制，利用Docker缓存）
COPY .env.example /app/
COPY start_with_env.sh /app/

# 设置启动脚本权限
RUN chmod +x  start_with_env.sh

# 复制应用代码（最后复制，避免频繁变更影响缓存）
COPY app.py file_converter.py download_models.py /app/

# 设置基础环境变量
ENV PYTHONPATH=/app

# 设置默认运行时环境变量（与 start_with_env.sh 保持一致）
ENV API_PORT=8888
ENV STARTUP_WAIT_TIME=15
ENV LOG_LEVEL=INFO
ENV MINERU_DEVICE_MODE=cpu
ENV MINERU_MODELS_DIR=/app/models
# 下载默认模型文件实现离线部署
RUN python3 download_models.py

# 暴露端口（默认端口，可通过环境变量在运行时修改）
EXPOSE 8888 8888

# 添加健康检查（检查API服务）
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:$API_PORT/health || exit 1

# 使用支持环境变量的启动脚本（默认）
# 用户可以通过 docker run 命令覆盖为 ./start_services.sh 使用简单模式
CMD ["./start_with_env.sh"]