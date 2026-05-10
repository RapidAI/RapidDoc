FROM docker.1ms.run/library/python:3.10.16-slim

ARG DEBIAN_FRONTEND=noninteractive

# Install libgl for opencv support & Noto fonts for Chinese characters
RUN sed -i 's|http://deb.debian.org/debian|https://mirrors.tuna.tsinghua.edu.cn/debian|g; s|http://deb.debian.org/debian-security|https://mirrors.tuna.tsinghua.edu.cn/debian-security|g' /etc/apt/sources.list.d/debian.sources && \
    apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update && \
    apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libreoffice-writer \
        libreoffice-calc \
        libreoffice-impress \
        fonts-noto-core \
        fonts-noto-cjk \
        fontconfig \
        wget \
        curl \
        ca-certificates \
        libgl1 && \
    fc-cache -fv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app

# 配置 pip 国内镜像与超时，降低大包下载超时概率
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_DEFAULT_TIMEOUT=300
ENV PIP_RETRIES=10

# 安装 Python 依赖
RUN python3 -m pip install --upgrade pip setuptools wheel --break-system-packages && \
    python3 -m pip install --no-cache-dir --prefer-binary --break-system-packages \
        --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
        'rapid-doc[cpu,api]==0.9.3' && \
    python3 -m pip cache purge

# 复制配置文件和脚本（优先复制，利用Docker缓存）
COPY docker/.env.example /app/
COPY docker/start_with_env.sh /app/

# 设置启动脚本权限
RUN sed -i 's/\r$//' /app/start_with_env.sh && chmod +x /app/start_with_env.sh

# 复制应用代码（最后复制，避免频繁变更影响缓存）
COPY docker/app.py docker/file_converter.py docker/download_file.py docker/download_models.py docker/models_download_utils.py /app/

# 设置基础环境变量
ENV PYTHONPATH=/app

# 设置默认运行时环境变量（与 start_with_env.sh 保持一致）
ENV API_PORT=8888
ENV STARTUP_WAIT_TIME=15
ENV LOG_LEVEL=INFO
ENV MINERU_DEVICE_MODE=cpu
ENV RAPID_MODELS_DIR=/app/models
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
