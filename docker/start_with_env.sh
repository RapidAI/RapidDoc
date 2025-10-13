#!/bin/bash
set -euo pipefail

# 加载环境变量文件（如果存在）
if [ -f "/app/.env" ]; then
    echo "Loading environment variables from .env file..."
    export $(grep -v '^#' /app/.env | xargs)
fi

# 设置默认值
export PYTHONPATH=/app
export API_PORT=${API_PORT:-8888}
export STARTUP_WAIT_TIME=${STARTUP_WAIT_TIME:-15}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

echo "API Port: $API_PORT"
echo "Log Level: $LOG_LEVEL"
echo "======================================"

# 启动主应用
echo "Starting RapidDoc Web API on port $API_PORT..."
echo "API Documentation: http://localhost:$API_PORT/docs"
echo "Health Check: http://localhost:$API_PORT/health"
echo "======================================"

# 启动 API 服务
cd /app
python3 app.py