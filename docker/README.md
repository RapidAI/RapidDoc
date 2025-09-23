# RapidDoc 镜像部署指南

## 镜像构建

### 执行构建命令

```bash
cd docker

## 下载所有模型文件实现完全离线部署（Dockerfile里如果需要可取消注释）
#RUN python3 download_models.py

# 1. CPU 模式
docker build -f Dockerfile -t volador/rapid-doc:v0.1.0 .

# 2. GPU 模式 TODO
# docker build -f DockerfileGPU -t volador/rapid-doc-gpu:v0.1.0 .
```


## 运行部署

### 1. CPU 模式

仅CPU推理，资源占用较少：
```bash
docker-compose -f docker-compose.yml up -d
```
### 2. GPU 模式
```bash
# TODO
```

## 服务端口

- **8888**: RapidDoc Web API 服务端口

## API 使用

### 健康检查

```bash
curl http://localhost:8888/health
```

### 文档解析 API

```bash
# 上传文档进行解析
curl -X POST "http://localhost:8888/parse" \
     -F "file=@document.pdf" \
     -F "mode=pipeline"
```

## 配置文件详解

### .env 环境变量配置文件

`.env` 文件用于配置服务器和系统运行参数，支持以下配置项：

#### 基础配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `API_PORT` | `8888` | RapidDoc Web API 端口 |


### 系统配置

| 变量名 | 默认值 | 说明 |
|--------|--------|------|
| `STARTUP_WAIT_TIME` | `15` | 启动等待时间（秒） |
| `LOG_LEVEL` | `INFO` | 日志级别 |
| `VERBOSE` | `false` | 详细日志输出 |
