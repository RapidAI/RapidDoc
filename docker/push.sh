docker build -f .\docker\Dockerfile -t="hzkitty/rapid-doc:0.9.11" .
docker push hzkitty/rapid-doc:0.9.11



docker build -f .\docker\DockerfileGPU -t="hzkitty/rapid-doc:0.9.11-gpu" .
docker push hzkitty/rapid-doc:0.9.11-gpu


