docker build -f .\docker\Dockerfile -t="hzkitty/rapid-doc:0.9.5" .
docker push hzkitty/rapid-doc:0.9.5



docker build -f .\docker\DockerfileGPU -t="hzkitty/rapid-doc:0.9.5-gpu" .
docker push hzkitty/rapid-doc:0.9.5-gpu


