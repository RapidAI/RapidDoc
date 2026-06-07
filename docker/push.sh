docker build -f .\docker\Dockerfile -t="hzkitty/rapid-doc:0.9.7" .
docker push hzkitty/rapid-doc:0.9.7



docker build -f .\docker\DockerfileGPU -t="hzkitty/rapid-doc:0.9.7-gpu" .
docker push hzkitty/rapid-doc:0.9.7-gpu


