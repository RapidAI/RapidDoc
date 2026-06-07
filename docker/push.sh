docker build -f .\docker\Dockerfile -t="hzkitty/rapid-doc:0.9.6" .
docker push hzkitty/rapid-doc:0.9.6



docker build -f .\docker\DockerfileGPU -t="hzkitty/rapid-doc:0.9.6-gpu" .
docker push hzkitty/rapid-doc:0.9.6-gpu


