docker build -f .\docker\Dockerfile -t="hzkitty/rapid-doc:0.9.9" .
docker push hzkitty/rapid-doc:0.9.9



docker build -f .\docker\DockerfileGPU -t="hzkitty/rapid-doc:0.9.9-gpu" .
docker push hzkitty/rapid-doc:0.9.9-gpu


