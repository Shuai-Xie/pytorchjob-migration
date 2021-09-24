FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime

WORKDIR /opt/mnist
COPY . .

# docker build -f Dockerfile . -t shuaix/pytorchjob-migration:1.0
# docker push shuaix/pytorchjob-migration:1.0
# docker pull shuaix/pytorchjob-migration:1.0
