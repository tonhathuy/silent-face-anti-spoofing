FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-runtime
WORKDIR /root
COPY ./ ./
RUN apt-get update && apt-get install -y wget llvm
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD nvidia-smi; sh start_service.sh

