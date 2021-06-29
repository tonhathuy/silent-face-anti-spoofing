FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04
WORKDIR /root
COPY ./ ./
RUN apt-get update && apt-get install -y python3 python3-pip cmake wget llvm
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN pip3 install numpy==1.16.1
RUN pip3 install opencv-python==4.2.0.32
RUN pip3 install configparser==5.0.2
RUN pip3 install six==1.16.0
RUN pip3 install future==0.18.2
RUN pip3 install python-multipart
RUN pip3 install uvicorn==0.14.0 fastapi==0.65.2
CMD nvidia-smi; sh start_service.sh

