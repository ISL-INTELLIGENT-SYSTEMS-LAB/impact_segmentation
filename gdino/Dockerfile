# Use the official CUDA base image with Ubuntu and CUDA support
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

# Set environment variables for Python installation
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y 
RUN apt-get update && apt-get install -y git bzip2 nodejs libssl-dev
RUN apt-get update && apt-get install -y cmake
RUN apt-get update && apt-get install -y libgtk2.0-dev pkg-config libcairo2-dev
RUN apt-get update && apt-get install -y libglfw3-dev libgles2-mesa-dev
RUN apt-get update && apt-get install -y x11-apps nano vim gedit eog
RUN apt-get update && apt-get install -y protobuf-compiler wget
RUN apt-get update && apt-get install -y ninja-build

# Install Python 3.7 and other necessary packages
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -y python3.7 
RUN apt-get update && apt-get install -y python3.7-venv 
RUN apt-get update && apt-get install -y python3.7-dev
RUN apt-get update && apt-get install -y python3-pip
RUN rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default python3
RUN ln -s /usr/bin/python3 /usr/local/bin/python

# Verify Python and CUDA installation
RUN python --version 
#RUN nvcc --version

# Set a working directory
WORKDIR /workspace

# Install any required Python packages (uncomment if you have a requirements.txt)
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --upgrade --force-reinstall setuptools
RUN pip install cython
RUN pip install packaging wheel
RUN pip install pycocotools faster-coco-eval onnx
RUN pip install torchmetrics
RUN pip install filetype
#RUN git clone https://github.com/fundamentalvision/Deformable-DETR.git
#RUN pip install Deformable-DETR/models/ops
RUN pip install -r requirements.txt
#RUN git clone https://github.com/longzw1997/Open-GroundingDino.git /workspace/gdino
RUN wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth -P /workspace/weights/gdino
RUN wget https://huggingface.co/google-bert/bert-base-uncased/blob/main/config.json -P /workspace/weights/bert-base-uncased
RUN wget https://huggingface.co/google-bert/bert-base-uncased/blob/main/pytorch_model.bin -P /workspace/weights/bert-base-uncased

# If you want to run a script, you can add it like this
# COPY your_script.py .
# CMD ["python", "./your_script.py"]

# Optionally, copy your project files into the container
 COPY . /workspace

# Default command
CMD ["bash"]
