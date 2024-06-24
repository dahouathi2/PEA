# Use the base image with CUDA support
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime
 
# Install OpenMPI
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*
 
# Install mpi4py using conda
RUN pip install --no-cache-dir \
    conda && \
    conda install -c conda-forge mpi4py
 
# Install other dependencies from requirements.txt
WORKDIR /PEA
COPY requirement.txt /PEA/
RUN pip install --no-cache-dir -r requirement.txt
 
# Install specific versions of PyTorch, torchvision, and torchaudio
RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
 
# Copy the rest of the application code
COPY . /PEA/

 
# Set environment variable for CUDA
ENV CUDA_LAUNCH_BLOCKING=1
 
# Specify the entrypoint for the container
ENTRYPOINT ["/bin/bash"]