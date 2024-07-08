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
# Install db-dtypes separately using conda
RUN conda install -c conda-forge db-dtypes
# Install other dependencies from requirements.txt
WORKDIR /PEA
COPY requirement.txt /PEA/
RUN pip install --no-cache-dir -r requirement.txt

# Install specific versions of PyTorch, torchvision, and torchaudio
RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

# Copy only the necessary directories and files
COPY models /PEA/models
COPY utils /PEA/utils
COPY layers /PEA/layers
COPY data_provider /PEA/data_provider
COPY ds_config_zero2.json /PEA/ds_config_zero2.json
COPY LICENSE /PEA/LICENSE
COPY dataset/prompt_bank /PEA/dataset/prompt_bank
# Install Google Cloud SDK
RUN apt-get update && apt-get install -y curl && \
    curl https://sdk.cloud.google.com | bash && \
    /root/google-cloud-sdk/install.sh --quiet

# Ensure gsutil is in PATH
ENV PATH $PATH:/root/google-cloud-sdk/bin
# Copy the startup script and make it executable

# Install Google Cloud BigQuery and other required libraries
RUN pip install google-cloud-bigquery
COPY startup.sh /PEA/
RUN chmod +x /PEA/startup.sh

# Set environment variable for CUDA
ENV CUDA_LAUNCH_BLOCKING=1

# Use the startup script as the entrypoint
ENTRYPOINT ["/PEA/startup.sh"]
