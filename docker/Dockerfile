FROM python:3.10.0
WORKDIR /PEA
# Install OpenMPI
RUN apt-get update && apt-get install -y \
    libopenmpi-dev \
    openmpi-bin \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install mpi4py
RUN pip install mpi4py



COPY ../requirements.txt /PEA/

RUN pip install -r requirements.txt
RUN pip install pandas
COPY ../ /PEA/
RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
