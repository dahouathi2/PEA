FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime
RUN pip install pandas
WORKDIR /PEA
COPY requirements.txt /PEA/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /PEA/