#!/bin/bash

# Download the main script from GCS
gsutil cp gs://timellmframework-itg-mediabook-gbl-ww-dv-unique/pytorch-gcs/task_framework.py /PEA/task_framework.py

# Execute the main script with the passed arguments
python /PEA/task_framework.py "$@"
