PROJECT="itg-bpma-gbl-ww-np" # Put your project id
ZONE="europe-west1-d" # For example
gcloud compute machine-types list \
--project=${PROJECT} \
--filter=zone=${ZONE}
