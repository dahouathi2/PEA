# Makefile for Docker repository initialization, building, and deployment
include ./docker/.env
export

# Targets
init:
	gcloud config set project $(PROJECT_ID)
	gcloud auth configure-docker $(REGION)-docker.pkg.dev
	- gcloud artifacts repositories create $(REPO) --repository-format=docker --location=europe-west1 --description="Docker repository"

build:
	docker build -f docker/Dockerfile . -t $(TAG)
	docker tag $(TAG) $(TRAIN_DOCKER_URI)

deploy:
	docker push $(TRAIN_DOCKER_URI)