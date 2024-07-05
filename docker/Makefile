# Makefile for Docker repository initialization, building, and deployment

# Load environment variables from .env file
include docker/.env
export

# Set default values if not defined in .env
PROJECT_ID ?= your-project-id
REGION ?= europe-west1
REPO ?= your-repo-name
TAG ?= gcr.io/$(PROJECT_ID)/your-image:latest
TRAIN_DOCKER_URI ?= $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPO)/your-image:latest

# Targets
init:
	gcloud config set project $(PROJECT_ID)
	gcloud auth configure-docker $(REGION)-docker.pkg.dev
	- gcloud artifacts repositories create $(REPO) --repository-format=docker --location=$(REGION) --description="Docker repository"

build:
	docker build -f docker/Dockerfile . -t $(TAG)
	docker tag $(TAG) $(TRAIN_DOCKER_URI)

deploy:
	docker push $(TRAIN_DOCKER_URI)
