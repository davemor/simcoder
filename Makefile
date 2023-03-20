#################################################################################
# GLOBALS                                                                       #
#################################################################################
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = simcoder
PWD := $(shell pwd)

# docker options
DOCKER_IMAGE_NAME = simcoder

# docker build process info
LOCAL_USER = $(USER)
LOCAL_UID = $(shell id -u)
LOCAL_GID = $(shell id -g)

#################################################################################
# CONTAINER COMMANDS                                                            #
#################################################################################
docker_image:
	docker build -t $(DOCKER_IMAGE_NAME) .

docker_run_elnuevo_interactive:
	docker run \
		--gpus all \
		--name $(LOCAL_USER)-$(DOCKER_IMAGE_NAME) \
		-v /home/$(LOCAL_USER)/development/simcoder:/workspace/${PROJECT_NAME} \
		-v /home/$(LOCAL_USER)/datasets/similarity:/input \
		-v /home/$(LOCAL_USER)/results/similarity:/output \
		-it $(DOCKER_IMAGE_NAME):latest

run_batch:
	docker run \
		--rm \
		-u $(LOCAL_UID):$(LOCAL_GID) \
		--gpus all \
		--name $(LOCAL_USER)-$(DOCKER_IMAGE_NAME) \
		-v /home/$(LOCAL_USER)/datasets/mf/images:/input \
		-v /home/$(LOCAL_USER)/results/similarity:/output \
		-it $(DOCKER_IMAGE_NAME):latest \
		/input /output --format=csv --chunksize=10000