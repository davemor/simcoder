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
DATA_WRITERS_GID = $(shell getent group icaird | cut -d: -f3)

#################################################################################
# CONTAINER COMMANDS                                                            #
#################################################################################
docker_image:
	docker build --build-arg LOCAL_USER=$(LOCAL_USER) \
				 --build-arg LOCAL_UID=$(LOCAL_UID) \
				 --build-arg LOCAL_GID=$(LOCAL_GID) \
				 --build-arg PROJECT_NAME=$(PROJECT_NAME) \
				 -t $(DOCKER_IMAGE_NAME)_$(LOCAL_USER) .

docker_run_elnuevo_interactive:
	docker run \
		--gpus all \
		--name $(LOCAL_USER)-$(DOCKER_IMAGE_NAME) \
		--user $(LOCAL_UID):$(LOCAL_GID) \
		-v /home/$(LOCAL_USER)/development/simcoder:/home/${LOCAL_USER}/${PROJECT_NAME} \
		-v /home/$(LOCAL_USER)/datasets/similarity:/input \
		-v /home/$(LOCAL_USER)/results/similarity:/output \
		-it $(DOCKER_IMAGE_NAME)_$(LOCAL_USER):latest