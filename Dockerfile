FROM nvcr.io/nvidia/pytorch:20.08-py3

LABEL maintainer="David Morrison"

# arguments
ARG LOCAL_USER
ARG LOCAL_UID
ARG LOCAL_GID
ARG PROJECT_NAME

# update the image and install some basic software
# so we can install sudo
RUN apt -y update && apt -y upgrade
RUN apt -y install sudo

# add the local user and disable their password
RUN adduser -uid ${LOCAL_UID} --disabled-password --gecos '' ${LOCAL_USER}

# add the local user to the sudo and data-writers groups
RUN usermod -a -G sudo ${LOCAL_USER}

# remove the password from sudo operations
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# copy the local files required to install the requirements and local package
# making sure they are copied over with the local user and group as the owners
WORKDIR "/home/${LOCAL_USER}/${PROJECT_NAME}"

# note that the whole contents of the dir should be copies incase the home dir is not accessible for mounting
COPY --chown=${LOCAL_UID}:${LOCAL_GID} Makefile environment.yml setup.py ./
COPY --chown=${LOCAL_UID}:${LOCAL_GID} ./simcoder ./simcoder

# ensure this directory is owned by the local user
RUN chown ${LOCAL_UID}:${LOCAL_GID} .

# change to the ubuntu user
USER ${LOCAL_USER}

# clone the root conda env
# clone should allow for "hard linking" to packages
# and not make copies of the packages
RUN conda create --name ${PROJECT_NAME} --clone root

# install any project specific dependencies
# including the local package in develop mode
# so it can be edited during development
RUN conda env update --name ${PROJECT_NAME} --file environment.yml

RUN conda init bash
RUN echo "conda activate ${PROJECT_NAME}" >> ~/.bashrc
ENV PATH /opt/conda/envs/${PROJECT_NAME}/bin:$PATH
ENV CONDA_DEFAULT_ENV ${PROJECT_NAME}