FROM nvcr.io/nvidia/tensorflow:23.02-tf2-py3

LABEL maintainer="David Morrison"

RUN pip install tqdm

COPY . /workspace/simcoder

WORKDIR /workspace/simcoder
# ENTRYPOINT [ "python", "simcoder" ]