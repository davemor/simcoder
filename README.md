# Simcoder
A utility for encoding images using the SimCLR2 pretrained model weigths.

# Installation
The tool is designed to run inside a docker container and requires a build step to use. There are a set of make targets that automate these steps. They are in the Makefile. You may need to modify them for your machine.

Here is an example of installation:

```bash
git clone https://github.com/davemor/simcoder.git
cd simcoder
make docker_image
```

# Usage
To run a container and execute the program, there is a handy make target called run_batch.
```bash
make run_batch
```
This assumes that a host directory containing images is mapped to `/input` inside the container and that the directory to save the embeddings. You will need to remap these directories in order to use it on your specific machine. This can be done inside the run_batch target in the Makefile.

The program accepts two options:
- --format - specifies the output format which can be csv, npy, or mat.
- --chunksize - the number of image embeddings per output file.