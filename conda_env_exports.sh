#!/bin/bash

# Use these env variables in case you encounter compiler issues
# when installing submodules (simple-knn, diff-gaussian-rasteriser)

# 1. Re-set the essential CUDA variables for this session
export CUDA_HOME=$CONDA_PREFIX
export NVCC_APPEND_FLAGS="-allow-unsupported-compiler"

# 2. Ensure the C++ compiler can find the CUDA math headers (Thrust/CUB)
export CPATH=$CONDA_PREFIX/include:$CPATH