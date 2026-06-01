#!/bin/bash
# T244.2 — DGX Spark CUDA env wrapper.
#
# The system has CUDA 13.2 toolkit (in /usr/local/cuda symlink) but driver
# 580.142 only supports CUDA 13.0 PTX. Without forcing 13.0 nvrtc, every
# kernel load fails with CUDA_ERROR_UNSUPPORTED_PTX_VERSION.
#
# Usage : source scripts/dgx-env.sh && cargo test ...
# Or    : scripts/dgx-env.sh cargo test ...
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/targets/sbsa-linux/lib:${LD_LIBRARY_PATH}"
if [ "$#" -gt 0 ]; then
    exec "$@"
fi
