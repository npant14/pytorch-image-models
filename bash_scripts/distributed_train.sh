#!/bin/bash
NUM_PROC=$1
shift

# Find a free port in the range 29500-65535
FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

torchrun --nproc_per_node=$NUM_PROC --master_port=$FREE_PORT train_skeleton.py "$@"
