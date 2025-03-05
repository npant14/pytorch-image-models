#!/bin/bash

# parameters: model_name, ip_bands, cl_lambda, cls_input_size, gpus, batch_size, bypass, time

# chresmax_v2
# ./submit_job.sh chresmax_v2 3 0 9216 8 32 False "120:00:00"
# ./submit_job.sh chresmax_v2 3 0 18432 8 32 True "120:00:00"
# ./submit_job.sh chresmax_v2 3 1 9216 8 32 False "120:00:00"
# ./submit_job.sh chresmax_v2 3 1 18432 8 32 True "144:00:00"

# resmax_v2
./submit_job.sh resmax_v2 1 0 6400 8 128 False "48:00:00"
./submit_job.sh resmax_v2 1 0 15616 8 128 True "72:00:00"
./submit_job.sh resmax_v2 3 0 9216 8 64 False "48:00:00"
./submit_job.sh resmax_v2 3 0 18432 8 64 True "72:00:00"

# for imgscale in 160 192 227 270 322 382 454
# do
#     # ./val_job.sh resmax_v2 3 0 9216 2 32 False $imgscale "1:00:00"
#     # ./val_job.sh resmax_v2 1 0 6400 2 64 False $imgscale "1:00:00"
#     # ./val_job.sh resmax_v2 3 0 18432 2 32 True $imgscale "1:00:00"
#     # ./val_job.sh resmax_v2 1 0 15616 2 64 True $imgscale "1:00:00"
#     # ./val_job.sh chresmax_v2 3 0 9216 2 32 False $imgscale "1:00:00"
#     # ./val_job.sh chresmax_v2 3 0 18432 2 32 True $imgscale "1:00:00"
#     # ./val_job.sh chresmax_v2 3 1 9216 2 32 False $imgscale "1:00:00"
#     # ./val_job.sh chresmax_v2 3 1 18432 2 32 True $imgscale "1:00:00"
#     ./val_job.sh chresmax_v2 3 1 18432 2 32 True $imgscale "1:00:00"
# done

# ./val_job.sh resmax_v2 3 0 9216 2 32 False 322 "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/validation" "local"