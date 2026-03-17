#!/bin/bash

# train parameters: model_name, ip_bands, cl_lambda, cls_input_size, gpus, batch_size, bypass, time, lr, gpu partition, scale, input_size
# val parameters: model_name, ip_bands, cl_lambda, cls_input_size, gpus, batch_size, bypass, time (or local), ckpt_dir, results_dir

########## IMAGENET TRAINING AND VALIDATION EXP ##########

# ./bash_scripts/submit_job.sh alexnet_s 0 0 0 2 128 False "48:00:00" 0.01 gpu-he 1.0 322
# ./bash_scripts/submit_job.sh alexnet 0 0 0 2 128 False "72:00:00" 0.01 gpu-he 1.0 227
# ./bash_scripts/submit_job.sh alexnet_nopool 0 0 0 2 128 False "72:00:00" 0.01 gpu-he 1.0
# ./bash_scripts/submit_job.sh resnet18 0 0 512 8 32 False "48:00:00" 0.1 gpu
# ./bash_scripts/submit_job.sh resnet18 0 0 512 8 32 False "48:00:00" 0.1 gpu 1.0 227
# ./bash_scripts/submit_job.sh vgg11 0 0 512 8 32 False "96:00:00" 0.01 gpu 0.08 227
# ./bash_scripts/submit_job.sh vgg11 0 0 512 8 32 False "96:00:00" 0.01 gpu 1.0 227

# ./bash_scripts/submit_job.sh chresmax_v3_2 3 0.1 18432 8 16 True "192:00:00" 0.01 gpu
# ./bash_scripts/submit_job.sh chresmax_v3_2 3 0.1 18432 8 16 True "192:00:00" 0.01 gpu 0.08 322
# ./bash_scripts/submit_job.sh chresmax_v3_2_abs 3 0.1 18432 8 16 True "192:00:00" 0.01 gpu 0.08 322

# ./bash_scripts/submit_job.sh hmax3 3 0.1 18432 8 16 True "192:00:00" 0.01 gpu
# ./bash_scripts/submit_job.sh hmax_2_5 18 0.5 0000 8 16 True "192:00:00" 0.01 gpu
# ./bash_scripts/submit_job.sh hmax_v3_adj 3 0.1 18432 8 16 True "192:00:00" 0.01 gpu

# Validation commands moved to imagenet/submit_validation.sh
# Run with:
# ./imagenet/submit_validation.sh
