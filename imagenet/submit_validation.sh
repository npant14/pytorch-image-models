#!/bin/bash

IMAGE_SCALE_LIST="160 192 227 270 322 382 454"

########## IMAGENET VALIDATION EXP ##########

# ./imagenet/val_job.sh alexnet 0 0 0 1 256 False "160 192 227 270 322 382 454" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_wo_aug/ip_0_alexnet_gpu_2_cl_0_ip_3_322_322_0_c1[_6,3,1_]"

# ./imagenet/val_job.sh resnet18 0 0 0 1 256 False "160 192 227 270 322 382 454" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/sep/resnet_18_fair_comparasion"
# ./imagenet/val_job.sh alexnet 0 0 0 1 256 False "160 192 227 270 322 382 454" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/sep/alexnet_fair_comparasion"

# ./imagenet/val_job.sh chresmax_v3_2 3 0.1 18432 1 128 True "160 192 227 270 322 382 454" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/5/ip_3_chresmax_v3_2_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass"
# ./imagenet/val_job.sh chresmax_v3_2_abs 3 0.1 18432 1 64 True "160 192 227 270 322 382 454" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/5/ip_3_chresmax_v3_2_abs_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass"
./imagenet/val_job.sh hmax_v3_adj 3 0.1 18432 1 128 True "160 192 227 270 322 382 454" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/5/ip_3_hmax_v3_adj_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass"
./imagenet/val_job.sh alexnet_timm 0 0 0 1 128 False "160 192 227 270 322 382 454" "1:00:00" ""
./imagenet/val_job.sh resnet18_timm 0 0 0 1 128 False "160 192 227 270 322 382 454" "1:00:00" ""
./imagenet/val_job.sh vit_base_patch16_224 0 0 0 1 128 False "160 192 227 270 322 382 454" "1:00:00" ""

########## BACKGROUND COLOR EXP ##########

# for PAD in "blue" "gray" "noise" "constant"
# do
#     ./imagenet/val_job.sh chresmax_v3_blue 3 0.1 18432 1 256 True "$IMAGE_SCALE_LIST" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/4/ip_3_chresmax_v3_blue_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass" "$PAD" "gpu-he"
#     ./imagenet/val_job.sh chresmax_v3_gray 3 0.1 18432 1 256 True "$IMAGE_SCALE_LIST" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/4/ip_3_chresmax_v3_gray_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass" "$PAD" "gpu-he"
#     ./imagenet/val_job.sh chresmax_v3_noise 3 0.1 18432 1 256 True "$IMAGE_SCALE_LIST" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/4/ip_3_chresmax_v3_noise_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass" "$PAD" "gpu-he"
#     ./imagenet/val_job.sh chresmax_v3 3 0.1 18432 1 256 True "$IMAGE_SCALE_LIST" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/0/models_wo_aug/ip_3_chresmax_v3_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass" "$PAD" "gpu-he"
# done

########## MNIST EXP ##########

# for cl_lambda in $(seq 0.505 0.005 0.545); do
#     ./imagenet/val_job.sh chresmax_v3_bypass_only 11 $cl_lambda 4096 1 64 True "112 134 158 188 224 266 316 376 448" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_11_chresmax_v3_bypass_only_gpu_8_cl_${cl_lambda}_ip_3_224_224_4096_c1[_6,3,1_]_bypass"
# done

# ./imagenet/val_job.sh alexnet 0 0 0 1 256 False "192 227" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_w_aug/ip_0_alexnet_gpu_2_cl_0_ip_3_322_322_0_c1[_6,3,1_]_scale_0.08"
