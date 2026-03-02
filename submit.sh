#!/bin/bash

IMAGE_SCALE_LIST="160 192 227 270 322 382 454"

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

# ./bash_scripts/val_job.sh alexnet 0 0 0 1 256 False "160 192 227 270 322 382 454" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_wo_aug/ip_0_alexnet_gpu_2_cl_0_ip_3_322_322_0_c1[_6,3,1_]"

# ./bash_scripts/val_job.sh resnet18 0 0 0 1 256 False "160 192 227 270 322 382 454" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/sep/resnet_18_fair_comparasion"
# ./bash_scripts/val_job.sh alexnet 0 0 0 1 256 False "160 192 227 270 322 382 454" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/sep/alexnet_fair_comparasion"

# ./bash_scripts/submit_job.sh chresmax_v3_2 3 0.1 18432 8 16 True "192:00:00" 0.01 gpu
# ./bash_scripts/submit_job.sh chresmax_v3_2 3 0.1 18432 8 16 True "192:00:00" 0.01 gpu 0.08 322
# ./bash_scripts/submit_job.sh chresmax_v3_2_abs 3 0.1 18432 8 16 True "192:00:00" 0.01 gpu 0.08 322

# ./bash_scripts/submit_job.sh hmax3 3 0.1 18432 8 16 True "192:00:00" 0.01 gpu
# ./bash_scripts/submit_job.sh hmax_2_5 18 0.5 0000 8 16 True "192:00:00" 0.01 gpu
# ./bash_scripts/submit_job.sh hmax_v3_adj 3 0.1 18432 8 16 True "192:00:00" 0.01 gpu

# ./bash_scripts/val_job.sh chresmax_v3_2 3 0.1 18432 1 128 True "160 192 227 270 322 382 454" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/5/ip_3_chresmax_v3_2_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass"
# ./bash_scripts/val_job.sh chresmax_v3_2_abs 3 0.1 18432 1 64 True "160 192 227 270 322 382 454" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/5/ip_3_chresmax_v3_2_abs_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass" 
./bash_scripts/val_job.sh hmax_v3_adj 3 0.1 18432 1 128 True "160 192 227 270 322 382 454" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/5/ip_3_hmax_v3_adj_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass"
# ./bash_scripts/val_job.sh vit_base_patch16_224 0 0 0 1 128 False "160 192 227 270 322 382 454" "1:00:00" ""
./bash_scripts/val_job.sh vit_base_patch16_224 0 0 0 1 128 False "192" "1:00:00" ""


# 
########## BACKGROUND COLOR EXP ##########

# ./bash_scripts/submit_job.sh chresmax_v3_blue 3 0.1 18432 8 32 True "144:00:00" 0.01 gpu 1.0 322
# ./bash_scripts/submit_job.sh chresmax_v3_noise 3 0.1 18432 8 32 True "144:00:00" 0.01 gpu 1.0 322
# ./bash_scripts/submit_job.sh chresmax_v3_gray 3 0.1 18432 8 32 True "144:00:00" 0.01 gpu 1.0 322

# for PAD in "blue" "gray" "noise" "constant"
# do
#     ./bash_scripts/val_job.sh chresmax_v3_blue 3 0.1 18432 1 256 True "$IMAGE_SCALE_LIST" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/4/ip_3_chresmax_v3_blue_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass" "$PAD" "gpu-he"
#     ./bash_scripts/val_job.sh chresmax_v3_gray 3 0.1 18432 1 256 True "$IMAGE_SCALE_LIST" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/4/ip_3_chresmax_v3_gray_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass" "$PAD" "gpu-he"
#     ./bash_scripts/val_job.sh chresmax_v3_noise 3 0.1 18432 1 256 True "$IMAGE_SCALE_LIST" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/4/ip_3_chresmax_v3_noise_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass" "$PAD" "gpu-he"
#     ./bash_scripts/val_job.sh chresmax_v3 3 0.1 18432 1 256 True "$IMAGE_SCALE_LIST" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/0/models_wo_aug/ip_3_chresmax_v3_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass" "$PAD" "gpu-he"
# done


########## MNIST EXP ##########

# for cl_lambda in $(seq 0.505 0.005 0.545); do
#     # ./bash_scripts/submit_job.sh chresmax_v3_bypass_only 11 $cl_lambda 4096 8 4 True "48:00:00" 0.001 gpu 1.0 224
#     ./bash_scripts/val_job.sh chresmax_v3_bypass_only 11 $cl_lambda 4096 1 64 True "112 134 158 188 224 266 316 376 448" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_11_chresmax_v3_bypass_only_gpu_8_cl_${cl_lambda}_ip_3_224_224_4096_c1[_6,3,1_]_bypass"
# done
# 

# ./bash_scripts/submit_job.sh chresmax_v3_bypass_only 11 0.5 4096 8 4 True "48:00:00" 0.001 gpu 1.0 224
# ./bash_scripts/submit_job.sh hmax_old_deep 18 0.5 0000 8 4 True "48:00:00" 0.001 gpu 1.0 224


# ./bash_scripts/val_job.sh chresmax_v3_bypass_only 11 0.5 4096 1 64 True "112 134 158 188 224 266 316 376 448" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/mnist/ip_11_chresmax_v3_bypass_only_gpu_8_cl_0.5_ip_3_224_224_4096_c1[_6,3,1_]_bypass"

# ./bash_scripts/val_job.sh hmax_old_deep 18 0.5 0000 1 8 True "112 134 158 188 224 266 316 376 448 634 896 1268 1792" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_18_hmax_old_deep_gpu_2_cl_0.5_ip_3_224_224_0000_c1[_6,3,1_]_bypass_11"

# ./bash_scripts/val_job.sh alexnet 0 0 0 1 256 False "192 227" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_w_aug/ip_0_alexnet_gpu_2_cl_0_ip_3_322_322_0_c1[_6,3,1_]_scale_0.08"

#