#!/bin/bash

# train parameters: model_name, ip_bands, cl_lambda, cls_input_size, gpus, batch_size, bypass, time, lr

# chresmax_v3
# ./submit_job.sh chresmax_v3 3 0 9216 8 32 False "120:00:00"
# ./submit_job.sh chresmax_v3 3 0.1 9216 8 32 False "120:00:00"
# ./submit_job.sh chresmax_v3 3 0.5 9216 8 32 False "120:00:00"
# ./submit_job.sh chresmax_v3 3 1 9216 8 32 False "120:00:00"
# ./submit_job.sh chresmax_v3 3 0 18432 8 32 True "144:00:00"
# ./submit_job.sh chresmax_v3 3 0.1 18432 8 32 True "144:00:00" 0.01
# ./submit_job.sh chresmax_v3 3 0.5 18432 8 32 True "144:00:00"
# ./submit_job.sh chresmax_v3 3 1 18432 8 32 True "144:00:00"

# ./submit_job.sh contrastive_resmax 3 0.1 9216 8 32 False "96:00:00"
# ./submit_job.sh contrastive_resmax 3 0.5 9216 8 32 False "96:00:00"
# ./submit_job.sh contrastive_resmax 3 1 9216 8 32 False "96:00:00"
# ./submit_job.sh contrastive_resmax 3 1 18432 8 32 True "120:00:00"


# resmax_v2
# ./submit_job.sh resmax_v2 1 0 6400 8 128 False "48:00:00"
# ./submit_job.sh resmax_v2 1 0 15616 8 128 True "72:00:00"
# ./submit_job.sh contrastive_resmaxv1 3 0.1 18432 1 32 True "local" 0.01 gpu
# ./submit_job.sh resmax_v2 3 0 9216 1 64 Falsec "48:00:00"
# ./submit_job.sh resmax_v2 3 0 18432 8 64 True "72:00:00"

# 
# ./submit_job.sh alexnet_s 0 0 0 2 128 False "48:00:00" 0.01 gpu-he 1.0 322
# ./submit_job.sh alexnet 0 0 0 2 128 False "72:00:00" 0.01 gpu-he 1.0 227
# ./submit_job.sh alexnet_nopool 0 0 0 2 128 False "72:00:00" 0.01 gpu-he 1.0
# ./submit_job.sh resnet18 0 0 512 8 32 False "48:00:00" 0.1 gpu
# ./submit_job.sh resnet18 0 0 512 8 32 False "48:00:00" 0.1 gpu 1.0 227
# ./submit_job.sh resmax_v3 1 0 512 8 32 False "72:00:00" 0.1 gpu
# ./submit_job.sh resmax_v3 1 0 512 8 32 True "72:00:00" 0.1 gpu 0.08
# ./submit_job.sh resmax_v3 3 0 512 1 32 False "local" 0.1 gpu
# ./submit_job.sh resmax_v3 3 0 512 8 32 False "72:00:00" 0.1 gpu 0.08
# ./submit_job.sh resmax_v3 3 0 512 8 32 True "96:00:00" 0.1 gpu 0.08
# ./submit_job.sh vgg11 0 0 512 8 32 False "96:00:00" 0.01 gpu 0.08 227
# ./submit_job.sh vgg11 0 0 512 8 32 False "96:00:00" 0.01 gpu 1.0 227
# ./submit_job.sh chresmax_v5 3 0.1 512 8 32 False "48:00:00" 0.1 gpu 0.08
# ./submit_job.sh chresmax_v5 3 0.1 512 8 32 True "72:00:00" 0.1 gpu
# ./submit_job.sh chresmax_v5 5 0.1 512 4 64 False "72:00:00" 0.1 gpu-he 0.08
# ./submit_job.sh chresmax_v5 5 0.1 512 4 32 True "72:00:00" 0.05 gpu-he 0.08

# ./submit_job.sh contrastive_resmaxv1 3 0.1 9216 8 32 False "96:00:00" 0.0001 
# ./submit_job.sh contrastive_resmaxv1 3 0.5 9216 8 32 False "96:00:00" 0.0001 
# ./submit_job.sh contrastive_resmaxv1 3 1 9216 8 32 False "96:00:00" 0.0001 

# ./submit_job.sh chresmax_v4 3 0.1 9216 8 32 False "120:00:00"
# ./submit_job.sh chresmax_v4 3 0.5 9216 8 32 False "120:00:00"
# ./submit_job.sh chresmax_v4 3 1 9216 8 32 False "120:00:00"\

# ./submit_job.sh chresmax_v3 3 0.01 18432 8 32 True "192:00:00" 0.01 gpu
# ./submit_job.sh chresmax_v3 3 0.1 18432 4 64 True "144:00:00" 0.01 gpu-he 0.08
# ./submit_job.sh chresmax_v3_1 3 0.1 18432 8 32 True "144:00:00" 0.01
# ./submit_job.sh chresmax_v3_2 3 0.1 18432 8 16 True "192:00:00" 0.01 gpu 0.08
# ./submit_job.sh chresmax_v3_blue 3 0.1 18432 8 32 True "144:00:00" 0.01 gpu 1.0 322
# ./submit_job.sh chresmax_v3_noise 3 0.1 18432 8 32 True "144:00:00" 0.01 gpu 1.0 322
# ./submit_job.sh chresmax_v3_gray 3 0.1 18432 8 32 True "144:00:00" 0.01 gpu 1.0 322
# ./submit_job.sh chresmax_v3_bypass_only 3 0.1 9216 8 32 True "192:00:00" 0.01 gpu 1.0 322
# ./submit_job.sh chresmax_v3_a 3 0.1 18432 8 32 True "192:00:00" 0.01 gpu 0.08 322
# ./submit_job.sh chresmax_v3_a_2 3 0.1 18432 8 32 True "192:00:00" 0.01 gpu 1.0 322
# ./submit_job.sh ft_resmax_v2 3 0.1 18432 8 32 True "144:00:00" 0.0001 gpu 1.0 322
# ./submit_job.sh chresmax_v3_2_abs 3 0.1 18432 8 16 True "192:00:00" 0.01 gpu


# ./submit_job.sh chresmax_v3_2_rand 11 0.1 18432 8 16 True "192:00:00" 0.01 gpu






# MNIST
# ./submit_job.sh chresmax_v3 3 0.1 512 2 128 True "local" 0.01 gpu-he 1.0 96
# ./submit_job.sh chresmax_v3_bypass_only 11 0.1 9216 4 64 True "192:00:00" 0.01 gpu 1.0 224
# ./submit_job.sh chresmax_v3 11 0.1 10496 2 16 True "48:00:00" 0.0001 gpu-he 1.0 224
# ./submit_job.sh chresmax_v3 16 0.1 10496 2 8 True "48:00:00" 0.001 gpu-he 1.0 224
# ./submit_job.sh chresmax_v3_a_2 11 0.1 10496 2 16 True "48:00:00" 0.001 gpu-he 1.0 224
# ./submit_job.sh chresmax_v3_a_2 18 0.1 10496 2 8 True "48:00:00" 0.01 gpu-he 1.0 224
# ./submit_job.sh chresmax_v3_2_abs 11 0.1 15616 2 16 True "48:00:00" 0.01 gpu-he 1.0 224
# ./submit_job.sh chresmax_v3_2_abs 18 0.1 15616 2 8 True "48:00:00" 0.01 gpu-he 1.0 224
# ./submit_job.sh chresmax_v3_2_abs 18 1 15616 2 8 True "48:00:00" 0.01 gpu-he 1.0 224
# ./submit_job.sh chresmax_v3_2_rand 11 0.1 13312 2 16 True "48:00:00" 0.0001 gpu-he 1.0 224

# ./submit_job.sh chresmax_v3_bypass_only 16 1 4096 8 4 True "48:00:00" 0.001 gpu 1.0 224

# ./submit_job.sh chresmax_v3_bypass_only_1 16 0.5 4096 2 4 True "48:00:00" 0.001 gpu 1.0 224

# ./submit_job.sh chresmax_abs_bypass_only 16 1 9216 4 8 True "48:00:00" 0.001 gpu-he 1.0 224 
# ./submit_job.sh chresmax_abs_bypass_only 16 0.1 9216 8 4 True "48:00:00" 0.001 gpu 1.0 224 

# ./submit_job.sh chresmax_v3_bypass_only_o 16 0.1 9216 2 8 True "48:00:00" 0.001 gpu-he 1.0 224



# val parameters: model_name, ip_bands, cl_lambda, cls_input_size, gpus, batch_size, bypass, time (or local), ckpt_dir, results_dir

# IMAGE_SCALE_LIST="160 192 227 270 322 382 454"

# for PAD in "blue" "gray" "noise" "constant"
# do
#     ./val_job.sh chresmax_v3_blue 3 0.1 18432 1 256 True "$IMAGE_SCALE_LIST" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/4/ip_3_chresmax_v3_blue_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass" "$PAD" "gpu-he"
#     ./val_job.sh chresmax_v3_gray 3 0.1 18432 1 256 True "$IMAGE_SCALE_LIST" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/4/ip_3_chresmax_v3_gray_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass" "$PAD" "gpu-he"
#     ./val_job.sh chresmax_v3_noise 3 0.1 18432 1 256 True "$IMAGE_SCALE_LIST" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/4/ip_3_chresmax_v3_noise_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass" "$PAD" "gpu-he"
#     ./val_job.sh chresmax_v3 3 0.1 18432 1 256 True "$IMAGE_SCALE_LIST" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/0/models_wo_aug/ip_3_chresmax_v3_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass" "$PAD" "gpu-he"
# done

# ./val_job.sh chresmax_v3_2 3 0.1 18432 1 256 True "160 192 227 270 322 382 454" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/0/models_wo_aug/ip_3_chresmax_v3_2_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass" "constant" "gpu-he"

# ./val_job.sh chresmax_v3_a 3 0.1 18432 1 256 True "160 192 227 270 322 382 454" "1:00:00" "/oscar/data/tserre/xyu110/pytorch-output/train/0/models_wo_aug/ip_3_chresmax_v3_a_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass" "constant" "gpu-he"

# for CL in 0.1 0.5 1
# do
#     ./val_job.sh contrastive_resmaxv1 3 $CL 9216 1 128 False "$IMAGE_SCALE_LIST" "local"
#     wait
# done

# for CL in 0.1
# do
#     ./val_job.sh chresmax_v5 3 $CL 512 1 64 False "$IMAGE_SCALE_LIST" "local"
#     wait
# done

# for CL in 0.1 0.5 1
# do
#     ./val_job.sh chresmax_v3 3 $CL 18432 1 32 True "$IMAGE_SCALE_LIST" "local"
#     wait
# done

    

# ./val_job.sh vggmax_v1 3 0 20736 1 32 True "160 192 227 270 322 382 454" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/others/batch_size_128/ip_3_vggmax_v1_gpu_8_cl_0_ip_3_322_322_20736_c1[_6,3,1_]"
# ./val_job.sh alexnet 0 0 16384 1 32 False "160 192 227 270 321 382 454" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/alexnets/alexnet_size_321"
# ./val_job.sh alexnet 0 0 9216 1 128 False "160 192 227 270 321 382 454" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/alexnets/alexnet_size_227"

# ./val_job.sh alexnet 0 0 0 1 256 False "160" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_wo_aug/ip_0_alexnet_gpu_2_cl_0_ip_3_227_227_0_c1[_6,3,1_]"



# ./val_job.sh alexnet 0 0 0 1 256 False "160 192 227 270 322 382 454" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_wo_aug/ip_0_alexnet_gpu_2_cl_0_ip_3_322_322_0_c1[_6,3,1_]"

# ./val_job.sh alexnet 0 0 0 1 256 False "160 192 227 270 322 382 454" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_w_aug/ip_0_alexnet_gpu_2_cl_0_ip_3_322_322_0_c1[_6,3,1_]_scale_0.08"

# ./val_job.sh resnet18 0 0 0 1 128 False "160 192 227 270 322 382 454" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_w_aug/ip_0_resnet18_gpu_8_cl_0_ip_3_322_322_512_c1[_6,3,1_]_scale_0.08"

# ./val_job.sh resnet18 0 0 0 1 128 False "160 192 227 270 322 382 454" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/baseline_wo_aug/ip_0_resnet18_gpu_8_cl_0_ip_3_322_322_512_c1[_6,3,1_]"

# ./val_job.sh chresmax_v3 3 0.1 18432 1 128 True "160" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/models_wo_aug/ip_3_chresmax_v3_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass"

# ./val_job.sh chresmax_v3_1 3 0.1 18432 1 128 True "160" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/4/ip_3_chresmax_v3_1_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass"

# ./val_job.sh chresmax_v3_2 3 0.1 18432 1 128 True "160" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/4/ip_3_chresmax_v3_2_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass"



# ./val_job.sh chresmax_v3_blue 3 0.1 18432 1 128 True "160" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/4/ip_3_chresmax_v3_blue_gpu_8_cl_0.1_ip_3_322_322_18432_c1[_6,3,1_]_bypass" "blue"

# ./val_job.sh chresmax_v5 3 0.1 512 1 128 False "160 192 227 270 322 382 454" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/4/ip_3_chresmax_v5_gpu_8_cl_0.1_ip_3_322_322_512_c1[_6,3,1_]_scale_0.08"

# ./val_job.sh resmax_v2 3 0 18432 1 64 True "160 192 227 270 322 382 454" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/models_wo_aug/ip_3_resmax_v2_gpu_8_cl_0_ip_3_322_322_18432_c1[_6,3,1_]_bypass"

# ./val_job.sh resmax_v2 3 0 9216 1 64 False "160 192 227 270 322 382 454" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/models_wo_aug/ip_3_resmax_v2_gpu_8_cl_0_ip_3_322_322_9216_c1[_6,3,1_]"

# ./val_job.sh resmax_v2 11 0 10496 1 64 True "224" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/mnist/ip_11_chresmax_v3_gpu_2_cl_0.1_ip_3_224_224_10496_c1[_6,3,1_]_bypass_3/checkpoint-10.pth.tar"


##### MNIST SCALE INVARIANCE ###########

# ./val_job.sh chresmax_v3_2_abs 11 0 15616 1 64 True "188 224 266 316 376 448" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_11_chresmax_v3_2_abs_gpu_2_cl_0.1_ip_3_224_224_15616_c1[_6,3,1_]_bypass" &

# ./val_job.sh chresmax_v3_2_rand 11 0 13312 1 64 True "112 134 158 188 224 266 316 376 448" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_11_chresmax_v3_2_rand_gpu_2_cl_0.1_ip_3_224_224_13312_c1[_6,3,1_]_bypass" &

# ./val_job.sh chresmax_v3_a_2 11 0 10496 1 64 True "112 134 158 188 224 266 316 376 448" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_11_chresmax_v3_a_2_gpu_2_cl_0.1_ip_3_224_224_10496_c1[_6,3,1_]_bypass" &

# ./val_job.sh chresmax_v3 11 0 10496 1 64 True "112 134 158 188 224 266 316 376 448" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_11_chresmax_v3_gpu_2_cl_0.1_ip_3_224_224_10496_c1[_6,3,1_]_bypass" &

# ./val_job.sh chresmax_v3 16 0 10496 1 64 True "112 134 158 188 224 266 316 376 448" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_16_chresmax_v3_gpu_2_cl_0.1_ip_3_224_224_10496_c1[_6,3,1_]_bypass" &

# ./val_job.sh hmax_old 18 0 0000 1 32 True "112 134 158 188 224 266 316 376 448" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_18_hmax_old_gpu_1_cl_0.5_ip_3_224_224_0000_c1[_6,3,1_]_bypass_1" &

# ./val_job.sh hmax_old 18 0 0000 1 32 True "634 896 1268 1792" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_18_hmax_old_gpu_1_cl_0.5_ip_3_224_224_0000_c1[_6,3,1_]_bypass_1" &

# Jun 9th 2025

# ./val_job.sh chresmax_v3_2_abs 18 0.1 15616 1 64 True "112 134 158 188 224 266 316 376 448" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_18_chresmax_v3_2_abs_gpu_2_cl_0.1_ip_3_224_224_15616_c1[_6,3,1_]_bypass_2" &

# ./val_job.sh chresmax_v3_2_abs 18 1 15616 1 64 True "112 134 158 188 224 266 316 376 448" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_18_chresmax_v3_2_abs_gpu_2_cl_1_ip_3_224_224_15616_c1[_6,3,1_]_bypass_2" &

# ./val_job.sh chresmax_v3_bypass_only_1 11 0.1 4096 1 64 True "112 134 158 188 224 266 316 376 448" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_11_chresmax_v3_bypass_only_gpu_2_cl_0.1_ip_3_224_224_4096_c1[_6,3,1_]_bypass_3" &

# ./val_job.sh chresmax_abs_bypass_only 16 1 9216 1 64 True "112 134 158 188 224 266 316 376 448" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_16_chresmax_abs_bypass_only_gpu_4_cl_1_ip_3_224_224_9216_c1[_6,3,1_]_bypass_1" &

# ./val_job.sh chresmax_abs_bypass_only 16 0.1 9216 1 64 True "112 134 158 188 224 266 316 376 448" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_16_chresmax_abs_bypass_only_gpu_8_cl_0.1_ip_3_224_224_9216_c1[_6,3,1_]_bypass" &

# ./val_job.sh chresmax_v3_bypass_only 16 1 4096 1 64 True "112 134 158 188 224 266 316 376 448" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_16_chresmax_v3_bypass_only_gpu_8_cl_1_ip_3_224_224_4096_c1[_6,3,1_]_bypass" &

# ./val_job.sh chresmax_v3_bypass_only 16 0.1 4096 1 64 True "112 134 158 188 224 266 316 376 448" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/0/mnist/ip_16_chresmax_v3_bypass_only_gpu_8_cl_0.1_ip_3_224_224_4096_c1[_6,3,1_]_bypass" &