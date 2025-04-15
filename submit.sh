#!/bin/bash

# train parameters: model_name, ip_bands, cl_lambda, cls_input_size, gpus, batch_size, bypass, time, lr

# chresmax_v3
# ./submit_job.sh chresmax_v3 3 0 9216 8 32 False "120:00:00"
# ./submit_job.sh chresmax_v3 3 0.1 9216 8 32 False "120:00:00"
# ./submit_job.sh chresmax_v3 3 0.5 9216 8 32 False "120:00:00"
# ./submit_job.sh chresmax_v3 3 1 9216 8 32 False "120:00:00"
# ./submit_job.sh chresmax_v3 3 0 18432 8 32 True "144:00:00"
# ./submit_job.sh chresmax_v3 3 0.1 18432 8 32 True "144:00:00"
# ./submit_job.sh chresmax_v3 3 0.5 18432 8 32 True "144:00:00"
# ./submit_job.sh chresmax_v3 3 1 18432 8 32 True "144:00:00"

# ./submit_job.sh contrastive_resmax 3 0.1 9216 8 32 False "96:00:00"
# ./submit_job.sh contrastive_resmax 3 0.5 9216 8 32 False "96:00:00"
# ./submit_job.sh contrastive_resmax 3 1 9216 8 32 False "96:00:00"
# ./submit_job.sh contrastive_resmax 3 1 18432 8 32 True "120:00:00"


# resmax_v2
# ./submit_job.sh resmax_v2 1 0 6400 8 128 False "48:00:00"
# ./submit_job.sh resmax_v2 1 0 15616 8 128 True "72:00:00"
# ./submit_job.sh resmax_v2 3 0 9216 1 64 False "48:00:00"
# ./submit_job.sh resmax_v2 3 0 18432 8 64 True "72:00:00"



# ./submit_job.sh resmax_v3 1 0 512 8 128 False "48:00:00"
# ./submit_job.sh resmax_v3 3 0 512 8 64 False "48:00:00"

# ./submit_job.sh contrastive_resmaxv1 3 0.1 9216 8 32 False "96:00:00" 0.0001 
# ./submit_job.sh contrastive_resmaxv1 3 0.5 9216 8 32 False "96:00:00" 0.0001 
# ./submit_job.sh contrastive_resmaxv1 3 1 9216 8 32 False "96:00:00" 0.0001 

# val parameters: model_name, ip_bands, cl_lambda, cls_input_size, gpus, batch_size, bypass, time (or local), ckpt_dir, results_dir

IMAGE_SCALE_LIST="160 192 227 270 322 382 454"

for CL in 0.1 0.5 1
do
    ./val_job.sh contrastive_resmaxv1 3 $CL 9216 1 128 False "$IMAGE_SCALE_LIST" "local"
    wait
done

# for CL in 0.1 0.5 1
# do
#     ./val_job.sh chresmax_v3 3 $CL 9216 1 32 False "$IMAGE_SCALE_LIST" "local"
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

# ./val_job.sh alexnet 0 0 16384 1 32 False "160 192 227 270 321 382 454" "local" "/oscar/data/tserre/xyu110/pytorch-output/train/alexnets/alexnet_size_321_scale_0.08"

# ./val_job.sh chresmax_v3 3 0.5 18432 1 32 True "454" "local"