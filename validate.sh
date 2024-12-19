
DATASET="torch/imagenet"
MODEL="alexmax_v2_Cmid"
BIG_SIZE=322
SMALL_SIZE=227
PYRAMID="True"
CLASSIFIER_INPUT_SIZE=9216
CL_LAMBDA=0
INPUT_SIZE="3 322 322"

sh distributed_val.sh 2 validate.py  \
    --data-dir /gpfs/data/tserre/npant1/ILSVRC/\
    --model $MODEL\
    --model-kwargs big_size=$BIG_SIZE small_size=$SMALL_SIZE pyramid=$PYRAMID classifier_input_size=$CLASSIFIER_INPUT_SIZE \
    --input-size 3 227 227\
    --image-scale 3 170 170 \
    -b 32  \
    --checkpoint output/train/12_24_fix/noclamp_2scale_debug_scoring_alexmax_v2_Cmid_cl_0_ip_3_322_322_9216_c1\[_6\,3\,1_\]/model_best.pth.tar \



# DATASET="torch/imagenet"
# MODEL="alexmax"
# BIG_SIZE=322
# SMALL_SIZE=322
# PYRAMID="True"
# CLASSIFIER_INPUT_SIZE=9216
# CL_LAMBDA=0
# INPUT_SIZE="3 322 322"

# sh distributed_val.sh 2 validate.py  \
#     --data-dir /gpfs/data/tserre/npant1/ILSVRC/\
#     --model $MODEL\
#     --model-kwargs pyramid=$PYRAMID ip_scale_bands=1 classifier_input_size=$CLASSIFIER_INPUT_SIZE \
#     --input-size 3 322 322\
#     --image-scale 3 322 322 \
#     -b 32  \
#     --checkpoint output/train/debug_alexmax_cl_0_ip_1_322_9216/checkpoint-41.pth.tar \