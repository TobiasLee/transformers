TASK_NAME="sst-2"
DATA_DIR=glue_data
GPU=3
# --metric_name "mcc"
CUDA_VISIBLE_DEVICES=$GPU python head_prune.py \
    --data_dir $DATA_DIR/SST-2 --model_name_or_path  "${TASK_NAME}_output_1234"  \
    --task_name $TASK_NAME --batch_size 16  --predictor_lr 1e-2\
    --max_seq_length 128 --cache_dir ~/.cache/torch/transformers \
    --output_dir head_prune_${TASK_NAME}_output_1234/ 
