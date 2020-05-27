TASK_NAME="qqp"
DATA_DIR=glue_data
GPU=3
SR=0.1
# --metric_name "mcc" for cola
for seed in 1234 2345 3456 4567  5678 
do
CUDA_VISIBLE_DEVICES=$GPU python head_prune.py \
    --data_dir $DATA_DIR/QQP --model_name_or_path  "${TASK_NAME}_output_$seed"  \
    --task_name $TASK_NAME --batch_size 32  --predictor_lr 1e-3  --sparse_ratio $SR  --epoch_num 10 \
    --max_seq_length 128 --cache_dir ~/.cache/torch/transformers  \
    --output_dir head_prune_${TASK_NAME}_output_${seed}_SR$SR/
done 
