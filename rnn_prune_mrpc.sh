TASK_NAME="qqp"
TASK_DATA="QQP"
DATA_DIR=glue_data
GPU=1
SR=0.1
BSZ=16
# --metric_name "mcc" for cola $  --metric_name 'pearson' sts-b
#for SR in 0.0 0.1 0.5 1
#do
for seed in 1234 2345 3456 4567 5678 
do
CUDA_VISIBLE_DEVICES=$GPU python head_prune.py \
    --data_dir $DATA_DIR/$TASK_DATA --model_name_or_path  "results/${TASK_NAME}_output_$seed"  \
    --task_name $TASK_NAME --batch_size $BSZ  --predictor_lr 1e-3  --sparse_ratio $SR  --epoch_num 10 \
    --max_seq_length 128 --cache_dir ~/.cache/torch/transformers  \
    --output_dir results/half_${BSZ}_head_prune_${TASK_NAME}_output_${seed}_SR$SR/
#done 
done 
