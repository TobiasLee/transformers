TASK_NAME="mnli"
TASK_DATA="MNLI"
HEAD_NUM=88
DATA_DIR=glue_data
GPU=1
for seed in 1234 2345 3456 4567 5678 # 1234 
do
CUDA_VISIBLE_DEVICES=$GPU python ./examples/bertology/run_bertology.py \
    --data_dir $DATA_DIR/$TASK_DATA --head_num $HEAD_NUM  --per_iter_mask 2 --model_name_or_path  "results/${TASK_NAME}_output_$seed" \
    --task_name $TASK_NAME --batch_size 16  \
    --max_seq_length 128 --cache_dir ~/.cache/torch/transformers \
    --output_dir results/${TASK_NAME}_importance_score_output_$seed/ \
    --try_masking
done
