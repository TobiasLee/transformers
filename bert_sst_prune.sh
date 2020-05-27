TASK_NAME="sst-2"
DATA_DIR=glue_data
GPU=3
for seed in 1234 # 2345 3456 4567 5678
do
CUDA_VISIBLE_DEVICES=$GPU python ./examples/bertology/run_bertology.py \
    --data_dir $DATA_DIR/SST-2 --model_name_or_path  "${TASK_NAME}_output_$seed"  \
    --task_name $TASK_NAME --batch_size 16 --head_num 80 --masking_amount 0.0001 \
    --max_seq_length 128 --cache_dir ~/.cache/torch/transformers \
    --output_dir ${TASK_NAME}_compare_output_$seed/ \
    --try_masking
done
