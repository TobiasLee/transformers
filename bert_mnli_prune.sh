TASK_NAME="mnli"
DATA_DIR=glue_data
GPU=7
for seed in 1234 # 2345 3456 4567 5678 # 1234 
do
CUDA_VISIBLE_DEVICES=$GPU python ./examples/bertology/run_bertology.py \
    --data_dir $DATA_DIR/MNLI --model_name_or_path  "mnli_output_$seed"  \
    --task_name $TASK_NAME --batch_size 16 \
    --max_seq_length 128 --cache_dir ~/.cache/torch/transformers \
    --output_dir mnli_output_$seed/ \
    --try_masking
done
