TASK_NAME="mrpc"
TASK_DATA="MRPC"
DATA_DIR=glue_data
BSZ=4
MAX_LEN=128
EPOCH=3.0
LR=2e-5
GPU="2,3,4,5"
for seed in 1234 #2345 3456 4567 5678
do
CUDA_VISIBLE_DEVICES=$GPU python run_glue_mixed.py \
  --base_model_name_or_path bert-base-uncased \
  --large_model_name_or_path bert-large-uncased --cache_dir ~/.torch/transformers/mixed/ \
  --mixed_model_name_or_path mixed-bert \
  --task_name $TASK_NAME \
  --do_train --save_total_limit 10 --save_steps 1000\
  --do_eval \
  --data_dir $DATA_DIR/$TASK_DATA \
  --max_seq_length $MAX_LEN \
  --per_gpu_train_batch_size $BSZ \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --output_dir mixed_${TASK_NAME}_epoch${EPOCH}_LR${LR}_BSZ${BSZ}_LEN${MAX_LEN}_seed$seed/ \
  --fp16 --seed $seed
done
