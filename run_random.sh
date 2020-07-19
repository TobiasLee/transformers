TASK_NAME="sst-2"
TASK_DATA="SST-2"
DATA_DIR=glue_data
BSZ=16
MAX_LEN=128
EPOCH=3.0
LR=2e-5
GPU="0,1,2,3"
BIG_MODEL="roberta"
SMALL_MODEL="roberta"
for EPOCH in 3.0 5.0 # 6.0 8.0 10.0 #2.0 3.0 5.0
do
for LR in 2e-5 1e-5 #5e-5 1e-4
do
for seed in 1234 2345 3456 #4567 5678
do
CUDA_VISIBLE_DEVICES=$GPU python run_glue_random_path.py \
  --base_model_name_or_path distilroberta-base   \
  --large_model_name_or_path roberta-base --cache_dir ~/.torch/transformers/mixed/ \
  --base_model_handler $SMALL_MODEL  --large_model_handler $BIG_MODEL \
  --mixed_model_name_or_path mixed-roberta \
  --task_name $TASK_NAME \
  --do_train --save_total_limit 5 --save_steps 1000\
  --do_eval \
  --data_dir $DATA_DIR/$TASK_DATA \
  --max_seq_length $MAX_LEN \
  --per_device_train_batch_size $BSZ \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --output_dir results/random_path_${BIG_MODEL}-distil${SMALL_MODEL}_${TASK_NAME}_epoch${EPOCH}_LR${LR}_BSZ${BSZ}_LEN${MAX_LEN}_seed$seed/ \
  --fp16 --seed $seed
done
done 
done 
