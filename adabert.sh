export GLUE_DIR=glue_data
GPU="4,5,6,7"
TASK_NAME=qqp
TASK_DATA=QQP
BSZ=32
MAX_LEN=128
EPOCH=5.0
LR=2e-5


# bert-base-cased
for seed in 1234 #2345 3456 # 4567 5678
do

SAVE_PATH=ada_results/${TASK_NAME}_${seed}_sep_cls
CUDA_VISIBLE_DEVICES=$GPU python run_adabert.py \
  --model_name_or_path  bert-base-cased  \
  --task_name $TASK_NAME \
  --save_total_limit 5 --save_steps 1000\
  --do_eval --do_train  \
  --data_dir $GLUE_DIR/$TASK_DATA \
  --max_seq_length $MAX_LEN \
  --per_device_train_batch_size $BSZ \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --output_dir $SAVE_PATH \
  --fp16 --seed $seed 

echo "Evaluating Path" 
for idx in 0 1 2 
do
echo $SAVE_PATH 
CUDA_VISIBLE_DEVICES=$GPU python run_adabert.py \
  --model_name_or_path  $SAVE_PATH \
  --task_name $TASK_NAME \
  --model_infer_idx $idx  --save_total_limit 5 --save_steps 1000\
  --do_eval  \
  --data_dir $GLUE_DIR/$TASK_DATA \
  --max_seq_length $MAX_LEN \
  --per_device_train_batch_size $BSZ \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --output_dir $SAVE_PATH/infer$idx \
  --fp16 --seed $seed 
done
done 

