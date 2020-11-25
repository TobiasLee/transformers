GLUE_DIR=glue_data_dif
GPU="3,4,5,6"
task=mnli-difaware #mnli-dif
TASK_DATA=mnli-multitask # -mtl-aug
# -multitask-2cls #  mnli
EPOCH=10.0
LR=5e-5
BSZ=64
WU=0
for seed in 666 # 2345 3456 4567 5678
do
SMALL_MODEL=/home/lilei/small-bert/uncased_L-2_H-768_A-12
SAVE_PATH=dif_aware_result/${task}_e${EPOCH}_lr${LR}_bsz${BSZ}_${seed}_wu${WU}_only2#new_sampling
CUDA_VISIBLE_DEVICES=$GPU python run_dif_aware.py \
  --model_name_or_path $SMALL_MODEL --fp16  \
  --task_name $task  \
  --save_total_limit 10 --save_steps 1000 --logging_steps 500 --warmup_steps $WU --evaluate_during_training\
  --do_train  --do_eval --do_predict \
  --data_dir $GLUE_DIR/$TASK_DATA  --predict_file "eval" \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ --logging_dir ${SAVE_PATH}/logs \
  --learning_rate $LR \
  --num_train_epochs $EPOCH --overwrite_output_dir \
  --output_dir  $SAVE_PATH \
  --seed $seed 
done

