# For compression with a constant replacing rate
export GLUE_DIR=/home/lilei/transformers/glue_data
export TASK_NAME=qqp
export TASK_DATA=QQP
SEED=1234
export PRE_PATH=/home/lilei/BERT-of-Theseus/sst-2_output_$SEED/
GPU="0,1,2,3,4,5"
  # $PRE_PATH  \
#for idx in $(seq 0 63) #62 60 56 48 32 63 # 0 1 3 7 15 31 
#
#do
FINE_TUNED_PATH=results/${TASK_NAME}_output_$SEED
MULTIPLE_OUTPUT_PATH=multiple_result/${TASK_NAME}_multiple_linear_${SEED}
CUDA_VISIBLE_DEVICES=$GPU python ./run_multiple_distillation.py \
  --model_name_or_path $FINE_TUNED_PATH \
  --task_name $TASK_NAME --seed $SEED --fp16 \
  --do_eval --do_train  \
  --data_dir "$GLUE_DIR/$TASK_DATA" \
  --max_seq_length 128 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --save_steps 500 --save_total_limit 10  \
  --num_train_epochs 10 \
  --output_dir $MULTIPLE_OUTPUT_PATH \
  --replacing_rate 0.3 \
  --scheduler_type linear \
  --scheduler_linear_k 0.0006


for idx in -1 0 1 2  # 2 3 4 5
do
echo "selected model $idx"
CUDA_VISIBLE_DEVICES=$GPU python ./run_multiple_distillation.py \
  --model_name_or_path $MULTIPLE_OUTPUT_PATH \
  --task_name $TASK_NAME --seed $SEED --fp16 \
  --do_eval --small_model_index $idx  \
  --data_dir "$GLUE_DIR/$TASK_DATA" \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --save_steps 500 --save_total_limit 10  \
  --num_train_epochs 10 \
  --output_dir ${MULTIPLE_OUTPUT_PATH}/model_idx$idx \
  --replacing_rate 0.3 \
  --scheduler_type linear \
  --scheduler_linear_k 0.0006
done 
