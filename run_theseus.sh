# For compression with a constant replacing rate
export GLUE_DIR=/home/lilei/transformers/glue_data
export TASK_NAME=SST-2
SEED=1234
export PRE_PATH=/home/lilei/BERT-of-Theseus/sst-2_output_$SEED/
GPU="2,3"
  # $PRE_PATH  \
for idx in $(seq 0 63) #62 60 56 48 32 63 # 0 1 3 7 15 31 
do
CUDA_VISIBLE_DEVICES=$GPU python ./run_glue_theseus.py \
  --model_name_or_path successor_result/${TASK_NAME}_successor_linear_$SEED/ \
  --task_name $TASK_NAME --seed $SEED --fp16 \
  --switch_pattern $idx \
  --do_eval \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --learning_rate 2e-5 \
  --save_steps 500 \
  --num_train_epochs 15 \
  --output_dir successor_result/${TASK_NAME}_successor_linear_${SEED}_idx${idx}/ \
  --evaluate_during_training \
  --replacing_rate 0.3 \
  --scheduler_type linear \
  --scheduler_linear_k 0.0006
done 
