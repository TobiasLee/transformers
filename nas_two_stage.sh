export GLUE_DIR=/home/lilei/transformers/glue_data
export TASK_NAME=SST-2
export TASK_DATA="sst-2"
export GPU="0" # ,1,2,3" # ,1,2,3" #,1,2,3"
EXIT_EPOCH=5
AGENT_EPOCH=8
BSZ=32
EXIT_LR=5e-5
AGENT_LR=5e-5
for SEED in 1234 #2345 3456
do
OUTPUT_PATH=results/${TASK_NAME}_theseus_linear_${SEED}_bsz64
for pr in 0.001 0.005 0.01 0.05 # 0.1  0.2 0.5  2.0 5.0 10.0  # 10.0 50.0 100.0 1000.0
do
FIRST_STAGE_OUTPUT=${OUTPUT_PATH}_nas_penalty_ratio${pr}_epoch${EPOCH}_exit_lr${EXIT_LR}_fix_scc_early_exit_first_stage/
CUDA_VISIBLE_DEVICES=$GPU python ./run_glue_theseus.py \
  --model_name_or_path  ${OUTPUT_PATH} \
  --task_name $TASK_NAME --seed $SEED --fp16 --switch_mode --first_stage --fix_scc_layer --early_exit  \
  --do_eval --do_train \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ \
  --per_device_eval_batch_size $BSZ \
  --learning_rate $EXIT_LR \
  --save_steps 1000 --save_total_limit 5 \
  --num_train_epochs $EXIT_EPOCH --path_penalty_ratio $pr \
  --output_dir "$FIRST_STAGE_OUTPUT" \
  --evaluate_during_training \
  --replacing_rate 0.3 \
  --scheduler_type linear \
  --scheduler_linear_k 0.0006

SECOND_STAGE_OUTPUT=${OUTPUT_PATH}_nas_penalty_ratio${pr}_epoch${EPOCH}_agent_lr${AGENT_LR}_fix_scc_early_exit_second_stage/
CUDA_VISIBLE_DEVICES=$GPU python ./run_glue_theseus.py \
  --model_name_or_path "$FIRST_STAGE_OUTPUT" \
  --task_name $TASK_NAME --seed $SEED --fp16 --switch_mode --second_stage --fix_scc_layer --early_exit  \
  --do_eval --do_train \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ \
  --per_device_eval_batch_size $BSZ \
  --learning_rate $AGENT_LR \
  --save_steps 1000 --save_total_limit 5 \
  --num_train_epochs $AGENT_EPOCH --path_penalty_ratio $pr \
  --output_dir  "$SECOND_STAGE_OUTPUT" \
  --evaluate_during_training \
  --replacing_rate 0.3 \
  --scheduler_type linear \
  --scheduler_linear_k 0.0006

#done
done
done



