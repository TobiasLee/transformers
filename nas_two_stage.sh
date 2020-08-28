export GLUE_DIR=/home/lilei/transformers/glue_data
export TASK_NAME=SST-2
export TASK_DATA="sst-2"
export GPU="3" # ,1,2,3" # ,1,2,3" #,1,2,3"
EXIT_EPOCH=1.0
AGENT_EPOCH=8.0
BSZ=32
EXIT_LR=5e-5

train_early_exit=1
train_agent=0

AGENT_LR=2e-5
for SEED in 1234 # 3456 # $2345 #1234 #2345 3456
do
OUTPUT_PATH=successor_result/${TASK_NAME}_successor_linear_${SEED}_bsz64

FIRST_STAGE_OUTPUT=${OUTPUT_PATH}_nas_epoch${EXIT_EPOCH}_exit_lr${EXIT_LR}_fix_scc_early_exit_trained/
if [ $train_early_exit -gt 0 ]
then



CUDA_VISIBLE_DEVICES=$GPU python ./run_glue_theseus.py \
  --model_name_or_path  ${OUTPUT_PATH} \
  --task_name $TASK_NAME --seed $SEED --fp16 --switch_mode --train_early_exit --fix_scc_layer --early_exit  \
  --do_eval --do_train \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ \
  --per_device_eval_batch_size $BSZ \
  --learning_rate $EXIT_LR \
  --save_steps 1000 --save_total_limit 5 \
  --num_train_epochs $EXIT_EPOCH  \
  --output_dir "$FIRST_STAGE_OUTPUT" \
  --evaluate_during_training \
  --replacing_rate 0.3 \
  --scheduler_type linear \
  --scheduler_linear_k 0.0006
fi 

if [ $train_agent -gt 0 ]
then

for pr in  0.0002 0.0003 0.0005 0.0007 #0.1 0.01 0.001 # 5 #  0.0002 # 0.001 0.0001  #0.0002 #0.0005 0.0007  # 0.0005 0.0009 #  0.001 # 0.01 0.1 1.0 #  0.005 0.01 0.05 # 0.1  0.2 0.5  2.0 5.0 10.0  # 10.0 50.0 100.0 1000.0
do
SECOND_STAGE_OUTPUT=${OUTPUT_PATH}_nas_penalty_ratio${pr}_epoch${EPOCH}_agent_lr${AGENT_LR}_fix_scc_early_exit_second_stage/
CUDA_VISIBLE_DEVICES=$GPU python ./run_glue_theseus.py \
  --model_name_or_path  "$SECOND_STAGE_OUTPUT"  \
  --task_name $TASK_NAME --seed $SEED  --switch_mode --train_agent --fix_scc_layer --early_exit  \
  --do_eval --fp16 \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ \
  --per_device_eval_batch_size 1\
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

fi 
done



