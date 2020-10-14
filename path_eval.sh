export GLUE_DIR=/home/lilei/transformers/glue_data
export TASK_NAME=SST-2
export TASK_DATA="SST-2-3200"
export GPU="6"
EXIT_EPOCH=30.0
AGENT_EPOCH=900.0
BSZ=32
EXIT_LR=2e-5 
# set train_early_exit to 1 to train early exit 
train_early_exit=0
train_agent=1
AGENT_LR=5e-5
for SEED in 1234 #2345 #1234  # 3456 # $2345 #1234 #2345 3456
do
OUTPUT_PATH=successor_result/${TASK_NAME}_successor_linear_${SEED}_bsz64

#FIRST_STAGE_OUTPUT=${OUTPUT_PATH}_nas_epoch${EXIT_EPOCH}_exit_lr${EXIT_LR}_fix_scc_early_exit_all-large-decay-fix_trained
FIRST_STAGE_OUTPUT=${OUTPUT_PATH}_nas_epoch${EXIT_EPOCH}_exit_lr${EXIT_LR}_fix_scc_early_exit_mixed-decay-fix_trained
if [ $train_early_exit -gt 0 ]
then

CUDA_VISIBLE_DEVICES=$GPU python ./run_glue_theseus.py \
  --model_name_or_path  $OUTPUT_PATH --init_highway \
  --task_name $TASK_NAME --seed $SEED --fp16 --switch_mode --logging_steps 100 --train_early_exit --fix_scc_layer --early_exit  \
  --do_eval --do_train \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ \
  --per_device_eval_batch_size $BSZ \
  --learning_rate $EXIT_LR \
  --save_steps 1000 --save_total_limit 5 \
  --num_train_epochs $EXIT_EPOCH  \
  --output_dir "${FIRST_STAGE_OUTPUT}" \
  --evaluate_during_training \
  --replacing_rate 0.3 \
  --scheduler_type linear \
  --scheduler_linear_k 0.0006

# code for eval learned early exit 
for early_idx in $(seq 0 5)
do

CUDA_VISIBLE_DEVICES=$GPU python ./run_glue_theseus.py \
  --model_name_or_path  $FIRST_STAGE_OUTPUT\
  --task_name $TASK_NAME --seed $SEED --fp16 --switch_mode --logging_steps 50 --train_early_exit --fix_scc_layer --early_exit  \
  --do_eval --early_exit_idx $early_idx \
  --data_dir "$GLUE_DIR/$TASK_NAME" \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ \
  --per_device_eval_batch_size $BSZ \
  --learning_rate $EXIT_LR \
  --save_steps 1000 --save_total_limit 5 \
  --num_train_epochs $EXIT_EPOCH  \
  --output_dir "${FIRST_STAGE_OUTPUT}/$early_idx" \
  --evaluate_during_training \
  --replacing_rate 0.3 \
  --scheduler_type linear \
  --scheduler_linear_k 0.0006
done 
fi 

if [ $train_agent -gt 0 ]
then

CL_INIT=0 # 5 # only learn the last layer  switch
CL_IDX=-1 # 5
CL_INTERVAL=2 
WU_STEPS=8000

echo "Running Path Evaluation"
for PATH_IDX in $(seq 1 126)
do
echo $PATH_IDX 
SECOND_STAGE_OUTPUT=${FIRST_STAGE_OUTPUT}_path_idx_3200_${PATH_IDX}
CUDA_VISIBLE_DEVICES=$GPU python ./run_glue_theseus.py  --path_idx $PATH_IDX --logging_paths --cl_idx 100  --logging_steps 1000 \
  --model_name_or_path  "$FIRST_STAGE_OUTPUT" --fp16  \
  --task_name $TASK_NAME --seed $SEED  --switch_mode --train_agent --fix_scc_layer --early_exit  \
  --do_predict  \
  --data_dir "$GLUE_DIR/$TASK_DATA" --warmup_steps $WU_STEPS \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ \
  --per_device_eval_batch_size $BSZ \
  --learning_rate $AGENT_LR \
  --save_steps 2000 --save_total_limit 5 \
  --num_train_epochs $AGENT_EPOCH  \
  --output_dir  "$SECOND_STAGE_OUTPUT" \
  --evaluate_during_training \
  --replacing_rate 0.3 \
  --scheduler_type linear \
  --scheduler_linear_k 0.0006

done 


fi 
done



