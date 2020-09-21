export GLUE_DIR=/home/lilei/transformers/glue_data
export TASK_NAME=SST-2
export TASK_DATA="sst-2"
export GPU="2" # ,1,2,3" # ,1,2,3" #,1,2,3"
EXIT_EPOCH=10.0
AGENT_EPOCH=100.0
BSZ=32
EXIT_LR=2e-5 
# set train_early_exit to 1 to train early exit 
train_early_exit=0
train_agent=1
CL_IDX=2 # only learn the last layer  switch

AGENT_LR=5e-5
for SEED in 1234 #2345 #1234  # 3456 # $2345 #1234 #2345 3456
do
OUTPUT_PATH=successor_result/${TASK_NAME}_successor_linear_${SEED}_bsz64

FIRST_STAGE_OUTPUT=${OUTPUT_PATH}_nas_epoch${EXIT_EPOCH}_exit_lr${EXIT_LR}_fix_scc_early_exit_all-large-decay-fix_trained

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
for early_idx in $(seq 0 5 )
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
echo "Train switch agent"
for ERROR_PENALTY in -1.0 #0.0 -0.1 -0.5 -1.0 #0.11 0.115 0.12  #0.105  #0.001 0.01 0.1 #0.5 1.0 10.0 #0.1 0.01 0.001  #0.0002 0.0003 0.0005 0.0007 #0.1 0.01 0.001 # 5 #  0.0002 # 0.001 0.0001  #0.0002 #0.0005 0.0007  # 0.0005 0.0009 #  0.001 # 0.01 0.1 1.0 #  0.005 0.01 0.05 # 0.1  0.2 0.5  2.0 5.0 10.0  # 10.0 50.0 100.0 1000.0
do
for BD_ALPHA in 0.8 #-1.0 #0.6 # -1.0 # 0.8 # 0.6 0.5  #  0.8 0.5
do
SECOND_STAGE_OUTPUT=${OUTPUT_PATH}_nas_error_penalty${ERROR_PENALTY}_BD_ALPHA${BD_ALPHA}_epoch${AGENT_EPOCH}_agent_lr${AGENT_LR}_fix_scc_early_exit_second_stage_action3_block-drop-CL${CL_IDX}_no_square_default2/
CUDA_VISIBLE_DEVICES=$GPU python ./run_glue_theseus.py --cl_idx $CL_IDX --bound_alpha $BD_ALPHA --logging_paths  --logging_steps 800 --error_penalty $ERROR_PENALTY  \
  --model_name_or_path  "$FIRST_STAGE_OUTPUT" --fp16  \
  --task_name $TASK_NAME --seed $SEED  --switch_mode --train_agent --fix_scc_layer --early_exit  \
  --do_eval  --do_train \
  --data_dir "$GLUE_DIR/$TASK_NAME" --warmup_steps 8000 \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ \
  --per_device_eval_batch_size 4 \
  --learning_rate $AGENT_LR \
  --save_steps 2000 --save_total_limit 5 \
  --num_train_epochs $AGENT_EPOCH  \
  --output_dir  "$SECOND_STAGE_OUTPUT" \
  --evaluate_during_training \
  --replacing_rate 0.3 \
  --scheduler_type linear \
  --scheduler_linear_k 0.0006

done
done

fi 
done



