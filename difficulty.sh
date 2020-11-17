export GLUE_DIR=glue_data_dif
GPU="0"
task=sst2-dif #mnli-dif
TASK_DATA=sst2 #  mnli
EPOCH=30.0
LR=1e-3
BSZ=64
WU=1000
AGENT="bow"
for seed in 1234 # 2345 3456 4567 5678
do
SAVE_PATH=dif_result/${task}_e${EPOCH}_lr${LR}_bsz${BSZ}_${seed}_unbalance_wu${WU}_regression_${AGENT}
CUDA_VISIBLE_DEVICES=$GPU python run_diff.py \
  --model_name_or_path bert-base-uncased --agent_type $AGENT\
  --task_name $task \
  --save_total_limit 10 --save_steps 1000 --logging_steps 500 --warmup_steps $WU --evaluate_during_training \
  --do_train --do_eval --do_predict \
  --data_dir $GLUE_DIR/$TASK_DATA  \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ --logging_dir ${SAVE_PATH}/logs \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --output_dir  $SAVE_PATH \
  --seed $seed 
done

