export GLUE_DIR=glue_data_dif_bal
GPU="2"
task=mnli-dif-cls #mnli-dif
TASK_DATA=mnli #  mnli
EPOCH=30.0
LR=1e-3
BSZ=64
WU=2000
AGENT="transformer"
for seed in 1234 # 2345 3456 4567 5678
do
SAVE_PATH=dif_result/${task}_e${EPOCH}_lr${LR}_bsz${BSZ}_${seed}_balance_wu${WU}_cls_${AGENT}
CUDA_VISIBLE_DEVICES=$GPU python run_diff.py \
  --model_name_or_path bert-base-uncased \
  --task_name $task \
  --save_total_limit 5 --save_steps 2000 --logging_steps 500 --warmup_steps $WU --evaluate_during_training \
  --do_train --do_eval --do_predict \
  --data_dir $GLUE_DIR/$TASK_DATA  \
  --max_seq_length 128 --agent_type $AGENT  \
  --per_device_train_batch_size $BSZ --logging_dir ${SAVE_PATH}/logs \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --output_dir  $SAVE_PATH \
  --seed $seed 
done

