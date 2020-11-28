GLUE_DIR=glue_data_dif
GPU="0,1,2,3"
task=mnli-difaware #mnli-dif
TASK_DATA=mnli-multitask # -mtl-aug
# -multitask-2cls #  mnli
EPOCH=5.0
LR=2e-5
BSZ=64
WU=0
for seed in 666 # 2345 3456 4567 5678
do
#for EPOCH in 1 2 #5 7 9 10 15 20 30  
#do
for half_idx in 0 1
do
for layer in 12 
do
SMALL_MODEL=/home/lilei/small-bert/uncased_L-${layer}_H-768_A-12
SAVE_PATH=normal_result/${task}_e${EPOCH}_lr${LR}_bsz${BSZ}_${seed}_wu${WU}_half${half_idx}_${layer}L 
#no_awarenew_sampling
echo $SMALL_MODEL
echo "Half Idx:${half_idx}"
CUDA_VISIBLE_DEVICES=$GPU python run_dif_aware.py --half_index $half_idx --only_ce  \
  --model_name_or_path bert-base-uncased --fp16  \
  --task_name $task  \
  --save_total_limit 10 --save_steps 1000 --logging_steps 1000 --warmup_steps $WU --evaluate_during_training\
  --do_train  --do_eval --do_predict \
  --data_dir $GLUE_DIR/$TASK_DATA  --predict_file "train" --per_device_eval_batch_size $BSZ \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ --logging_dir ${SAVE_PATH}/logs \
  --learning_rate $LR \
  --num_train_epochs $EPOCH --overwrite_output_dir \
  --output_dir  $SAVE_PATH \
  --seed $seed 
done
done 
done

