export GLUE_DIR=glue_data
GPU="2,3"
TASK_NAME=sst-2
TASK_DATA=SST-2
for LAYER_NUM in 2 4 6 8 12
do
for seed in 1234 #2345 3456 # 4567 5678
do
CUDA_VISIBLE_DEVICES=$GPU python run_glue_nl.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train --save_total_limit 5 --save_steps 1000\
  --do_eval --layer_limit $LAYER_NUM  \
  --data_dir $GLUE_DIR/$TASK_DATA \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir nl_results/${TASK_NAME}_${LAYER_NUM}L_${seed}/ \
  --fp16 --seed $seed 
done
done 

