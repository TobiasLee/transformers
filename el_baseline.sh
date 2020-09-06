export GLUE_DIR=glue_data
GPU="5,6,7"
task="sst-2"
task_data="SST-2"
LOSS_TYPE="dl"
DL_GMA=1.0
BSZ=32
for seed in 1234 2345 3456 #  4567 5678
do
CUDA_VISIBLE_DEVICES=$GPU python run_el_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $task \
  --do_train --save_total_limit 10 --save_steps 1000\
  --do_eval --loss_type $LOSS_TYPE --dl_gamma $DL_GMA\
  --data_dir $GLUE_DIR/$task_data \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ \
  --learning_rate 2e-5 \
  --num_train_epochs 4.0 \
  --output_dir ${task}_output_${seed}_${LOSS_TYPE}_GMA${DL_GMA}/ \
  --fp16 --seed $seed
done
