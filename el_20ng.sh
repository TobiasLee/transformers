export GLUE_DIR=20NG_data
GPU="2,3,4,5,6,7"
task="twentyng"
task_data="cleaned"
LOSS_TYPE="default"
DL_GMA=1.0
BSZ=32
GMA=1.0
EPOCH=20.0
LR=2e-5

for EPOCH in 200 #120 150 200 #60.0 80.0 100.0 #10.0 15.0 20.0 # 5.0 7.0 10.0 # 5.0 7.0 10.0
do
for BSZ in 16 #  32 #16 32  #64 
do
for LR in 2e-5 #5e-5 2e-5 1e-4 # 5e-5 1e-4  1e-3
do
for BETA in 0.999 # 0.9999
do
for seed in 1234 2345 3456 #1234 # 2345 3456 #1234 #2345 3456 #1234 #2345 3456 #  4567 5678
do
#for LOSS_TYPE in "dl" 
#do
#for GMA in 0.1 0.2 0.5 1.0 1.5 2.0 # 0.5 #0.1 0.2 0.5 1.0 1.5 2.0 # 1.5 2.0  
#do
OUTPUT_PATH=el_results/2L_${task}_${task_data}_output_${seed}_${LOSS_TYPE}_gamma${GMA}_beta${BETA}_uncased_epoch${EPOCH}_lr${LR}_bsz${BSZ}/
CUDA_VISIBLE_DEVICES=$GPU python run_el_glue_2l.py \
  --model_name_or_path bert-base-cased  \
  --task_name $task  \
  --save_total_limit 5 --save_steps 2000 \
  --do_eval --do_train --loss_type $LOSS_TYPE --dl_gamma $GMA --fl_gamma $GMA --rl_beta $BETA --mixed_gamma $GMA  \
  --data_dir $GLUE_DIR/$task_data \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ \
  --learning_rate $LR  \
  --num_train_epochs $EPOCH \
  --output_dir $OUTPUT_PATH  \
  --fp16 --seed $seed
# done
done
done 
done
done 
done 
