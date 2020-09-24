export GLUE_DIR=20NG_data
GPU="0,1,2,3,4,5"
task="twentyng"
task_data="cleaned"
LOSS_TYPE="el"
BSZ=32
LR=2e-5
for seed in 2345 3456 #1234 #2 345 3456 #  4567 5678
do
for EPOCH in 200.0 # 5.0 7.0 10.0 # 5.0 7.0 10.0
do
for BSZ in 16 #  32 #16 32  #64 
do
for LR in 2e-5 #5e-5 2e-5 1e-4 # 5e-5 1e-4  1e-3
do
for BETA in 0.999 # 0.9999 #0.9999
do
for GMA in -1.0 #  1.0 2.0 3.0 4.0 #  0.1 0.2 0.5 1.0 1.5 2.0 # 1.5 2.0  
do
for DEFER_START in 80 160 #0 20 40 80 160 # 0 
do
OUTPUT_PATH=el_results/${task}_output_${seed}_${LOSS_TYPE}_gamma${GMA}_beta${BETA}_uncased_epoch${EPOCH}_lr${LR}_bsz${BSZ}_START${DEFER_START}/ 
CUDA_VISIBLE_DEVICES=$GPU python run_el_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $task  \
  --save_total_limit 10 --save_steps 1000  --el_start $DEFER_START --el_end $EPOCH  \
  --do_eval --do_train --loss_type $LOSS_TYPE --el_gamma $GMA --el_beta $BETA \
  --data_dir $GLUE_DIR/$task_data \
  --max_seq_length 128 \
  --per_device_train_batch_size $BSZ \
  --per_device_eval_batch_size 32 \
  --learning_rate $LR  \
  --num_train_epochs $EPOCH \
  --output_dir $OUTPUT_PATH \
  --fp16 --seed $seed
done
done
done
done 
done
done 
done 
