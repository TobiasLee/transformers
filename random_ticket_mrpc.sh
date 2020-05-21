export GLUE_DIR=glue_data
GPU="0"
for seed in 1234 2345 3456 4567 5678
do
CUDA_VISIBLE_DEVICES=$GPU run_glue_mask.py \
  --model_name_or_path bert-base-cased \
  --mask_file "mrpc_random_ticket.npy" \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/MRPC/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 --save_steps 1000 --save_total_limit 5 \
  --num_train_epochs 3.0 \
  --output_dir random_ticket_mrpc_output_$seed/ \
  --fp16 --seed $seed 
done
