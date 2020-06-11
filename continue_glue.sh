export GLUE_DIR=glue_data
TASK_NAME="mnli"
TASK_DATA="MNLI"
GPU="2"
SR=0.128
BSZ=16
for seed in 1234 2345 3456 4567 5678
do
CUDA_VISIBLE_DEVICES=$GPU python3 run_glue_mask.py \
  --model_name_or_path  "results/${TASK_NAME}_output_$seed" \
  --mask_file "results/half_${BSZ}_head_prune_${TASK_NAME}_output_${seed}_SR$SR/learned_mask.npy"\
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_DATA \
  --max_seq_length 128 \
  --per_gpu_train_batch_size $BSZ \
  --learning_rate 1e-5 --save_steps 1000 --save_total_limit 5 \
  --num_train_epochs 2.0 \
  --output_dir results/continue_training_${TASK}_BSZ${BSZ}_output_$seed/ \
  --fp16 --seed $seed
done
