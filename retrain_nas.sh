TASK="sts-b"
TASK_DATA="STS-B"
export GLUE_DIR=glue_data
GPU="2"
for seed in 1234 2345 3456 4567 5678
do
CUDA_VISIBLE_DEVICES=$GPU python3 run_glue_mask.py \
  --model_name_or_path bert-base-cased \
  --mask_file "results/head_prune_${TASK}_output_${seed}_SR0.1/learned_mask.npy"\
  --task_name $TASK \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_DATA \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32\
  --learning_rate 2e-5 --save_steps 1000 --save_total_limit 5 \
  --num_train_epochs 3.0 \
  --output_dir results/retrain_nas_${TASK}_output_$seed/ \
  --fp16 --seed $seed 
done
