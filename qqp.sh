export GLUE_DIR=glue_data
GPU="0,1,2"
task=qqp
for seed in 1234 2345 3456 4567 5678
do
CUDA_VISIBLE_DEVICES=$GPU python examples/text-classification/run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name qqp \
  --do_train --save_total_limit 10 --save_steps 1000\
  --do_eval \
  --data_dir $GLUE_DIR/QQP \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 16 \
  --learning_rate 2e-5 \
  --num_train_epochs 4.0 \
  --output_dir qqp_output_$seed/ \
  --fp16 --seed $seed 
done
DATA_DIR=glue_data
GPU=0
for seed in 1234 2345 3456 4567 5678 # 1234 
do
CUDA_VISIBLE_DEVICES=$GPU python ./examples/bertology/run_bertology.py \
    --data_dir $DATA_DIR/QQP --model_name_or_path  "qqp_output_$seed"  \
    --task_name $task --batch_size 16 \
    --max_seq_length 128 --cache_dir ~/.cache/torch/transformers \
    --output_dir qqp_output_$seed/ \
    --try_masking
done
