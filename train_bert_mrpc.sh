export GLUE_DIR=glue_data
GPU="5,6"
for seed in 1234 2345 3456 4567 5678
do
CUDA_VISIBLE_DEVICES=$GPU python examples/text-classification/run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name MRPC \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/MRPC/ \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir mrpc_output_$seed/ \
  --fp16 --seed $seed 
done
