export DATA_PATH=/home/lilei/transformers/data/
export TASK_NAME=lanco
GPU="0,1,2,3,4,5"

#    --do_train  --evaluate_during_training \
CUDA_VISIBLE_DEVICES=$GPU python ./examples/run_glue.py \
    --model_type bert \
    --model_name_or_path bert-large-uncased \
    --task_name $TASK_NAME \
    --no_cuda \
    --do_eval --eval_all_checkpoints \
    --data_dir /home/lilei/transformers/data/lanco  \
    --max_seq_length 16 \
    --per_gpu_eval_batch_size=32  --save_total_limit 20 --save_steps 4000 \
    --fp16 \
    --per_gpu_train_batch_size=32   \
    --learning_rate 1e-5 \
    --num_train_epochs 5.0 \
    --output_dir $TASK_NAME
