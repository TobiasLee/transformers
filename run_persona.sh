export DATA_DIR=./persona
export TASK_NAME=persona
GPU="4,5,6"
MODEL="roberta"
CUDA_VISIBLE_DEVICES=$GPU python persona_classifier.py \
  --model_type $MODEL \
  --model_name_or_path roberta-large \
  --task_name $TASK_NAME --fp16 \
  --do_eval --do_train --should_continue --overwrite_output_dir \
  --data_dir $DATA_DIR \
  --max_seq_length 32 \
  --per_gpu_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 5.0 \
  --output_dir ckpt/$TASK_NAME-$MODEL-large
