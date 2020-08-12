TASK_NAME="sst-2"
TASK_DATA="SST-2"
DATA_DIR=glue_data
BSZ=64
MAX_LEN=128
EPOCH=3.0
LR=2e-5
GPU="0,1,2,3"
BIG_MODEL="roberta"
SMALL_MODEL="roberta"
pattern_idx=-1 # dynamic
NP=3
for tkw in 1.0 # 0.5 0.2 0.1 # 0.01 # 0.001 # 0.02 0.05
do
for th in 0.5 # 0.7 0.8 0.9 #  0.001 0.1 # 0.01 0.05 #  0.2 0.3 # 0.001 0.01  0.05 0.1 
do
for EPOCH in 3.0 # 5.0  # 5.0 #  7.0 10.0 # 3.0 5.0 # 6.0 8.0 10.0 #2.0 3.0 5.0
do
for LR in  5e-5 # 1e-4 5e-5 #1e-3 1e-4 #  1e-3 # 1e-5 # 5e-5 #  1e-4
do
for seed in 1234 2345 3456 # 1234 # 2345 3456 #4567 5678
do
for pattern_idx in 1 #2 3 4  # $(seq 1 6) #1 3 7 15 31 32 48 56 60 62 #62 61 59 55 47 31 # 1 2 4 8 16 32  #$(seq 1 31) --non_linear_tl 
do
CUDA_VISIBLE_DEVICES=$GPU python run_glue_nas.py --fp16  --tl_kd_weight $tkw --entropy_threshold $th --unfreeze_base_encoder  --freeze_trained_models --switch_pattern_idx $pattern_idx --num_parts $NP  \
  --base_model_name_or_path results/${TASK_NAME}_distilroberta-base_$seed   \
  --large_model_name_or_path results/${TASK_NAME}_roberta-base_$seed --cache_dir ~/.torch/transformers/mixed/ \
  --base_model_handler $SMALL_MODEL  --large_model_handler $BIG_MODEL --mixed_model_name_or_path mixed-roberta \
  --task_name $TASK_NAME \
  --saved_path results/fix-pattern1-unfreeze-base_roberta-distilroberta_wiki-2-mlm_epoch3.0_LR5e-5_BSZ8_LEN128_seed${seed}_num_parts3 \
  --save_total_limit 3 --save_steps 5000\
   --do_eval  --do_train  \
  --data_dir $DATA_DIR/$TASK_DATA \
  --max_seq_length $MAX_LEN \
  --per_device_train_batch_size $BSZ \
  --per_device_eval_batch_size $BSZ  \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --output_dir   results/fix-pattern${pattern_idx}-unfreeze-base-after-mlm_${BIG_MODEL}-distil${SMALL_MODEL}_${TASK_NAME}_epoch${EPOCH}_LR${LR}_BSZ${BSZ}_LEN${MAX_LEN}_seed${seed}_num_parts${NP} \
  --seed $seed
done
done
done 
done
done
done 
