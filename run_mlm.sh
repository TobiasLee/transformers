TASK_NAME="wiki-103-mlm"
TASK_DATA="wiki-103-raw"
DATA_DIR=wiki_data/wikitext-103-raw/
BSZ=24
EPOCH=3.0
LR=2e-5
GPU="5,6,7"
BIG_MODEL="roberta"
SMALL_MODEL="roberta"
pattern_idx=-1 # dynamic
NP=3
for tkw in  1.0 # 0.5 0.2 0.1  # 0.5 # 0.2 0.1 # 0.01 # 0.001 # 0.02 0.05
do
for th in 0.5 # 0.7 0.8 0.9 #  0.001 0.1 # 0.01 0.05 #  0.2 0.3 # 0.001 0.01  0.05 0.1 
do
for EPOCH in 6.0 #  5.0 #  7.0 10.0 # 3.0 5.0 # 6.0 8.0 10.0 #2.0 3.0 5.0
do
for LR in 5e-5 # 5e-5 # 1e-5 1e-4 1e-3  #   1e-3 1e-4 #  1e-3 # 1e-5 # 5e-5 #  1e-4
do
for seed in 1234 2345 3456 #1234 2345 3456 # 1234 # 2345 3456 #4567 5678
do
for pattern_idx in 0 7 #$(seq 1 6) #62 #1 3 7 15 31 32 48 56 60 62 #  $(seq 1 6) # 1 2 5 6 # $(seq 1 6) ##62 61 59 55 47 31 # 1 2 4 8 16 32  #$(seq 1 31)
# --iterative_mlm
do
CUDA_VISIBLE_DEVICES=$GPU python run_lm_switch.py --fp16 --mlm   --unfreeze_base_encoder  --tl_kd_weight $tkw --entropy_threshold $th   --freeze_trained_models --switch_pattern_idx $pattern_idx --num_parts $NP  \
  --base_model_name_or_path distilroberta-base   \
  --large_model_name_or_path roberta-base --cache_dir ~/.torch/transformers/mixed/ \
  --base_model_handler $SMALL_MODEL  --large_model_handler $BIG_MODEL --mixed_model_name_or_path mixed-roberta \
  --save_total_limit 3 --save_steps 10000\
   --do_eval  --block_size 128 \
    --train_data_file=$DATA_DIR/wiki.train.raw  \
    --eval_data_file=$DATA_DIR/wiki.test.raw \
  --per_device_train_batch_size $BSZ \
  --per_device_eval_batch_size $BSZ  \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --saved_path results/random_pattern-1-kdtl-unfreeze-base-loaded_${BIG_MODEL}-distil${SMALL_MODEL}_${TASK_NAME}_epoch${EPOCH}_LR${LR}_BSZ${BSZ}_seed${seed}_num_parts${NP}_128\
  --output_dir  results/random_pattern-unfreeze-base-loaded_${BIG_MODEL}-distil${SMALL_MODEL}_${TASK_NAME}_epoch${EPOCH}_LR${LR}_BSZ${BSZ}_seed${seed}_num_parts${NP}_128_idx${pattern_idx}  \
  --seed $seed
done
done
done 
done
done
done 
