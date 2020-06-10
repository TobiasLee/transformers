OUTPUT_DIR=germeval-model
BATCH_SIZE=32
NUM_EPOCHS=3
SAVE_STEPS=750
MAX_LENGTH=128
DATA_DIR="ner_data/data_germ_14"
BERT_MODEL=bert-base-multilingual-cased
GPU="7"
HEAD_NUM=64

for SEED in 1234 2345 3456 4567 5678  
do
CUDA_VISIBLE_DEVICES=$GPU python3 examples/token-classification/run_ner_head_prune.py --data_dir $DATA_DIR \
--labels $DATA_DIR/labels.txt --head_num $HEAD_NUM  --per_iter_mask  2 \
--model_name_or_path  ckpt/${OUTPUT_DIR}_seed$SEED \
--output_dir result/random_mask_NER_genereval_${SEED}_head_num$HEAD_NUM \
--max_seq_length  $MAX_LENGTH \
--per_device_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--random_masking 

done 
