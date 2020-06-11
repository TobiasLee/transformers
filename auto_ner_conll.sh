OUTPUT_DIR=conll2003-en-model  #germeval-model
BATCH_SIZE=32
NUM_EPOCHS=3
SAVE_STEPS=750
MAX_LENGTH=128
DATA_DIR="ner_data/conll-2003"
BERT_MODEL=bert-base-cased
GPU="0"


for SEED in 1234 2345 3456 4567 5678  
do
CUDA_VISIBLE_DEVICES=$GPU python3 examples/token-classification/auto_prune.py --data_dir $DATA_DIR \
--labels $DATA_DIR/labels.txt --metric_name 'f1'\
--model_name_or_path   ckpt/${OUTPUT_DIR}_seed$SEED \
--output_dir result/auto_prune_NER_conll2003en_${SEED} \
--max_seq_length  $MAX_LENGTH \
--per_device_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--sparse_ratio 0.1 --epoch_num 10 --predictor_lr 1e-3 

done 
