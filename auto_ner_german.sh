OUTPUT_DIR=germeval-model
BATCH_SIZE=32
NUM_EPOCHS=3
SAVE_STEPS=750
MAX_LENGTH=128
DATA_DIR="ner_data/data_germ_14"
BERT_MODEL=bert-base-multilingual-cased
GPU="0"
PREDICTOR="mlp" # hc 
for SR in 0.0 0.05 0.1 0.2 0.3 0.5 1 1.5 2 3 5 10 20 50 100 1000 1000  
do 
#for MODEL_SEED in 1234 2345 3456 4567 5678
#do   --seed $SEED \
for SEED in 1234 2345 3456 4567 5678  
do
CUDA_VISIBLE_DEVICES=$GPU python3 examples/token-classification/auto_prune.py --data_dir $DATA_DIR \
--labels $DATA_DIR/labels.txt \
--model_name_or_path   ckpt/${OUTPUT_DIR}_seed$SEED \
--output_dir result/auto_prune_NER_german_model_seed${SEED}_${PREDICTOR}_SR$SR \
--max_seq_length  $MAX_LENGTH \
--per_device_train_batch_size $BATCH_SIZE --predictor $PREDICTOR \
--sparse_ratio $SR --epoch_num 10 --predictor_lr 1e-3 
done 
done
#done 
